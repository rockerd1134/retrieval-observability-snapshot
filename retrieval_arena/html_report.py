from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import ValidationError


HTML_REPORT_SCHEMA_VERSION = "retrieval_arena.html_observability_report.v1"


def build_html_observability_report(
    bundle_root: Path,
    out_path: Path | None = None,
    *,
    created_at: str | None = None,
) -> dict[str, Any]:
    bundle_root = bundle_root.resolve()
    if not bundle_root.is_dir():
        raise ValidationError(f"Report bundle root not found: {bundle_root}")
    report = assemble_report_data(bundle_root, created_at=created_at)
    output = out_path or bundle_root / "observability_report.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_html_report(report), encoding="utf-8")
    return {
        "ok": True,
        "summary": f"HTML observability report wrote {output}",
        "report": report,
        "written_artifacts": [str(output)],
        "report_path": output,
    }


def assemble_report_data(bundle_root: Path, *, created_at: str | None = None) -> dict[str, Any]:
    pilot_manifest = _read_json(bundle_root / "pilot_manifest.json")
    plan = _read_json(bundle_root / "plan_resolved.json")
    snapshot_diff = _read_json(bundle_root / "snapshot_comparison" / "snapshot_diff.json")
    before_corpus = _read_json(bundle_root / "snapshot_comparison" / "before" / "snapshot_manifests" / "corpus_snapshot_manifest.json")
    after_corpus = _read_json(bundle_root / "snapshot_comparison" / "after" / "snapshot_manifests" / "corpus_snapshot_manifest.json")
    before_graph = _read_json(bundle_root / "snapshot_comparison" / "before" / "snapshot_manifests" / "graph_snapshot_manifest.json")
    after_graph = _read_json(bundle_root / "snapshot_comparison" / "after" / "snapshot_manifests" / "graph_snapshot_manifest.json")
    before_support = _read_json(bundle_root / "snapshot_comparison" / "before" / "snapshot_manifests" / "support_surface_manifest.json")
    after_support = _read_json(bundle_root / "snapshot_comparison" / "after" / "snapshot_manifests" / "support_surface_manifest.json")
    comparisons = _collect_comparisons(bundle_root)
    data = {
        "schema_version": HTML_REPORT_SCHEMA_VERSION,
        "created_at": created_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "bundle_root": str(bundle_root),
        "pilot_id": pilot_manifest.get("pilot_id") or plan.get("pilot_id") or bundle_root.name,
        "corpus_id": pilot_manifest.get("corpus_id") or plan.get("corpus_id") or snapshot_diff.get("before", {}).get("corpus_id"),
        "query_set_id": pilot_manifest.get("query_set_id") or plan.get("query_set_id"),
        "pilot_manifest": pilot_manifest,
        "plan": plan,
        "snapshot_diff": snapshot_diff,
        "snapshot_manifests": {
            "before_corpus": before_corpus,
            "after_corpus": after_corpus,
            "before_graph": before_graph,
            "after_graph": after_graph,
            "before_support": before_support,
            "after_support": after_support,
        },
        "comparisons": comparisons,
    }
    data["insights"] = _derive_insights(data)
    return data


def render_html_report(report: dict[str, Any]) -> str:
    snapshot = report.get("snapshot_diff", {})
    summary = snapshot.get("summary", {}) if isinstance(snapshot.get("summary"), dict) else {}
    manifests = report.get("snapshot_manifests", {})
    comparisons = report.get("comparisons", [])
    before = manifests.get("before_corpus", {}) if isinstance(manifests, dict) else {}
    after = manifests.get("after_corpus", {}) if isinstance(manifests, dict) else {}
    title = f"{report.get('corpus_id') or 'Corpus'} Observability Report"
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>{_esc(title)}</title>",
            f"<style>{_css()}</style>",
            "</head>",
            "<body>",
            "<main>",
            '<section class="hero">',
            '<div class="eyebrow">Retrieval Audit Framework anonymous review</div>',
            f"<h1>{_esc(title)}</h1>",
            f"<p>{_esc(str(report.get('pilot_id') or 'unknown pilot'))}</p>",
            '<div class="metric-row">',
            _metric("Snapshots", f"{before.get('snapshot_id', 'unknown')} -> {after.get('snapshot_id', 'unknown')}"),
            _metric("Queries", _value_from_comparison(comparisons, "query_count") or report.get("query_set_id") or "unknown"),
            _metric("Families", len(comparisons)),
            _metric("Generated", report.get("created_at")),
            "</div>",
            "</section>",
            _section("Generated Insights", _insight_list(report.get("insights", []))),
            _section("Corpus Snapshot Summary", _snapshot_summary(before, after, summary)),
            _section("Graph And Support Drift", _graph_support(snapshot, summary, manifests)),
            _section("Retrieval Family Matrix", _family_matrix(comparisons)),
            _section("Retrieval Drift Metrics", _drift_table(comparisons)),
            _section("Regression Audit Labels", _audit_table(comparisons)),
            _section("Query Case Studies", _case_studies(comparisons)),
            _section("Systems And Storage Measurements", _systems_table(comparisons)),
            _section("Provenance And Replay", _provenance(report, comparisons)),
            "</main>",
            "</body>",
            "</html>",
        ]
    )


def _collect_comparisons(bundle_root: Path) -> list[dict[str, Any]]:
    comparisons_root = bundle_root / "comparisons"
    if not comparisons_root.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in comparisons_root.iterdir() if item.is_dir()):
        comparison_id = path.name
        drift = _read_json(path / "retrieval_drift_summary.json")
        audit = _read_json(path / "regression_audit_summary.json")
        systems = _read_json(path / "systems_measurements.json")
        before_replay = _read_json(bundle_root / "retrieval" / comparison_id / "before" / "retrieval_replay_manifest.json")
        after_replay = _read_json(bundle_root / "retrieval" / comparison_id / "after" / "retrieval_replay_manifest.json")
        rows.append(
            {
                "comparison_id": comparison_id,
                "paths": {
                    "comparison_dir": _rel(path, bundle_root),
                    "drift_summary": _rel(path / "retrieval_drift_summary.json", bundle_root),
                    "drift_rows": _rel(path / "retrieval_drift.jsonl", bundle_root),
                    "audit_summary": _rel(path / "regression_audit_summary.json", bundle_root),
                    "audit_rows": _rel(path / "regression_audit.jsonl", bundle_root),
                    "systems_measurements": _rel(path / "systems_measurements.json", bundle_root),
                    "before_replay": _rel(bundle_root / "retrieval" / comparison_id / "before" / "retrieval_replay_manifest.json", bundle_root),
                    "after_replay": _rel(bundle_root / "retrieval" / comparison_id / "after" / "retrieval_replay_manifest.json", bundle_root),
                },
                "drift": drift,
                "audit": audit,
                "systems": systems,
                "case_studies": _select_case_studies(path / "regression_audit.jsonl"),
                "before_replay": before_replay,
                "after_replay": after_replay,
            }
        )
    return rows


def _derive_insights(report: dict[str, Any]) -> list[str]:
    summary = report.get("snapshot_diff", {}).get("summary", {})
    comparisons = report.get("comparisons", [])
    insights: list[str] = []
    page_delta = _num(summary.get("page_count_delta"))
    if page_delta:
        insights.append(f"Corpus page count changed by {page_delta:+g} pages across the snapshot pair.")
    changed_pages = _num(summary.get("changed_file_count"))
    if changed_pages:
        insights.append(f"{changed_pages:g} pages changed content hash between snapshots.")
    edge_delta = _num(summary.get("added_edge_count")) - _num(summary.get("removed_edge_count"))
    if edge_delta:
        insights.append(f"Graph links changed by {_num(summary.get('added_edge_count')):g} added and {_num(summary.get('removed_edge_count')):g} removed edges.")
    support_changed = _num(summary.get("changed_support_question_count"))
    if support_changed:
        insights.append(f"Support targets changed for {support_changed:g} questions.")
    if comparisons:
        lowest = min(comparisons, key=lambda row: _num(row.get("drift", {}).get("mean_top_k_jaccard"), default=1.0))
        value = lowest.get("drift", {}).get("mean_top_k_jaccard")
        if value is not None:
            insights.append(f"{lowest['comparison_id']} has the lowest mean top-k Jaccard overlap at {_fmt(value)}.")
        label_counts = [(row["comparison_id"], _sum_labels(row.get("audit", {}).get("cause_label_counts", {}))) for row in comparisons]
        label_counts.sort(key=lambda item: item[1], reverse=True)
        if label_counts and label_counts[0][1]:
            insights.append(f"{label_counts[0][0]} has the most audit labels recorded ({label_counts[0][1]:g}).")
    if not insights:
        insights.append("Required report artifacts were available, but no nonzero drift or audit count crossed the report callout thresholds.")
    return insights


def _section(title: str, body: str) -> str:
    return f'<section><h2>{_esc(title)}</h2>{body}</section>'


def _snapshot_summary(before: dict[str, Any], after: dict[str, Any], summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            '<div class="metric-row">',
            _metric("Before Pages", before.get("page_count")),
            _metric("After Pages", after.get("page_count")),
            _metric("Page Delta", _signed(summary.get("page_count_delta"))),
            _metric("Content Bytes Delta", _signed(summary.get("corpus_size_delta_bytes"))),
            "</div>",
            _table(
                ["Field", "Before", "After"],
                [
                    ["Snapshot ID", before.get("snapshot_id"), after.get("snapshot_id")],
                    ["Source Commit", _short(before.get("source_commit")), _short(after.get("source_commit"))],
                    ["Content Hash", _short(before.get("content_hash")), _short(after.get("content_hash"))],
                    ["Manifest Hash", _short(before.get("manifest_hash")), _short(after.get("manifest_hash"))],
                ],
            ),
            _table(
                ["Corpus Drift", "Count"],
                [
                    ["Added pages", summary.get("added_file_count")],
                    ["Removed pages", summary.get("removed_file_count")],
                    ["Changed pages", summary.get("changed_file_count")],
                    ["Stable manifest differences", summary.get("stable_manifest_difference_count")],
                ],
            ),
        ]
    )


def _graph_support(snapshot: dict[str, Any], summary: dict[str, Any], manifests: dict[str, Any]) -> str:
    before_graph = manifests.get("before_graph", {})
    after_graph = manifests.get("after_graph", {})
    before_support = manifests.get("before_support", {})
    after_support = manifests.get("after_support", {})
    support = snapshot.get("support_surface_result", {}) if isinstance(snapshot.get("support_surface_result"), dict) else {}
    return "\n".join(
        [
            '<div class="metric-row">',
            _metric("Graph Available", summary.get("graph_available")),
            _metric("Support Available", summary.get("support_surface_available")),
            _metric("Added Edges", summary.get("added_edge_count")),
            _metric("Removed Edges", summary.get("removed_edge_count")),
            _metric("Changed Support Questions", summary.get("changed_support_question_count")),
            "</div>",
            _table(
                ["Surface", "Before", "After"],
                [
                    ["Graph nodes", before_graph.get("node_count"), after_graph.get("node_count")],
                    ["Graph edges", before_graph.get("edge_count"), after_graph.get("edge_count")],
                    ["Support targets", before_support.get("support_target_count"), after_support.get("support_target_count")],
                    ["Supported questions", before_support.get("supported_query_count"), after_support.get("supported_query_count")],
                ],
            ),
            _list("Changed support questions", [item.get("question_id") for item in support.get("changed_questions", [])[:8]]),
        ]
    )


def _family_matrix(comparisons: list[dict[str, Any]]) -> str:
    return _table(
        ["Family", "Queries", "Mean Top-k Jaccard", "Support Recall Regressions", "Labeled Audit Rows"],
        [
            [
                row["comparison_id"],
                row.get("drift", {}).get("query_count"),
                _bar(row.get("drift", {}).get("mean_top_k_jaccard")),
                row.get("drift", {}).get("support_recall_regression_count"),
                row.get("audit", {}).get("labeled_query_count"),
            ]
            for row in comparisons
        ],
    )


def _drift_table(comparisons: list[dict[str, Any]]) -> str:
    return _table(
        ["Family", "Ordered Overlap", "Rank Displacement", "Support Exposure Regr.", "Evidence Coverage Regr.", "Action Trace Available"],
        [
            [
                row["comparison_id"],
                _fmt(row.get("drift", {}).get("mean_ordered_top_k_overlap")),
                _fmt(row.get("drift", {}).get("mean_rank_displacement")),
                row.get("drift", {}).get("support_exposure_regression_count"),
                row.get("drift", {}).get("evidence_coverage_regression_count"),
                _availability(row.get("drift", {}).get("optional_signal_availability", {}).get("action_trace", {})),
            ]
            for row in comparisons
        ],
    )


def _audit_table(comparisons: list[dict[str, Any]]) -> str:
    blocks = [
        _table(
            ["Family", "Top Audit Labels", "Evidence Availability"],
            [
                [
                    row["comparison_id"],
                    _label_counts(row.get("audit", {}).get("cause_label_counts", {})),
                    _label_counts({key: value.get("available") for key, value in row.get("audit", {}).get("evidence_availability_counts", {}).items() if isinstance(value, dict)}, limit=4),
                ]
                for row in comparisons
            ],
        )
    ]
    top_cases = []
    for row in comparisons:
        for case in row.get("case_studies", []):
            top_cases.append([row["comparison_id"], case.get("question_id"), ", ".join(case.get("cause_labels", [])[:5])])
    blocks.append(_table(["Family", "Question", "Labels"], top_cases[:12]))
    return "\n".join(blocks)


def _case_studies(comparisons: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for row in comparisons:
        for case in row.get("case_studies", [])[:2]:
            metrics = case.get("drift_metrics", {}) if isinstance(case.get("drift_metrics"), dict) else {}
            labels = ", ".join(case.get("cause_labels", [])[:6]) if isinstance(case.get("cause_labels"), list) else "none"
            blocks.append(
                "\n".join(
                    [
                        '<article class="case">',
                        f"<h3>{_esc(row['comparison_id'])} / {_esc(case.get('question_id'))}</h3>",
                        f'<p class="muted">{_esc(labels)}</p>',
                        _case_signal_table(metrics),
                        _table(
                            ["Before retrieved docs", "After retrieved docs"],
                            [[_doc_list(case.get("before", {}).get("doc_ids", [])), _doc_list(case.get("after", {}).get("doc_ids", []))]],
                        ),
                        _case_rank_score_table(metrics),
                        _case_evidence_table(case.get("associated_evidence", {})),
                        "</article>",
                    ]
                )
            )
    if not blocks:
        return '<p class="muted">No regression audit case studies available.</p>'
    return "\n".join(blocks)


def _case_signal_table(metrics: dict[str, Any]) -> str:
    support_exposure = metrics.get("support_exposure", {}) if isinstance(metrics.get("support_exposure"), dict) else {}
    coverage = metrics.get("evidence_coverage", {}) if isinstance(metrics.get("evidence_coverage"), dict) else {}
    recall = metrics.get("support_recall", {}) if isinstance(metrics.get("support_recall"), dict) else {}
    distance = metrics.get("distance_to_support", {}) if isinstance(metrics.get("distance_to_support"), dict) else {}
    action = metrics.get("action_trace", {}) if isinstance(metrics.get("action_trace"), dict) else {}
    top_k = metrics.get("top_k_jaccard", {}) if isinstance(metrics.get("top_k_jaccard"), dict) else {}
    ordered = metrics.get("ordered_top_k_overlap", {}) if isinstance(metrics.get("ordered_top_k_overlap"), dict) else {}
    return _table(
        ["Signal", "Before", "After", "Delta"],
        [
            ["Top-k Jaccard", _fmt(top_k.get("value")), "same query pair", ""],
            ["Ordered top-k overlap", _fmt(ordered.get("value")), "same query pair", ""],
            ["Support exposure", support_exposure.get("before_exposed_count"), support_exposure.get("after_exposed_count"), _signed_or_missing(support_exposure.get("exposed_count_delta"))],
            ["Evidence coverage", _fmt(coverage.get("before_coverage")), _fmt(coverage.get("after_coverage")), _signed_or_missing(coverage.get("coverage_delta"))],
            ["Support recall", _fmt(recall.get("before_recall")), _fmt(recall.get("after_recall")), _signed_or_missing(recall.get("recall_delta"))],
            ["Distance to support", distance.get("before_min_distance"), distance.get("after_min_distance"), _signed_or_missing(distance.get("min_distance_delta"))],
            ["Action trace steps", action.get("before_action_count"), action.get("after_action_count"), _signed_or_missing(action.get("action_count_delta"))],
            ["Final context Jaccard", _fmt(action.get("final_context_jaccard")), "same query pair", ""],
        ],
    )


def _case_rank_score_table(metrics: dict[str, Any]) -> str:
    rank = metrics.get("rank_displacement", {}) if isinstance(metrics.get("rank_displacement"), dict) else {}
    scores = metrics.get("retained_score_delta", {}) if isinstance(metrics.get("retained_score_delta"), dict) else {}
    score_by_doc = {
        str(item.get("doc_id")): item
        for item in scores.get("documents", [])
        if isinstance(item, dict) and item.get("doc_id") is not None
    }
    rows = []
    for item in rank.get("documents", [])[:8]:
        if not isinstance(item, dict):
            continue
        doc_id = item.get("doc_id")
        score = score_by_doc.get(str(doc_id), {})
        rows.append(
            [
                doc_id,
                item.get("before_rank"),
                item.get("after_rank"),
                _signed_or_missing(item.get("rank_delta")),
                _fmt(score.get("before_score")) if score else "missing",
                _fmt(score.get("after_score")) if score else "missing",
                _signed_or_missing(score.get("score_delta")) if score else "missing",
            ]
        )
    return _table(["Doc", "Before rank", "After rank", "Rank delta", "Before score", "After score", "Score delta"], rows)


def _case_evidence_table(evidence: Any) -> str:
    evidence = evidence if isinstance(evidence, dict) else {}
    corpus = evidence.get("corpus", {}) if isinstance(evidence.get("corpus"), dict) else {}
    graph = evidence.get("graph", {}) if isinstance(evidence.get("graph"), dict) else {}
    support = evidence.get("support_surface", {}) if isinstance(evidence.get("support_surface"), dict) else {}
    return _table(
        ["Evidence", "Nearby or changed items"],
        [
            ["Added corpus docs", _doc_list(corpus.get("added_doc_ids", []))],
            ["Changed corpus docs", _doc_list(corpus.get("changed_doc_ids", []))],
            ["Removed corpus docs", _doc_list(corpus.get("removed_doc_ids", []))],
            ["Added graph edges", _edge_list(graph.get("added_edges", []))],
            ["Removed graph edges", _edge_list(graph.get("removed_edges", []))],
            ["Support surface", _support_summary(support)],
        ],
    )


def _systems_table(comparisons: list[dict[str, Any]]) -> str:
    return _table(
        ["Family", "Pages", "Graph Edges", "Audit Rows", "Report Bytes / Query", "Artifact Bytes / Page"],
        [
            [
                row["comparison_id"],
                row.get("systems", {}).get("workload_metrics", {}).get("page_count"),
                row.get("systems", {}).get("workload_metrics", {}).get("graph_edge_count"),
                row.get("systems", {}).get("workload_metrics", {}).get("audit_row_count"),
                _ratio(row, "report_bytes_per_query"),
                _ratio(row, "artifact_bytes_per_page"),
            ]
            for row in comparisons
        ],
    )


def _provenance(report: dict[str, Any], comparisons: list[dict[str, Any]]) -> str:
    pilot = report.get("pilot_manifest", {})
    env = pilot.get("environment", {}) if isinstance(pilot.get("environment"), dict) else {}
    git = env.get("retrieval_arena_git_provenance", {}) if isinstance(env.get("retrieval_arena_git_provenance"), dict) else {}
    rows = [
        ["Pilot manifest", "pilot_manifest.json"],
        ["Resolved plan", "plan_resolved.json"],
        ["Snapshot diff", "snapshot_comparison/snapshot_diff.json"],
        ["Tool commit", _short(git.get("commit"))],
        ["Tool branch", git.get("branch")],
        ["Dirty worktree at run time", git.get("dirty")],
    ]
    for row in comparisons:
        rows.append([f"{row['comparison_id']} before replay", row["paths"].get("before_replay")])
        rows.append([f"{row['comparison_id']} after replay", row["paths"].get("after_replay")])
    return _table(["Evidence", "Reference"], rows)


def _insight_list(insights: list[str]) -> str:
    return '<div class="insights">' + "".join(f"<div>{_esc(item)}</div>" for item in insights) + "</div>"


def _metric(label: str, value: Any) -> str:
    return f'<div class="metric"><span>{_esc(label)}</span><strong>{_esc(_display(value))}</strong></div>'


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th>{_esc(item)}</th>" for item in headers)
    body = "".join("<tr>" + "".join(f"<td>{_cell(item)}</td>" for item in row) + "</tr>" for row in rows)
    if not rows:
        body = f'<tr><td colspan="{len(headers)}" class="muted">No rows available.</td></tr>'
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _list(title: str, items: list[Any]) -> str:
    if not items:
        return ""
    return f"<h3>{_esc(title)}</h3><ul>" + "".join(f"<li>{_esc(_display(item))}</li>" for item in items) + "</ul>"


def _bar(value: Any) -> str:
    if value is None:
        return "missing"
    number = max(0.0, min(1.0, _num(value)))
    width = int(round(number * 100))
    return f'<div class="bar"><span style="width:{width}%"></span><em>{_esc(_fmt(value))}</em></div>'


def _label_counts(counts: dict[str, Any], *, limit: int = 5) -> str:
    if not isinstance(counts, dict) or not counts:
        return "none"
    pairs = sorted(((key, _num(value)) for key, value in counts.items()), key=lambda item: (-item[1], item[0]))[:limit]
    return ", ".join(f"{key}: {_fmt(value)}" for key, value in pairs)


def _doc_list(doc_ids: Any, *, limit: int = 8) -> str:
    if not isinstance(doc_ids, list) or not doc_ids:
        return "none"
    items = "".join(f"<li>{_esc(item)}</li>" for item in doc_ids[:limit])
    if len(doc_ids) > limit:
        items += f"<li class=\"muted\">+{len(doc_ids) - limit} more</li>"
    return f"<ol>{items}</ol>"


def _edge_list(edges: Any, *, limit: int = 6) -> str:
    if not isinstance(edges, list) or not edges:
        return "none"
    items = []
    for edge in edges[:limit]:
        if isinstance(edge, dict):
            source = edge.get("source_doc_id") or edge.get("source") or edge.get("from") or edge.get("src") or "unknown"
            target = edge.get("target_doc_id") or edge.get("target") or edge.get("to") or edge.get("dst") or "unknown"
            relation = edge.get("relation") or edge.get("edge_type") or edge.get("type")
            label = f"{source} -> {target}" + (f" ({relation})" if relation else "")
        else:
            label = str(edge)
        items.append(f"<li>{_esc(label)}</li>")
    if len(edges) > limit:
        items.append(f"<li class=\"muted\">+{len(edges) - limit} more</li>")
    return f"<ol>{''.join(items)}</ol>"


def _support_summary(support: dict[str, Any]) -> str:
    if not isinstance(support, dict) or not support:
        return "none"
    rows = []
    for key in sorted(support):
        value = support[key]
        if isinstance(value, list):
            rows.append(f"{key}: {len(value)} item(s)")
        elif isinstance(value, dict):
            rows.append(f"{key}: {len(value)} field(s)")
        else:
            rows.append(f"{key}: {_display(value)}")
    return "<br>".join(_esc(item) for item in rows[:8])


def _select_case_studies(path: Path, *, limit: int = 4) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    rows.sort(key=lambda row: (-_case_score(row), str(row.get("question_id") or "")))
    return rows[:limit]


def _case_score(row: dict[str, Any]) -> float:
    labels = row.get("cause_labels", [])
    score = float(len(labels)) if isinstance(labels, list) else 0.0
    metrics = row.get("drift_metrics", {}) if isinstance(row.get("drift_metrics"), dict) else {}
    score += max(0.0, 1.0 - _metric_value(metrics, "top_k_jaccard", "value")) * 10
    score += max(0.0, 1.0 - _metric_value(metrics, "ordered_top_k_overlap", "value")) * 5
    score += max(0.0, -_metric_value(metrics, "support_recall", "recall_delta")) * 10
    score += max(0.0, -_metric_value(metrics, "evidence_coverage", "coverage_delta")) * 10
    score += max(0.0, -_metric_value(metrics, "support_exposure", "exposed_count_delta"))
    score += abs(_metric_value(metrics, "distance_to_support", "min_distance_delta"))
    score += abs(_metric_value(metrics, "action_trace", "action_count_delta"))
    return score


def _metric_value(metrics: dict[str, Any], name: str, key: str) -> float:
    raw = metrics.get(name, {}) if isinstance(metrics.get(name), dict) else {}
    return _num(raw.get(key))


def _ratio(row: dict[str, Any], key: str) -> str:
    ratio = row.get("systems", {}).get("artifact_metrics", {}).get("ratios", {}).get(key, {})
    if not isinstance(ratio, dict) or not ratio.get("available"):
        return "missing"
    return _fmt(ratio.get("value"))


def _availability(raw: dict[str, Any]) -> str:
    if not isinstance(raw, dict):
        return "missing"
    return f"{_display(raw.get('available_count'))} available, {_display(raw.get('unavailable_count'))} unavailable"


def _value_from_comparison(comparisons: list[dict[str, Any]], key: str) -> Any:
    for row in comparisons:
        value = row.get("drift", {}).get(key)
        if value is not None:
            return value
    return None


def _sum_labels(counts: dict[str, Any]) -> float:
    return sum(_num(value) for value in counts.values()) if isinstance(counts, dict) else 0.0


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError(f"Invalid JSON report artifact {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValidationError(f"JSON report artifact must contain an object: {path}")
    return raw


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"Invalid JSONL report artifact {path}:{line_number}: {exc}") from exc
        if isinstance(raw, dict):
            rows.append(raw)
    return rows


def _rel(path: Path, base: Path) -> str | None:
    if not path.exists():
        return None
    try:
        return str(path.resolve().relative_to(base.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def _cell(value: Any) -> str:
    text = _display(value)
    if isinstance(text, str) and text.startswith("<"):
        return text
    return _esc(text)


def _display(value: Any) -> str:
    if value is None:
        return "missing"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return _fmt(value)
    return str(value)


def _fmt(value: Any) -> str:
    if value is None:
        return "missing"
    number = _num(value)
    if abs(number) >= 100:
        return f"{number:,.0f}"
    if abs(number) >= 10:
        return f"{number:,.1f}"
    return f"{number:,.3f}".rstrip("0").rstrip(".")


def _signed(value: Any) -> str:
    number = _num(value)
    return f"{number:+,.0f}" if float(number).is_integer() else f"{number:+,.3f}"


def _signed_or_missing(value: Any) -> str:
    if value is None:
        return "missing"
    return _signed(value)


def _num(value: Any, *, default: float = 0.0) -> float:
    try:
        if value is None or isinstance(value, bool):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _short(value: Any) -> str:
    if not value:
        return "missing"
    text = str(value)
    return text[:12] if len(text) > 12 else text


def _esc(value: Any) -> str:
    return html.escape(_display(value), quote=True)


def _css() -> str:
    return """
:root { color-scheme: light; --ink:#18202a; --muted:#5e6a75; --line:#d9e0e7; --paper:#f7f8fa; --accent:#246b5f; --warn:#a85f1a; --blue:#315d8c; }
* { box-sizing: border-box; }
body { margin:0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color:var(--ink); background:var(--paper); line-height:1.45; }
main { max-width:1180px; margin:0 auto; padding:28px; }
.hero { padding:30px 0 18px; border-bottom:2px solid var(--ink); }
.eyebrow { text-transform:uppercase; letter-spacing:0; color:var(--accent); font-weight:700; font-size:12px; }
h1 { font-size:36px; line-height:1.05; margin:8px 0; letter-spacing:0; }
h2 { font-size:22px; margin:32px 0 12px; letter-spacing:0; }
h3 { font-size:16px; margin:18px 0 8px; letter-spacing:0; }
p { color:var(--muted); margin:0 0 10px; }
.metric-row { display:grid; grid-template-columns:repeat(auto-fit, minmax(170px, 1fr)); gap:10px; margin:14px 0; }
.metric { background:white; border:1px solid var(--line); border-radius:6px; padding:12px; min-height:78px; }
.metric span { display:block; color:var(--muted); font-size:12px; margin-bottom:6px; }
.metric strong { display:block; font-size:20px; overflow-wrap:anywhere; }
section { margin-top:18px; }
.insights { display:grid; grid-template-columns:repeat(auto-fit, minmax(260px, 1fr)); gap:10px; }
.insights div { background:#eef5f2; border-left:4px solid var(--accent); padding:12px; border-radius:4px; }
table { width:100%; border-collapse:collapse; background:white; border:1px solid var(--line); margin:10px 0 16px; table-layout:fixed; }
th, td { text-align:left; vertical-align:top; padding:9px 10px; border-bottom:1px solid var(--line); overflow-wrap:anywhere; font-size:13px; }
th { color:#23313f; background:#edf1f4; font-size:12px; text-transform:uppercase; }
.muted { color:var(--muted); }
ul { margin:8px 0 18px; padding-left:20px; }
ol { margin:0; padding-left:18px; }
.case { background:white; border:1px solid var(--line); border-radius:6px; padding:14px; margin:12px 0 18px; }
.case h3 { margin-top:0; }
.case table { margin-bottom:12px; }
.bar { position:relative; height:22px; background:#eef1f3; border:1px solid var(--line); border-radius:4px; overflow:hidden; min-width:100px; }
.bar span { display:block; height:100%; background:linear-gradient(90deg, var(--accent), var(--blue)); }
.bar em { position:absolute; left:8px; top:1px; font-style:normal; font-size:12px; color:#111; }
@media (max-width: 700px) { main { padding:16px; } h1 { font-size:28px; } th, td { font-size:12px; padding:8px; } }
"""
