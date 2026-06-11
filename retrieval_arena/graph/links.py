from __future__ import annotations

import csv
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from ..hashing import sha256_file, sha256_json
from ..snapshots import corpus_doc_id


GRAPH_EXTRACTION_VERSION = "markdown-links-v1"
INLINE_LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
REFERENCE_DEF_RE = re.compile(r"^\s*\[[^\]]+\]:\s*(\S+)", re.MULTILINE)
HTML_EXTENSIONS = {".html", ".htm"}


def write_markdown_link_graph(dataset_path: Path) -> dict[str, Any]:
    corpus_dir = dataset_path / "corpus"
    path_to_doc = _path_to_doc_id(corpus_dir)
    edges = sorted(_extract_edges(corpus_dir, path_to_doc))
    edge_path = dataset_path / "graph_edges.csv"
    with edge_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source", "target"])
        writer.writeheader()
        for source, target in edges:
            writer.writerow({"source": source, "target": target})
    metrics = graph_metrics(edges, sorted(set(path_to_doc.values())))
    metrics_path = dataset_path / "graph_metrics.json"
    metrics_path.write_text(_json(metrics), encoding="utf-8")
    return {
        "edge_file": "graph_edges.csv",
        "edge_file_hash": sha256_file(edge_path),
        "graph_metrics_file": "graph_metrics.json",
        "graph_metrics_file_hash": sha256_file(metrics_path),
        "graph_hash": sha256_json({"edges": [{"source": source, "target": target} for source, target in edges]}),
        "node_count": metrics["node_count"],
        "edge_count": metrics["edge_count"],
        "graph_metrics": metrics,
    }


def graph_metrics(edges: list[tuple[str, str]], corpus_nodes: list[str]) -> dict[str, Any]:
    nodes = sorted(set(corpus_nodes) | {node for edge in edges for node in edge})
    weak = _weak_components(nodes, edges)
    strong = _strong_components(nodes, edges)
    out_degree: dict[str, int] = {node: 0 for node in nodes}
    in_degree: dict[str, int] = {node: 0 for node in nodes}
    for source, target in edges:
        out_degree[source] = out_degree.get(source, 0) + 1
        in_degree[target] = in_degree.get(target, 0) + 1
    return {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "weak_components": [{"size": len(component), "nodes": component} for component in weak],
        "strong_components": [{"size": len(component), "nodes": component} for component in strong],
        "weak_component_count": len(weak),
        "strong_component_count": len(strong),
        "largest_component_size": max((len(component) for component in weak), default=0),
        "isolated_node_count": sum(1 for node in nodes if in_degree.get(node, 0) == 0 and out_degree.get(node, 0) == 0),
        "max_in_degree": max(in_degree.values(), default=0),
        "max_out_degree": max(out_degree.values(), default=0),
    }


def _extract_edges(corpus_dir: Path, path_to_doc: dict[str, str]) -> set[tuple[str, str]]:
    edges: set[tuple[str, str]] = set()
    for path_key, source_doc_id in sorted(path_to_doc.items()):
        source_path = corpus_dir / path_key
        text = source_path.read_text(encoding="utf-8", errors="replace")
        for href in [*_inline_hrefs(text), *_reference_hrefs(text)]:
            target = _resolve_link(href, Path(path_key).parent, path_to_doc)
            if target and target != source_doc_id:
                edges.add((source_doc_id, target))
    return edges


def _inline_hrefs(text: str) -> list[str]:
    return [match.group(1).strip("<>") for match in INLINE_LINK_RE.finditer(text)]


def _reference_hrefs(text: str) -> list[str]:
    return [match.group(1).strip("<>") for match in REFERENCE_DEF_RE.finditer(text)]


def _resolve_link(href: str, source_parent: Path, path_to_doc: dict[str, str]) -> str | None:
    parsed = urlparse(href)
    if parsed.scheme or parsed.netloc or href.startswith(("mailto:", "tel:", "#")):
        return None
    raw_path = unquote(parsed.path)
    if not raw_path:
        return None
    raw_path = raw_path.replace("{{ page.lang }}", "").replace("{{page.lang}}", "")
    raw_path = raw_path.replace("{{ site.lang }}", "").replace("{{site.lang}}", "")
    raw_path = raw_path.strip("/")
    if not raw_path:
        return None
    candidate = Path(raw_path)
    if not href.startswith("/"):
        candidate = source_parent / candidate
    normalized = _normalize_candidate(candidate.as_posix())
    for option in _candidate_paths(normalized):
        if option in path_to_doc:
            return path_to_doc[option]
    return None


def _candidate_paths(path: str) -> list[str]:
    base = path.strip("/")
    suffix = Path(base).suffix.lower()
    candidates = [base]
    if suffix in HTML_EXTENSIONS:
        candidates.append(str(Path(base).with_suffix(".md")).replace("\\", "/"))
    elif suffix == "":
        candidates.extend([f"{base}.md", f"{base}/index.md"])
    return list(dict.fromkeys(_normalize_candidate(candidate) for candidate in candidates if candidate))


def _normalize_candidate(path: str) -> str:
    parts: list[str] = []
    for part in path.replace("\\", "/").split("/"):
        if part in {"", "."}:
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)
    return "/".join(parts)


def _path_to_doc_id(corpus_dir: Path) -> dict[str, str]:
    return {
        path.relative_to(corpus_dir).as_posix(): corpus_doc_id(path.relative_to(corpus_dir).as_posix())
        for path in sorted(corpus_dir.rglob("*"))
        if path.is_file()
    }


def _weak_components(nodes: list[str], edges: list[tuple[str, str]]) -> list[list[str]]:
    neighbors: dict[str, set[str]] = defaultdict(set)
    for source, target in edges:
        neighbors[source].add(target)
        neighbors[target].add(source)
    seen: set[str] = set()
    components: list[list[str]] = []
    for node in nodes:
        if node in seen:
            continue
        queue: deque[str] = deque([node])
        seen.add(node)
        component: list[str] = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in sorted(neighbors[current]):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        components.append(sorted(component))
    return sorted(components, key=lambda component: (-len(component), component))


def _strong_components(nodes: list[str], edges: list[tuple[str, str]]) -> list[list[str]]:
    graph: dict[str, list[str]] = {node: [] for node in nodes}
    reverse: dict[str, list[str]] = {node: [] for node in nodes}
    for source, target in edges:
        graph.setdefault(source, []).append(target)
        reverse.setdefault(target, []).append(source)
    visited: set[str] = set()
    order: list[str] = []

    def visit(node: str) -> None:
        visited.add(node)
        for neighbor in sorted(graph.get(node, [])):
            if neighbor not in visited:
                visit(neighbor)
        order.append(node)

    for node in nodes:
        if node not in visited:
            visit(node)
    components: list[list[str]] = []
    visited.clear()

    def assign(node: str, component: list[str]) -> None:
        visited.add(node)
        component.append(node)
        for neighbor in sorted(reverse.get(node, [])):
            if neighbor not in visited:
                assign(neighbor, component)

    for node in reversed(order):
        if node in visited:
            continue
        component: list[str] = []
        assign(node, component)
        components.append(sorted(component))
    return sorted(components, key=lambda component: (-len(component), component))


def _json(value: dict[str, Any]) -> str:
    import json

    return json.dumps(value, indent=2, sort_keys=True) + "\n"
