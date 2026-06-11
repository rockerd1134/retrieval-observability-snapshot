# HTML Observability Report

Generate a self-contained report from a completed longitudinal audit pilot bundle:

```powershell
python -m retrieval_arena.cli report html --bundle calibration\review2026\pilot_express_docs
```

The default output is:

```text
calibration/review2026/pilot_express_docs/observability_report.html
```

Use `--out` to write elsewhere:

```powershell
python -m retrieval_arena.cli report html --bundle calibration\review2026\pilot_docker_docs --out reports\docker_observability.html
```

The report is generated from existing pipeline artifacts:

- `pilot_manifest.json`
- `plan_resolved.json`
- `snapshot_comparison/snapshot_diff.json`
- corpus, graph, and support snapshot manifests
- per-family retrieval drift summaries and rows
- regression audit summaries and rows
- systems measurements
- before/after replay manifests

Sections include generated insights, corpus snapshot summary, graph/support
drift, retrieval-family matrix, drift metrics, audit labels, query case
studies, systems/storage measurements, and provenance/replay references.

Query case studies are selected deterministically from audit rows. They show
before/after retrieved document IDs, rank and score deltas, support exposure,
evidence coverage, support recall, distance-to-support, action-trace summaries,
and nearby corpus/graph/support evidence when available.
