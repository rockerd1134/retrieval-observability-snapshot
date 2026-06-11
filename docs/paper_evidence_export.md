# Paper Evidence Export

Export manuscript-facing evidence from the same assembled data used by the
HTML observability report:

```powershell
python -m retrieval_arena.cli report evidence --bundle calibration\review2026\pilot_express_docs --out-dir review_artifacts\evidence\review2026\express_docs
python -m retrieval_arena.cli report evidence --bundle calibration\review2026\pilot_docker_docs --out-dir review_artifacts\evidence\review2026\docker_docs
python -m retrieval_arena.cli report evidence --bundle calibration\review2026\pilot_kubernetes_website_docs --out-dir review_artifacts\evidence\review2026\kubernetes_website_docs
```

Each export writes:

- `corpus_graph_support_summary.csv`
- `family_drift_matrix.csv`
- `audit_cause_labels.csv`
- `systems_storage_measurements.csv`
- `selected_case_studies.csv`
- `provenance_replay_summary.md`

These files are intended as paper inputs for replayability, longitudinal drift,
regression audit, and systems/storage sections. They preserve source bundle and
artifact-path references so manuscript tables can be traced back to generated
pipeline outputs.

Do not treat the exports as static benchmark results. They are derived evidence
from selected calibration bundles and should be regenerated when the bundle or
report schema changes.
