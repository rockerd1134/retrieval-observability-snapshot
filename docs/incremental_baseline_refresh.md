# Incremental review Baseline Refresh

Use incremental refresh when a longitudinal audit pilot baseline already exists and the
plan changes only by adding or invalidating selected retrieval tests.

```powershell
python -m retrieval_arena.cli study review pilot --plan configs\review2026\pilot_express_docs.json --refresh-baseline
```

The refresh mode validates existing run manifests per selected dataset/test. A
run may be reused only when the recorded dataset name, test name, query-set ID,
query-set hash, corpus/graph/support snapshot manifest hashes, scoring hash,
resolved run identity hash, and required output files still match the active
plan. Older run manifests without a resolved run identity fall back to the full
retrieval config hash.

Missing or invalidated tests are rerun for the affected before/after retrieval
config. Valid run directories are preserved, and the experiment manifest is
rewritten to index both reused and newly generated run manifests.

Downstream comparison reports are reused only when their before/after selected
runs were both reused and the required drift, audit, or measurement files are
present. Any new or invalidated comparison refreshes its drift, regression
audit, systems measurements, and copied baseline artifacts. The pilot manifest
records reuse decisions with concise reasons and hash-check details.

The mode is conservative by design: if validation cannot prove compatibility,
the affected family is regenerated.
