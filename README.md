# Retrieval Audit Framework Review Snapshot

This repository snapshot contains the retrieval audit tool used by the
accompanying anonymous manuscript on replayable retrieval audits for evolving
digital documentation collections.

The snapshot is intended for anonymous review through a service such as
`anonymous.4open.science`. It contains tool source code, configurations,
examples, documentation, and tests. It intentionally does not include the
manuscript PDF/source, author notes, project planning history, raw third-party
documentation snapshots, or full generated run outputs.

## Contents

- `retrieval_arena/`: implementation source.
- `configs/`: corpus source descriptors and pipeline configuration files.
- `docs/`: tool usage notes.
- `examples/`: small example inputs.
- `tests/`: retriever/plugin fixtures.
- `unit_tests/`: implementation tests.
- `pyproject.toml`: Python package metadata.
- `LICENSE`: license for this tool snapshot.
- `SNAPSHOT_MANIFEST.md`: review-scope and exclusion details.

## Install

From the snapshot root:

```powershell
python -m pip install -e .
```

## Test

```powershell
python -m pytest unit_tests
```

## Review Scope

This snapshot supports inspection of the implementation that produces:

- source descriptors and import/preparation manifests;
- graph and support-surface manifests;
- retrieval replay manifests;
- replay comparison reports;
- retrieval drift reports;
- regression audit reports;
- artifact footprint summaries;
- paper-facing evidence exports.

Full corpus regeneration requires public third-party documentation repositories
and local generated artifact directories that are not redistributed in this
anonymous review snapshot.

## Anonymous Hosting Note

For review, upload this snapshot to a clean repository with no identifying Git
history, then submit the anonymized URL produced by `anonymous.4open.science`.
Do not upload the original working repository directly.
