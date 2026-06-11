# Snapshot Manifest

## Included

This anonymous snapshot includes:

- Retrieval Audit Framework Python source under `retrieval_arena/`;
- source descriptors and experiment configs under `configs/`;
- lightweight examples and documentation under `examples/` and `docs/`;
- retriever fixtures under `tests/`;
- implementation tests under `unit_tests/`;
- package metadata and license files.

## Excluded

The snapshot intentionally excludes:

- `.git/` history and remotes;
- virtual environments;
- Python caches;
- raw third-party documentation snapshots;
- generated `raw/`, `processed/`, `graphs/`, `reports/`, `results/`,
  `analysis/`, and calibration artifact directories;
- manuscript PDF/source and paper-facing evidence exports;
- project planning notes, blog entries, policy reviews, and author-facing
  sprint history.

## Third-Party Documentation Boundary

Some configurations reference public Express, Docker, and Kubernetes
documentation repositories and revision identifiers. These references are
corpus provenance, not author identity. Raw documentation snapshots are not
redistributed here.

## Review Namespace

Configuration, study, and generated-output examples use `review2026` and
longitudinal-audit labels so the snapshot uses neutral review vocabulary.

## Suggested Anonymous URL Workflow

1. Create a clean temporary Git repository containing this snapshot.
2. Commit only the files in this snapshot.
3. Push it to a repository that does not expose author identity in file content.
4. Use `anonymous.4open.science` to produce an anonymized review URL.
5. Submit the anonymized URL with the review materials.
