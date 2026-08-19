# Vendored Hermes Agent source

- Package: `hermes-agent`
- Version: `0.17.0`
- Upstream: https://github.com/NousResearch/hermes-agent
- License: MIT; see [`LICENSE`](LICENSE)
- Snapshot source: `/opt/hermes` in the verified five-role demo container
- Snapshot date: 2026-08-19
- Export archive SHA-256: `db098436cc14d21e5d832d54d54db6707e4844164c918b2ff0be5b20a5e43358`

This snapshot preserves the source used by the running demo. Rebuildable and
machine-local artifacts were deliberately excluded: `.venv/`, `venv/`,
`node_modules/`, `.playwright/`, `.pytest-cache/`, `__pycache__/`, `*.pyc`,
`*.egg-info/`, `.install_method`, and `.env`.

The expert-team configuration, role SOUL files, and role-specific Skills are
maintained at the repository root and are not part of the upstream Hermes
Agent project.

## Upstream tag boundary

Hermes uses date-based Git tags rather than a `v0.17.0` tag. The public
`v2026.6.19` tag also declares package version 0.17.0, but it is not
byte-identical to this running snapshot: a file-level comparison found 137
container-only files, 3,070 tag-only development/repository files, and 240
changed common files. This repository therefore does not claim that the
snapshot is the public `v2026.6.19` commit. The container snapshot, its package
metadata, the preserved MIT license, and the export hash above are the release
provenance available for reproducing the demonstrated runtime.
