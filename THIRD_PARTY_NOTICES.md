# Third-party software and Skills

## Hermes Agent

This repository includes a source snapshot of Hermes Agent 0.17.0 under
`vendor/hermes-agent/`. Hermes Agent is copyright Nous Research and licensed
under the MIT License. The upstream project is
[NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent); the
vendored license is preserved at `vendor/hermes-agent/LICENSE`, and snapshot
provenance is recorded in `vendor/hermes-agent/VENDORED-SOURCE.md`.

## Agent Skills

This repository bundles 201 public Agent Skills so that the five-role demo can
be installed reproducibly. The project-level `LICENSE` does **not** relicense
those third-party files. Their upstream copyright and license terms continue to
apply. Exact assignment, path, source group, and source commit are recorded in
[`manifests/skills.csv`](manifests/skills.csv).

| Upstream source | Pinned commit | Bundled Skills | License information found at the pinned revision |
|---|---:|---:|---|
| [Ascend/agent-skills](https://atomgit.com/Ascend/agent-skills) | `6a6e9af256d866316cccc4b53966b146c4f536f2` | 31 | Repository `LICENSE`: Apache-2.0 for code/configuration and CC-BY-SA-4.0 for documentation/other material |
| [Ascend/model-agent](https://atomgit.com/Ascend/model-agent) | `c417b66fc29b859cc6b6ccfeab292c64bebdd919` | 168 | No repository-level `LICENSE`, `NOTICE`, or `COPYING` file was found during packaging; individual files may declare their own terms |
| [hw-pbclouds/agent-skills](https://gitcode.com/hw-pbclouds/agent-skills) | `2c49e24fe86392ea8ce1b0311bfb80701660ba23` | 2 | No repository-level `LICENSE`, `NOTICE`, or `COPYING` file was found during packaging; individual files may declare their own terms |

Before redistributing a derived bundle, review the current upstream terms. Any
third-party trademarks and product names belong to their respective owners.
