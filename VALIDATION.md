# Release validation

The public bundle is gated by executable checks, not by documentation alone.

Validated on 2026-08-19:

- exact role set: default Leader plus `deployer`, `tester`, `profiler`, `analyst`;
- vendored Hermes Agent metadata: `hermes-agent` 0.17.0, MIT license,
  required source/install files present, and rebuilt dependencies/runtime state absent;
- manifest assignment: 201 Skills (`1 + 127 + 19 + 5 + 49`);
- all packaged Skill entry points exist;
- five SOUL bases, 201 Skill packages, 12 progress Python modules, lifecycle
  Hook, command wrapper, and Hook merger match the deployed five-role system;
- SHA-256 inventory covers all 1,539 bundled Skill files;
- no `.env`, credential file, SQLite runtime database, private server name,
  historical job ID, private-network address, or private-key block is included;
- all role and protocol Markdown decodes as UTF-8 without replacement bytes;
- shell entry points pass Git Bash `bash -n`;
- Python modules compile;
- 58 `team_progress`, release inventory, and Hermes source-contract unit tests pass;
- Hermes installer dry-run, existing-version reuse, version-mismatch rejection,
  and top-level `--install-hermes` routing pass the shell integration test.

Re-run locally:

```bash
python3 tools/validate_release.py
python3 tools/verify_skill_hashes.py
python3 -m compileall -q team_progress tools tests
python3 -m unittest discover -s tests -t . -q
bash tests/test_install_hermes.sh
bash -n install.sh scripts/*.sh team_progress/assets/team-progress \
  team_progress/assets/hermes-kanban-progress-hook
```

Runtime installation must additionally be verified inside the target Hermes
container with `scripts/verify-team.sh` because local release checks cannot
prove provider credentials, NPU drivers, CANN, container paths, or Gateway
state.
