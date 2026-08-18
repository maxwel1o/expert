# Skill Installation Flow (agent-skills repo → Hermes)

## Key Paths

| Variable | Value | Notes |
|----------|-------|-------|
| AGENT_HOME | `/opt/data` | NOT `/opt/data/home/.hermes/` |
| Skills dir | `/opt/data/skills/` | Hermes only scans this directory |
| Repo root | `/opt/data/home/.hermes/code/agent-skills/` | Git clone of hw-pbclouds/agent-skills |
| Source skills | `skills0-dev/` | Flat layout in repo, one dir per skill |
| Target layout | `/opt/data/skills/<category>/` | Category subdirs: mlops, software-development, etc. |

## Installation Steps

1. Identify skill in `skills0-dev/<skill-name>/`
2. Determine category (mlops, software-development, etc.)
3. Copy **entire** directory (SKILL.md + references/ + scripts/ + templates/ + tests/):
   ```bash
   cp -r skills0-dev/<skill-name>/ /opt/data/skills/<category>/<skill-name>/
   ```
4. Verify with `skill_view('<skill-name>')` — must show readiness=available
5. Do NOT create files/directories without explicit user approval

## Common Mistakes

- Installing to `/opt/data/home/.hermes/skills/` — invisible to Hermes
- Copying only SKILL.md without references/scripts/templates — skill is incomplete
- Creating symlinks or alternative directories — causes confusion
- Guessing AGENT_HOME — always verify: it's `/opt/data` in this deployment

## Usage Hook Integration

- Usage events logged to `~/.hermes/logs/skill-usage.jsonl`
- Aggregated view in repo's `.skill-usage.json`
- `agent-skill-commit` Step 5 reads jsonl, re-counts, writes `.skill-usage.json`
- Three Gates control lifecycle: Gate 1 (usage validation), Gate 2 (doc completeness), Gate 3 (review readiness)
