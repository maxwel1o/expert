---
name: gitcode-repo-bootstrap
description: Automatically create a GitCode repository, configure PAT-based remote, push code, and tag the repo with #+NPU. Use when the user asks to submit adaptation results to GitCode/AtomGit, publish a model card, or push deliverables to a remote repository for community review.
---

# GitCode Repo Bootstrap Skill

One-shot automation for publishing Ascend NPU adaptation results to GitCode (AtomGit community). Covers repo creation, `.gitignore` setup, PAT authentication, branch push, and merge-request initiation.

## When to Invoke

- User says "提交到 GitCode", "push to gitcode", "publish to AtomGit"
- Need to create a public repo under a GitCode organization or personal namespace
- Need to tag the repository description with `#+NPU`
- Adaptation deliverables (scripts, logs, README) are ready and need remote backup

## Prerequisites

- `ATOMGIT_USER_TOKEN` or equivalent PAT exported in the environment
- GitCode API base URL: `https://api.gitcode.com/api/v5`
- Git installed locally with user identity configured

## Workflow

### Step 1: Discover or Create the Repository

**Option A: Create a new repo via API**

```bash
TOKEN="${ATOMGIT_USER_TOKEN}"
REPO_NAME="MyModel-NPU"
DESCRIPTION="MyModel Ascend NPU Adaptation #+NPU"

curl -s -X POST "https://api.gitcode.com/api/v5/user/repos" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${TOKEN}" \
  -d "{
    \"name\": \"${REPO_NAME}\",
    \"description\": \"${DESCRIPTION}\",
    \"private\": false,
    \"auto_init\": false
  }"
```

If the name already exists in your namespace, GitCode returns `422` with message `Project with same name or path in this namespace already exists`. In that case:
- Check existing forks: `GET /api/v5/user/repos`
- Reuse the existing fork, or append a suffix to the repo name.

**Option B: Reuse an existing fork**

List repos and pick the one whose `parentfull_name` matches the target upstream:

```bash
curl -s -H "Authorization: Bearer ${TOKEN}" \
  "https://api.gitcode.com/api/v5/user/repos" | \
  python3 -c "import sys,json; [print(r['full_name'], r.get('parentfull_name','')) for r in json.load(sys.stdin)]"
```

### Step 2: Clone the Fork Locally

Use the PAT embedded in the HTTPS URL to avoid interactive password prompts:

```bash
git clone "https://oauth2:${ATOMGIT_USER_TOKEN}@gitcode.com/${USER}/${REPO_NAME}.git" my-repo
cd my-repo
```

### Step 3: Prepare Deliverables

Create a `.gitignore` that excludes large artifacts:

```bash
cat > .gitignore <<'EOF'
model_weights/
*.safetensors
*.bin
*.pyc
__pycache__/
EOF
```

Stage the adaptation files (do NOT stage model weights):

```bash
git add inference.py eval_accuracy.py benchmark.sh readme.md \
        validation_report.json scripts/ logs/ screenshots/ .gitignore
```

### Step 4: Commit and Push

```bash
git commit -m "feat: add NPU adaptation deliverables

- inference scripts
- accuracy / performance benchmarks
- validation report and documentation

#+NPU"

git remote add origin "https://oauth2:${ATOMGIT_USER_TOKEN}@gitcode.com/${USER}/${REPO_NAME}.git"
git push -u origin master
```

### Step 5: Ensure Description Contains #+NPU

If the repo was created without the tag, patch the description via API:

```bash
curl -s -X PATCH "https://api.gitcode.com/api/v5/repos/${USER}/${REPO_NAME}" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${TOKEN}" \
  -d '{
    "description": "MyModel Ascend NPU Adaptation #+NPU"
  }'
```

### Step 6: Open Merge Request to Upstream (Optional)

If the repo is a fork of an organization repo (e.g. `Ascend/model-agent`) and you need to contribute skills back:

```bash
# Create a feature branch for the skill
git checkout -b feat/my-skill
# ... add skill files ...
git add .
git commit -m "skill: add my-skill"
git push origin feat/my-skill
```

Then create the MR via API:

```bash
curl -s -X POST "https://api.gitcode.com/api/v5/repos/Ascend/model-agent/pulls" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${TOKEN}" \
  -d "{
    \"title\": \"skill: add my-skill\",
    \"head\": \"${USER}:feat/my-skill\",
    \"base\": \"master\",
    \"body\": \"Add new skill for ... #+NPU\"
  }"
```

## Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `422 Project with same name or path already exists` | Repo or fork already exists | List user repos and reuse the existing one |
| `401 Unauthorized` | PAT missing or expired | Export `ATOMGIT_USER_TOKEN` and verify with `curl /api/v5/user` |
| `403 Forbidden` | PAT lacks `repo` scope | Regenerate the PAT with full repo permissions |
| Push rejected for large files | Accidentally staged `.safetensors` | Add `model_weights/` to `.gitignore`, `git rm --cached`, recommit |
| Cannot open MR to upstream | Not a fork, or no permission on upstream org | Verify `parentfull_name` exists; if not, the repo is standalone |

## Security Note

- Never hardcode the PAT in shell history. Prefer environment variables.
- If the PAT must appear in a remote URL, strip the remote after push or use a credential helper.
- The PAT in `https://oauth2:TOKEN@gitcode.com/...` will be logged by Git in plain text if `GIT_TRACE=1` is enabled; avoid tracing in shared environments.

## One-liner Checklist

```text
[ ] PAT exported as ATOMGIT_USER_TOKEN
[ ] Repo created or existing fork identified
[ ] .gitignore excludes model_weights/
[ ] Commit message references #+NPU
[ ] Description updated via API to include #+NPU
[ ] Push succeeded to origin
[ ] MR opened to upstream (if contributing back)
```
