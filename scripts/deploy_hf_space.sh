#!/usr/bin/env bash
#
# Manually deploy the RoverDevKit webapp to a dedicated Hugging Face
# Space (Docker SDK).
#
# This is a *manual* deploy: it does not run on `git push`. Invoke it
# (or `make deploy-space`) whenever you want the hosted demo to track
# the current committed HEAD of this repo.
#
# What it does:
#   1. Clones the Space repo into a local mirror (.hf-space/, gitignored).
#   2. Exports this repo's committed tree (HEAD) into the mirror.
#   3. Materializes the two Space-only files that intentionally do not
#      live on the GitHub main branch:
#        - a real root `Dockerfile` (a copy of webapp/Dockerfile, which
#          already uses repo-root-relative COPY paths), because the HF
#          builder only looks for ./Dockerfile;
#        - a front-matter `README.md` (from deploy/huggingface/README.md)
#          declaring `sdk: docker` / `app_port: 8000`.
#   4. Tracks the >10 MB surrogate bundle with Git LFS (HF rejects large
#      files on plain git; GitHub does not, which is why LFS is applied
#      only here on the Space side).
#   5. Commits and pushes to the Space's main branch.
#
# Prerequisites:
#   - Create the Space first (https://huggingface.co/new-space, SDK: Docker).
#   - export HF_SPACE_REMOTE=https://huggingface.co/spaces/<user>/roverdevkit
#   - git-lfs installed (https://git-lfs.com ; e.g. `brew install git-lfs`).
#   - Be logged in to HF for the push (a token with write scope; e.g. set
#     a credential helper or use a remote URL of the form
#     https://<user>:<hf_token>@huggingface.co/spaces/<user>/roverdevkit).

set -euo pipefail

SPACE_REMOTE="${HF_SPACE_REMOTE:-}"
if [[ -z "${SPACE_REMOTE}" ]]; then
  cat >&2 <<'EOF'
error: HF_SPACE_REMOTE is not set.

Set it to your Space's git URL, for example:

  export HF_SPACE_REMOTE=https://huggingface.co/spaces/<user>/roverdevkit

Create the Space first at https://huggingface.co/new-space (SDK: Docker).
EOF
  exit 1
fi

if ! command -v git-lfs >/dev/null 2>&1; then
  cat >&2 <<'EOF'
error: git-lfs is required.

The surrogate bundle (models/surrogate_v9/quantile_bundles.joblib, ~26 MB)
exceeds Hugging Face's 10 MB plain-git limit and must be pushed via LFS.

Install git-lfs: https://git-lfs.com  (e.g. `brew install git-lfs`).
EOF
  exit 1
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
SOURCE_REF="${HF_SPACE_REF:-HEAD}"
MIRROR="${REPO_ROOT}/.hf-space"

SHA="$(git -C "${REPO_ROOT}" rev-parse --short "${SOURCE_REF}")"

echo ">> Deploying roverdevkit @ ${SHA} to ${SPACE_REMOTE}"

# Fresh checkout of the Space repo each run keeps state predictable and
# avoids drift from a stale local mirror.
rm -rf "${MIRROR}"
git clone --quiet "${SPACE_REMOTE}" "${MIRROR}"

# Export the committed tree of this repo into the mirror, on top of the
# Space's existing .git. `git archive` only emits tracked, committed
# content, so uncommitted edits and gitignored junk never leak out.
git -C "${REPO_ROOT}" archive "${SOURCE_REF}" | tar -x -C "${MIRROR}"

cd "${MIRROR}"
git lfs install --local >/dev/null 2>&1

# HF Spaces (Docker SDK) only reads ./Dockerfile. webapp/Dockerfile
# already expects the repo root as its build context, so a plain copy
# at the root works without any path edits.
#
# These two Space-only files are read from the working tree (not the
# exported HEAD) so the deploy config does not need to be committed to
# ship — only the app content carried by `git archive` does.
cp "${REPO_ROOT}/webapp/Dockerfile" Dockerfile

# The Space README carries the YAML front matter HF needs; it is kept
# out of the GitHub main branch on purpose.
SPACE_README="${REPO_ROOT}/deploy/huggingface/README.md"
if [[ ! -f "${SPACE_README}" ]]; then
  echo "error: missing ${SPACE_README} (the Space's front-matter README)." >&2
  exit 1
fi
cp "${SPACE_README}" README.md

# HF rejects >10 MB files over plain git, so track the heavy binaries
# with LFS on the Space side only.
cat > .gitattributes <<'EOF'
*.joblib filter=lfs diff=lfs merge=lfs -text
EOF

git add -A
if git diff --cached --quiet; then
  echo ">> Space already matches ${SHA}; nothing to push."
  exit 0
fi

git commit --quiet -m "Deploy roverdevkit @ ${SHA}"
git push origin HEAD:main

echo ">> Done. Watch the build at the Space's 'Logs' tab; /healthz should turn green."
