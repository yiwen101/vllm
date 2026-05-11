#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-}"
BRANCH="${BRANCH:-probe-workbench}"
REPO_DIR="${REPO_DIR:-/workspace/vllm}"
REMOTE="${REMOTE:-origin}"

if [[ -z "${REPO_URL}" ]]; then
  echo "Set REPO_URL first. Example:"
  echo "  export REPO_URL=https://github.com/YOUR_GITHUB_USER/vllm.git"
  exit 2
fi

echo "Repo URL: ${REPO_URL}"
echo "Branch: ${BRANCH}"
echo "Repo dir: ${REPO_DIR}"
echo

if [[ ! -d "${REPO_DIR}/.git" ]]; then
  mkdir -p "$(dirname "${REPO_DIR}")"
  git clone --branch "${BRANCH}" "${REPO_URL}" "${REPO_DIR}"
else
  cd "${REPO_DIR}"
  git remote set-url "${REMOTE}" "${REPO_URL}"
  git fetch "${REMOTE}" "${BRANCH}"
  git checkout -B "${BRANCH}" "${REMOTE}/${BRANCH}"
  git reset --hard "${REMOTE}/${BRANCH}"
fi

cd "${REPO_DIR}"
echo
echo "Current state:"
git status --short --branch
git log --oneline -3

