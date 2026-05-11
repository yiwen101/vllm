#!/usr/bin/env bash
set -euo pipefail

REMOTE="origin"
BRANCH="probe-workbench"

echo "Repo: $(pwd)"
echo "Remote: ${REMOTE}"
echo "Branch: ${BRANCH}"
echo
echo "This will hard-reset the current repo to ${REMOTE}/${BRANCH}."
echo "It will not delete ignored logs under logs/."
echo "Press Ctrl-C within 5 seconds to cancel."
sleep 5

git fetch "${REMOTE}" "${BRANCH}"
git checkout -B "${BRANCH}" "${REMOTE}/${BRANCH}"
git reset --hard "${REMOTE}/${BRANCH}"

echo
echo "Current state:"
git status --short --branch
git log --oneline -3
