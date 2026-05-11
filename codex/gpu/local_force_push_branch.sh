#!/usr/bin/env bash
set -euo pipefail

REMOTE="origin"
BRANCH="$(git branch --show-current)"

if [[ -z "${BRANCH}" ]]; then
  echo "Could not determine current branch. Pass branch explicitly:"
  echo "  git checkout probe-workbench"
  exit 2
fi

echo "Local branch: ${BRANCH}"
echo "Remote: ${REMOTE}"
echo
echo "This will force-update ${REMOTE}/${BRANCH} using --force-with-lease."
echo "Press Ctrl-C within 5 seconds to cancel."
sleep 5

git push --force-with-lease -u "${REMOTE}" "${BRANCH}"

echo
echo "Pushed ${BRANCH} to ${REMOTE}/${BRANCH}"
