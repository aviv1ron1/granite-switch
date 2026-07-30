#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Handle a /gpu-test PR comment: verify the commenter holds Maintain/Admin, then
# dispatch gpu-tests.yaml against the PR's head commit. On rejection, react 👎 and
# reply naming the requirement.
#
# Deployed to granite-switch as .github/scripts/gpu_test_command.sh.
# It runs on a GitHub-hosted runner (no /opt/gsw), which is why it is checked in
# rather than being baked into the runner image. See
# gpu-test-command.yaml for the full rationale.
#
# This check is fast-fail UX. The authoritative gate is /opt/gsw/check_role.sh
# inside gpu-tests.yaml, which also covers direct workflow_dispatch.
#
# All GitHub-controlled values arrive as positional args from quoted env in the
# workflow — never interpolated into this script — so a crafted login cannot
# inject shell.
#
# Usage: gpu_test_command.sh <actor-login> <pr-number> <comment-id>
# Env:   GH_TOKEN, GITHUB_REPOSITORY, DEFAULT_BRANCH, SCRIPT_DIR
set -euo pipefail

ACTOR="${1:?usage: gpu_test_command.sh <actor-login> <pr-number> <comment-id>}"
PR_NUMBER="${2:?missing pr number}"
COMMENT_ID="${3:?missing comment id}"

REPO="${GITHUB_REPOSITORY:?}"
DEFAULT_BRANCH="${DEFAULT_BRANCH:?}"
SCRIPT_DIR="${SCRIPT_DIR:?}"

react() {
  gh api -X POST "repos/${REPO}/issues/comments/${COMMENT_ID}/reactions" \
    -f content="$1" >/dev/null
}

# check_role.sh exits non-zero (and prints the role) when not authorized.
if ! ROLE_MSG="$("${SCRIPT_DIR}/check_role.sh" "$ACTOR" 2>&1)"; then
  react '-1'
  gh api -X POST "repos/${REPO}/issues/${PR_NUMBER}/comments" \
    -f body="@${ACTOR} \`/gpu-test\` requires the **Maintain** or **Admin** role. Not launching."
  echo "$ROLE_MSG" >&2
  exit 1
fi

SHA="$(gh api "repos/${REPO}/pulls/${PR_NUMBER}" --jq '.head.sha')"

react 'rocket'

gh workflow run gpu-tests.yaml \
  --ref "$DEFAULT_BRANCH" \
  -f sha="$SHA" \
  -f pr_number="$PR_NUMBER"

echo "Dispatched gpu-tests.yaml for PR #${PR_NUMBER} at ${SHA} (by ${ACTOR})"
