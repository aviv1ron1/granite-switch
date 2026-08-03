#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Handle a /gpu-test* PR comment: verify the commenter holds Maintain/Admin, work
# out which test scope was asked for, then dispatch gpu-tests.yaml against the PR's
# head commit. On rejection, react 👎 and reply naming the requirement.
#
# Three commands, differing only in the `suite` input they dispatch:
#   /gpu-test        full   — all five suites, most of a day
#   /gpu-test-short  short  — tests/vllm/ + tests/integration/
#   /gpu-test-dev    dev    — one fast GPU file, a couple of minutes
#
# The suite NAME is dispatched, never a path list: gpu-tests.yaml owns the mapping
# and validates the name against a fixed set. See the `suite` input there.
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
# Usage: gpu_test_command.sh <actor-login> <pr-number> <comment-id> <comment-body>
# Env:   GH_TOKEN, GITHUB_REPOSITORY, DEFAULT_BRANCH, SCRIPT_DIR
set -euo pipefail

ACTOR="${1:?usage: gpu_test_command.sh <actor-login> <pr-number> <comment-id> <comment-body>}"
PR_NUMBER="${2:?missing pr number}"
COMMENT_ID="${3:?missing comment id}"
# May legitimately be empty or multi-line, so no :? guard.
BODY="${4:-}"

REPO="${GITHUB_REPOSITORY:?}"
DEFAULT_BRANCH="${DEFAULT_BRANCH:?}"
SCRIPT_DIR="${SCRIPT_DIR:?}"

react() {
  gh api -X POST "repos/${REPO}/issues/comments/${COMMENT_ID}/reactions" \
    -f content="$1" >/dev/null
}

reply() {
  gh api -X POST "repos/${REPO}/issues/${PR_NUMBER}/comments" -f body="$1" >/dev/null
}

# check_role.sh exits non-zero (and prints the role) when not authorized.
if ! ROLE_MSG="$("${SCRIPT_DIR}/check_role.sh" "$ACTOR" 2>&1)"; then
  react '-1'
  reply "@${ACTOR} the GPU test commands require the **Maintain** or **Admin** role. Not launching."
  echo "$ROLE_MSG" >&2
  exit 1
fi

# Which scope? First whitespace-delimited token of the FIRST line, so
# "/gpu-test-dev please" works and a command followed by prose or a second
# paragraph still parses. \r is stripped because GitHub sends CRLF line endings.
CMD="$(printf '%s' "$BODY" | head -n1 | tr -d '\r' | awk '{print $1}')"

# Matched EXACTLY, not by prefix. The calling workflow's `if:` guard is
# startsWith(body, '/gpu-test'), which is a cheap prefilter and nothing more --
# under it, "/gpu-testing on this later" reaches us and used to launch a full
# five-suite run. Anything not on this list is now declined.
case "$CMD" in
  /gpu-test)       SUITE="full"  ;;
  /gpu-test-short) SUITE="short" ;;
  /gpu-test-dev)   SUITE="dev"   ;;
  *)
    react 'confused'
    reply "@${ACTOR} \`${CMD}\` is not a GPU test command. Try one of:
- \`/gpu-test\` — all five suites (hours)
- \`/gpu-test-short\` — \`tests/vllm/\` + \`tests/integration/\`
- \`/gpu-test-dev\` — one fast GPU file (a couple of minutes)"
    # Exit 0: a mistyped command is user error, not a broken workflow. A red X on
    # the launcher run would send someone looking for a bug that isn't there.
    echo "Not a recognised command: '$CMD' — declined." >&2
    exit 0
    ;;
esac

SHA="$(gh api "repos/${REPO}/pulls/${PR_NUMBER}" --jq '.head.sha')"

react 'rocket'

gh workflow run gpu-tests.yaml \
  --ref "$DEFAULT_BRANCH" \
  -f sha="$SHA" \
  -f pr_number="$PR_NUMBER" \
  -f suite="$SUITE"

echo "Dispatched gpu-tests.yaml (suite=${SUITE}) for PR #${PR_NUMBER} at ${SHA} (by ${ACTOR})"
