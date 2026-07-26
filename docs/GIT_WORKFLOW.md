# Contributing Guide

Guidelines for contributing to Granite Switch.

## Quick Start

1. Fork the repository
2. Create a branch: `git checkout -b feature/your-feature`
3. Make changes and commit
4. Push and open a Pull Request

## Branching

- **`main`**: Stable branch, always ready for release
- **Feature branches**: `feature/short-description`
- **Bugfix branches**: `bugfix/short-description`

## Workflow

```bash
# 1. Create branch from main
git checkout main
git pull origin main
git checkout -b feature/your-feature

# 2. Make changes and commit (-s adds your DCO sign-off)
git add <files>
git commit -s -m "Add feature X"

# 3. Keep up-to-date with main
git fetch origin
git rebase origin/main

# 4. Push and create PR
git push origin feature/your-feature
```

## Commit Messages

Write clear commit messages that explain **what** changed and **why**:

```
Short summary (50 chars or less)

Longer explanation if needed. Explain what changed and why,
not how (the diff shows how).

Fixes #123
```

**Good examples:**
- "Fix batch indexing for variable sequence lengths"
- "Add serialization roundtrip test"
- "Update supported models documentation"

**Avoid:**
- "fix bug" (what bug?)
- "update code" (what changed?)
- "WIP" (squash before merging)

## Sign-off (DCO)

Every commit must carry a `Signed-off-by:` trailer to certify you agree to the
[Developer Certificate of Origin](https://developercertificate.org/). The DCO check
blocks any PR containing an unsigned (non-merge) commit.

```bash
# Sign off a single commit as you make it
git commit -s -m "Your commit message"

# Sign off every commit on an existing branch
git rebase --signoff origin/main
```

The pre-commit hooks add and verify the sign-off for you automatically once installed.
See [CICD.md](CICD.md) for the full hook and CI setup.

## Code Quality

Before committing:

1. **Run tests**: `uv run pytest tests/ -v`
2. **Check comments match code** — stale comments are worse than no comments
3. **Update docs** if behavior changed

## Pull Requests

- Target the `main` branch
- Include a clear description of changes
- Reference related issues
- Ensure tests pass

## Questions?

Open an issue or start a discussion.
