# CI/CD Plan

This document describes the CI/CD setup for granite-switch — what runs locally, what runs on every PR, and the roadmap for future automation.

## Local: Pre-commit Hooks

Pre-commit hooks enforce code quality before a commit lands. They run automatically on `git commit` after a one-time setup.

### Setup

```bash
uv run pre-commit install                      # hooks for staged files
uv run pre-commit install --hook-type commit-msg   # DCO hook for commit messages
```

### What runs on every commit

| Hook | What it does | Auto-fix? |
|------|-------------|-----------|
| `ruff-format` | Formats Python files | Yes |
| `ruff` | Lints Python files (imports, style) | Yes (fixable issues) |
| `check-headers` | Ensures every `.py` file starts with `# SPDX-License-Identifier: Apache-2.0` | Yes |
| `check-dco` | Validates commit message has `Signed-off-by: Name <email>` | No — blocks commit |
| `check-toml` / `check-yaml` | Validates config file syntax | No |
| `end-of-file-fixer` / `trailing-whitespace` | Hygiene | Yes |
| `uv-lock` | Ensures `uv.lock` is in sync with `pyproject.toml` | No — run `uv lock` to fix |

### DCO sign-off

Every commit must be signed off to certify you agree to the [Developer Certificate of Origin](https://developercertificate.org/). Add it automatically with:

```bash
git commit -s -m "Your commit message"
```

To sign off all commits in an existing branch:

```bash
git rebase --signoff origin/main
```

### Running hooks manually

```bash
# Run all hooks on all files
uv run pre-commit run --all-files

# Run a specific hook
uv run pre-commit run ruff --all-files
uv run pre-commit run check-headers --all-files
```

---

## On Every PR: GitHub Actions

Four workflows run automatically when a PR is opened against `main` or a commit is pushed.

### `ci.yaml` — Lint + CPU tests + coverage

**Trigger:** Pull request or push to `main`

**Jobs:**
1. `lint` — runs `ruff format --check` and `ruff check`
2. `test-cpu` — runs `tests/unit/`, `tests/composer/`, and `tests/hf/` on Python 3.11 and 3.12 in parallel

Coverage is uploaded to [Codecov](https://codecov.io) after each test run (see [Coverage](#coverage) below).

### `check-headers.yaml` — SPDX header check

**Trigger:** Pull request to `main`

Runs `ci/check_headers.py` against all `.py` files in `src/` and `tests/`. Fails if any file is missing the SPDX header. The pre-commit hook auto-fixes this locally; the workflow is a safety net for PRs from contributors who haven't installed hooks.

### `dco.yaml` — DCO sign-off check

**Trigger:** Pull request to `main`

Checks that every non-merge commit in the PR has a `Signed-off-by:` line. Skips merge commits automatically.

---

## Manual: GPU Tests

GPU tests (`tests/vllm/`, `tests/integration/`) cannot run on GitHub-hosted runners. They are triggered manually by repository admins via the `gpu-tests.yaml` workflow.

**Trigger:** `workflow_dispatch` — only users with write access to the repo can trigger this from the GitHub UI (`Actions` → `GPU Tests` → `Run workflow`).

**Current state:** The workflow is scaffolded but inert until a [self-hosted runner](https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners/about-self-hosted-runners) with the `gpu` label is registered on a machine inside the IBM network. No code changes are needed when the runner is set up — just register it.

In the meantime, run GPU tests manually on the HPC environment:

```bash
make test-gpu        # vllm + integration tests
make test-gpu-full   # + regression suite
```

---

## Coverage

CI runs tests with `pytest-cov` and uploads `coverage.xml` to [Codecov](https://codecov.io). Codecov posts a per-PR diff comment showing which lines added in the PR are covered.

**One-time setup (repo admin):**
1. Sign up at [codecov.io](https://codecov.io) and link the repository
2. Add the `CODECOV_TOKEN` secret to the repository (`Settings` → `Secrets and variables` → `Actions`)

Until the token is configured, coverage upload silently no-ops (`fail_ci_if_error: false`) and does not block CI.

---

## Releases & Publishing

granite-switch publishes to [public PyPI](https://pypi.org/project/granite-switch/). Releases are created via GitHub Releases; publishing happens automatically.

### Versioning

Semantic versioning: `MAJOR.MINOR.PATCH`. The version lives in `pyproject.toml` as the single source of truth:

```toml
version = "0.2.0"
```

While in pre-1.0, increment `MINOR` for new features and `PATCH` for bug fixes. Bump to `1.0.0` when the public API stabilizes.

### Release process

1. **Open a release PR** from a `release/vX.Y.Z` branch:
   - Bump `version` in `pyproject.toml`
   - Add a section to `CHANGELOG.md`

2. **Merge to `main`**

3. **Create a GitHub Release** (`Releases` → `Draft a new release`):
   - Tag: `vX.Y.Z` (created from `main`)
   - Title: `vX.Y.Z`
   - Body: paste the `CHANGELOG.md` section for this release

4. **Click Publish release** — `publish.yaml` triggers automatically

### What `publish.yaml` does

Builds wheel + sdist with `uv build`, then uploads with `uv publish`. Uses **PyPI Trusted Publisher (OIDC)** — no token required.

### One-time PyPI setup (repo admin)

1. Create the project on PyPI (first release can be done manually)
2. Go to `pypi.org` → project page → `Manage` → `Publishing`
3. Add a Trusted Publisher:
   - Owner: `generative-computing`
   - Repository: `granite-switch`
   - Workflow: `publish.yaml`
   - Environment: `pypi`
4. Create a `pypi` environment in the GitHub repo (`Settings` → `Environments`)

---

## Roadmap

### Phase 2: Mypy Type Checking

Mypy is deferred because the codebase has no existing `[tool.mypy]` configuration and retrofitting type annotations across an existing codebase is a significant effort worth its own PR.

#### Strictness level: curated middle ground (not `--strict`)

`--strict` is the wrong choice here. PyTorch, HuggingFace, and vLLM all have incomplete or missing type stubs — `Any` leaks in from those library boundaries and propagates everywhere, meaning most mypy errors would be fighting stub quality rather than catching real bugs in granite-switch code. SPINE uses `strict = true` because it was designed with types in mind from day one; this codebase is retrofitting.

The recommended config enforces the flags that deliver the highest signal-to-noise ratio:

**`pyproject.toml`:**
```toml
[tool.mypy]
python_version = "3.11"
pretty = true
ignore_missing_imports = true   # vllm has no stubs; HF stubs are partial
disallow_untyped_defs = true    # every function must be annotated — highest-value flag
warn_return_any = true          # catch Any leaking out of our own functions
warn_unused_ignores = true      # keep # type: ignore comments from going stale
exclude = ["^tests/"]
```

`disallow_untyped_defs` is the core flag: it ensures every function added going forward is annotated, which is where mypy earns its keep. Tighten toward `--strict` incrementally as stub quality improves and the codebase is fully annotated.

**`.pre-commit-config.yaml`** (add to `local` repo block):
```yaml
- id: mypy
  name: MyPy
  entry: uv run --no-sync mypy src
  pass_filenames: false
  language: system
  files: '\.py$'
```

**`ci.yaml`** (add job):
```yaml
mypy:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: astral-sh/setup-uv@v5
    - run: uv sync --frozen --extra hf --extra compose
    - run: uv run mypy src
```
