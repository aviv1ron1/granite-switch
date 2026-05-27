---
name: validate-links
description: Validate local file links AND first-party Python imports across an entire repo (notebooks and markdown) and propose fixes for broken targets. Catches the kind of breakage that happens after renames, renumbering, or directory moves -- `[foo](./old_name.ipynb)` style links that silently 404 from GitHub/Colab/nbviewer, plus `from pkg.old_module import ...` imports that fail at notebook runtime. Use when the user asks to "validate links", "check links", "audit links", "verify links", "find broken links", "validate imports", or "make sure filenames align" after any restructuring.
---

# Link Validation Skill

Find every local link in the repo's `.ipynb` and `.md` files, flag the ones whose targets don't exist on disk, and propose fixes. Also validate first-party Python imports inside notebook code cells and `.py` files -- a `from granite_switch.tutorials.rag_display import ask` line silently breaks the same way a stale markdown link does when a module moves or gets renamed. Read-only by default; fixes happen only after the user confirms.

## What counts as a "local link"

- Markdown link syntax `[text](target)` where `target` does **not** start with `http://` or `https://`.
- Targets that point at a file with extension `.ipynb`, `.md`, `.py`, `.png`, `.jpg`, `.svg`, `.json`, or `.sh` (extend the list if the repo uses others — ask if unsure).
- Inside `.ipynb`: only **markdown cells** (`cell_type == "markdown"`). Code cells are skipped — strings inside Python aren't links.
- `attachment:` references (notebook-embedded images) are **not** local file links — skip them.

## What counts as a "stale label"

Display labels that look like a filename — `[`old_name.ipynb`](new_name.ipynb)`, `[old_name.ipynb](new_name.ipynb)`, or `[Title (old_name.ipynb)](new_name.ipynb)` — but where the filename in the label doesn't match the URL. After a rename, the URL often gets fixed while the label keeps the old name and silently lies about what the link points at. Treat these as fixable in the same pass as broken targets.

A label is considered "filename-shaped" when it contains a token that ends in one of the tracked extensions (`.ipynb`, `.md`, `.py`, ...). Plain prose labels like `"the simple pipeline"` are not stale even if the URL changes -- only fix labels that purport to name the file.

## What counts as a "broken import"

A first-party Python import is **broken** when its dotted module path does not resolve to a file or package on disk under the repo's package roots. Example: `from granite_switch.tutorials.rag_display import ask` is valid only when `<root>/granite_switch/tutorials/rag_display.py` (or `<root>/granite_switch/tutorials/rag_display/__init__.py`) exists, where `<root>` is one of the configured package roots (typically `src/` for src-layout repos, or `.` otherwise).

Scope:

- Only **first-party** packages count. Determine the set of first-party top-level package names by listing the immediate child directories of each package root that contain an `__init__.py` (e.g., `src/granite_switch/` -> first-party name `granite_switch`). Imports whose top-level name is not in this set (`numpy`, `torch`, `os`, `json`, ...) are skipped -- this is not a substitute for a real linter.
- Both forms are checked: `from A.B.C import name` and `import A.B.C [as alias]`.
- Inside `.ipynb`: only **code cells** (`cell_type == "code"`). Skip cells whose first non-empty source line is a Jupyter magic (`%`, `%%`, `!`).
- Inside `.py`: parse with `ast.parse`; this naturally ignores strings and comments and handles multi-line imports, parenthesized import lists, and relative imports. Relative imports (`from .foo import x`) are resolved against the importing file's package and are checked the same way.
- A dotted path resolves if walking it from a package root lands on a directory with `__init__.py` at every intermediate step and a `.py` file or package directory at the leaf. The imported names themselves (`ask`, `show_answer`) are **not** verified -- that needs real import-time analysis.

Discovering package roots:

1. If `pyproject.toml` has `[tool.setuptools.packages.find] where = ["src"]` (or `[tool.hatch.build.targets.wheel] packages = ["src/foo"]`, or `[tool.poetry] packages = [{ include = "foo", from = "src" }]`), use those.
2. Otherwise default to `.` and `src/` if either contains a top-level dir with `__init__.py`.
3. If the user's repo uses a layout the heuristic misses, ask before guessing.

## Workflow

### 1. Discover

Run from the repo root:

- List all `.ipynb`, `.md`, and `.py` files (respect `.gitignore` -- use `git ls-files '*.ipynb' '*.md' '*.py'` so you don't audit vendored copies in `node_modules/`, `.venv/`, etc.).
- Build a set of every existing file path in the repo (`git ls-files`) -- this is what link targets are checked against.
- Determine the import package roots and first-party package names per "What counts as a broken import" above. If `pyproject.toml` is missing or unreadable, fall back to `.` + `src/` and report which roots/packages were used so the user can correct the assumption.

### 2. Scan

For each file:

- For `.md` -- read the raw text and run the **link** scan only.
- For `.ipynb` -- parse JSON, iterate `cells`. Run the **link** scan against `markdown` cells (join `source` to a string). Run the **import** scan against `code` cells (skip cells whose first non-empty line is a `%`, `%%`, or `!` magic; otherwise concatenate `source` and feed to the import scanner).
- For `.py` -- run the **import** scan only (parse with `ast.parse`; record the line number from the AST node for reporting).

**Link scan:**

- Run the link regex `\[([^\]]+)\]\(([^)]+)\)` against the text.
- For each match, take the target, drop any `#anchor` fragment, resolve the path **relative to the file's directory** (so `../foo.md` from `tutorials/notebooks/x.ipynb` resolves to `tutorials/foo.md`).
- A target is **broken** if the resolved path doesn't exist on disk.
- Independently, flag the link as having a **stale label** when:
  - the label contains a filename-shaped token (ends in a tracked extension), AND
  - that token is not equal to `Path(target).name` (the basename of the URL, after stripping any `#anchor`).

  Stale labels are reported even when the target itself resolves cleanly -- the URL works, but the label lies about what it points at.

**Import scan:**

- For notebook code cells, parse the joined source with `ast.parse`. Wrap in `try/except SyntaxError` and skip cells that fail to parse (rare, usually transient half-edited cells); note the cell index in the skip log so the user can investigate.
- For `.py` files, parse the whole file the same way.
- Walk `ast.Import` and `ast.ImportFrom` nodes. For each, build the dotted module path:
  - `import a.b.c` -> `a.b.c` per alias.
  - `from a.b import c, d` -> check `a.b` resolves; the imported names are not verified, but if `a.b.c` *also* resolves as a submodule path, prefer that interpretation when reporting (it makes the suggestion more specific).
  - `from . import x` and `from .. import x` -> compute the absolute package by walking up from the file's package, then check that resolves.
- Filter to first-party top-level names. Skip everything else.
- A first-party dotted path is **broken** when no package root contains a matching directory-or-file chain (intermediate dirs need `__init__.py`; leaf can be either `<name>.py` or `<name>/__init__.py`).

### 3. Report

Present a single report grouped by source file, in this shape:

```
BROKEN LINKS

tutorials/notebooks/00_hello_adapter.ipynb (cell 0)
  ./hello_mellea.ipynb                          -> closest match: 01_hello_mellea.ipynb
  ../notebooks/03_compose_granite_switch.ipynb  -> closest match: 04_compose_granite_switch.ipynb

docs/SOMETHING.md (line 42)
  ../old/path/file.md                           -> no close match found

STALE LABELS (target works, but the label names the wrong file)

tutorials/README.md (line 15)
  [03_01_old_name.ipynb](notebooks/03_01_new_name.ipynb)
    -> label should be `03_01_new_name.ipynb`

BROKEN IMPORTS (first-party module path does not resolve on disk)

tutorials/notebooks/03_01_rag_101.ipynb (cell 4)
  from granite_switch.tutorials.rag_displays import ask
    -> closest match: granite_switch.tutorials.rag_display

src/granite_switch/composer/old_helpers.py (line 12)
  from granite_switch.composer.weight_remap import AdapterRemapper
    -> closest match: granite_switch.composer.weight_remapper

(package roots used: src/  |  first-party packages: granite_switch)
```

For each broken link, compute a "closest match" by:

1. Take the basename of the broken target (`hello_mellea.ipynb`).
2. Among all existing files in the repo with the same extension, prefer the one whose basename has the smallest edit distance (or contains the broken basename as a substring, or vice versa). Renumbering cases -- `03_compose_x.ipynb` vs `04_compose_x.ipynb` -- should match strongly.
3. If no candidate is closer than ~50% similar, report "no close match found" rather than guessing.

For each broken import, compute a closest match against the set of all valid first-party dotted paths (every `.py` file and package directory under each package root, expressed in dotted form). Use the same edit-distance heuristic, but match on the **full dotted path**, not just the leaf, so `granite_switch.tutorials.rag_displays` correctly suggests `granite_switch.tutorials.rag_display` rather than some unrelated `rag_display` elsewhere. As with links, suppress suggestions weaker than ~50% similar.

Also note when a `BROKEN` link or import has multiple plausible matches (e.g., `02_govt_rag_pipeline.ipynb` is gone and the repo now has `03_01_*`, `03_02_*`, `03_03_*`) -- list them all and ask the user which one to use.

### 4. Propose fixes

After the report, ask the user:

- **High-confidence renames** (single obvious match, just a number prefix change): show the exact replacements you'd make as a list and ask for approval as a batch.
- **Ambiguous cases**: ask one question per ambiguous link or import, presenting candidates as options.
- **No-match cases**: ask whether to drop the link/import, leave it, or point it somewhere else.
- **Stale labels**: include label-only fixes in the same approval batch as the URL fixes. When a single broken link has both a broken URL *and* a stale label (common after a rename), propose fixing both at once - the user shouldn't have to approve the URL, run the skill again, and approve the label separately. Default to "yes, fix labels too" for filename-shaped labels; only ask separately when the label is something other than a bare filename (e.g. a sentence that happens to mention the old filename).
- **Broken imports**: treat the same as broken links. High-confidence module-rename fixes (`weight_remap` -> `weight_remapper`) go in the batch; ambiguous ones become individual questions. When the same broken import appears in many files, propose a single repo-wide find-and-replace for that exact `from ... import` / `import ...` line and apply it everywhere at once -- a typo'd module name is almost never correct in one file and wrong in another.

Do not edit anything until the user confirms.

### 5. Apply fixes

For `.md` and `.py` files, use `Edit` with a precise `old_string` that includes enough surrounding context to be unique.

For `.ipynb` files, use `NotebookEdit` - `Edit` will refuse on notebooks. You'll need the cell's `id`, which you already saw in step 2; pass it as `cell_id`. Replace the full cell `new_source` with the corrected text.

When a cell has multiple fixes pending (broken URL + stale label, or several broken imports, or a mix), apply them in the **same** `NotebookEdit`/`Edit` call. Two passes through the same cell wastes tool calls and risks the second edit racing a linter that reformats the file between Reads.

After each edit, do not re-read the file - `NotebookEdit`/`Edit` errors loudly if the change failed.

### 6. Verify

Re-run the scanner from step 2. The report should show **0 broken links**, **0 stale labels**, *and* **0 broken imports**. If it still shows some, investigate - don't declare done. As a belt-and-braces check, also `git ls-files | xargs grep -l <old_token>` for any string fragment that was renamed (e.g. `govt_rag`); the scanner only catches strings inside `[...](...)` syntax and parsed `import` statements, and the same token may appear elsewhere (Colab badge URLs, prose, code comments, dynamic `importlib.import_module(...)` calls) where it's just as broken.

## Reference scanner

This Python snippet implements steps 1-3 and is safe to copy verbatim into a `Bash` call:

```python
import json, re, subprocess
from pathlib import Path

repo = Path('.').resolve()
tracked = subprocess.check_output(
    ['git', 'ls-files'], cwd=repo, text=True
).splitlines()
existing = {(repo / p).resolve() for p in tracked}
# Directories that contain tracked files - so dir-style links like
# `[scripts/](scripts/)` or `[docs/](../docs/)` aren't false-positived.
existing_dirs = set()
for p in tracked:
    for parent in (repo / p).resolve().parents:
        existing_dirs.add(parent)

link_re = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
ext_ok = {'.ipynb', '.md', '.py', '.png', '.jpg', '.jpeg', '.svg', '.json', '.sh'}

def scan_text(text, source_path, source_label):
    """Return (broken, stale_labels) tuples for one file.

    broken       : (source_label, target, basename)
    stale_labels : (source_label, label_text, target, expected_label_token)
    """
    broken = []
    stale = []
    # Token in the label that looks like a filename (ends in a tracked ext).
    label_filename_re = re.compile(
        r'[\w./-]+\.(?:ipynb|md|py|png|jpg|jpeg|svg|json|sh)\b',
        re.IGNORECASE,
    )
    for m in link_re.finditer(text):
        label_text = m.group(1)
        target = m.group(2).strip()
        if target.startswith(('http://', 'https://', 'mailto:', '#', 'attachment:')):
            continue
        bare = target.split('#')[0].split('?')[0]
        if not bare:
            continue
        ext = Path(bare).suffix.lower()
        if ext and ext not in ext_ok:
            continue
        resolved = (source_path.parent / bare).resolve()
        target_basename = Path(bare).name
        target_ok = resolved in existing or (not ext and resolved in existing_dirs)
        if not target_ok:
            broken.append((source_label, target, target_basename))
            # Don't double-report a broken link as also having a stale label;
            # fixing the URL is the load-bearing part. The label gets fixed
            # in the same edit per "What counts as a stale label" guidance.
            continue
        # Target resolves. Now check whether the label names a *different* file.
        for tok_match in label_filename_re.finditer(label_text):
            label_token = tok_match.group(0).split('/')[-1]
            if label_token != target_basename:
                stale.append((source_label, label_text, target, target_basename))
                break
    return broken, stale

broken = []
stale = []
for rel in tracked:
    p = repo / rel
    if not p.exists():
        continue
    if p.suffix == '.md':
        b, s = scan_text(p.read_text(), p, rel)
        broken += b; stale += s
    elif p.suffix == '.ipynb':
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        for ci, cell in enumerate(data.get('cells', [])):
            if cell.get('cell_type') != 'markdown':
                continue
            src = ''.join(cell.get('source', []))
            b, s = scan_text(src, p, f'{rel} (cell {ci})')
            broken += b; stale += s

print('BROKEN LINKS')
for label, target, _ in broken:
    print(f'{label}\n  {target}')
print(f'\n{len(broken)} broken link(s)')

print('\nSTALE LABELS')
for label, ltext, target, expected in stale:
    print(f'{label}\n  [{ltext}]({target})  -> label should name {expected}')
print(f'\n{len(stale)} stale label(s)')
```

For closest-match suggestions, extend the script to compute `difflib.get_close_matches(basename, [Path(f).name for f in tracked if Path(f).suffix == ext], n=3, cutoff=0.5)`.

## Reference import scanner

Drop-in companion to the link scanner above. Run from the repo root after the `tracked` / `existing` sets are built:

```python
import ast, json, tomllib
from pathlib import Path

def discover_package_roots(repo: Path):
    """Return (roots, first_party_names) using pyproject.toml when possible."""
    roots: list[Path] = []
    pyproject = repo / 'pyproject.toml'
    if pyproject.exists():
        cfg = tomllib.loads(pyproject.read_text())
        find = cfg.get('tool', {}).get('setuptools', {}).get('packages', {}).get('find', {})
        for w in find.get('where', []) or []:
            roots.append((repo / w).resolve())
        # hatch / poetry / flit fallbacks omitted for brevity -- add as needed.
    if not roots:
        for cand in ('.', 'src'):
            p = (repo / cand).resolve()
            if p.exists():
                roots.append(p)
    first_party = set()
    for r in roots:
        if not r.exists():
            continue
        for child in r.iterdir():
            if child.is_dir() and (child / '__init__.py').exists():
                first_party.add(child.name)
    return roots, first_party

def module_resolves(dotted: str, roots: list[Path]) -> bool:
    parts = dotted.split('.')
    for root in roots:
        cur = root
        ok = True
        for i, part in enumerate(parts):
            is_last = i == len(parts) - 1
            pkg_dir = cur / part
            if pkg_dir.is_dir() and (pkg_dir / '__init__.py').exists():
                cur = pkg_dir
                continue
            if is_last and (cur / f'{part}.py').exists():
                return True
            ok = False
            break
        if ok:
            return True
    return False

def resolve_relative(file_path: Path, level: int, module: str | None, roots: list[Path]) -> str | None:
    """Turn `from ..foo.bar import x` into an absolute dotted path, or None if outside any package root."""
    for root in roots:
        try:
            rel = file_path.resolve().relative_to(root)
        except ValueError:
            continue
        # Drop the file name; walk up `level` package boundaries.
        pkg_parts = list(rel.parts[:-1])
        if level - 1 > len(pkg_parts):
            return None
        base = pkg_parts[: len(pkg_parts) - (level - 1)] if level > 1 else pkg_parts
        tail = module.split('.') if module else []
        return '.'.join(base + tail)
    return None

def scan_imports(source: str, source_label: str, file_path: Path,
                 roots: list[Path], first_party: set[str]) -> list[tuple[str, str, int]]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split('.')[0]
                if top in first_party and not module_resolves(alias.name, roots):
                    out.append((source_label, f'import {alias.name}', node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative
                dotted = resolve_relative(file_path, node.level, node.module, roots)
                if dotted is None:
                    continue
            else:
                dotted = node.module or ''
            if not dotted:
                continue
            top = dotted.split('.')[0]
            if top not in first_party:
                continue
            # Prefer the more specific submodule form when a single `name` is imported
            # and `dotted.name` itself resolves -- it makes the suggestion sharper.
            if len(node.names) == 1 and module_resolves(f'{dotted}.{node.names[0].name}', roots):
                continue
            if not module_resolves(dotted, roots):
                names = ', '.join(a.name for a in node.names)
                out.append((source_label, f'from {dotted} import {names}', node.lineno))
    return out

# Usage:
# roots, first_party = discover_package_roots(repo)
# broken_imports: list[tuple[str, str, int]] = []
# for rel in tracked:
#     p = repo / rel
#     if p.suffix == '.py':
#         broken_imports += scan_imports(p.read_text(), rel, p, roots, first_party)
#     elif p.suffix == '.ipynb':
#         data = json.loads(p.read_text())
#         for ci, cell in enumerate(data.get('cells', [])):
#             if cell.get('cell_type') != 'code':
#                 continue
#             src_lines = cell.get('source', [])
#             if not src_lines:
#                 continue
#             first_nonblank = next((l for l in src_lines if l.strip()), '')
#             if first_nonblank.lstrip().startswith(('%', '!')):
#                 continue
#             src = ''.join(src_lines)
#             broken_imports += scan_imports(src, f'{rel} (cell {ci})', p, roots, first_party)
```

For closest-match import suggestions, build the set of all valid first-party dotted paths once (every `.py` file and package dir under the roots, expressed as dotted form), then `difflib.get_close_matches(broken_dotted, valid_dotted, n=3, cutoff=0.5)`.

## Hard rules

- **Never edit before the user confirms.** Even "obvious" renumbering fixes go through approval.
- **Never delete a link target or import.** If a broken link or import has no plausible replacement, ask the user -- don't silently strip the link or comment out the import.
- **Don't string-search code cells for links.** A string `"./old_name.ipynb"` inside a Python cell might be load-bearing test data, not a link. Code cells are scanned with `ast` for **imports only**, never with the link regex.
- **Don't follow symlinks blindly.** If `git ls-files` lists a symlink, treat the symlink path as the file location for resolution purposes.
- **Don't audit vendored trees.** `git ls-files` already excludes them; do not fall back to `find` or `glob` that would re-include `.venv/`, `node_modules/`, `dist/`, etc.
- **Fix labels alongside URLs in the same edit.** When a link's target is renamed, the display label often becomes stale at the same time. Don't make the user run the skill twice - propose URL + label fixes together, apply them in one `Edit`/`NotebookEdit` call per cell, and only break the work into separate approvals when a label is genuinely ambiguous (e.g. prose, not a bare filename).
- **Only validate first-party imports.** `numpy`, `torch`, stdlib, etc. are out of scope -- this skill is checking whether the repo's own module paths still resolve after a rename, not running a real linter.
- **Don't verify imported names.** `from pkg.mod import some_name` is checked only at the module level (`pkg.mod`). Confirming `some_name` actually exists requires importing the module, which is out of scope.

## When NOT to use this skill

- The user is asking about external URL liveness (HTTP 200 / 404) -- that's a different tool (link-checker against the network).
- The user wants to audit cross-references inside a single notebook (e.g., section anchors) -- that's narrower and the regex above won't cover it.
- The user wants a full static type/name check (verify imported attributes exist, catch unused imports, flag third-party version mismatches) -- use a real linter (`ruff`, `pyright`, `mypy`). This skill only checks that first-party module *paths* resolve on disk.
