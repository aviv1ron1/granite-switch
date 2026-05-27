---
name: tutorial-notebook
description: Polish an existing Jupyter tutorial notebook — or scaffold a new one from scratch — so it teaches a first-time reader clearly. Enforces a standard template (title → metadata → intro → prerequisites → numbered sections → next steps), catches common bugs (broken imports, silent data loads, stale thresholds, dead code), and adds the load-bearing polish items that actually move the needle (diagrams, explanatory comments, adapter descriptions, link validation). Use when the user asks to "improve a notebook", "make a notebook perfect", "apply the template", or is creating a new tutorial and wants it to match their existing ones.
---

# Tutorial Notebook Skill

This skill produces tutorial notebooks that a first-time reader can open, execute, and understand without needing to consult other docs. It was distilled from an end-to-end polish of `tutorials/notebooks/rag_flow.ipynb` and the lessons from what made each change earn its place.

## Core principle

**Comments, cells, and sections earn their place by answering WHY, not what.** If removing a comment or splitting a cell wouldn't help a future reader make or avoid a decision, don't add it. Apply every checklist item below through that lens — don't mechanically tick boxes.

## Two modes

- **Polish mode** — user hands you an existing notebook. Read it end-to-end first; don't edit before you understand the whole narrative. Make changes one at a time so each can be verified.
- **Scaffold mode** — user wants a new notebook from scratch. Start from the [template skeleton](#template-skeleton) below and fill in content. Still apply the full checklist before declaring done.

In both modes, always check against the same rubric — that's what makes notebooks feel like siblings instead of cousins.

## Interaction rhythm

- **Never batch large rewrites.** Propose changes one at a time, each with a brief "why." Let the user say yes/no/adjust before moving on.
- **Push back honestly when asked to do something that adds noise rather than clarity.** If a requested change would hurt the first-time reader (e.g., collapsing load-bearing reference material behind closed `<details>`, or wrapping unrelated functions in a namespace class), say so with reasoning — don't silently comply.
- **Show diffs, not summaries.** After an edit, show the actual changed lines. "Summary of changes: …" is less trustworthy than the diff itself.
- **Verify after each code edit.** `python3 -c "import ast; ast.parse(open(file).read())"` for Python files, `python3 -c "import json; json.load(open(nb))"` for notebooks. Also run `git diff --stat` to sanity-check scope.

---

## The checklist

Work through this in order. Each item has: **what**, **why**, and **how to check**.

### 1. Correctness and bugs (do these FIRST — everything else is polish)

These are the ones that make a notebook fail to run for a first-time user.

- **Broken import paths.** Imports that assume the repo is on `sys.path` (e.g., `from tutorials.scripts.xyz import ...` when there are no `__init__.py` files) will `ImportError` for any user who opens the notebook cold. Fix with `import sys; sys.path.insert(0, "../scripts")` + plain import. *Check:* open the notebook's directory, look at the import path, and ask "does this actually work without my PYTHONPATH being set right?"

- **Threshold / constant mismatches between logic and display.** A function that *decides* using threshold `0.5` while the *display helper* badges using `0.4` will produce contradictory output ("🟢 passed AND 🔴 blocked"). Grep for the same threshold across all cells; make sure they agree. This was a real bug in `rag_flow.ipynb`.

- **Unused constants and dead config.** Constants defined in the config cell that aren't referenced anywhere mislead readers into thinking they matter. Delete them. If the function they were meant for returns a string verdict instead of a score, the "threshold constant" is nonsense — that was the `ANSWERABILITY_THRESHOLD` case.

- **Stale comments and docstrings.** `# QC returns CLEAR when...` — what's QC? If a function name appears in a docstring, make sure it's the *current* name, not an internal abbreviation.

### 2. Template structure

Every tutorial notebook should follow this shape. Deviations need a reason.

```
# H1 Title                          ← cell 0 starts here
Metadata line (Duration only — full Prerequisites section follows below)
Intro paragraph (what it demonstrates, one or two sentences)
Why this approach (if the choice isn't obvious — e.g., "Why vLLM:")
What you'll learn (bullets — first bullet names the concrete deliverable as a learning outcome, remaining bullets are transferable conceptual takeaways)

## Prerequisites                   <- still cell 0, or next cell
1. Install
2. Get artifacts (models, data)
3. Start servers
4. Verify
Pointer to the softer-intro notebook and PREREQUISITES.md for depth

---                                 ← visual break; diagram lives in its own cell
Intrinsics / components used        ← if the tutorial exercises multiple adapters/tools
Pipeline / architecture diagram     ← image attachment, on its own cell

## 1 · <section name>               ← numbered H2s, numbered 1..N
[one-line intro explaining the section's purpose]
<code cell(s)>

## 2 · <section name>
...

## N · Next steps                   ← terminal section; numbering is a style call
- Adapt to your own app (point at the reusable function)
- Related tutorials / how-tos
- External references (library docs, model cards)
```

**Subsection rules (H3):** use sparingly, only when one H2 has multiple distinct helpers. In `rag_flow.ipynb`, §5 splits into `5a · Display helpers (printing only - not part of the pipeline)` because the display utilities are conceptually separate from the pipeline function above them. Don't force subsections for sections that have one concept.

### 3. Intro cell (cell 0) — the highest-leverage surface

A cold reader decides in 10 seconds whether the notebook is for them. That decision happens in cell 0.

- **H1 title** matches the subject of the tutorial, not the repo.
- **Metadata line** directly under the title: `**Duration:** ~X min (first run)`. The full `## Prerequisites` section is right below, so no need to link to it from the metadata line.
- **Motivation paragraphs** should be two short paragraphs, not one 90-word wall. First paragraph: what this demonstrates. Second paragraph (optional, italicized): the constraint that explains *why this approach*. Example: `*Why vLLM:* the mellea intrinsics API currently supports vLLM only.`
- **What you'll learn:** 3-5 bullets - one consolidated list, no separate "What you'll build" section. Lead with a bullet that names the concrete deliverable phrased as a learning outcome (e.g., `"How to build a 7-turn conversation that exercises every step of the pipeline"`), then follow with bullets about transferable conceptual takeaways. Bullets should not be a list of cells - `"how to call foo()"` is too mechanical; `"how to chain multiple intrinsics into one RAG pipeline"` is right.
- **Adapters used callout:** directly after the "What you'll learn" bullets, add a one-line `**Adapters used:**` paragraph that names which adapter libraries (and specific intrinsics within them) the notebook exercises, each linked to its HuggingFace repo. Example: `**Adapters used:** intrinsics from the [Core](https://huggingface.co/ibm-granite/granitelib-core-r1.0) library (\`context-attribution\`, \`uncertainty\`) and the [Guardian](https://huggingface.co/ibm-granite/granitelib-guardian-r1.0) library (\`guardian-core\`).` This lets a reader skimming the intro instantly see whether the notebook touches the capability they care about, without reading the full body. Keep it to one sentence; list only adapters the notebook actually *invokes* (not ones mentioned in reference tables). If the notebook has no "What you'll learn" list (freeform intro), place the callout immediately before the Prerequisites section. Keep this list in sync with any top-level README's "where used in tutorials" column — mismatches surface fast.
- **Prerequisites section:** numbered checklist with copy-pasteable commands. Every installation step, every "start this server" step, and a verification command (`curl ...`) the reader can run before moving on. Don't just link to `PREREQUISITES.md` - inline what they need, *then* link to the full doc for depth.

### 4. Component/adapter introduction table

If the tutorial uses more than 2–3 distinct libraries, adapters, or intrinsics, add a compact reference table near the top (right before the pipeline diagram). Two columns:

| Component | Role |
|-----------|------|
| `foo.bar` | One-line description of what it does in *this* tutorial. |

Not an API reference — just enough that a reader skimming the notebook knows what each name means before they hit it in code.

### 5. Pipeline / architecture diagram

A diagram is usually worth including. Default to adding one when the notebook executes a multi-step flow with branching, or when a conceptual illustration would help a reader form the right mental model before reading code. Skip a diagram only when the flow is trivially linear (e.g., a two-cell "load model, run inference" demo) and a picture would add nothing a section header doesn't already convey.

**Two acceptable formats, both in use in this repo:**

1. **Image cell attachment** - markdown cell containing `![image.png](attachment:image.png)` with the PNG embedded under that cell's `attachments` metadata. Used by `granite_switch_with_hf.ipynb`. Renders everywhere (GitHub, nbviewer, JupyterLab, VS Code).
2. **Mermaid rendered from a code cell** - a Python cell that defines a Mermaid source string and renders it (e.g., via `IPython.display`). Used by `rag_flow.ipynb`. Easier to keep in sync with code labels and edit in-place.

Pick whichever fits the diagram and the notebook. Don't mix both styles within one notebook.

**The skill cannot generate or attach a PNG itself.** When the chosen format is an image attachment, describe to the user exactly what the diagram should show (the steps, branches, terminal states, labels you'd want), leave a placeholder markdown cell with a TODO comment to reserve the slot, and ask the user to produce and attach the image. For Mermaid, the skill *can* author the source directly.

**Diagram content rules** (apply when describing what the image should show):
- **Include every early-exit branch**, not just the happy path. A reader scanning the diagram should see all possible terminal states (e.g., `BLOCKED`, `UNANSWERABLE`, `DONE`).
- **Match node labels to code.** If the display helper calls steps `[1a]`, `[1b]`, `[2]`... the diagram nodes should use the same tags. Shared vocabulary is the point.
- **Match terminal emoji to the code's print output.** If `show_answer` prints ⛔ for blocks and 🔍 for unanswerable, the diagram's terminal nodes use the same glyphs. This makes the diagram a legend for the runtime output.
- **Keep the diagram on its own cell.** Cell 0 has enough to carry without a diagram inside it.

### 6. Code cells — structure

- **Each `## N · ...` section has at most one concept per code cell.** If a cell is >80 lines doing two distinct things (e.g., `run_pipeline` + display helpers), split it with a markdown divider (`### Na · ...`). Subsections need a short intro markdown cell describing what the code does.

- **Extract helpers when extracting *gains* clarity.** The 7-line `ChatContext` build logic at the top of `run_pipeline` was pure bootstrapping; lifting it into `_build_context(history, query)` let the main function read as a clean 7-step sequence. Extract when it tightens; don't extract for the sake of extraction.

- **Don't extract "for namespacing."** A `Show` class containing three unrelated static methods adds ceremony without structure. Python's namespacing tool is the module/cell, not the class.

- **Shell-out lines — `%pip` for installs, `!` for everything else.** Installer lines (`pip install`, `pip uninstall`, `conda install`) use the Jupyter line magic: `%pip install -q -e "/content/granite-switch[vllm]"`. Every other shell-out — `!git clone ...`, `!python -m ...`, `!python script.py`, `!huggingface-cli ...`, `!curl ...`, `!ls`, `!head`, etc. — keeps the `!` prefix. **Why:** `%pip` is a Jupyter line magic that always installs into the kernel running the notebook; `!pip` shells out to whatever `pip` is on PATH, which in Colab and managed-Jupyter environments often differs from the kernel's interpreter and produces the classic "installed fine, still ImportError" failure for the reader. Only `%pip`/`%conda` get this treatment — the others are not line magics and `%`-prefixing them would either no-op or error.

  ```
  %pip install -q -e "/content/granite-switch[vllm]"   # good — targets the running kernel
  !pip install -q -e "/content/granite-switch[vllm]"   # bad — may install into the wrong interpreter
  !git clone https://github.com/...                    # good — git is not a line magic
  !python -m granite_switch.composer.compose_granite_switch ...   # good — same reason
  ```

### 7. Imports

**Consolidate imports into a single cell near the top of the notebook, not scattered across cells.** One dedicated imports cell (typically right after the config cell, or merged with it if config is small) makes dependencies visible at a glance and matches standard Python/Jupyter convention. Readers scanning the notebook can see the full set of external dependencies in one place instead of discovering them cell-by-cell.

**Placement:**
- Put the imports cell early — after the intro/prerequisites markdown, before the first substantive code section.
- If the config cell is small (a few constants), imports can live with it; otherwise keep imports in their own cell so neither drowns the other.
- Group imports conventionally: stdlib first, third-party next, local/project last, with blank lines between groups.

**Narrow exceptions — keep an import local to its cell only when:**
- The import has a heavy side effect at import time (registers a plugin, mutates global state) and the reader needs to see it happen at that point in the narrative.
- The import is genuinely optional/conditional (inside a `try/except` or guarded by a feature flag).

**Never** write `# MelleaDocument is used later in §4` above an import as a workaround for scattered placement — consolidate instead. Pointer comments are a smell.

### 8. Comments — earn or delete

Default to writing no comments. Add one only when:

- **The value is non-obvious.** `TOP_K = 20` deserves "balances recall against context budget; mt-rag-benchmark default." `VLLM_PORT = 8000` does not deserve a comment.
- **The ordering matters and a refactor could break it.** `# Harm check must run BEFORE scope check so harmful+out-of-scope queries are labeled harmful, not merely out-of-scope.` Without this, someone will swap them for "fail fast" and introduce a silent regression.
- **The value is a knob readers will tune.** `temperature=0.0` deserves "grounded RAG — we want the model to repeat the docs, not paraphrase. Also makes demos reproducible." Someone will bump it to 0.7 and break grounding.

**Do not** write comments that restate what the code says (`# retrieve top-K documents` above `retrieve_top_k_documents(...)`). Delete them.

### 9. Reference tables — every row parallel

Reference tables (like "what `show_intermediates` displays at each step") must have every row in the same shape. If row 1 says `"badge + raw score. Exits early if ≥ 0.5"`, row 4 should say `"badge + verdict string. Exits early if unanswerable"` — same structure, same verb, same vocabulary. Asymmetric rows look like bugs to a reader.

Also: the text in reference tables must match what the code prints. If the code renders `🟢 safe / 🔴 harmful`, the table says `🟢 safe / 🔴 harmful`, not `safe / flagged`.

### 10. Display rendering

- **Use `display(Markdown(...))`** for rich output in notebooks. Don't use ANSI color codes (`\x1b[32m...`) — they render in terminal only and look like garbage in rendered notebooks or exported HTML.
- **For collapsible detail** (large outputs, reference tables that take vertical space), use `<details open>...<summary>...</summary>...</details>`. Default to `<details open>` for load-bearing content (users can collapse it); use closed `<details>` only for truly optional depth.
- **Standard emoji glyphs** for status — keep them consistent across the notebook series: ⛔ block/refuse · 🔍 empty/unanswerable · ❓ clarification needed · ✅ pass/done · 🟢/🔴 safe/danger binary · 📄 document · 📚 collection · 🔖 citation/reference.

### 11. Helper scripts (supporting `.py` files)

If the tutorial loads data or does heavy setup in a sibling script:

- **Progress feedback for any operation > 5 seconds.** `tqdm` for downloads (use `httpx.stream()` with `Content-Length`), `tqdm` for batch processing, progress prints for shorter waits. Silent multi-minute operations make users think the notebook froze.
- **Atomic writes for persistent state.** When writing a file the notebook will later re-read (e.g., extracted jsonl), write to `path.tmp` first, then `os.replace(tmp, path)`. A Ctrl-C mid-write produces a truncated file that silently breaks subsequent runs — one of the worst classes of bug to debug.
- **Validate non-empty output loudly.** After parsing/loading, if the result has zero rows, raise `RuntimeError` with actionable guidance (`"Delete X and rerun"`), not a silent empty return.
- **Split timeouts.** `httpx.Timeout(total_seconds, connect=10.0)` instead of a flat `timeout=120`. Fails fast on unreachable servers; patient on slow transfers.
- **Escalate GPU/CPU warnings.** `print("Notice: ...")` gets lost in notebook output. Use `warnings.warn(...)` with a concrete time estimate ("~10 min on GPU vs. hours on CPU") so users can abort before committing.

### 12. Queries / demos — design intentionally

If the notebook ends with runnable demo cells, the demo set should *tour every exit path* in the system. For a pipeline with {happy path, ambiguous, unanswerable, out-of-scope, harmful} outcomes, include one demo of each. A demo that only shows the happy path teaches half the system.

Add one-line intent comments per demo: `# Q3 — resolves clarification: query rewrite uses history to reconstruct full question`. These teach what each demo *is testing*, beyond what the query text alone conveys.

### 13. Next steps section

Close every notebook with 3–5 bullets pointing the reader somewhere concrete:

1. **Adapt-to-your-app pointer:** name the reusable function/class and remind the reader it's lift-able. `run_pipeline(query, history)` is stateless — copy it as a starting point.
2. **Go deeper on this topic:** related how-to or tutorial in the repo.
3. **Extend with custom content:** how to bring your own adapter / corpus / model.
4. **Library deep-dive:** link to the framework's main repo/docs.
5. **Browse alternatives:** catalog of other adapters/models the reader could try.

Two bullets that both say "go compose your own model" is one bullet wasted — make sure every bullet opens a *distinct* next direction.

**Inter-notebook wiring rule:**

The granite-switch tutorial set uses descriptive filenames (no numeric prefixes), so wiring is judged by *content*, not by index. Two principles for the next-steps bullets:

1. **The producer is reachable from every consumer.** `compose_granite_switch.ipynb` produces the checkpoint that every other notebook consumes. Every notebook except the producer itself should include a "compose your own checkpoint" bullet pointing to it. A reader who lands on any consumer should be one click from the producer - they shouldn't have to discover it by reading every sibling.
2. **Don't link backward to softer-intro notebooks.** If a notebook is a deeper or harder version of another (e.g., `granite_switch_with_hf.ipynb` is the long-form version of `hello_adapter.ipynb`; `rag_flow.ipynb` is the long-form of `rag_101.ipynb`), the long-form should *not* link back to its softer sibling - the reader already passed it. The softer notebook *can* link forward to the long-form.

Every notebook should also link to whatever logical follow-ups exist for the reader (the next pipeline to try, the comparison/race demo, the framework's main repo). Three to five bullets is the right shape - see section 13's general rules.

**Use same-directory relative paths** (`./name.ipynb`) when all notebooks live in one folder - not `../notebooks/name.ipynb`. After editing, run a link-resolution check to catch typos and stale filenames:

```python
import json, re, pathlib
nbdir = pathlib.Path("tutorials/notebooks")
for nb_path in sorted(nbdir.glob("*.ipynb")):
    nb = json.loads(nb_path.read_text())
    for c in nb["cells"]:
        if c["cell_type"] != "markdown": continue
        src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        if "Next steps" not in src: continue
        for href in re.findall(r"\]\((\./[^)]+\.ipynb)\)", src):
            assert (nb_path.parent / href).resolve().exists(), f"{nb_path.name}: broken {href}"
print("all next-steps links resolve")
```

When notebooks get renamed or split, the next-steps sections of *every other notebook in the series* go stale silently. Always re-run the link check after any rename. The repo also has a `validate-links` skill that runs this check across notebooks and markdown together - prefer it for cross-cutting validation.

### 14. Links

- **Every external link:** verify with `curl -s -o /dev/null -w "%{http_code}" <url>`. Expect 200/301/302.
- **Every internal link:** verify the file exists on disk. Relative paths should work from the notebook's directory (notebooks live in `tutorials/notebooks/`, so `../PREREQUISITES.md` resolves to `tutorials/PREREQUISITES.md`).
- **Anchor links:** Markdown lowercases heading text to form anchors. `## Prerequisites` produces `#prerequisites`, not `#Prerequisites`. Check every in-notebook anchor.

### 15. Prose — light touch

For prose-clarity passes: fix stale references (section counts that have changed, helper function signatures that have changed), tighten walls of text (split 90-word single paragraphs into two), and make sure section intros say *what* and *why*, not just the section name restated. Don't rewrite prose that's already clear — every unnecessary change is a chance to introduce a regression.

---

## Template skeleton (scaffold mode)

For a brand-new notebook, start from this and fill it in. Delete sections that truly don't apply (e.g., no diagram if the flow is linear and has one step).

```markdown
# <Title — what the notebook accomplishes>

**Duration:** ~N min (first run)

This notebook demonstrates <one-sentence concrete pitch>. <One more sentence on scope.>

*Why <key choice>:* <one-line constraint explanation, if the choice isn't self-evident>

**What you'll learn:**
- How to build <concrete deliverable - "a 7-turn conversation that exercises every step", "a composed model checkpoint with two adapters", etc.>
- <Transferable takeaway 1>
- <Transferable takeaway 2>

**Adapters used:** intrinsics from the [<Library>](<hf-url>) library (`<adapter_1>`, `<adapter_2>`)<, and the [<Library>](<hf-url>) library (`<adapter_3>`)>.

## Prerequisites

1. **Install dependencies** (<GPU? CPU? which>):
   ```bash
   pip install "<extras>"
   ```
2. **Get <artifact>.** <How to obtain it, pointer to a ready-made option, pointer to a "compose your own" tutorial.>
3. **Start <service>** (if applicable):
   ```bash
   <start command>
   ```
4. **Verify:** `<verification command>`

<Pointer to softer-intro notebook if one exists, pointer to PREREQUISITES.md for depth.>
```

Then a second markdown cell with (if applicable) an intrinsics/components table and the diagram (image attachment — see section 5). Then numbered `## 1 · Section`, `## 2 · Section`, etc., each with a one-line intro markdown cell before its code. End with `## N · Next steps`.

---

## When working on an existing notebook

1. **Read the whole thing first.** Don't edit until you understand the arc: what it teaches, what the demos tour, what the reader is expected to walk away with.
2. **Identify the real bugs before the polish.** Run through section 1 of the checklist. A broken import is worth ten prose tweaks.
3. **Propose changes one at a time.** Each should be justifiable in one sentence. If you can't justify it, don't do it.
4. **After each change:** verify JSON validity, syntax, and that the diff scope matches what was planned. Show the diff.
5. **When multiple tasks conflict:** defer to the principle that serves the first-time reader. For example, a reader scanning a notebook benefits from one consolidated imports cell (full dependency list visible at a glance) more than from imports scattered next to their use sites.

---

## Universal anti-patterns — push back if the user asks for these

- **Classes that are just namespaces.** `class Show: @staticmethod def answer(r): ...` adds ceremony. Use functions in cells.
- **Scattering imports across cells near their first use.** Hurts scan-ability; readers lose the single-glance view of what the notebook depends on. Consolidate into one imports cell near the top.
- **Collapsing load-bearing reference material behind closed `<details>`.** `<details open>` is fine; closed is only for optional depth.
- **Splitting "What you'll build" out as its own section.** Use one consolidated "What you'll learn" list; if the concrete deliverable is load-bearing, make it the first bullet phrased as a learning outcome (`"How to build <deliverable>"`) rather than a separate header.
- **Renaming a section just because another notebook uses a different name.** Template consistency matters, but forcing "Prerequisites" when the existing "Before you start" reads better locally is cargo-cult.
- **Numbering every heading mechanically.** "Next steps" as "`## 6 · Next steps`" vs. unnumbered `## Next steps` is a style call, not a correctness one. Only enforce when the user has said they want strict numbering.

---

## Verification checklist (before declaring done)

- [ ] Notebook is valid JSON: `python3 -c "import json; json.load(open(PATH))"`.
- [ ] Every code cell parses: walk cells, `ast.parse(source)` each one.
- [ ] Structural overview (`for i, c in enumerate(nb['cells']): print(i, c['cell_type'], first_line)`) shows one H1, numbered H2s, H3s only under H2s that have them, code-cell length sensible (no >120-line monsters unless justified).
- [ ] All external URLs return 2xx/3xx.
- [ ] All internal links point at files/anchors that exist.
- [ ] Reference tables' badge glyphs match the code's actual print statements.
- [ ] Diagram terminals match the code's actual exit names (`blocked`, `unanswerable`, etc.).
- [ ] Intro cell has an **Adapters used:** callout naming every adapter library the notebook actually invokes, each linked to its HuggingFace repo.
- [ ] No `!pip install` / `!pip uninstall` / `!conda install` lines anywhere — installer lines use `%pip` / `%conda` so they target the running kernel.
- [ ] Imports are consolidated into one cell near the top, not scattered across cells next to their first use (narrow exceptions: side-effectful or conditional imports — see section 7).
- [ ] Running the notebook top-to-bottom with "Run All" should complete cleanly (requires the runtime environment — document this as a manual step for the user).

If any of these fails, fix before handing off. The skill is "produce a notebook that runs cleanly on first try for a cold reader" — missing verification undermines the whole exercise.
