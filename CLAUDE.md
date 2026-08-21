# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working with the user

Before planning or executing a task, ask clarifying questions whenever any aspect of the request is ambiguous (scope, target file, audience level, language, length, intended unit, etc.). Do not guess and proceed silently. If the request is fully unambiguous, you may skip this and act directly.

## Repository purpose

LaTeX source for a Brazilian (Portuguese) "Introdução a Deep Learning" course at PUCRS, taught by Prof. Lucas S. Kupssinskü. The LaTeX artifact families are:

- `slides/` — Beamer presentations, one `.tex` per lecture, organised by unit (`01-mlp` … `06-topicos`).
- `problems/` — Problem sets (`listaNN_*.tex`), organised by the same unit numbering.
- `exams/` — Provas. Each prova has a paired `_gabarito.tex` (answer key) that CI builds locally but excludes from the public deploy.
- `projects/` — Course project specs (`projetoNN_*.tex`), flat folder (projects span units). Same `\documentclass{dlproblemset}` as listas; deliberately contain only technical requirements, no assessment context (grading, deadlines, delivery mechanics live on moodle, not in the PDF).

## Build commands

Everything is LuaLaTeX + latexmk + `-shell-escape` (minted needs it). Always run from the **repo root**, never from inside a subdirectory, because slides reference repo-relative `\input{slides/utils/preamble}` paths and problem sets resolve `\documentclass{dlproblemset}` via the `TEXINPUTS=./problems/style//` extension set in `latexmkrc`.

```bash
make problems          # compile every problems/**/*.tex with \documentclass
make exams             # compile every exams/**/*.tex (includes _gabarito.tex)
make projects          # compile every projects/**/*.tex with \documentclass
make clean-problems    # latexmk -C on problems + remove build/
make clean-exams       # latexmk -C on exams
make clean-projects    # latexmk -C on projects

# Single file (slides, problems, or exams — same invocation):
latexmk -lualatex slides/01-mlp/02-perceptron.tex
```

PDFs land in `build/` (set by `latexmkrc`'s `$out_dir = 'build'`). The same `latexmkrc` enables `-shell-escape` for lualatex and points biber at the `slides/` directory for the shared `reference.bib`.

There is no `make slides` target, slides are compiled individually or through CI. CI compiles slides, problems, exams, and projects in one pass (`.github/workflows/latex-ci.yml`); project PDFs deploy flat to `deploy/projects/`.

## Rendering PDFs and PPTX to PNG (visual inspection)

When the user asks for a visual inspection of a compiled deck or a legacy `.pptx`, render to PNG and use the `Read` tool on the resulting file (it accepts images).

**Output directory rule:** always write inspection PNGs into `build/` (e.g. `build/inspect-<name>/`), never into the repo root. This keeps temporary debug images out of the working tree.

**PDF → PNG**, use `pdftoppm` (poppler, ships with TeX Live 2025, already on PATH):

```bash
# all pages
pdftoppm -png -r 120 build/01-gans.pdf build/inspect-gans/gans
# specific range (first arg after -f / -l is 1-indexed)
pdftoppm -png -r 120 -f 1 -l 3 build/01-gans.pdf build/inspect-gans/gans
```

Output filenames are `<prefix>-NN.png` (zero-padded). `-r 120` is a good default DPI for inspection, bump to 150 for fine detail.

**PPTX → PNG**, no LibreOffice on this machine, use Microsoft PowerPoint via COM from PowerShell. Run via the `PowerShell` tool:

```powershell
$ppt = New-Object -ComObject PowerPoint.Application
$pres = $ppt.Presentations.Open("C:\absolute\path\to\file.pptx", $true, $true, $false)
$out = "C:\absolute\path\to\out_dir"
New-Item -ItemType Directory -Force -Path $out | Out-Null
# single slide
$pres.Slides.Item(1).Export("$out\slide1.png", "PNG", 1280, 720)
# or all slides
# foreach ($s in $pres.Slides) { $s.Export("$out\slide$($s.SlideIndex).png", "PNG", 1280, 720) }
$pres.Close(); $ppt.Quit()
[System.Runtime.InteropServices.Marshal]::ReleaseComObject($ppt) | Out-Null
```

Notes: `Open(file, ReadOnly, Untitled, WithWindow)` with `WithWindow=$false` keeps PowerPoint headless. Always pass absolute Windows paths (the COM call does not understand MSYS `/c/...` paths). Always `Close()` and `Quit()` to avoid orphaned PowerPoint processes.

## CI behaviour worth knowing

- The compile loop in `latex-ci.yml` finds candidates with `find … -name '*.tex' | xargs grep -l '\documentclass'`, then **excludes** `slides/deprecated/`, `slides/teste.tex`, `slides/testetikz.tex`, and any `*_gabarito.tex`. New scratch files matching those patterns won't be built; new top-level `.tex` files anywhere else will.
- Lint step parses every `build/*.log` for undefined references, undefined citations, and overfull hboxes. These are advisory (`::warning`) only, they never fail the build. Only a non-zero `latexmk` exit fails CI.
- Deploy to `gh-pages` happens only on `release: published` (not on push). Compiled PDFs deploy from `deploy/` (slides flat, problems grouped by module subdir, exams flat, gabaritos excluded). Pushes and PRs only compile/validate; markdown-only changes (`**.md`) trigger nothing.

## Document architecture

**Slides** (`slides/**/*.tex`) all start with:
```latex
\documentclass[aspectratio=169,xcolor=dvipsnames]{beamer}
\input{slides/utils/preamble}
\input{slides/utils/custom-commands}
```
- `slides/utils/preamble.tex` is the single source of truth for fonts (XCharter via fontspec + matched `newtxmath[xcharter]` for math, requires LuaLaTeX), the SimplePlus beamer theme (loaded from `slides/SimplePlusTheme/`), pgfplots/tikz libraries, the `vscode-light` listings style for Python, and minted configuration. Don't re-`\usepackage` things already in the preamble.
- `slides/reference.bib` is the unified bibliography. `biber` is configured with `--input-directory slides` so all decks resolve it.
- `slides/img/` is the shared figure root, `slides/code/` holds Python snippets referenced from `\lstinputlisting`, `slides/ppt/` holds source PowerPoint/TikZ originals (not compiled).

**Problems** (`problems/**/*.tex`) use `\documentclass{dlproblemset}`. The class at `problems/style/dlproblemset.cls` bundles XCharter serif + matched `newtxmath`, KOMA `scrartcl` base, gray 1pt rules above the running header and below the footer, the split header (title left / subtitle right, in sans 8pt gray), and the bottom-left "Prof. Lucas Silveira Kupssinkü" footer. Each lista declares `\title{}` and `\subtitle{}` in the preamble; the body starts directly with content (no `\chapter*` or `\section*{Lista de Exercícios}` heading, no `\maketitle`). The class loads `graphicx`, `float`, `subfig`, `emoji`, `hyperref`, `amsmath`/`amssymb`/`amsthm`, `xcolor`, `microtype`, and `enumerate` already — listas only re-load what they genuinely need beyond that. Note `tikz`, `booktabs`, and `pgfplots` are **not** bundled, so a lista that uses tables or diagrams must `\usepackage` them explicitly (e.g. `\usepackage{booktabs}` for `\toprule`/`\midrule`, otherwise compilation fails with "Undefined control sequence").

**Projects** (`projects/*.tex`) also use `\documentclass{dlproblemset}` with `\title{}`/`\subtitle{}` (subtitle "Projeto N"), so everything said above about the class applies to them too.

**Exams** are standalone `article` documents (no shared style file). The `_gabarito.tex` variant of each prova holds the answer key and is filtered out of the public deploy pipeline.

**Interactive demos** (`docs/`) are the course's static website, the fourth artifact family. `docs/index.html` is the landing page (lists every lecture with PDF links plus an `exercise-chip chip-demo` linking its demo); each lecture demo lives at `docs/<slug>/` (kebab-case, e.g. `self-attention`, `lstm-gru`). A demo is `index.html` + `js/sectionN-*.js`, vanilla JS + canvas, **no external dependencies**, pt-BR prose, 3–4 sections each pushed onto `DL.sections` as `{name, init}`. Shared assets are `docs/shared/demo.css` and `docs/shared/js/{utils,plot,boot}.js`; `boot.js` must be the **last** deferred script. Always read colors from `DL.plot.theme()` inside `draw()` so the light/dark toggle works. Canonical reference demo: `docs/gradient-descent/`. Exceptions: `diffusion` and `lm` are self-contained (own `utils`/`plot`/`boot`, not `shared/js/boot.js`). The CI deploy step copies `docs/.` wholesale into `deploy/`, so new demos publish to gh-pages automatically with no allowlist to update. There is no build step for demos; serve `docs/` with `python3 -m http.server` to preview locally (see the interactive-demo-pattern memory for WSL screenshotting via headless Edge).

**Helper scripts** (`scripts/`) are Python utilities that generate figures consumed by slides, e.g. `train_vae_mnist.py` trains a small conv VAE on MNIST and writes `slides/img/05-geradores/vae-{samples,interpolation}.png`. Run from the repo root (`python scripts/train_vae_mnist.py --epochs 15`); not part of the LaTeX or CI pipelines.

## Skills available for content authoring

Three project-specific skills generate course material end-to-end and follow the conventions above. Prefer them over hand-writing scaffolding:

- `create-beamer-slide` — new lecture slide deck.
- `create-problem-set` — new `lista` for a unit.
- `create-exam` — new prova + gabarito pair.

## Conventions

- Course content is written in **Portuguese (pt-BR)**. Code identifiers, file names, and git history stay in English/snake_case (`06-llm_training.tex`, `lista03_backprop.tex`).
- File numbering inside a unit reflects lecture order (`01-`, `02-`, …). Keep it sequential when adding new material.
- The author block (`\author{Lucas Silveira Kupssinskü}`, `\institute{… PUCRS}`) and `\date{\mespor de \the\year}` macro are conventions across decks, the `\mespor` command is defined in `preamble.tex`.

### Prose style for course content

Avoid `-` and `---` as parenthetical or explanatory punctuation in slide body text, problem statements, exam questions, and gabarito explanations. Introduce clarifications with a comma, with parentheses, or after a colon. This rule applies to authored prose only, hyphens in compound terms (`self-attention`, `encoder-only`), filenames, code listings, math, and numeric ranges stay as written.

### Objective-question authoring (problem sets and exams)

Objective questions (multiple choice, true/false) must not let a student guess the answer without understanding the content. Two common leakage patterns to avoid:

1. **Information asymmetry between alternatives.** All distractors should look as plausible and as informative as the correct option. The correct alternative cannot be the one that names the right concept while the others are vague or visibly wrong, and it cannot be markedly longer or more detailed (the "longest answer is correct" heuristic). Match length, register, and specificity across all options.
2. **Over-explained correct alternative.** Do not embed the justification, definition, or hint inside the correct option. Reasoning, derivations, and "why this is right" belong in the gabarito (the `_gabarito.tex` answer key), never in the question stem or the alternative text shown to students.

When generating or reviewing objective items, sanity-check by reading only the alternatives (without the stem): if the correct one is identifiable from length, vocabulary, or self-contained explanation alone, rewrite it.
