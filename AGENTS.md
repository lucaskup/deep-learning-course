# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, Codex, Cursor, Copilot, and others that follow the AGENTS.md convention) when working with code in this repository.

## Working with the user

Before planning or executing a task, ask clarifying questions whenever any aspect of the request is ambiguous (scope, target file, audience level, language, length, intended unit, etc.). Do not guess and proceed silently. If the request is fully unambiguous, you may skip this and act directly.

## Repository purpose

LaTeX source for a Brazilian (Portuguese) "Introdução a Deep Learning" course at PUCRS, taught by Prof. Lucas S. Kupssinskü. The LaTeX artifact families are:

- `slides/` — Beamer presentations, one `.tex` per lecture, organised by unit (`01-mlp` … `06-topicos`).
- `problems/` — Problem sets (`listaNN_*.tex`), organised by the same unit numbering.
- `exams/` — Provas, grouped by semester (`exams/2026-1/`, `exams/2026-2/`, …). Each prova has a paired `_gabarito.tex` (answer key) that CI builds locally but excludes from the public deploy. The two `*_formulario.tex` formula sheets sit at the `exams/` root because they are shared across semesters.
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

PDFs land in `build/` (set by `latexmkrc`'s `$out_dir = 'build'`).

**Always clean up after compiling.** Every time you compile anything (single file or `make` target), finish by removing every auxiliary artifact so that `build/` holds only PDFs. Read the `.log` first if you need to check for errors or warnings, then run from the repo root:

```bash
find build -mindepth 1 ! -name '*.pdf' -exec rm -rf {} +
```

This deletes `.aux`, `.log`, `.fls`, `.fdb_latexmk`, `.bbl`/`.bcf`/`.run.xml`, `.nav`/`.snm`/`.toc`/`.out`, `_minted-*` directories, and any `inspect-*` PNG folders (run it only after you have finished viewing them), keeping all `*.pdf` files. Do this even after a failed build, once the error has been diagnosed. The same `latexmkrc` enables `-shell-escape` for lualatex and points biber at the `slides/` directory for the shared `reference.bib`.

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

- The compile loop in `latex-ci.yml` finds candidates with `find … -name '*.tex' | xargs grep -l '^[^%]*\documentclass'` (a `\documentclass` that only appears after a `%` on its line, i.e. inside a comment, does not count), then **excludes** `slides/deprecated/` (directory no longer exists, filter kept), `slides/teste.tex`, `slides/testetikz.tex`, and any `*_gabarito.tex`. New scratch files matching those patterns won't be built; new `.tex` files with a `\documentclass` anywhere else will. Fragments without `\documentclass` (e.g. `slides/05-geradores/colors.tex` and `trilemma.tex`, which the four unit-5 decks `\input`) are ignored by the loop.
- Lint step parses every `build/*.log` for undefined references, undefined citations, and overfull hboxes. These are advisory (`::warning`) only, they never fail the build. Only a non-zero `latexmk` exit fails CI.
- Deploy to `gh-pages` happens on `release: published` or on manual `workflow_dispatch` (Actions tab / `gh workflow run latex-ci.yml`), never on plain push. Compiled PDFs deploy from `deploy/` (slides flat, problems grouped by module subdir, exams flat, gabaritos excluded). Pushes and PRs only compile/validate; markdown-only changes (`**.md`) trigger nothing.

## Document architecture

**Slides** (`slides/**/*.tex`) all start with:
```latex
\documentclass[aspectratio=169,xcolor=dvipsnames]{beamer}
\input{slides/utils/preamble}
\input{slides/utils/custom-commands}
```
- `slides/utils/preamble.tex` is the single source of truth for fonts (XCharter via fontspec + matched `newtxmath[xcharter]` for math, requires LuaLaTeX), the SimplePlus beamer theme (loaded from `slides/SimplePlusTheme/`), pgfplots/tikz libraries, the `vscode-light` listings style for Python, and minted configuration. Don't re-`\usepackage` things already in the preamble.
- `slides/reference.bib` is the unified bibliography. `biber` is configured with `--input-directory slides` so all decks resolve it.
- `slides/img/` is the shared figure root, `slides/code/` holds Python snippets referenced from `\lstinputlisting[style=vscode-light]{slides/code/NN_name.py}`, `slides/ppt/` holds source PowerPoint/TikZ originals (not compiled). All `\includegraphics`/`\lstinputlisting`/`\input` paths are repo-relative (`slides/img/01-mlp/…`), which is why compilation must start from the repo root.
- Snippets in `slides/code/` are prefixed with the unit number (`01_forward_pass.py`, `05_bce_loss.py`); keep that prefix when adding one.

**Problems** (`problems/**/*.tex`) use `\documentclass{dlproblemset}`. The class at `problems/style/dlproblemset.cls` bundles XCharter serif + matched `newtxmath`, KOMA `scrartcl` base, gray 1pt rules above the running header and below the footer, the split header (title left / subtitle right, in sans 8pt gray), and the bottom-left "Prof. Lucas Silveira Kupssinkü" footer. Each lista declares `\title{}` and `\subtitle{}` in the preamble; the body starts directly with content (no `\chapter*` or `\section*{Lista de Exercícios}` heading, no `\maketitle`). The class loads `graphicx`, `float`, `subfig`, `emoji`, `hyperref`, `amsmath`/`amssymb`/`amsthm`, `xcolor`, `microtype`, and `enumerate` already — listas only re-load what they genuinely need beyond that. Note `tikz`, `booktabs`, and `pgfplots` are **not** bundled, so a lista that uses tables or diagrams must `\usepackage` them explicitly (e.g. `\usepackage{booktabs}` for `\toprule`/`\midrule`, otherwise compilation fails with "Undefined control sequence").

**Projects** (`projects/*.tex`) also use `\documentclass{dlproblemset}` with `\title{}`/`\subtitle{}` (subtitle "Projeto N"), so everything said above about the class applies to them too.

**Exams** are standalone `article` documents (no shared style file). Every prova, gabarito, and formulário repeats the same preamble block (XCharter + `newtxmath[xcharter,bigdelims,vvarbb]` + the `OT1/XCharter(0)` font-substitution lines that stop math digits from silently falling back to Computer Modern), so copy the preamble from an existing exam rather than writing a new one. File naming is `<YYYY>_<S>_unidade<N>[e<M>].tex` (e.g. `2026_1_unidade2e3.tex`); the semester prefix stays in the filename even inside the semester folder because the deploy flattens `exams/` into one directory. Second-sitting exams use the `_G2` suffix (`2026_1_unidades1-3_G2.tex`). The `_gabarito.tex` is a **separate document, not an `\input` of the prova**: it opens with a `Quadro de respostas` table (question number → letter), then one `\subsection*{Questão N \textemdash{} Resposta: (x)}` per question with the derivation and a one-line refutation of each wrong alternative by letter. Because it refers to alternatives only by letter, reordering or relettering alternatives in the prova silently desynchronises the gabarito; update both files together. `exams/README.md` and `problems/README.md` restate these conventions for the public repo; update them if a convention changes.

**Interactive demos** (`docs/`) are the course's static website, the fourth artifact family. `docs/index.html` is the landing page (lists every lecture with PDF links plus an `exercise-chip chip-demo` linking its demo); each lecture demo lives at `docs/<slug>/` (kebab-case, e.g. `self-attention`, `lstm-gru`). A demo is `index.html` + `js/sectionN-*.js`, vanilla JS + canvas, **no external dependencies**, pt-BR prose, 3–4 sections each pushed onto `DL.sections` as `{name, init}`. Shared assets are `docs/shared/demo.css` and `docs/shared/js/{utils,plot,boot}.js`; `boot.js` must be the **last** deferred script. Always read colors from `DL.plot.theme()` inside `draw()` so the light/dark toggle works. Canonical reference demo: `docs/gradient-descent/`. Exceptions: `diffusion` and `lm` are self-contained (own `utils`/`plot`/`boot`, not `shared/js/boot.js`). The CI deploy step copies `docs/.` wholesale into `deploy/`, so new demos publish to gh-pages automatically with no allowlist to update. `docs/index.html` links PDFs by the flat deploy layout (`slides/<basename>.pdf`, `problems/<unit>/<basename>.pdf`, `exams/<basename>.pdf`), so a new lecture, lista, or prova only appears on the site once its entry is added there by hand. There is no build step for demos; serve `docs/` with `python3 -m http.server` to preview locally (see the interactive-demo-pattern memory for WSL screenshotting via headless Edge).

**Helper scripts** (`scripts/`) are Python utilities that generate figures consumed by slides, e.g. `train_vae_mnist.py` trains a small conv VAE on MNIST and writes `slides/img/05-geradores/vae-{samples,interpolation}.png`. Run from the repo root (`python scripts/train_vae_mnist.py --epochs 15`); not part of the LaTeX or CI pipelines. `export_shattered_gradients.py` (numpy only, ~2 min) writes the `slides/data/02-cnn/shattered-*.csv` files plotted by the residual-connections deck. `export_init_histograms.js` is the exception: a node script that runs the `docs/weight-init` demo's own `winit-core.js` to write the CSVs and σ macros in `slides/data/01-mlp/` that the weight-init deck plots with pgfplots (`node scripts/export_init_histograms.js`); rerun it whenever the demo's numerics change.

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

Avoid constructions that first state what something is not and only then state what it is ("X não é A, é B", "não se trata de A, mas de B", "mais do que A, é B"). State the positive claim directly ("X é B"). If contrasting with a common misconception is pedagogically necessary, make it a separate sentence after the direct statement rather than the lead-in.

Avoid noun + adjective pairs where the adjective only signals importance, judgement, or degree instead of adding technical content: "problema concreto", "ideia central", "diferença crítica", "assimetria importante", "propriedade interessante", "consequência absurda", "componente fundamental", "pico claro", "pequeno MLP", "cuidado extra", "especialmente fácil", "exemplo real". Drop the adjective or rewrite the sentence so the claim stands on its own ("um MLP aplicado a cada posição", "O modelo aprende a copiar"). Keep adjectives that carry a technical meaning the sentence needs: "produto escalar", "máscara causal", "limite superior bem definido", "$d$ grande", "dicionário suave" (as the name of an analogy). Block and frame titles follow the same rule ("Consequência", not "Consequência absurda").

Avoid "wrap-up" labels and sentences that announce a summary instead of making a claim: "A frase que resume tudo", "Em uma frase", "A lição", "O ponto principal", "Resumindo", "the key takeaway", "the one-sentence summary". Block titles and lead-ins of this kind add no content and read as machine-generated. Give the block a title that names the concept ("Fronteira de decisão") and state the claim directly.

Avoid indirect, inverted sentences that delay the subject or hide it behind a quantifier or an appositive: "Quatro escolhas, todas hiperparâmetros, definem o classificador", "Três fatores, combinados, explicam o erro", "Duas ideias sustentam o método". Write the subject first and the claim right after it: "O MLP tem quatro hiperparâmetros", "O erro tem três causas", "O método se apoia em duas ideias". The same applies to sentences that open with a dangling participle or a long adverbial before naming what they are about.

Every lecture, problem set, and exam is self-contained. Do not point the reader to other parts of the course by position: no "próximo slide", "slide anterior", "na próxima aula", "no próximo deck", "da aula anterior", "na unidade anterior", "voltaremos a isso na unidade 4", "veremos adiante", "como vimos", and no frame titles built on that ("O MLP da aula anterior como grafo computacional"). Lecture order changes between semesters, decks are published individually, and students open PDFs out of order, so such references go stale or promise content the student may not have. If a concept from another lecture is needed, name the concept directly ("o MLP de duas camadas", "a perda half SSE") and restate the one-line definition if the slide depends on it, without saying where it was introduced. If a topic is deliberately postponed, name the topic without saying where it lives ("a derivação completa do backprop fica para a próxima aula" becomes "a derivação completa do backprop não é tratada aqui"). Speaker notes (`\note{}`) follow the same rule. Problem sets, exams, and gabaritos follow it too, except that a gabarito may cite the lecture by title.

### Objective-question authoring (problem sets and exams)

Objective questions (multiple choice, true/false) must not let a student guess the answer without understanding the content. Two common leakage patterns to avoid:

1. **Information asymmetry between alternatives.** All distractors should look as plausible and as informative as the correct option. The correct alternative cannot be the one that names the right concept while the others are vague or visibly wrong, and it cannot be markedly longer or more detailed (the "longest answer is correct" heuristic). Match length, register, and specificity across all options.
2. **Over-explained correct alternative.** Do not embed the justification, definition, or hint inside the correct option. Reasoning, derivations, and "why this is right" belong in the gabarito (the `_gabarito.tex` answer key), never in the question stem or the alternative text shown to students.

When generating or reviewing objective items, sanity-check by reading only the alternatives (without the stem): if the correct one is identifiable from length, vocabulary, or self-contained explanation alone, rewrite it.
