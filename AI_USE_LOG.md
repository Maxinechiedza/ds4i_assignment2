# AI use log

Running log of AI use as we go. Add a row whenever you use an AI tool for something
that ends up in the code, analysis, or report — don't wait until the end to
reconstruct this. At submission time, summarize this into the table in `index.qmd`'s
"AI use statement" appendix (that table wants `tool / used for / how checked / how
much used`, not raw prompts, but it's much faster to write once this log exists).

Columns:
- **Task**: which of the 9 brief tasks this relates to (1-9)
- **Tool**: name of the AI tool used
- **Prompt / what for**: the actual prompt, or a short description if the exchange was long
- **Checked how**: how you verified the output (re-derived it by hand, tested on a known case, compared to the lecture notes, ran it and inspected results, etc.)
- **Used**: none / some, edited / most, as-is — rough honesty check, not a precise %

## Jesse (Role A)

| Date | Task | Tool | Prompt / what for | Checked how | Used |
|---|---|---|---|---|---|
| | | | | | |

## Maxine (Role B)

Draft rows written with Claude on 2026-10-20 from our conversation; cells marked [check] still need my confirmation.

| Date | Task | Tool | Prompt / what for | Checked how | Used |
|---|---|---|---|---|---|
| 2026-09-28 | 7, 8 | Claude (Anthropic) | Planned the 3-person work split for Tasks 1–6 (Work Split doc) and drafted the report template: `index.qmd` with owner boxes and figure labels, `_quarto.yml` | We read and adjusted the plan; template rendered with Quarto | Most, edited |
| 2026-09-28 to 2026-10-02 | 7, 9 | Claude | GitHub set-up help: repo layout, `README.md` workflow rules, `submission.txt`; fixing the `usethis`/`gh` "check_string" error and the GitHub Pages "No such file docs" error | Repo works for all three of us; Pages site loaded in the browser | Most, edited |
| 2026-10-12 | 2 | Claude | Day 1 CNN code: `get_split()` and `build_cnn()` in `R/cnn_functions.R`; explanations of one-hot labels, pixel scaling and the layer sizes | Ran it in RStudio; checked image dimensions and 750 training images per writer | Most, as-is |
| 2026-10-12 to 2026-10-13 | 5 | Claude | Tuning design and scripts `R/tune_cnn.R` and `R/tune_cnn_valacc.R`; reading effects against seed noise; diagnosing early-stopping problems (validation loss vs accuracy, patience, epoch cap) | Ran every fit myself and read the learning curves; re-running a seed reproduced the same result | Most, edited |
| 2026-10-14 | 7 | Claude | Git help: commit, push and merge commands; tidied `index.qmd` after RStudio's Visual mode reformatted other members' sections | Diff against `main` touched only my sections; pull requests merged without conflicts | Most, as-is |
| 2026-10-14 to 2026-10-15 | 5, 6 | Claude | `R/tune_cnn_reg.R` (regularisation check with a decision rule fixed before the runs) and `R/cnn_digit_session.R` (leave-one-session-out and digit-only CNNs, 3 seeds)| Ran both scripts; the session-5 fold reproduced the main 56.6%; the current setting re-run matched earlier runs exactly | Most, edited |
| 2026-10-13 to 2026-10-18 | 2, 5, 6 | Claude | Explanations to help me interpret results: top-3 accuracy, overfitting vs choosing on validation, which exhibit answers which brief question, digit-only vs all-digit models | Checked the numbers against the output files | Some, edited |
| 2026-10-18 | 2, 5, 6 | Claude | Review of my draft text: fact-check against the results, word counts | I corrected the facts and rewrote the flagged passages myself | Some, edited |

## Nicole (Role C)

| Date | Task | Tool | Prompt / what for | Checked how | Used |
|---|---|---|---|---|---|
| | | | | | |
