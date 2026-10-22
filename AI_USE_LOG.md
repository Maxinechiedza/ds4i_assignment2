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
| 2026-10-18 | 4 | Claude Code | Debugging, editing and refining `R/optimism_baseline.R` — my own k-NN baseline script comparing a random split vs the session-held-out split to quantify the optimism gap | Wrote and ran the script myself, retesting after each change; read the printed accuracies and re-derived the gap by hand; sanity-checked the split sizes and that the random split truly ignores session | Edited, some |
| 2026-10-18 | 6 | Claude Code | Debugging, editing and refining `R/eval_embeddings.R` — my own script for PCA baseline embeddings, pairwise distances on `data/test_pairs.csv`, and AUC/EER/nearest-centroid accuracy for PCA, CNN and Siamese embeddings, by digit and by session | Wrote and ran the script myself, retesting after each change; inspected `outputs/comparison.csv`; cross-checked the AUC/EER definitions against the method Nicole already used in `Siamese.R` for consistency | Edited, some |
| 2026-10-18 | 1, 4, 6 | Claude Code | Draft prose for the Data and exploration, Evaluation design, and Side-by-side sections of `index.qmd`, citing numbers computed from the data/output files | Re-ran the underlying R code myself to confirm every cited number; rewriting into my own words before submission | Edited, some |

## Maxine (Role B)

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
| 2026-10-14 | 3 | ChatGPT | Helped troubleshoot the Siamese network R code, including pair construction, model training, contrastive loss and evaluation metrics. | Inspected all suggested edits to the code and ran in RStudio, inspected model outputs, pair construction and evaluation results. | Edited, some |
| 2026-10-18 | 5 | ChatGPT | Assisted with hyperparameter tuning, and plotting learning rate, margin and embedding dimension sensitivity. | Inspected all suggested edits to the tuning experiments and ran in R, checked the CSV outputs, seed variability and plotted results. | Edited, most |
| 2026-10-19 | 3, 6 | ChatGPT | Helped refine Siamese figures, including positive/negative pairs, distance distributions, ROC curves, digit-level performance and session effects. | Ran all suggested edits to the plotting code, inspected the figures and compared results with the saved output files. | Edited, some |
| 2026-10-20 | 3, 5, 6 | ChatGPT | Reviewed and revised the Siamese analysis and results write-up for clarity, structure, consistency and interpretation. | Reviewed wording and incorporated edits. | Edited, some |
| 2026-10-21 | 8 | ChatGPT | Assisted with drafting and refining the report introduction and conclusion, including summarising the CNN and Siamese findings.. | Reviewed the suggestions against the report's results and model comparisons. | Edited, some |
