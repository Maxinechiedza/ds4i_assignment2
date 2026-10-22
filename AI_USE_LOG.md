------------------------------------------------------------------------

editor: markdown: wrap: 72 ---

# AI use log

Running log of AI use as we go.
Add a row whenever you use an AI tool for something that ends up in the code, analysis, or report — don't wait until the end to reconstruct this.
At submission time, summarize this into the table in `index.qmd`'s "AI use statement" appendix (that table wants `tool / used for / how checked / how much used`, not raw prompts, but it's much faster to write once this log exists).

Columns: - **Task**: which of the 9 brief tasks this relates to (1-9) - **Tool**: name of the AI tool used - **Prompt / what for**: the actual prompt, or a short description if the exchange was long - **Checked how**: how you verified the output (re-derived it by hand, tested on a known case, compared to the lecture notes, ran it and inspected results, etc.) - **Used**: none / some, edited / most, as-is — rough honesty check, not a precise %

## Jesse (Role A)

| Date | Task | Tool | Prompt / what for | Checked how | Used |
|--------|--------|--------|----------------------|------------------|--------|
| 2026-10-18 | 4 | Claude Code | Write `R/optimism_baseline.R`: a k-NN baseline on simple image features, run on a random split vs the session-held-out split, to quantify the optimism gap | Ran the script, read the printed accuracies and re-derived the gap by hand; sanity-checked the split sizes and that the random split truly ignores session | Edited, most |
| 2026-10-18 | 6 | Claude Code | Write `R/eval_embeddings.R`: PCA baseline embeddings, pairwise distances on `data/test_pairs.csv`, AUC/EER/nearest-centroid accuracy for PCA, CNN and Siamese embeddings, by digit and by session | Ran the script and inspected `outputs/comparison.csv`; cross-checked the AUC/EER definitions against the method Nicole already used in `Siamese.R` for consistency | Edited, most |
| 2026-10-18 | 1, 4, 6 | Claude Code | Draft prose for the Data and exploration, Evaluation design, and Side-by-side sections of `index.qmd`, citing numbers computed from the data/output files | Re-ran the underlying R code myself to confirm every cited number; rewriting into my own words before submission | Edited, some |

## Maxine (Role B)

| Date | Task | Tool | Prompt / what for | Checked how | Used |
|------|------|------|-------------------|-------------|------|
|      |      |      |                   |             |      |

## Nicole (Role C)

| Date | Task | Tool | Prompt / what for | Checked how | Used |
|---------|---------|---------|--------------------|-------------------|---------|
| 2026-10-14 | 3 | ChatGPT | Helped troubleshoot the Siamese network R code, including pair construction, model training, contrastive loss and evaluation metrics. | Inspected all suggested edits to the code and ran in RStudio, inspected model outputs, pair construction and evaluation results. | Edited, some |
| 2026-10-18 | 5 | ChatGPT | Assisted with hyperparameter tuning, and plotting learning rate, margin and embedding dimension sensitivity. | Inspected all suggested edits to the tuning experiments and ran in R, checked the CSV outputs, seed variability and plotted results. | Edited, most |
| 2026-10-19 | 3, 6 | ChatGPT | Helped refine Siamese figures, including positive/negative pairs, distance distributions, ROC curves, digit-level performance and session effects. | Ran all suggested edits to the plotting code, inspected the figures and compared results with the saved output files. | Edited, some |
| 2026-10-20 | 3, 5, 6 | ChatGPT | Reviewed and revised the Siamese analysis and results write-up for clarity, structure, consistency and interpretation. | Reviewed wording and incorporated edits. | Edited, some |
| 2026-10-21 | 8 | ChatGPT | Assisted with drafting and refining the report introduction and conclusion, including summarising the CNN and Siamese findings.. | Reviewed the suggestions against the report's results and model comparisons. | Edited, some |
