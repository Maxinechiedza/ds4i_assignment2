# DS4I Assignment 2: writer identification with neural networks

STA5073Z Data Science for Industry 2026.
Group: Maxine Senderayi, Nicole Thomas, [third member].

Website: https://maxinechiedza.github.io/ds4i_assignment2/  
Repository: https://github.com/Maxinechiedza/ds4i_assignment2

## What lives where

| Path | What it is | Owner |
|---|---|---|
| `index.qmd` | The report. Every section is marked with its owner. | Everyone, own sections only |
| `_quarto.yml` | Turns the report into a website that renders into `docs/` | Maxine |
| `docs/` | The rendered website GitHub Pages shows. Never edit by hand. | Made by `quarto render` |
| `data/handwriting.rds` | The raw data from Amathuba | – |
| `R/00_data.R` | Loads the data and writes the shared split files | Maxine |
| `data/splits.csv`, `data/test_pairs.csv` | Shared split and test pairs, written by `R/00_data.R` | Maxine |
| `R/cnn.R`, `R/tune_cnn.R`, `R/cnn_digit_session.R` | CNN code | Role B |
| `R/siamese.R`, `R/tune_siamese.R` (or `python/`) | Siamese network code | Role C |
| `R/optimism_baseline.R`, `R/eval_embeddings.R` | Split experiment and model comparison | Maxine |
| `outputs/` | CSV results that the report reads | Whoever's script writes them |
| `submission.txt` | Submission deliverable 2: names, roles, links | Maxine |

## How we work

1. **Clone once.** In RStudio: File > New Project > Version Control > Git, and paste the repo URL. Save it in a folder that is not synced by OneDrive, iCloud or Dropbox.
2. **Pull first.** Every time you sit down, press Pull (blue arrow) in the Git pane.
3. **Use your own branch** (e.g. `cnn`, `siamese`, `data`): Git pane > New Branch.
4. **Commit small, push often.** Tick your files, write a short message, Commit, then Push (green arrow).
5. **Merge through a pull request.** On GitHub, open a pull request into `main`; another member reviews and merges it.
6. **Stay in your lane.** Edit only your own sections of `index.qmd` and your own scripts, so merge conflicts stay rare.
7. **Leave `docs/` to Maxine.** Only Maxine renders the site and commits `docs/`, from `main`. If you render to preview, don't tick the `docs/` files when you commit.

## Publishing the website

In RStudio's Terminal tab (not the Console) run `quarto render`, then commit and push `docs/`.
GitHub Pages is set to deploy from the `main` branch, `/docs` folder.
