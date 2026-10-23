# DS4I Assignment 2: writer identification with neural networks

STA5073Z Data Science for Industry 2026.
Group: Jesse MacDonald (Role A), Maxine Senderayi (Role B, CNN), Nicole Thomas (Role C, Siamese network).

Website: https://maxinechiedza.github.io/ds4i_assignment2/  
Repository: https://github.com/Maxinechiedza/ds4i_assignment2

## What lives where

| Path | What it is | Owner |
|---|---|---|
| `index.qmd` | The report (Quarto), including the plagiarism declaration and AI use statement | Everyone, own sections only |
| `_quarto.yml` | Turns the report into a website that renders into `docs/` | Maxine |
| `assets/` | Website look: `report.scss` (styling), `title-block.html` (title page), `fonts/` | Maxine |
| `docs/` | The rendered website GitHub Pages shows. Never edit by hand. | Made by `quarto render` |
| `data/handwriting.rds` | The raw data from Amathuba | – |
| `R/00_data.R` | Loads the data and writes the shared split files | Jesse |
| `data/splits.csv`, `data/test_pairs.csv` | Shared split and test pairs, written by `R/00_data.R` | Jesse |
| `R/cnn_functions.R`, `R/cnn.R`, `R/tune_cnn.R`, `R/tune_cnn_valacc.R`, `R/tune_cnn_reg.R`, `R/cnn_digit_session.R` | CNN code | Maxine (Role B) |
| `Siamese.R`, `Tune_Siamese.R`, `Siamese_Unseen_Writers.R` | Siamese network code | Nicole (Role C) |
| `R/optimism_baseline.R`, `R/eval_embeddings.R` | Split experiment and model comparison | Jesse |
| `outputs/` | CSV results that the report reads | Whoever's script writes them |
| `project-details.txt` | Submission file: names, roles, website and repository links | Maxine |
| `README.txt` | Submission file: what each file in `project.zip` contains | Maxine |
| `AI_USE_LOG.md` | Dated log of everyone's AI use, summarised in the report | Everyone |

## How we work

1. **Clone once.** In RStudio: File > New Project > Version Control > Git, and paste the repo URL. Save it in a folder that is not synced by OneDrive, iCloud or Dropbox.
2. **Pull first.** Every time you sit down, press Pull (blue arrow) in the Git pane.
3. **Use your own branch** (e.g. `cnn`, `siamese`, `data`): Git pane > New Branch.
4. **Commit small, push often.** Tick your files, write a short message, Commit, then Push (green arrow).
5. **Merge through a pull request.** On GitHub, open a pull request into `main`; another member reviews and merges it.
6. **Stay in your lane.** Edit only your own sections of `index.qmd` and your own scripts, so merge conflicts stay rare.
7. **Website renders go through a pull request too.** Anyone can render the site, but only after everyone's pull requests are merged into `main`. Render on a short-lived `render` branch and merge it through a pull request, like any other change. If you render to preview on your own branch, don't tick the `docs/` files when you commit.

