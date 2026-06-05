# Paper sources

LaTeX sources for the MSCA-Net paper.

## Build

```bash
cd paper
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Or open `main.tex` on Overleaf.

## Figures

`figures/` holds the publication figures copied from `outputs/msca_net/`
(confusion matrix, ROC curves, t-SNE, training history). Regenerate them with:

```bash
python -m src.evaluate --config configs/msca_net.yaml --checkpoint outputs/msca_net/best.pt
python scripts/export_paper_figures.py
```

## Results

The numbers in Tables 1–2 are taken from `outputs/model_results.csv` and the
ablation runs. The `\result{}` macro in `main.tex` marks the cells that are
populated from those CSVs.

## Porting to a journal template

The single-column `article` layout maps cleanly onto:
- **Elsevier** (`elsarticle`, e.g. *Mechanical Systems and Signal Processing*,
  *Engineering Structures*): replace the preamble/`\author` block.
- **MDPI** (`Sensors`, `Applied Sciences`): use the MDPI class.
- **IEEE** (`IEEEtran`, two-column): use `\documentclass[journal]{IEEEtran}`.
The body, equations, tables, and TikZ figure carry over unchanged.
