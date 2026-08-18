# INT396 — Unsupervised Learning: 14 Practicals

All 14 practicals for the Unsupervised Learning course, done on real data, tested and working.

## What's inside

Each folder is one practical. Each one has:
- The code (`.py` file) — short, tested, works
- A `README.md` explaining it in plain language
- Charts made from real data

## The 14 practicals

| # | What it's about | Data used |
|---|---|---|
| 1 | Exploring customer data + distance measures | Real mall customers |
| 2 | k-Means vs k-Medoids clustering | Real mall customers |
| 3 | Building k-Means from scratch | Real mall customers |
| 4 | Why scaling your data matters | Real mall customers |
| 5 | Picking the right number of clusters | Real mall customers |
| 6 | Hierarchical clustering + dendrograms | Real mall customers |
| 7 | DBSCAN (density-based clustering) | Real mall customers |
| 8 | Comparing all 3 clustering methods | Real mall customers |
| 9 | PCA (shrinking your data down) | Real wine data |
| 10 | PCA vs t-SNE vs UMAP | Real handwritten digits |
| 11 | Market basket analysis (Apriori/FP-Growth) | Real grocery receipts |
| 12 | Finding unusual customers | Real mall customers |
| 13 | Checking if your clustering is trustworthy | Real mall customers |
| 14 | Full project — everything combined | Real mall customers |

## Data

Everything is real, nothing made up:

- `mall_customers.csv` — 200 real shoppers (age, income, spending score)
- `market_basket.csv` — 7,501 real grocery store receipts
- Wine and Digits datasets — built into scikit-learn, both real

See [`datasets/README.md`](datasets/README.md) for where each one came from.

## How to run any practical

```bash
git clone <this-repo-url>
cd <repo-folder>
pip install -r requirements.txt
cd Practical-01-EDA-Similarity-Measures
python similarity_demo.py
```

Every practical works the same way — go into its folder, run the `.py` file.

## Putting this on your own GitHub

```bash
git init
git add .
git commit -m "INT396 practicals"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

## If something doesn't work

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | `pip install -r requirements.txt` |
| Can't find a data file | Make sure you `cd` into the practical's folder first |
| Charts don't pop up | That's fine — they save to an `images/` folder instead |

## License

MIT — free to use and share.
