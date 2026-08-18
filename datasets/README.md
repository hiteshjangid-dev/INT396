# Datasets

All real data. Nothing made up.

| File | What it is | Real source |
|---|---|---|
| `mall_customers.csv` | 200 real shoppers — age, income, spending score | Kaggle / public GitHub mirrors |
| `market_basket.csv` | 7,501 real grocery store receipts | Public GitHub mirror |

Two more datasets are used straight from scikit-learn (also real, not made up):

| Dataset | How it's loaded | What it is |
|---|---|---|
| Wine | `sklearn.datasets.load_wine()` | 178 real Italian wines, chemical measurements |
| Digits | `sklearn.datasets.load_digits()` | 1,797 real handwritten digit scans |

## Re-downloading the data

```bash
curl -o mall_customers.csv "https://raw.githubusercontent.com/gakudo-ai/open-datasets/refs/heads/main/Mall_Customers.csv"
curl -o market_basket.csv "https://raw.githubusercontent.com/BejaminNaibei/dataset/main/Market_Basket_Optimisation.csv"
```
