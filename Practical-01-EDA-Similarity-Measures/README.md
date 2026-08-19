# Practical 1 - Exploring Customer Data + Distance Measures

Unit I - Foundations of Unsupervised Learning

## What this does

Before we can group customers, we need to know: how do we measure if two customers are "similar"? This practical tries 3 different ways to measure that, on real shopper data.

## Data

`mall_customers.csv` - 200 real shoppers, with age, income, and a spending score.

## First, just look at the data

![Distributions](images/01_eda_distributions.png)

Ages range from 18 to 70. Income ranges from $15k to $137k. Spending score goes from 1 to 99.

![Income vs spending](images/02_income_vs_spending.png)

## Three ways to measure "closeness"

- **Euclidean distance** - straight-line distance, like measuring with a ruler.
- **Manhattan distance** - add up the difference in each feature, like walking city blocks instead of cutting through buildings.
- **Cosine similarity** - ignores size, only cares about the *pattern*. Good for comparing shapes, not amounts.

## Why scaling matters

Before scaling, the distance between two real customers was **42.05**. After scaling, it was **1.64**.

Why the big change? Income goes up to $137,000 while Age only goes up to 70. Without scaling, Income completely drowns out Age in the distance calculation. Scaling fixes this by putting every feature on the same footing.

## Do different measures actually give different answers?

We picked one real customer and asked: who are their 5 closest neighbors?

| Measure | Closest 5 customers |
|---|---|
| Euclidean | 4, 17, 47, 16, 48 |
| Manhattan | 4, 17, 2, 16, 20 |
| Cosine | 69, 48, 49, 47, 4 |

Euclidean and Cosine only agree on 3 of 5. **Yes, the measure you pick really does change the answer.**

## Run it

```bash
cd Practical-01-EDA-Similarity-Measures
python similarity_demo.py
```

Tested and works. Takes about 2 seconds.

## If something goes wrong

| Problem | Fix |
|---|---|
| File not found | Make sure you `cd` into this folder first |
| Cosine gives `nan` | Happens if a row is all zeros - check your data |
| Distances look huge or tiny compared to a tutorial | Check whether you scaled the data or not - it changes everything |

## Files here

| File | What it is |
|---|---|
| `similarity_demo.py` | The code |
| `images/01_eda_distributions.png` | Age, income, spending charts |
| `images/02_income_vs_spending.png` | The main scatter plot |

Next: [Practical 2](../Practical-02-KMeans-vs-KMedoids/README.md)
