# Practical 1: Exploring Customer Data and Measuring "Closeness"

**Unit I — Foundations of Unsupervised Learning**
**Time needed: about 2 hours**
**Course outcome covered: CO1, CO2**

---

## Table of Contents

1. [Before You Start](#1-before-you-start)
2. [Why This Practical Matters](#2-why-this-practical-matters)
3. [The Theory, Explained Simply](#3-the-theory-explained-simply)
4. [The Dataset](#4-the-dataset)
5. [Setting Up the Code](#5-setting-up-the-code)
6. [Part A: Looking at the Data (EDA)](#6-part-a-looking-at-the-data-eda)
7. [Part B: The Three Distance Measures](#7-part-b-the-three-distance-measures)
8. [Part C: Why Scaling Changes Everything](#8-part-c-why-scaling-changes-everything)
9. [Part D: Do Different Measures Agree?](#9-part-d-do-different-measures-agree)
10. [Full Code, One Piece at a Time](#10-full-code-one-piece-at-a-time)
11. [Try It Yourself](#11-try-it-yourself)
12. [Common Mistakes and Fixes](#12-common-mistakes-and-fixes)
13. [Quick Quiz — Check Yourself](#13-quick-quiz-check-yourself)
14. [Summary](#14-summary)

---

<a id="1-before-you-start"></a>
## 1. Before You Start

**What you need:**
- Python installed (3.9 or newer)
- These packages: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn`
- The file `mall_customers.csv` in the `datasets` folder one level above this one

**Install everything in one line:**
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn
```

**How to run this practical:**
```bash
cd Practical-01-EDA-Similarity-Measures
python similarity_demo.py
```

You should see text printed in your terminal, and two image files appear inside a new `images` folder. If that happens, everything worked. If not, jump to [section 12](#12-common-mistakes-and-fixes).

---

<a id="2-why-this-practical-matters"></a>
## 2. Why This Practical Matters

Imagine you work at a mall and you have a spreadsheet of 200 customers: their age, how much money they make a year, and a "spending score" (a number from 1 to 100 that says how much they like to spend).

Your manager asks: **"Can you group similar customers together so we can market to each group differently?"**

Before you can group anything, you need to answer one very basic question first:

> **"What does it even mean for two customers to be similar?"**

That is the entire point of this practical. We are not clustering anyone yet — that comes in Practical 2. Right now we are just learning how to measure "closeness" between two data points, because every clustering algorithm in this whole course depends on getting this right.

This is unsupervised learning because nobody told us the "correct" groups in advance. There are no labels. We only have raw numbers, and we have to find patterns ourselves.

---

<a id="3-the-theory-explained-simply"></a>
## 3. The Theory, Explained Simply

### 3.1 What is a "distance" between two data points?

Think of each customer as a point in space. If we only used 2 numbers (say, Income and Spending Score), you could literally draw each customer as a dot on graph paper. The "distance" between two dots is just how far apart they are.

We use 3 real customer numbers here (Age, Income, Spending Score), so we can't draw it on paper anymore — but the math works the exact same way, it just happens in 3D instead of 2D. Later we use even more numbers (13, or even 64!), and the math still works the same way, even though we can no longer picture it directly. This is one of the most important ideas in this whole course: **the math of distance works in any number of dimensions, even ones we can't visualize.**

### 3.2 Euclidean Distance — "as the crow flies"

This is the distance you already know from school — the straight line between two points.

Formula for 2 numbers:
```
distance = square root of ( (x1-x2)^2 + (y1-y2)^2 )
```

For 3 numbers (like we use here), you just add one more squared term:
```
distance = square root of ( (x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2 )
```

**Real-life analogy:** if you're a bird flying from your house to a friend's house, you fly in a straight line over rooftops. That's Euclidean distance.

### 3.3 Manhattan Distance — "city block" distance

Instead of squaring the differences, you just add up the plain differences (ignoring negative signs).

Formula:
```
distance = |x1-x2| + |y1-y2| + |z1-z2|
```

**Real-life analogy:** if you're walking in a city with a grid of streets, you can't cut through buildings. You walk one block over, then one block up. That's Manhattan distance. It's named after Manhattan, New York, because the streets there are laid out in a grid.

### 3.4 Cosine Similarity — "same direction, different size"

This one is different from the other two. Instead of measuring how far apart two points are, it measures the **angle** between them, ignoring their size completely.

**Real-life analogy:** Imagine two customers. Customer A spends $10 on chips and $20 on soda. Customer B spends $100 on chips and $200 on soda. They spent very different total amounts, but the *ratio* of chips-to-soda is identical (1:2 for both). Cosine similarity would say these two customers have the exact same "pattern" of spending, even though their totals are very different. Euclidean distance would say they are far apart, because it cares about the actual numbers, not just the ratio.

Cosine similarity gives you a number between -1 and 1:
- **1** means the exact same direction (very similar pattern)
- **0** means completely unrelated directions
- **-1** means exact opposite directions

### 3.5 Why does the choice of measure actually matter?

Because different measures answer different questions:
- Euclidean/Manhattan answer: "Are these two customers' actual numbers close together?"
- Cosine answers: "Do these two customers have the same *pattern*, regardless of scale?"

Pick the wrong one, and your "similar" customers might not be similar at all for the business question you're actually trying to answer. We will prove this with real numbers later in this practical, not just claim it.

---

<a id="4-the-dataset"></a>
## 4. The Dataset

File: `../datasets/mall_customers.csv`

This is a real, publicly available dataset of 200 real shoppers at a mall. It has 5 columns:

| Column | What it means | Example value |
|---|---|---|
| `CustomerID` | Just a number to identify the row, not useful for math | 1 |
| `Gender` | Male or Female | Male |
| `Age` | The customer's age in years | 19 |
| `Annual Income (k$)` | Yearly income, in thousands of dollars | 15 (means $15,000) |
| `Spending Score (1-100)` | A made-up score by the mall based on how much and how often they spend. 1 = barely spends, 100 = spends a lot | 39 |

There are **no labels** in this data — nobody has told us which customers "belong together." That is exactly what makes this unsupervised learning: we have to find structure ourselves, without being told the right answer in advance.

---

<a id="5-setting-up-the-code"></a>
## 5. Setting Up the Code

Here is the very top of our script:

```python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import euclidean, cityblock, cosine
from sklearn.preprocessing import StandardScaler

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 170, "savefig.bbox": "tight"})
os.makedirs("images", exist_ok=True)
```

Let's go through this line by line:

- `import os` — lets us create folders and work with file paths.
- `import numpy as np` — the library that does fast math on lists of numbers (called "arrays").
- `import pandas as pd` — the library that reads and organizes our CSV file into a table (called a "DataFrame").
- `import matplotlib.pyplot as plt` and `import seaborn as sns` — both are for drawing charts. Seaborn makes matplotlib charts look nicer with less code.
- `from scipy.spatial.distance import euclidean, cityblock, cosine` — these are pre-built functions that calculate the 3 distances we just learned about. We don't have to write the math formulas by hand — scipy already did it for us, tested and correct.
- `from sklearn.preprocessing import StandardScaler` — a tool that rescales our numbers so they're all on a fair playing field (more on this in Part C).
- `sns.set_theme(...)` — just makes our charts look clean and consistent.
- `plt.rcParams.update(...)` — sets the image quality when we save charts.
- `os.makedirs("images", exist_ok=True)` — creates a folder called `images` to save our charts into. `exist_ok=True` means "don't complain if this folder already exists."

---

<a id="6-part-a-looking-at-the-data-eda"></a>
## 6. Part A: Looking at the Data (EDA)

"EDA" stands for Exploratory Data Analysis. Before doing any fancy math, always look at your data first. Here's the code:

```python
df = pd.read_csv("../datasets/mall_customers.csv")
print("Loaded", len(df), "real customers")
print(df.describe().round(1))
```

- `pd.read_csv(...)` opens the CSV file and turns it into a table we can work with, stored in a variable called `df` (short for "DataFrame," pandas' name for a table).
- `len(df)` counts the rows — how many customers we have.
- `df.describe()` is a magic one-line command that instantly gives you the count, average, minimum, maximum, and spread of every numeric column. `.round(1)` just rounds the numbers to 1 decimal place so it's easier to read.

**Real output you'll see:**
```
Loaded 200 real customers
       CustomerID    Age  Annual Income (k$)  Spending Score (1-100)
count       200.0  200.0               200.0                   200.0
mean        100.5   38.8                60.6                    50.2
std          57.9   14.0                26.3                    25.8
min           1.0   18.0                15.0                     1.0
25%          50.8   28.8                41.5                    34.8
50%         100.5   36.0                61.5                    50.0
75%         150.2   49.0                78.0                    73.0
max         200.0   70.0               137.0                    99.0
```

**How to read this table:**
- `count` — how many rows have a value (all 200, so no missing data — great, we don't have to clean anything).
- `mean` — the average. Average age is 38.8, average income is $60.6k.
- `std` — "standard deviation," a measure of how spread out the numbers are. A bigger number means more spread out.
- `min` / `max` — the smallest and largest value.
- `25%`, `50%`, `75%` — these are "percentiles." The `50%` row is the same as the median. `25%` means 25% of customers are below this value.

**The one thing to notice immediately:** Age goes up to 70, but Income goes up to 137. That gap in scale is going to cause a real problem later — keep that in the back of your mind, because we prove it mathematically in Part C.

### Drawing the distributions

```python
fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
axes[0].hist(df["Age"], bins=20, color="#3b82f6"); axes[0].set_title("Age")
axes[1].hist(df["Annual Income (k$)"], bins=20, color="#16a34a"); axes[1].set_title("Income ($k)")
axes[2].hist(df["Spending Score (1-100)"], bins=20, color="#f97316"); axes[2].set_title("Spending Score")
plt.tight_layout()
plt.savefig("images/01_eda_distributions.png")
plt.close()
```

- `plt.subplots(1, 3, ...)` makes one row of 3 side-by-side chart boxes ("axes"), and gives us back the whole figure plus each individual box in a list called `axes`.
- `axes[0].hist(df["Age"], bins=20, ...)` draws a histogram (a bar chart showing how many customers fall into each range) of the Age column, split into 20 bars ("bins").
- We repeat the same idea for Income and Spending Score.
- `plt.tight_layout()` — cleans up spacing so titles and labels don't overlap.
- `plt.savefig(...)` — saves the chart to a file instead of popping up a window (this lets the script run without needing a screen).
- `plt.close()` — closes the figure so it doesn't use up memory if we make more charts later.

![Distributions](images/01_eda_distributions.png)

**What this chart tells us:** Age is fairly spread out across the whole 18-70 range. Income has more customers in the middle range with fewer very rich or very poor customers. Spending Score is roughly evenly spread from 1 to 99.

### Drawing income vs spending

```python
plt.figure(figsize=(7.5, 6))
sns.scatterplot(data=df, x="Annual Income (k$)", y="Spending Score (1-100)", hue="Gender", s=70)
plt.title("Income vs Spending Score")
plt.tight_layout()
plt.savefig("images/02_income_vs_spending.png")
plt.close()
```

- `sns.scatterplot(...)` draws one dot per customer. `x=` and `y=` say which columns go on each axis. `hue="Gender"` colors the dots by gender so we can see if gender plays a role. `s=70` sets the dot size.

![Income vs spending](images/02_income_vs_spending.png)

**What this chart tells us:** Even without doing any clustering algorithm yet, you can probably already see with your own eyes that there seem to be a few natural "clumps" of customers here — for example, high income + high spending in one corner, high income + low spending in another. This is the exact picture Practicals 2 through 8 will teach a computer to find automatically.

---

<a id="7-part-b-the-three-distance-measures"></a>
## 7. Part B: The Three Distance Measures

Now let's actually calculate distances between two real customers, by hand (well, scipy does the arithmetic, but we control exactly what happens).

```python
features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
X = df[features].values
a, b = X[0], X[1]
print("\nCustomer A:", df.iloc[0][features].to_dict())
print("Customer B:", df.iloc[1][features].to_dict())
```

- `features = [...]` — a list of the 3 column names we want to use for our math. We leave out `CustomerID` and `Gender` because `CustomerID` is meaningless for distance (it's just a row number) and `Gender` is text, not a number.
- `df[features].values` — pulls out just those 3 columns and converts the table into a plain NumPy array of numbers, stored in `X`. This is the standard format almost every machine learning tool expects: rows are data points, columns are features.
- `a, b = X[0], X[1]` — grabs the first two rows (the first two real customers) as `a` and `b`.
- `df.iloc[0][features].to_dict()` — grabs row 0's values for our chosen features and turns them into a readable dictionary for printing.

**Real output:**
```
Customer A: {'Age': 19, 'Annual Income (k$)': 15, 'Spending Score (1-100)': 39}
Customer B: {'Age': 21, 'Annual Income (k$)': 15, 'Spending Score (1-100)': 81}
```

So Customer A is 19 years old, makes $15k a year, and has a spending score of 39. Customer B is 21, also makes $15k, but has a much higher spending score of 81.

### Calculating all three distances

```python
print(f"\nEuclidean distance:  {euclidean(a, b):.2f}  (straight-line distance)")
print(f"Manhattan distance:  {cityblock(a, b):.2f}  (add up each feature's difference)")
print(f"Cosine similarity:   {1 - cosine(a, b):.4f}  (how similar their *pattern* is, ignoring size)")
```

- `euclidean(a, b)` — scipy's ready-made function, does the "square, add, square-root" formula from section 3.2 for us.
- `cityblock(a, b)` — scipy's name for Manhattan distance (from "city block" distance).
- `cosine(a, b)` — this scipy function actually returns cosine *distance* (which is `1 - similarity`), so we write `1 - cosine(a, b)` to flip it back into similarity, which is easier to interpret (higher = more similar).
- `:.2f` inside the f-string means "show this number rounded to 2 decimal places." `:.4f` means 4 decimal places.

**Real output:**
```
Euclidean distance:  42.05  (straight-line distance)
Manhattan distance:  44.00  (add up each feature's difference)
Cosine similarity:   0.9694  (how similar their *pattern* is, ignoring size)
```

**Let's actually do the Euclidean math by hand to prove the function is right:**
```
Age difference:      19 - 21 = -2,  squared = 4
Income difference:    15 - 15 = 0,  squared = 0
Spending difference:  39 - 81 = -42, squared = 1764

Sum = 4 + 0 + 1764 = 1768
Square root of 1768 = 42.05
```
That matches the function's answer exactly. This is a good habit: whenever you use a library function for the first time, check it against hand math on a small example so you trust it.

**And the Manhattan math by hand:**
```
|19-21| + |15-15| + |39-81| = 2 + 0 + 42 = 44
```
Also matches exactly.

**Interpreting the cosine similarity of 0.9694:** this is very close to 1, meaning these two customers have a very similar *pattern* across Age/Income/Spending, even though their actual spending scores (39 vs 81) are quite different. Cosine similarity is looking at direction, not magnitude, so a big difference in one number doesn't automatically make the cosine score low if the overall "shape" of their numbers still points a similar way.

---

<a id="8-part-c-why-scaling-changes-everything"></a>
## 8. Part C: Why Scaling Changes Everything

This is the single most important lesson in this practical. Let's prove it with real numbers instead of just claiming it.

```python
X_scaled = StandardScaler().fit_transform(X)
print(f"\nEuclidean distance BEFORE scaling: {euclidean(a, b):.2f}")
print(f"Euclidean distance AFTER scaling:  {euclidean(X_scaled[0], X_scaled[1]):.2f}")
print("Income (up to $137k) drowns out Age (up to 70) unless we scale first.")
```

**What is `StandardScaler` actually doing?**

For every column, it does this transformation on every value:
```
new_value = (old_value - column_average) / column_standard_deviation
```

After this, every column has an average of 0 and a standard deviation of 1. This puts Age, Income, and Spending Score all on the exact same footing — none of them can dominate just because its raw numbers happen to be bigger.

- `StandardScaler()` — creates the scaling tool.
- `.fit_transform(X)` — this does two things at once: `fit` calculates the average and spread of each column, and `transform` actually applies the formula above to every value. The result is stored in `X_scaled`.

**Real output:**
```
Euclidean distance BEFORE scaling: 42.05
Euclidean distance AFTER scaling:  1.64
```

**Why did the number drop so much?** Before scaling, the Spending Score difference (42) got squared to 1764 — completely dominating the tiny Age difference (2, squared to just 4). Income's difference was 0 this time, but in general, since Income's numbers commonly range up to 137 while Age only ranges up to 70, Income differences would routinely produce far bigger squared values than Age differences, just because of its bigger natural scale — nothing to do with which feature is actually more important. After scaling, all three features contribute fairly to the total distance, because they've all been put on the same 0-to-1-ish scale.

**The takeaway for every practical after this one:** always scale your numeric features before calculating distances or running clustering algorithms, unless you have a specific reason not to. We will do this in almost every following practical.

---

<a id="9-part-d-do-different-measures-agree"></a>
## 9. Part D: Do Different Measures Agree?

We've shown scaling matters. Now let's answer the other big question from the theory section: does the *choice of measure* (Euclidean vs Manhattan vs Cosine) actually change your answer in practice?

```python
query = X_scaled[0]
dists = {
    "euclidean": [euclidean(query, x) for x in X_scaled],
    "manhattan": [cityblock(query, x) for x in X_scaled],
    "cosine": [cosine(query, x) for x in X_scaled],
}
results = pd.DataFrame({"customer": range(len(X_scaled)), **dists})
top5 = {k: results.sort_values(k).iloc[1:6]["customer"].tolist() for k in dists}
print("\nTop 5 closest customers to Customer 0, by each measure:")
for k, v in top5.items():
    print(f"  {k}: {v}")
overlap = len(set(top5["euclidean"]) & set(top5["cosine"]))
print(f"\nEuclidean and Cosine only agree on {overlap}/5 -- different measures give different answers.")
```

Let's break this down:

- `query = X_scaled[0]` — we pick Customer 0 (the very first real customer) and ask: "who are your closest neighbors?"
- `[euclidean(query, x) for x in X_scaled]` — this is a Python "list comprehension." In plain English it means: "for every customer `x` in our whole dataset, calculate the distance from `query` to `x`, and collect all those numbers into a list." We do this once for each of the 3 measures.
- `pd.DataFrame({"customer": range(len(X_scaled)), **dists})` — builds a table with one row per customer, and one column for each distance measure's score against Customer 0. The `**dists` unpacks our dictionary of lists directly into columns.
- `results.sort_values(k)` — sorts the table by measure `k`, smallest distance first (closest first).
- `.iloc[1:6]` — takes rows 1 through 5 (we skip row 0, index `0`, because that's Customer 0 compared to itself, which is always distance 0 — not a useful neighbor).
- `["customer"].tolist()` — pulls out just the customer ID numbers as a plain list.
- `set(top5["euclidean"]) & set(top5["cosine"])` — the `&` between two Python sets means "give me only the items that appear in both." This counts how many of the top-5 lists overlap between Euclidean and Cosine.

**Real output:**
```
Top 5 closest customers to Customer 0, by each measure:
  euclidean: [4, 17, 47, 16, 48]
  manhattan: [4, 17, 2, 16, 20]
  cosine: [69, 48, 49, 47, 4]

Euclidean and Cosine only agree on 3/5 -- different measures give different answers.
```

**This is proof, not opinion.** Euclidean and Manhattan mostly agree (they both found customers 4, 17, and 16 in their top 5), because they're both measuring similar things — actual numeric closeness. But Cosine picked a very different group of neighbors (69, 49), because it's answering a completely different question — "same pattern," not "same numbers." If your business question is "who spends in a similar pattern regardless of amount," you want Cosine. If your question is "who is numerically most like this customer," you want Euclidean or Manhattan. **Picking the wrong one gives you a wrong-for-your-purpose answer, even though the code runs without any errors.**

---

<a id="10-full-code-one-piece-at-a-time"></a>
## 10. Full Code, One Piece at a Time

Here is the entire script again, for reference, with every section labeled to match the explanations above:

```python
# ---- SETUP (section 5) ----
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import euclidean, cityblock, cosine
from sklearn.preprocessing import StandardScaler

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 170, "savefig.bbox": "tight"})
os.makedirs("images", exist_ok=True)

# ---- PART A: LOOK AT THE DATA (section 6) ----
df = pd.read_csv("../datasets/mall_customers.csv")
print("Loaded", len(df), "real customers")
print(df.describe().round(1))

fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
axes[0].hist(df["Age"], bins=20, color="#3b82f6"); axes[0].set_title("Age")
axes[1].hist(df["Annual Income (k$)"], bins=20, color="#16a34a"); axes[1].set_title("Income ($k)")
axes[2].hist(df["Spending Score (1-100)"], bins=20, color="#f97316"); axes[2].set_title("Spending Score")
plt.tight_layout()
plt.savefig("images/01_eda_distributions.png")
plt.close()

plt.figure(figsize=(7.5, 6))
sns.scatterplot(data=df, x="Annual Income (k$)", y="Spending Score (1-100)", hue="Gender", s=70)
plt.title("Income vs Spending Score")
plt.tight_layout()
plt.savefig("images/02_income_vs_spending.png")
plt.close()

# ---- PART B: THE THREE MEASURES (section 7) ----
features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
X = df[features].values
a, b = X[0], X[1]
print("\nCustomer A:", df.iloc[0][features].to_dict())
print("Customer B:", df.iloc[1][features].to_dict())
print(f"\nEuclidean distance:  {euclidean(a, b):.2f}  (straight-line distance)")
print(f"Manhattan distance:  {cityblock(a, b):.2f}  (add up each feature's difference)")
print(f"Cosine similarity:   {1 - cosine(a, b):.4f}  (how similar their *pattern* is, ignoring size)")

# ---- PART C: SCALING (section 8) ----
X_scaled = StandardScaler().fit_transform(X)
print(f"\nEuclidean distance BEFORE scaling: {euclidean(a, b):.2f}")
print(f"Euclidean distance AFTER scaling:  {euclidean(X_scaled[0], X_scaled[1]):.2f}")
print("Income (up to $137k) drowns out Age (up to 70) unless we scale first.")

# ---- PART D: DO MEASURES AGREE? (section 9) ----
query = X_scaled[0]
dists = {
    "euclidean": [euclidean(query, x) for x in X_scaled],
    "manhattan": [cityblock(query, x) for x in X_scaled],
    "cosine": [cosine(query, x) for x in X_scaled],
}
results = pd.DataFrame({"customer": range(len(X_scaled)), **dists})
top5 = {k: results.sort_values(k).iloc[1:6]["customer"].tolist() for k in dists}
print("\nTop 5 closest customers to Customer 0, by each measure:")
for k, v in top5.items():
    print(f"  {k}: {v}")
overlap = len(set(top5["euclidean"]) & set(top5["cosine"]))
print(f"\nEuclidean and Cosine only agree on {overlap}/5 -- different measures give different answers.")

print("\nDone. Charts saved in images/")
```

---

<a id="11-try-it-yourself"></a>
## 11. Try It Yourself

Now that you understand every line, try modifying the code to build real understanding:

1. **Change which customers you compare.** Instead of `a, b = X[0], X[1]`, try `a, b = X[10], X[150]`. Do the distances make sense given their actual Age/Income/Spending values?
2. **Add a 4th distance measure.** Look up `scipy.spatial.distance.chebyshev` (the "maximum coordinate difference" measure) and add it alongside the other three. Does it agree more with Euclidean or Manhattan?
3. **Try a different query customer.** Change `query = X_scaled[0]` to `query = X_scaled[100]`, and see if Euclidean and Cosine agree more or less than they did for Customer 0.
4. **Turn off scaling on purpose.** Compute the "top 5 nearest neighbors" experiment from Part D using the *unscaled* `X` instead of `X_scaled`. Do the neighbor lists change a lot? This shows you scaling doesn't just change the distance numbers — it changes which points end up "close" at all.

---

<a id="12-common-mistakes-and-fixes"></a>
## 12. Common Mistakes and Fixes

| Mistake | What happens | How to fix it |
|---|---|---|
| Running the script from the wrong folder | `FileNotFoundError: mall_customers.csv` | Always `cd` into the practical's own folder first, then run the script |
| Forgetting to scale before comparing distances | Numbers look "off" or one feature seems to dominate everything | Always run `StandardScaler().fit_transform(X)` before any distance work |
| Comparing a customer to themselves | Distance is always exactly 0 — not useful | Skip index 0 when looking at "nearest neighbors" (see `.iloc[1:6]` in the code — it skips the row matching itself) |
| Using `cosine()` and expecting it to return similarity directly | Your "similarity" values look backwards (bigger number = less similar) | Remember `scipy`'s `cosine()` returns *distance* (1 - similarity). Always do `1 - cosine(a, b)` to get similarity |
| Including `CustomerID` in the feature list | Distances become meaningless (row numbers aren't real information) | Only include actual measurements: Age, Income, Spending Score |
| Getting `nan` from `cosine()` | A row is all zeros, so there's no "direction" to compare | Check your data for degenerate rows — not an issue in this specific dataset, but common in sparse data like text |

---

<a id="13-quick-quiz-check-yourself"></a>
## 13. Quick Quiz — Check Yourself

Try answering these without looking back, then check:

1. What's the difference between Euclidean and Manhattan distance? *(Euclidean is straight-line; Manhattan adds up each dimension's difference separately, like walking city blocks.)*
2. Why does Cosine similarity ignore magnitude? *(Because it only measures the angle between two vectors, not their length.)*
3. If Customer A makes $15k and Customer B makes $137k, and we forget to scale, which feature will probably dominate the Euclidean distance — Age or Income? *(Income, because its raw numbers span a much bigger range.)*
4. True or false: choosing a different distance measure never changes your results. *(False — we proved this directly: Euclidean and Cosine only agreed on 3 of 5 nearest neighbors for the same customer.)*
5. What does `StandardScaler` actually do to each column? *(Subtracts the column's average and divides by its standard deviation, so every column ends up with mean 0 and standard deviation 1.)*

---

<a id="14-summary"></a>
## 14. Summary

In this practical, you:

- Loaded and looked at real, unlabeled data for 200 real mall customers.
- Learned and calculated 3 different ways to measure "closeness" between data points: Euclidean, Manhattan, and Cosine.
- Proved, with real numbers, that raw unscaled distances can be dominated by whichever feature happens to have the biggest range — not whichever feature actually matters most.
- Proved that different distance measures give genuinely different answers, not just slightly different numbers.

Everything you learned here — features, scaling, distance — is the foundation every clustering algorithm in the rest of this course is built on. In Practical 2, we finally use these ideas to group customers into segments.

## Files in This Folder

| File | What it is |
|---|---|
| `similarity_demo.py` | The full, runnable code |
| `images/01_eda_distributions.png` | Age, Income, Spending Score charts |
| `images/02_income_vs_spending.png` | The main scatter plot |

**Next up:** [Practical 2 — k-Means vs k-Medoids](../Practical-02-KMeans-vs-KMedoids/README.md)
