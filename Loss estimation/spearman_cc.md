# Spearman Correlation: A Mathematical Explanation

The **Spearman correlation** is a non-parametric measure of rank correlation. It assesses how well the relationship between two variables can be described using a monotonic function (i.e., a function that either never decreases or never increases). Unlike the Pearson correlation, which measures linear relationships, Spearman correlation works on the **ranks** of the data rather than the raw data itself.

Below is a step-by-step explanation of how Spearman correlation works, including the mathematical details of ranking:

## Step 1: Rank the Data

The first step in computing the Spearman correlation is to **rank** the data for each variable. Ranking assigns a unique position to each value in the dataset, with ties (duplicate values) receiving the average of their ranks.

### Example:

Consider two variables, \( X \) and \( Y \):

| Index | \( X \) | \( Y \) |
|-------|---------|---------|
| 1     | 10      | 20      |
| 2     | 15      | 25      |
| 3     | 10      | 30      |
| 4     | 20      | 35      |

#### Rank \( X \):
- Sort \( X \) in ascending order: `[10, 10, 15, 20]`.
- Assign ranks:
  - The first `10` gets rank \( 1 \).
  - The second `10` gets rank \( 2 \).
  - Since these values are tied, they receive the average rank: \( \frac{1 + 2}{2} = 1.5 \).
  - The value `15` gets rank \( 3 \).
  - The value `20` gets rank \( 4 \).

Final ranks for \( X \): `[1.5, 1.5, 3, 4]`.

#### Rank \( Y \):
- Sort \( Y \) in ascending order: `[20, 25, 30, 35]`.
- Assign ranks:
  - `20` gets rank \( 1 \).
  - `25` gets rank \( 2 \).
  - `30` gets rank \( 3 \).
  - `35` gets rank \( 4 \).

Final ranks for \( Y \): `[1, 2, 3, 4]`.

---

## Step 2: Compute the Difference in Ranks

For each pair of ranked values, compute the difference in ranks (\( d_i \)):

| Index | Rank \( X \) (\( R_X \)) | Rank \( Y \) (\( R_Y \)) | \( d_i = R_X - R_Y \) | \( d_i^2 \) |
|-------|--------------------------|--------------------------|------------------------|--------------|
| 1     | 1.5                      | 1                        | 0.5                    | 0.25         |
| 2     | 1.5                      | 2                        | -0.5                   | 0.25         |
| 3     | 3                        | 3                        | 0                      | 0            |
| 4     | 4                        | 4                        | 0                      | 0            |

---

## Step 3: Compute the Spearman Correlation

The Spearman correlation coefficient (\( \rho \)) is calculated using the formula:

\[
\rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}
\]

Where:
- \( \sum d_i^2 \) is the sum of squared differences in ranks.
- \( n \) is the number of observations.

### Apply the Formula:

From the table above:
- \( \sum d_i^2 = 0.25 + 0.25 + 0 + 0 = 0.5 \).
- \( n = 4 \).

Plugging into the formula:

\[
\rho = 1 - \frac{6 \times 0.5}{4(16 - 1)} = 1 - \frac{3}{60} = 1 - 0.05 = 0.95
\]

---

## Step 4: Interpretation

The Spearman correlation coefficient (\( \rho \)) ranges from \(-1\) to \(1\):
- \( \rho = 1 \): Perfect positive monotonic relationship.
- \( \rho = -1 \): Perfect negative monotonic relationship.
- \( \rho = 0 \): No monotonic relationship.

In this example, \( \rho = 0.95 \), indicating a strong positive monotonic relationship between \( X \) and \( Y \).

---

## Handling Ties

When there are ties (duplicate values), the ranks are assigned as the **average of their positions**. For example:
- If the sorted data is `[10, 10, 15, 20]`, the ranks are `[1.5, 1.5, 3, 4]`.

This ensures that tied values do not bias the correlation coefficient.

---

## Spearman Correlation in Python and R

Both Python's `scipy.stats.spearmanr` and R's `cor.test()` handle ties using the **average method** by default. However, if you want to manually compute the Spearman correlation (e.g., for educational purposes or to verify results), you can follow the steps above.

---

## Summary

1. **Rank the data** for each variable, handling ties by assigning average ranks.
2. **Compute the difference in ranks** (\( d_i \)) for each pair of observations.
3. **Square the differences** and sum them (\( \sum d_i^2 \)).
4. **Plug into the formula** to compute \( \rho \).

This process ensures that the Spearman correlation captures the strength and direction of a monotonic relationship between two variables.
