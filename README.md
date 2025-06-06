# Introduction and Question Identification

**Introduction**  
Online recipe platforms like AllRecipes and Food.com host hundreds of thousands of user-submitted recipes. However, many recipes receive few or no ratings, leaving users unsure which ones to trust. This project investigates whether we can predict a recipe's average user rating based solely on its metadata — such as ingredients, cook time, and nutritional values — before any user interaction has occurred.

We aim to answer:  
**Can we predict the average rating of a recipe based on its ingredients and preparation metadata?**

**Why This Matters**  
A reliable prediction model could:
- Recommend high-quality recipes even if they have few ratings
- Help food platforms surface new content more effectively
- Enhance the user experience by reducing decision fatigue
- Serve as a real-world case study for supervised learning using structured data

---

## Dataset Overview

This project uses two large datasets:

- **`RAW_recipes.csv`**  
  - 231,637 recipes × 17 columns  
  - Contains recipe-level metadata
- **`interactions.csv`**  
  - 731,927 user interactions × 5 columns  
  - Contains user ratings and reviews

---

## Relevant Columns and Their Roles

### From `RAW_recipes.csv`:
- **`name`**: Title of the recipe (used for context, not modeling)
- **`ingredients`**: List of ingredients (can be transformed into frequency/word vectors)
- **`minutes`**: Preparation time in minutes (numeric predictor)
- **`n_steps`**: Number of cooking steps (proxy for recipe complexity)
- **`submitted`**: Date the recipe was submitted (can be converted to recipe age)
- **`nutrition`**: List of 7 nutritional values — calories, fat, sugar, sodium, protein, etc.

### From `interactions.csv`:
- **`recipe_id`**: Foreign key used to join with `RAW_recipes.csv`
- **`rating`**: Target variable (1–5 rating)
- **`review`**: Free-text review (not used for modeling, but useful for exploring rating justification)
- **`date`**: Date of interaction (can be used to analyze temporal patterns)

---

## Summary

This project will explore how a combination of structured recipe metadata and historical interaction data can be used to predict user ratings. We will begin by cleaning and exploring the datasets, then assess missingness, perform hypothesis testing, and ultimately develop a supervised learning model.

## Data Cleaning and Exploratory Data Analysis

### Data Cleaning

We began with two datasets:

- `RAW_recipes.csv`: Contains recipe metadata like `name`, `ingredients`, `minutes`, `n_steps`, and `nutrition`.
- `interactions.csv`: Contains user-generated `rating`s and optional `review`s for each recipe.

Our first step was to clean the `interactions.csv` dataset. Some ratings were recorded as `0`, which is not a valid rating on the platform (ratings range from 1 to 5). These were replaced with `NaN` to properly represent missing or invalid values. We then calculated the **average rating** for each recipe across all its user reviews.

This new column, `avg_rating`, was merged into `recipes_df` using the `recipe_id` foreign key. This resulted in a single, enriched dataframe where each recipe has an associated average rating — our target variable.

Next, we unpacked the `nutrition` column from the `RAW_recipes.csv` file. This column originally contained a Python-style string representing a list of 7 nutrition-related values. We used `eval()` and `pd.Series` to extract these into their own structured columns:

- `calories`
- `fat_PDV`
- `sugar_PDV`
- `sodium_PDV`
- `protein_PDV`
- `sat_fat_PDV`
- `carbs_PDV`

These numeric columns are now ready for exploratory data analysis and modeling.

The result is a clean, feature-rich dataset containing both recipe metadata and user behavior metrics.

👉 **Below is the head of the cleaned `recipes_df` dataframe:**  
![Head of Cleaned DataFrame](images/head_df.jpg)

## Univariate Analysis

We began our exploratory analysis by examining the distribution of calories per recipe. Many recipes had extreme outliers, so we filtered out entries with more than 3000 calories to better capture the core distribution.

This skew suggests that user-submitted recipes tend to favor lighter or more everyday meals, while extremely high-calorie recipes are relatively rare. Understanding the distribution of calories helps inform feature scaling choices later and provides context for how nutrition varies across the dataset.

<iframe
  src="plots/calorie_dist.html"
  width="800"
  height="500"
  frameborder="0"
></iframe>

## Bivariate Analysis

To explore how recipe calorie content relates to user preferences, we plotted a scatterplot of `calories` versus `avg_rating`, filtering out recipes above 3000 calories to reduce distortion from extreme outliers.

The scatterplot reveals that **most ratings are clustered at the upper end (4.0 to 5.0)** regardless of calorie count, suggesting that calorie content may not be a strong driver of user satisfaction. However, we also see a small concentration of **lower ratings (1.0–3.0)** for recipes in the higher-calorie range (1500–3000), which could suggest either lower-quality indulgent recipes or less user engagement with such items. The lack of a clear downward or upward trend implies that **calories alone do not predict average rating well**, at least in a linear sense.

<iframe
  src="plots/calories_vs_avg_rating.html"
  width="800"
  height="500"
  frameborder="0"
></iframe>

## Interesting Aggregates

To explore whether recipe complexity affects user satisfaction, we grouped recipes by the number of preparation steps (`n_steps`) into four bins:  
**<5**, **5–10**, **10–20**, and **20+**.

For each bin, we calculated the **average user rating**. The bar chart below reveals that **the average ratings are nearly identical across all step groups**, ranging from 4.62 to 4.65. This suggests that the number of steps — a proxy for complexity — does **not have a strong influence** on how recipes are rated.

This result is itself significant: it implies that users may rate recipes based more on factors like taste, familiarity, or ingredient quality rather than how long or complicated the recipe is to make.

<iframe
  src="plots/avg_rating_by_steps.html"
  width="800"
  height="500"
  frameborder="0"
></iframe>

## Assessment of Missingness

We examined the dataset for missing values using `.isna().sum()`, and found that several columns contain `NaN` entries. The most notable missingness occurred in:

- **`avg_rating`**: This column was created by merging recipe data with user ratings. Many recipes had no user interactions, leading to missing average ratings.
- **Nutritional columns** (like `sugar_PDV` or `sat_fat_PDV`): Missing for some recipes where the `nutrition` list was malformed or incomplete.

---

### NMAR Column: `avg_rating`

We believe that the `avg_rating` column is **Not Missing at Random (NMAR)**. Its missingness is not due to chance or unrelated variables — rather, it's directly tied to a **lack of user engagement**. Recipes with no views, no ratings, or no popularity **will not have any average rating**, so the missingness is inherently linked to how visible or appealing a recipe is to users.

> This means the absence of data itself is informative — recipes with no ratings are likely unpopular or untested. This violates the assumptions of MAR or MCAR.

To possibly reclassify this as **MAR**, we would need **additional data** — for example:
- Impressions/clicks per recipe
- Whether the recipe was ever featured or recommended
- Author popularity or follower count

Such context could help explain the missing ratings based on observable factors, thus making the missingness **conditionally random**.

---

Understanding this NMAR pattern is important, as **excluding recipes with missing ratings** could bias our model toward popular recipes and overlook newer or niche content.

---

### Permutation Testing for Missingness Dependency

To further support our hypothesis that `avg_rating` is Not Missing at Random (NMAR), we performed permutation tests comparing recipes **with** and **without** missing `avg_rating` values across different features.

We computed a test statistic defined as the **absolute difference in mean values** between the missing and non-missing groups. We then compared this observed value to a distribution of the same statistic under random permutations of the missingness indicator.

---

**Test 1: `avg_rating` vs. `minutes`**
- **p-value = 0.0380**
- While the effect is modest, this test suggests that recipes with extremely short or long prep times are slightly more likely to have missing ratings — possibly due to user engagement differences.

**Test 2: `avg_rating` vs. `n_steps`**
- **p-value < 0.001**
- This test shows a strong relationship between number of steps and rating availability. Recipes with many steps are significantly less likely to be rated, which supports our earlier claim that `avg_rating` is NMAR — tied to a recipe's complexity and visibility.

---

### Plot: Empirical Distribution of Test Statistic

Below is a histogram of the test statistic from the `minutes` permutation test, with the observed value (in red) shown for comparison. The observed statistic is in the tail of the distribution, reinforcing the claim of dependency between `minutes` and the missingness of `avg_rating`.

<iframe
  src="plots/avg_rating_missingness_vs_minutes.html"
  width="800"
  height="500"
  frameborder="0"
></iframe>

The histogram above shows the empirical distribution of the test statistic (the absolute difference in mean `minutes`) generated by randomly permuting the missingness of `avg_rating` 1,000 times.

The red dashed line represents the **observed test statistic** of 117.34 — the actual difference in average preparation time between recipes that have a missing `avg_rating` and those that don’t.

Since the observed value lies in the far-right tail of the distribution, and the **p-value is 0.038**, we reject the null hypothesis that missingness in `avg_rating` is independent of `minutes`. This suggests that recipes with unusually short or long prep times are more likely to have missing ratings.

This supports our broader claim that `avg_rating` is **Not Missing at Random (NMAR)** — its absence is tied to recipe characteristics that influence user behavior, such as complexity or time investment.

---

### Hypotheses Testing

## Hypothesis Testing: Do Recipes with More Ingredients Receive Higher Ratings?

To investigate whether recipe complexity (measured by ingredient count) affects user satisfaction, we tested whether recipes with **more ingredients** tend to receive **higher average ratings**.

---

### Hypotheses

- **Null Hypothesis (H₀):** The number of ingredients and a recipe’s average rating are **independent** — any observed association is due to chance.
- **Alternative Hypothesis (H₁):** There is a **non-zero correlation** between number of ingredients and average rating.

---

### Test Design
We used a **permutation test** on the Pearson correlation coefficient between n_ingredients and avg_rating.

- **Test statistic**: Pearson correlation coefficient between n_ingredients and avg_rating
- **Significance level (α)**: 0.05
- **Observed test statistic**: 0.0030
- **p-value**: 0.3880

We computed the observed correlation and then permuted the avg_rating column 1,000 times to simulate what values would be expected under the null hypothesis of no association. We then compared our observed correlation to this distribution to obtain a p-value.

---

### Conclusion

The observed correlation of **0.0030** is extremely small, and the p-value of **0.3880** is **much larger** than our significance threshold of 0.05. Therefore, we fail to reject the null hypothesis.

This result suggests that, in this dataset, there is no meaningful relationship between the number of ingredients a recipe has and how highly it is rated. In other words, complexity does not guarantee better ratings, and simpler recipes may be just as satisfying to users.
---

### Permutation Test Plot

Below is a histogram showing the distribution of the test statistic under 1,000 permutations of ingredient group labels. The red dashed line represents the actual observed difference.

<iframe
  src="plots/ingredients_vs_rating.html"
  width="800"
  height="500"
  frameborder="0"
></iframe>


