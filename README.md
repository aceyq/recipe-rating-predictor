# Recipe Rating Predictor

A data science project for DSC 80 at UC San Diego exploring recipe quality and user ratings!

---

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
<iframe
  src="plots/calorie_dist.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

<!-- REPLACE THIS LINE with either a screenshot or a code snippet:
```python
recipes_df[['name', 'ingredients', 'minutes', 'n_steps', 'avg_rating', 'calories', 'fat_PDV', 'sugar_PDV']].head()

