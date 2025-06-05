---
layout: page
title: Predicting Recipe Ratings
---
# Introduction and Question Identification

**Introduction**  
Food is deeply personal — and yet, when we search for recipes online, we often rely on star ratings and vague comments to decide what to cook. This project explores how **recipe metadata** — like ingredients, cook time, and number of steps — can help us **predict recipe ratings**, potentially improving search results and recommendations on cooking platforms.

To investigate this, we use two large datasets:

- `RAW_recipes.csv`: Contains **231,637 recipes**, including details like:
  - `name`: Title of the recipe
  - `ingredients`: A list of ingredients used
  - `minutes`: Time it takes to prepare the recipe
  - `n_steps`: Number of steps in the preparation process
  - `submitted`: The date the recipe was added to the site
  - `nutrition`: A list of seven nutritional values (calories, fat, sugar, etc.)

- `interactions.csv`: Contains **731,927 user interactions** (ratings and reviews), with:
  - `recipe_id`: Foreign key linking to a recipe
  - `rating`: A score from 1 to 5
  - `review`: A free-text user review (can be blank)
  - `date`: The date the interaction occurred

**Core Question**  
> **Can we predict the average rating of a recipe based on its ingredients and preparation metadata?**

In other words — can we estimate how good a recipe is *before* anyone rates it?

**Why This Matters**  
Cooking websites host tens of thousands of recipes. Many are unrated or under-reviewed, leaving users guessing. A reliable prediction model could:

- Suggest high-quality new recipes faster
- Surface hidden gems with low visibility
- Help app developers recommend more relevant content to users

This question also offers a real-world application of data science principles — from data cleaning to prediction and fairness — in a way that’s tangible (and tasty!).

**Dataset Summary**
- `RAW_recipes.csv`: 231,637 rows × 17 columns  
- `interactions.csv`: 731,927 rows × 5 columns

**Relevant Columns Explained**

**RAW_recipes.csv**
- `name`: The title of the recipe  
- `ingredients`: A list of ingredients used  
- `minutes`: Time (in minutes) to prepare  
- `n_steps`: Number of steps in the instructions  
- `submitted`: Date the recipe was submitted  
- `nutrition`: List of nutritional values (calories, fat, sugar, etc.)

**interactions.csv**
- `recipe_id`: Links to a recipe in `RAW_recipes.csv`  
- `rating`: User’s rating (1–5)  
- `review`: Optional written review  
- `date`: Date of the interaction
