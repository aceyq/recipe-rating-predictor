# recipe-rating-predictor

A data science project for DSC 80 at UC San Diego exploring recipe quality and user ratings.

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
- `RAW_recipes.csv`: 231,637 rows by 17 columns
- `interactions.csv`: 731,927 rows by 5 columns

**Relevant Columns Explained**  
| Dataset         | Column       | Description                                               |
|----------------|--------------|-----------------------------------------------------------|
| RAW_recipes    | `name`       | The title of the recipe                                   |
| RAW_recipes    | `ingredients`| A list of ingredients used in the recipe                  |
| RAW_recipes    | `minutes`    | Time (in minutes) required to make the recipe             |
| RAW_recipes    | `n_steps`    | Number of steps in the cooking instructions               |
| RAW_recipes    | `submitted`  | Date the recipe was submitted                             |
| RAW_recipes    | `nutrition`  | A list of 7 values: calories, fat, sugar, sodium, etc.    |
| interactions   | `rating`     | User’s 1–5 rating of the recipe                           |
| interactions   | `recipe_id`  | ID that links to the corresponding recipe                 |
| interactions   | `review`     | Optional free-text comment left by the user               |

---

We begin this project by understanding and cleaning the data, then proceed to evaluate missingness, test hypotheses, and ultimately frame a prediction task for recipe ratings.