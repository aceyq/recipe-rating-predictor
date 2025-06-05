# recipe-rating-predictor
A data science project for DSC 80 at UC San Diego exploring recipe quality and user ratings.

## Step 1: Introduction and Question Identification

**Introduction**  
This project uses two datasets: `RAW_recipes.csv` and `interactions.csv`, both related to online recipes. `RAW_recipes.csv` contains detailed information on over 230,000 recipes, including attributes like name, ingredients, cook time, and nutritional values. `interactions.csv` logs user interactions such as ratings and reviews for these recipes.

**Question**  
The central question for this project is:  
**“Can we predict the average rating of a recipe based on its ingredients and preparation metadata?”**

**Why It Matters**  
With the rise of food-related apps and cooking platforms, helping users find high-quality recipes faster can have real-world impact — from saving time and reducing food waste, to improving dietary choices.

**Dataset Overview**  
- `RAW_recipes.csv`: 231,637 rows, 17 columns  
  - Relevant columns:
    - `name`: The name of the recipe  
    - `ingredients`: A list of ingredients used  
    - `minutes`: Total time to prepare  
    - `n_steps`: Number of cooking steps  
    - `submitted`: Date the recipe was added  
    - `nutrition`: A list with nutritional values (e.g., calories, fat, sugar, etc.)  
- `interactions.csv`: 731,927 rows, 5 columns  
  - Relevant columns:
    - `recipe_id`: Foreign key matching the recipe  
    - `rating`: The rating a user gave (1 to 5)  
    - `review`: Optional text review  
    - `date`: Date of interaction
