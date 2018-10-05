
# Repository Overview

In this repository you will find 3 notebooks, one for each of the midterm dataset options. While your model tuning is apt to differ (maybe even be better) use these as a general guidelines and ideas in data exploration, feature engineering, and model tuning. Key highlights and notes are included throughout. In general, remember that your process should generally include the following: 

* Define X and y
    * y should be a continuous numeric variable
    * X should be a number of numeric features
        *X features may need substantial preproccesing including:
            * Transforming datetime values
            * Normalizing features ranges/distributions
            * Creating dummy variables
            * Ensuring there are no categorical variables coded (misleadingly) as numbers
* Train / Test Split
* Fit algorithm on training data
    * Cross Validation
    * Feature Engineering
        * Synthetic Polynomial Features / Polynomial Regression
* Evaluate Model Performance on Test Set
    * Repeat process with seperate train/test split; do results hold?
* Continue feature engineering, tuning, etc.

Cheers!
