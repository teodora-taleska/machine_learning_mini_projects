# Classification Trees and Random Forests Implementation

## Project Overview
This project involves implementing classification trees and random forests in Python (version 3.12). The goal is to construct a classification tree and a random forest model to classify data with a binary target variable. The models will be tested on the TKI resistance FTIR spectral dataset, and performance will be evaluated using misclassification rates and standard errors.

## Features
1. **Tree Class**
   - Implements a classification tree using the Gini impurity criterion.
   - Attributes:
     - `rand`: Random generator for reproducibility.
     - `get_candidate_columns`: Function to select candidate columns for splitting.
     - `min_samples`: Minimum number of samples required to split a node.
   - Methods:
     - `build`: Constructs the tree.
     - `predict`: Predicts the class of input samples.

2. **RandomForest Class**
   - Implements a random forest using an ensemble of classification trees.
   - Attributes:
     - `rand`: Random generator for reproducibility.
     - `n`: Number of bootstrap samples.
   - Methods:
     - `build`: Constructs the forest with `n` trees (each using min_samples=2).
     - `predict`: Uses majority voting to predict class labels.

3. **Evaluation & Testing**
   - Compute **misclassification rates** and **standard errors** for both models on training and testing sets.
   - Implement unit tests (`MyTests`) for critical and edge cases.

4. **Permutation-Based Variable Importance**
   - Implemented in `importance()` method of `RandomForest`.
   - Computes feature importance by measuring accuracy drop after random permutation.
   - Generates a plot of feature importance.

5. **Variable Importance for Feature Combinations**
   - Extended variable importance to groups of 3 variables (`importance3()`).
   - Compares trees built on top-3 features from both importance methods.
   - Implements `importance3_structure()` to extract best variable combinations from pre-built trees.

## Dataset & Usage
- **Dataset Files:** `tki-train.tab` (training data), `tki-test.tab` (testing data).
- **Testing File:** `test_hw_tree.py` ensures correctness of implementation.

## Running the Code
Ensure Python 3.12 is installed and run:
```bash
python hw_tree.py
```

## Expected Outputs  
- **Misclassification Rates & Standard Errors:**  
  - Report misclassification rates and their uncertainties from `hw_tree_full` (classification tree with `min_samples=2`).  
  - Report misclassification rates and their uncertainties from `hw_randomforest` (random forest with `n=100` trees and `min_samples=2`).  

- **Performance Analysis:**  
  - Plot misclassification rates versus the number of trees (`n`).  
  - Compare classification trees built on the top 3 features from:  
    - Permutation-based variable importance (`importance()`).  
    - Extended variable importance for 3-variable combinations (`importance3()`).  
    - Best 3-variable selection based on tree structures (`importance3_structure()`).  

- **Variable Importance Visualization:**  
  - Plot feature importance for Random Forest with `n=100` trees, maintaining the variable order.  
  - Compare feature importance with variables from the roots of 100 non-random trees trained on randomized data.  

- **Final Comparison & Analysis:**  
  - Evaluate and report performance differences between models using different feature selection strategies.  
  - Justify the selection of best 3-variable combinations based on tree structures.  
