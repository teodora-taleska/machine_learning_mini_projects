# Machine Learning Mini Projects

A collection of small-scale machine learning projects and experiments that demonstrate core concepts across supervised learning, model evaluation, optimization, and kernel methods. Each project is self-contained inside its own folder and includes code, notebooks, a report, and notes to reproduce experiments and learn from the implementation details.

This README summarizes the contents of each project directory, installation and usage instructions and recommended experiments.

### Table of contents
- Overview
- Project summaries
  - ANN
  - GLM
  - Kernels
  - Loss estimation
  - Trees and Forests
- Getting started (setup & dependencies)
- How to run a project
- Recommended experiments and extensions
- Reproducibility
- Troubleshooting
- Acknowledgements & references

Overview
--------
These mini-projects are intended for hands-on learning and reproducible experimentation. They are focused on clarity and education rather than production-readiness. Typical contents of each project folder:
- Code (scripts and/or modules)
- Jupyter notebooks demonstrating step-by-step analysis and visualizations
- Small synthetic or sample datasets, or instructions to download public datasets
- Reports

Project summaries
-----------------

ANN (Artificial Neural Networks)
- Purpose: Implement basic feedforward neural networks and experiments to demonstrate how network depth, width, activation functions, initialization, learning rate, and regularization affect learning.
- Typical contents you should find: training scripts or notebooks, implementations of forward/backward passes (or use of frameworks such as PyTorch / TensorFlow), training/validation curves, and visualization of learned decision boundaries.
- How to run:
  - Inspect the notebook(s) for step-by-step examples.
- Suggested experiments:
  - Compare ReLU vs. tanh activations on the same task.
  - Explore effects of different optimizers (SGD, Adam).
  - Visualize loss surface slices / learning rate schedules.
- Expected learning outcomes: basic backpropagation intuition, overfitting vs underfitting, regularization techniques in small networks.

GLM (Generalized Linear Models)
- Purpose: Demonstrate linear regression, logistic regression, and other GLMs (e.g., Poisson) with synthetic and real datasets. Emphasis on interpretation, link functions, and diagnostics.
- Typical contents: notebooks showing model fitting, residual analysis, link function demonstrations, likelihood and deviance computations.
- How to run:
  - Open the GLM notebook for examples and explanatory visualizations.
- Suggested experiments:
  - Fit logistic regression with and without L1/L2 regularization; inspect coefficients.
  - Compare GLM results to more flexible models (e.g., small neural net) on the same dataset to discuss bias-variance tradeoff.
- Expected learning outcomes: familiarity with statistical model assumptions, link functions, and model diagnostics.

Kernels
- Purpose: Explore kernel methods (e.g., kernel ridge regression, kernel SVM) and kernel design (Gaussian, polynomial, linear), and show how kernels implicitly map data to higher-dimensional spaces.
- Typical contents: notebooks demonstrating kernel functions, Gram matrix computation, kernel PCA, and kernelized classifiers/regressors.
- How to run:
  - Use the notebook to step through kernel definitions and examples.
- Suggested experiments:
  - Visualize the effect of kernel width (gamma) on decision boundaries.
  - Compute kernel PCA embeddings and compare to standard PCA.
- Expected learning outcomes: intuition of kernel trick, kernel hyperparameter effects, trade-offs between linear and kernel methods.

Loss estimation
- Purpose: Investigate different loss functions (MSE, MAE, Huber, cross-entropy), robust estimation, and how the choice of loss affects model behavior.
- Typical contents: notebooks comparing optimization under different losses, robust regression examples, and discussion on outliers and influence functions.
- How to run:
  - Open the notebooks to see comparative experiments.
- Suggested experiments:
  - Compare MSE vs MAE in presence of outliers.
  - Plot gradients and their impacts for different losses.
- Expected learning outcomes: when to prefer robust losses, interpretation of loss gradients, and model sensitivity to noise.

Trees and Forests
- Purpose: Implement decision trees and random forests from scratch or via scikit-learn; illustrate impurity measures (Gini, entropy), bootstrap aggregation, feature importance, and out-of-bag estimation.
- Typical contents: implementation notebooks, scripts for training/evaluating decision trees and random forests, visualizations of tree structures and feature importance.
- How to run:
  - Check the project notebook for walkthroughs.
  - Note: directory name contains a space; when using CLI, quote the path or use escapes.
- Suggested experiments:
  - Build a random forest from scratch to understand bootstrapping and aggregation.
  - Compare performance of Gini vs. entropy and different max_depth settings.
- Expected learning outcomes: tree splitting criteria, ensemble benefits, and bias-variance behavior of tree-based models.

Getting started (setup & dependencies)
--------------------------------------
Recommended: create and use a virtual environment.

- Using venv:
  ```bash
  python -m venv venv
  source venv/bin/activate      # Linux / macOS
  venv\Scripts\activate         # Windows (PowerShell)
  ```

- Install core dependencies:
  ```bash
  pip install --upgrade pip
  pip install numpy pandas matplotlib scikit-learn jupyter notebook seaborn
  ```
- Optional (if projects use deep learning frameworks):
  ```bash
  pip install torch torchvision        # PyTorch
  # or
  pip install tensorflow               # TensorFlow
  ```

- If a project provides a requirements.txt inside its folder, install with:
  ```bash
  pip install -r <project-folder>/requirements.txt
  ```

How to run a project
--------------------
1. Open the folder for the project you want to run (e.g., `ANN/`, `GLM/`, `Kernels/`, `Loss estimation/`, `Trees and Forests/`).
2. Look for a README or a notebook (.ipynb).
3. Typical patterns:
   - Run a Jupyter notebook:
     ```bash
     jupyter notebook
     # then open the notebook in your browser and run cells interactively
     ```

Recommended experiments and extensions
-------------------------------------
- Hyperparameter sweeps: grid search and random search over learning rates, regularization, kernel parameters, and tree hyperparameters.
- Small-scale reproducibility: fix random seeds and document environment (Python version, package versions).
- Visual analysis: plot learning curves, decision boundaries, residuals, and confusion matrices.
- Compare algorithms: for a chosen dataset, compare GLM, small ANN, kernel SVM, and random forest to discuss strengths and weaknesses.
- Implement missing features: for example, if a tree implementation is missing pruning or splitting heuristics, add them and measure impact.

Reproducibility
-----------------------
- To ensure reproducibility:
  - Fix seeds for numpy, Python, and framework-specific RNGs.
  - Record package versions:
    ```bash
    pip freeze > requirements.lock
    ```
  - Use notebooks to record experiment notes and parameter settings.

Contributing
------------
Contributions are welcome. A suggested workflow:
1. Fork the repo and create a feature branch:
   ```bash
   git checkout -b feature/brief-description
   ```
2. Add code or documentation inside the relevant folder. Prefer notebooks for exploratory work and python modules for reusable code.
3. Add tests where appropriate and update the top-level README or the project's README with usage examples.
4. Open a pull request describing your changes and include reproducible steps / commands and expected outputs.

When adding new projects, include:
- A short README in the new folder describing purpose, files, dataset source, how to run, and expected outputs.
- Any required data or a script to download data.
- Example notebooks or scripts demonstrating the key experiments.

Troubleshooting
---------------------
- If a script fails due to missing packages: install dependencies with pip or check for a requirements file in the project folder.
- Path / filename issues: some folder names contain spaces (e.g., "Trees and Forests", "Loss estimation") — quote paths or use escaping when calling from shell.
- If outputs differ: check random seeds, package versions, and floating-point nondeterminism (especially for GPU frameworks).


Acknowledgements & references
-----------------------------
- scikit-learn documentation for quick references on implementations.
- Standard textbooks and online resources for theoretical background (Bishop, Hastie/Tibshirani/Friedman, Goodfellow et al. for deep learning).
- Papers or blogs linked in specific project notebooks when relevant.
  
