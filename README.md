# HIT140 – Foundations of Data Science
## Assignment 03: Multiple Linear Regression Analysis

### 1. Project Overview

This project focuses on developing and evaluating two multiple linear regression models using FIFA World Cup 2026 match data. The objective is to investigate how team-level and player-level statistics can be used to predict football match outcomes.

The assignment consists of two regression tasks:

- **Model 2.1:** Predicting the goal difference between two opposing teams.
- **Model 2.2:** Predicting the number of goals scored by an individual team.

Both models use eight explanatory variables based on information available before each match.

### 2. Model 2.1 – Predicting Goal Difference

**Dataset:** 104 observations (one per match)

**Dependent Variable:** Goal Difference (Team A Goals − Team B Goals)

| Variable | Explanatory Variable |
|---|---|
| X1 | Goals Scored Difference |
| X2 | Goals Conceded Difference |
| X3 | Forwards' Goals per Cap Difference |
| X4 | Attacking Average Caps Difference |
| X5 | Big 5 League Players Difference |
| X6 | FIFA Points Difference |
| X7 | Previous World Cups Difference |
| X8 | Host Nation |

This model examines how differences in attacking performance, defensive strength, international experience, squad quality, FIFA ranking points, and home advantage relate to goal difference.

### 3. Model 2.2 – Predicting Goals Scored

**Dataset:** 208 observations (one per team per match)

**Dependent Variable:** Number of Goals Scored

| Variable | Explanatory Variable |
|---|---|
| X1 | Team Goals Scored |
| X2 | Opponent Goals Conceded |
| X3 | Team Forwards' Goals per Cap |
| X4 | Team Attacking Average Caps |
| X5 | Forwards' Non-Penalty Goals per 90 |
| X6 | Midfielders' Assists per 90 |
| X7 | Team Big 5 League Players |
| X8 | Opponent Elo Rating |

This model investigates how attacking performance, opponent defensive strength, player scoring efficiency, midfield creativity, squad quality, and opponent strength relate to the number of goals scored.

### 4. Tools and Technologies

The project was developed using:

- **Python:** Data processing and statistical analysis.
- **Pandas:** Dataset loading, cleaning, and manipulation.
- **NumPy:** Numerical calculations and variable transformations.
- **Statsmodels:** Multiple linear regression and statistical testing.
- **Scikit-learn:** Model training, testing, and evaluation.
- **Matplotlib and Seaborn:** Data visualisation and exploratory analysis.
- **Visual Studio Code:** Python development and debugging.
- **GitHub:** Project organisation and version control.

### 5. Running the Python Scripts – Path Configuration

Before executing the Python scripts, users must ensure that the Python interpreter, project directory, and dataset file paths are correctly configured for their own computer.

The original development environment used Windows 11, Python 3.13, and Visual Studio Code. Therefore, the paths shown in the terminal may differ on another computer.

**Step 1: Configure the Python Interpreter**

Open the project in Visual Studio Code and select the installed Python interpreter using:

`Ctrl + Shift + P` → `Python: Select Interpreter`

Select the appropriate Python installation.

**Step 2: Change the Project and Script Paths**

The original project directory was:

```text
C:\CDU S2 2026\Foundation of data science Assignment3
```

Users must use the correct location where the repository has been saved on their own computer.

For example, navigate to the Model 2.2 directory:

```powershell
cd "C:\YourFolder\HIT140-Assignment-03\Linear Regression 2.2 predicting number of goals scored"
```

Run a Python script using:

```powershell
python "mlr_statsmodel.py"
```

Another example:

```powershell
python "multiple_linreg_rescaleT4_Z-Score.py"
```

The Python executable path displayed by VS Code will depend on the interpreter installed on the user's computer.

**Step 3: Check the Dataset File Paths**

The Python scripts use Pandas to read CSV datasets.

For Model 2.1:

```python
df = pd.read_csv("dataset_2_1_final.csv")
```

For Model 2.2:

```python
df = pd.read_csv("dataset_2_2_final.csv")
```

If the CSV files are stored in different locations, users must update these paths accordingly.

**Step 4: Install Required Libraries**

Install the required Python libraries using:

```powershell
python -m pip install pandas numpy scipy statsmodels scikit-learn matplotlib seaborn
```

**Important:** Users should run each Python script from its corresponding model folder. Incorrect interpreter, script, or dataset paths may cause execution errors such as `FileNotFoundError`.

### 6. Methodology

The following steps were performed:

1. Collected and organised FIFA World Cup 2026 match data and relevant pre-match statistics.
2. Prepared two datasets containing 104 and 208 observations.
3. Selected eight explanatory variables for each regression model.
4. Conducted exploratory data analysis and examined relationships between variables.
5. Developed multiple linear regression models using Ordinary Least Squares (OLS).
6. Examined correlation and multicollinearity among explanatory variables.
7. Tested logarithmic transformations to investigate potential improvements in model performance.
8. Evaluated models using R², adjusted R², AIC, BIC, MAE, MSE, and RMSE, where applicable.
9. Compared model results and interpreted the statistical findings.

### 7. Model Evaluation

Regression performance was examined using several statistical measures:

- **R²:** Measures the proportion of variation explained by the model.
- **Adjusted R²:** Evaluates explanatory power while accounting for the number of predictors.
- **MAE:** Measures the average absolute prediction error.
- **MSE:** Measures the average squared prediction error.
- **RMSE:** Measures prediction error in the same units as the dependent variable.
- **AIC and BIC:** Support comparisons between alternative regression models.
- **P-values:** Help assess statistical evidence for individual regression coefficients.

### 8. Key Findings

**Model 2.1:** The analysis examined the relationship between team-strength differences and match goal difference. Alternative variable transformations were investigated to evaluate their effects on model performance.

**Model 2.2:** The original regression model achieved an R² of approximately 0.327 and an adjusted R² of 0.300. Forwards' international goals per cap, Big 5 league representation, and opponent Elo rating showed statistically significant associations with goals scored at the 5% significance level.

A logarithmic transformation of attacking average caps was also tested, but it did not meaningfully improve the model's explanatory performance.

### 9. Use of Generative AI

Generative AI was used as a supporting tool for understanding statistical concepts, analysing regression results, and debugging Python code.

AI assistance included troubleshooting programming errors, interpreting regression outputs, understanding multicollinearity, examining logarithmic transformations, and comparing model evaluation metrics.

The analysis involved running Python scripts, examining statistical outputs, testing alternative approaches, and evaluating model performance.

### 10. Project Purpose

The purpose of this assignment is to apply multiple linear regression techniques to real-world football data, develop predictive models, interpret statistical relationships, and evaluate the usefulness and limitations of different explanatory variables.

### 11. Conclusion

This project demonstrates the application of Python-based data analysis and multiple linear regression to FIFA World Cup 2026 match statistics. Both models investigate the relationship between pre-match team and player characteristics and football match outcomes, while highlighting the importance of explanatory variable selection, model diagnostics, and statistical evaluation.
