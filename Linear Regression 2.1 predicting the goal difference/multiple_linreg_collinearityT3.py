import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Read dataset into a DataFrame
df = pd.read_csv("dataset_2_1_final.csv")

# Apply signed log transformation to X5 (from the previous step)
df["LOG_X5"] = np.sign(df["x5_big5_players_diff"]) * np.log1p(np.abs(df["x5_big5_players_diff"]))

# Keep X1-X8 and put goal_difference as the last column
df = df[['x1_goals_scored_diff', 'x2_goals_conceded_diff', 'x3_forwards_goals_per_cap_diff',
         'x4_attacking_avg_caps_diff', 'LOG_X5',
         'x6_fifa_points_diff', 'x7_previous_world_cups_diff',
         'x8_host_nation', 'goal_difference']]

# Plot correlation matrix
corr = df.corr()

# Make the figure bigger so all labels fit
plt.figure(figsize=(14, 10))

# Plot the pairwise correlation as heatmap
ax = sns.heatmap(
    corr,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=False,
    annot=True,
    fmt=".2f",
    annot_kws={"size": 9}
)

# customise the labels (full names in a smaller font)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right',
    fontsize=8
)
ax.set_yticklabels(
    ax.get_yticklabels(),
    fontsize=8
)

# Leave room for the labels at the bottom and left
plt.tight_layout()
plt.savefig("Correlation_Heatmap_2_1.png", dpi=300)
plt.show()


"""
BUILD AND EVALUATE LINEAR REGRESSION USING STATSMODELS
"""

# Separate explanatory variables (x) from the response variable (y)
x = df.iloc[:,:-1]
y = df.iloc[:,-1]

# Build and evaluate the linear regression model
x = sm.add_constant(x)
model = sm.OLS(y,x).fit()
pred = model.predict(x)
model_details = model.summary()
model_details.extra_txt = None
print(model_details)


"""
REBUILD AND REEVALUATE LINEAR REGRESSION USING STATSMODELS
WITH COLLINEARITY BEING FIXED
"""

# Drop one of the correlated variables (x6 and x7 have r = 0.78). Keep x6.
df = df.drop(["x7_previous_world_cups_diff"], axis=1)
print(df.info())

# Separate explanatory variables (x) from the response variable (y)
x = df.iloc[:,:-1]
y = df.iloc[:,-1]

# Build and evaluate the linear regression model
x = sm.add_constant(x)
model = sm.OLS(y,x).fit()
pred = model.predict(x)
model_details = model.summary()
model_details.extra_txt = None
print(model_details)