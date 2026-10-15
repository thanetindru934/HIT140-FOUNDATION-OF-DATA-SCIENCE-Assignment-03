import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Read dataset into a DataFrame
df = pd.read_csv("dataset_2_2_final.csv")

# Apply log transformation to X4 (from the previous step)
df["LOG_X4"] = np.log(df["x4_team_attacking_avg_caps"])

# Keep X1-X8 and put goals_scored as the last column
df = df[['x1_team_goals_scored', 'x2_opponent_goals_conceded',
         'x3_team_forwards_goals_per_cap', 'LOG_X4',
         'x5_team_forwards_npg_per90',
         'x6_team_midfielders_assists_per90',
         'x7_team_big5_players',
         'x8_opponent_elo', 'goals_scored']]

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

# Customise the labels (full names in a smaller font)
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
plt.savefig("Correlation_Heatmap_2_2.png", dpi=300)
plt.show()


"""
BUILD AND EVALUATE LINEAR REGRESSION USING STATSMODELS
"""

# Separate explanatory variables (x) from the response variable (y)
x = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Build and evaluate the linear regression model
x = sm.add_constant(x)
model = sm.OLS(y, x).fit()
pred = model.predict(x)
model_details = model.summary()
model_details.extra_txt = None
print(model_details)


"""
REBUILD AND REEVALUATE LINEAR REGRESSION USING STATSMODELS
WITH COLLINEARITY BEING FIXED
"""

# Drop one of the most correlated variables (x3 and x5 have r = 0.67). Keep x3.
df = df.drop(["x5_team_forwards_npg_per90"], axis=1)
print(df.info())

# Separate explanatory variables (x) from the response variable (y)
x = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Build and evaluate the linear regression model
x = sm.add_constant(x)
model = sm.OLS(y, x).fit()
pred = model.predict(x)
model_details = model.summary()
model_details.extra_txt = None
print(model_details)