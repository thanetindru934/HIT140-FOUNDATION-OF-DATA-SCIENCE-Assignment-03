import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np

# Read dataset into a DataFrame
df = pd.read_csv("dataset_2_1_final.csv")

# Keep only X1-X8 and Y (goal_difference as the last column)
df = df[['x1_goals_scored_diff', 'x2_goals_conceded_diff', 'x3_forwards_goals_per_cap_diff',
         'x4_attacking_avg_caps_diff', 'x5_big5_players_diff',
         'x6_fifa_points_diff', 'x7_previous_world_cups_diff',
         'x8_host_nation', 'goal_difference']]

# Separate explanatory variables (x) from the response variable (y)
x = df.iloc[:,:-1]
y = df.iloc[:,-1]

# Build and evaluate the linear regression model (original X5)
x = sm.add_constant(x)
model = sm.OLS(y,x).fit()
pred = model.predict(x)
model_details = model.summary()
print(model_details)


"""Non-linear transformation of X5 variable"""

# Re-read dataset into a DataFrame
df = pd.read_csv("dataset_2_1_final.csv")

# Apply signed log transformation (X5 has negative values, so plain np.log cannot be used)
df["LOG_X5"] = np.sign(df["x5_big5_players_diff"]) * np.log1p(np.abs(df["x5_big5_players_diff"]))

# Rearrange the variables so that goal_difference appears as the last column
df = df[['x1_goals_scored_diff', 'x2_goals_conceded_diff', 'x3_forwards_goals_per_cap_diff',
         'x4_attacking_avg_caps_diff', 'x5_big5_players_diff', 'LOG_X5',
         'x6_fifa_points_diff', 'x7_previous_world_cups_diff',
         'x8_host_nation', 'goal_difference']]

# Visualise the effect of the transformation
plt.figure(figsize=(20,10))

plt.subplot(1,2,1)
plt.scatter(df["x5_big5_players_diff"], df["goal_difference"], color="green")
plt.title("Original X5")
plt.xlabel("X5 - Big Five Players Difference")
plt.ylabel("Goal Difference")
b1, b0 = np.polyfit(df["x5_big5_players_diff"], df["goal_difference"], 1)
plt.plot([-25,25], [b0 + b1*(-25), b0 + b1*25])

plt.subplot(1,2,2)
plt.scatter(df["LOG_X5"], df["goal_difference"], color="red")
plt.title("Log Transformed X5")
plt.xlabel("LOG_X5")
plt.ylabel("Goal Difference")
b1, b0 = np.polyfit(df["LOG_X5"], df["goal_difference"], 1)
plt.plot([-3.5,3.5], [b0 + b1*(-3.5), b0 + b1*3.5])

plt.show()


"""
RE-RUN THE LINEAR REGRESSION MODEL WITH A TRANSFORMED VARIABLE
"""
# Drop the original X5 variable
df = df.drop("x5_big5_players_diff", axis=1)

# Separate explanatory variables (x) from the response variable (y)
x = df.iloc[:,:-1]
y = df.iloc[:,-1]

# Build and evaluate the linear regression model
x = sm.add_constant(x)
model = sm.OLS(y,x).fit()
pred = model.predict(x)
model_details = model.summary()
print(model_details)


