import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

"""
BUILD AND EVALUATE LINEAR REGRESSION USING STATSMODELS
"""

# read dataset into a DataFrame
df = pd.read_csv("dataset_2_1_final.csv")

# apply signed log transformation to X5 (from the previous step)
df["LOG_X5"] = np.sign(df["x5_big5_players_diff"]) * np.log1p(np.abs(df["x5_big5_players_diff"]))

# keep X1-X8 and put goal_difference as the last column
df = df[['x1_goals_scored_diff', 'x2_goals_conceded_diff', 'x3_forwards_goals_per_cap_diff',
         'x4_attacking_avg_caps_diff', 'LOG_X5',
         'x6_fifa_points_diff', 'x7_previous_world_cups_diff',
         'x8_host_nation', 'goal_difference']]

# separate explanatory variables (x) from the response variable (y)
x = df.iloc[:,:-1]
y = df.iloc[:,-1]

# build and evaluate the linear regression model
x = sm.add_constant(x)
model = sm.OLS(y,x).fit()
pred = model.predict(x)
model_details = model.summary()
print(model_details)


"""
APPLY Z-SCORE STANDARDISATION
"""
scaler = StandardScaler()

# Drop the previously added constant
x = x.drop(["const"], axis=1)

# Apply z-score standardisation to all explanatory variables
std_x = scaler.fit_transform(x.values)

# Restore the column names of each explanatory variable
std_x_df = pd.DataFrame(std_x, index=x.index, columns=x.columns)


"""
REBUILD AND REEVALUATE LINEAR REGRESSION USING STATSMODELS
USING STANDARDISED EXPLANATORY VARIABLES
"""

# Build and evaluate the linear regression model
std_x_df = sm.add_constant(std_x_df)

print(std_x_df)
model = sm.OLS(y,std_x_df).fit()
pred = model.predict(std_x_df)
model_details = model.summary()
print(model_details)