import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

"""
BUILD AND EVALUATE LINEAR REGRESSION USING STATSMODELS
"""

# read dataset into a DataFrame
df = pd.read_csv("dataset_2_2_final.csv")

# apply log transformation to X4 (from the previous step)
df["LOG_X4"] = np.log(df["x4_team_attacking_avg_caps"])

# keep X1-X8 and put goals_scored as the last column
df = df[['x1_team_goals_scored', 'x2_opponent_goals_conceded', 'x3_team_forwards_goals_per_cap',
         'LOG_X4', 'x5_team_forwards_npg_per90',
         'x6_team_midfielders_assists_per90', 'x7_team_big5_players',
         'x8_opponent_elo', 'goals_scored']]

# separate explanatory variables (x) from the response variable (y)
x = df.iloc[:,:-1]
y = df.iloc[:,-1]

# build and evaluate the linear regression model
x = sm.add_constant(x)
model = sm.OLS(y,x).fit()
pred = model.predict(x)
model_details = model.summary()
model_details.extra_txt = None
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

model = sm.OLS(y,std_x_df).fit()
pred = model.predict(std_x_df)
model_details = model.summary()
model_details.extra_txt = None
print(model_details)