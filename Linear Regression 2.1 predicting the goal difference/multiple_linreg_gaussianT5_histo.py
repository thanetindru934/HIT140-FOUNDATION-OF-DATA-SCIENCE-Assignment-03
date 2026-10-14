import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import PowerTransformer
import matplotlib.pyplot as plt
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
APPLY YEO-JOHNSON TRANSFORMATION TO LOG_X5 ONLY
"""
scaler = PowerTransformer()  # Yeo-Johnson by default

# Drop the previously added constant
x = x.drop(["const"], axis=1)

# Apply Yeo-Johnson transformation to LOG_X5 only
yj_x_df = x.copy()
yj_x_df["LOG_X5"] = scaler.fit_transform(x[["LOG_X5"]]).ravel()

"""
VISUALISE LOG_X5 BEFORE AND AFTER YEO-JOHNSON
"""
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.hist(x["LOG_X5"], bins=12, density=True)
plt.title(f"LOG_X5 (original) skew = {x['LOG_X5'].skew():.2f}")

plt.subplot(1,2,2)
plt.hist(yj_x_df["LOG_X5"], bins=12, density=True)
plt.title(f"LOG_X5 (Yeo-Johnson T.) skew = {yj_x_df['LOG_X5'].skew():.2f}")

plt.tight_layout()
plt.show()


"""
REBUILD AND REEVALUATE LINEAR REGRESSION USING STATSMODELS
USING TRANSFORMED LOG_X5
"""

# Build and evaluate the linear regression model
yj_x_df = sm.add_constant(yj_x_df)

print(yj_x_df)
model = sm.OLS(y,yj_x_df).fit()
pred = model.predict(yj_x_df)
model_details = model.summary()
model_details.extra_txt = None
print(model_details)