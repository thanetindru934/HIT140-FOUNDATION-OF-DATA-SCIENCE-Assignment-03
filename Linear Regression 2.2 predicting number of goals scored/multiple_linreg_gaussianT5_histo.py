import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import PowerTransformer
import matplotlib.pyplot as plt
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
APPLY YEO-JOHNSON TRANSFORMATION TO X5 ONLY
"""
scaler = PowerTransformer()  # Yeo-Johnson by default

# Drop the previously added constant
x = x.drop(["const"], axis=1)

# Apply Yeo-Johnson transformation to X5 only
yj_x_df = x.copy()
yj_x_df["x5_team_forwards_npg_per90"] = scaler.fit_transform(x[["x5_team_forwards_npg_per90"]]).ravel()

"""
VISUALISE X5 BEFORE AND AFTER YEO-JOHNSON
"""
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.hist(x["x5_team_forwards_npg_per90"], bins=12, density=True)
plt.title(f"X5 (original) skew = {x['x5_team_forwards_npg_per90'].skew():.2f}")

plt.subplot(1,2,2)
plt.hist(yj_x_df["x5_team_forwards_npg_per90"], bins=12, density=True)
plt.title(f"X5 (Yeo-Johnson T.) skew = {yj_x_df['x5_team_forwards_npg_per90'].skew():.2f}")

plt.tight_layout()
plt.savefig("X5_Before_After_Yeo_Johnson.png", dpi=300)
plt.show()


"""
REBUILD AND REEVALUATE LINEAR REGRESSION USING STATSMODELS
USING TRANSFORMED X5
"""

# Build and evaluate the linear regression model
yj_x_df = sm.add_constant(yj_x_df)

model = sm.OLS(y,yj_x_df).fit()
pred = model.predict(yj_x_df)
model_details = model.summary()
model_details.extra_txt = None
print(model_details)