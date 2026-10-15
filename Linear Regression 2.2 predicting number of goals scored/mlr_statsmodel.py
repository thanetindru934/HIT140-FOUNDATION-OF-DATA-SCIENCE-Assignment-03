import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

# Read dataset
df = pd.read_csv('dataset_2_2_final.csv')

# Select exactly 8 explanatory variables (X1-X8) and Y
df = df[['x1_team_goals_scored',
         'x2_opponent_goals_conceded',
         'x3_team_forwards_goals_per_cap',
         'x4_team_attacking_avg_caps',
         'x5_team_forwards_npg_per90',
         'x6_team_midfielders_assists_per90',
         'x7_team_big5_players',
         'x8_opponent_elo',
         'goals_scored']]

# Separate explanatory variables (x) from the response variable (y)
x = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Build and evaluate the linear regression model (original X4)
x = sm.add_constant(x)
model = sm.OLS(y, x).fit()
pred = model.predict(x)
model_details = model.summary()
model_details.extra_txt = None
print(model_details)


"""Non-linear transformation of X4 variable"""

# Apply log transformation to X4 (always positive, so np.log can be used)
df['LOG_X4'] = df['x4_team_attacking_avg_caps'].apply(np.log)

# Visualise the effect of the transformation
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.scatter(df['x4_team_attacking_avg_caps'], df['goals_scored'], color='green')
plt.title('Original X4')
plt.xlabel('X4 - Team Attacking Average Caps')
plt.ylabel('Goals Scored')

plt.subplot(1, 2, 2)
plt.scatter(df['LOG_X4'], df['goals_scored'], color='red')
plt.title('Log Transformed X4')
plt.xlabel('LOG_X4')
plt.ylabel('Goals Scored')

plt.tight_layout()
plt.savefig('X4_Before_After_Log_Transformation.png', dpi=300)
plt.show()


"""Re-run the linear regression model with the transformed variable"""

# Replace X4 with LOG_X4 and keep goals_scored as the last column
df = df[['x1_team_goals_scored',
         'x2_opponent_goals_conceded',
         'x3_team_forwards_goals_per_cap',
         'LOG_X4',
         'x5_team_forwards_npg_per90',
         'x6_team_midfielders_assists_per90',
         'x7_team_big5_players',
         'x8_opponent_elo',
         'goals_scored']]

x = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Build and evaluate the linear regression model (with LOG_X4)
x = sm.add_constant(x)
model = sm.OLS(y, x).fit()
pred = model.predict(x)
model_details = model.summary()
model_details.extra_txt = None
print(model_details)