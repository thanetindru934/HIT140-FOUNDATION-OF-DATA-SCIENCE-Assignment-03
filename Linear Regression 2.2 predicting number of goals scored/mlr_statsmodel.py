
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
         'goals_scored']].copy()


"""MODEL 1: ORIGINAL LINEAR REGRESSION"""

x = df.drop(columns=['goals_scored'])
y = df['goals_scored']

x = sm.add_constant(x)
model_original = sm.OLS(y, x).fit()

print("\n========== ORIGINAL MODEL (8 VARIABLES) ==========")
print(model_original.summary())


"""LOG TRANSFORMATION OF X4"""

# Apply logarithmic transformation to X4
df['LOG_X4'] = np.log(df['x4_team_attacking_avg_caps'])

# Visualise before and after transformation
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.scatter(
    df['x4_team_attacking_avg_caps'],
    df['goals_scored'],
    color='green'
)
plt.title('Original X4')
plt.xlabel('X4 - Team Attacking Average Caps')
plt.ylabel('Goals Scored')

plt.subplot(1, 2, 2)
plt.scatter(
    df['LOG_X4'],
    df['goals_scored'],
    color='red'
)
plt.title('Log Transformed X4')
plt.xlabel('LOG_X4')
plt.ylabel('Goals Scored')

plt.tight_layout()
plt.savefig('X4_Before_After_Log_Transformation.png', dpi=300)
plt.show()


"""MODEL 2: REGRESSION WITH LOG_X4"""

# Replace original X4 with LOG_X4
df_log = df.drop(columns=['x4_team_attacking_avg_caps']).copy()

x = df_log.drop(columns=['goals_scored'])
y = df_log['goals_scored']

x = sm.add_constant(x)
model_log = sm.OLS(y, x).fit()

print("\n========== MODEL WITH LOG_X4 (8 VARIABLES) ==========")
print(model_log.summary())


"""MODEL 3: DROP X5 AND KEEP X3"""

df_drop_x5 = df_log.drop(
    columns=['x5_team_forwards_npg_per90']
)

x = df_drop_x5.drop(columns=['goals_scored'])
y = df_drop_x5['goals_scored']

x = sm.add_constant(x)
model_drop_x5 = sm.OLS(y, x).fit()

print("\n========== DROP X5 (KEEP X3) ==========")
print(model_drop_x5.summary())


"""MODEL 4: DROP X3 AND KEEP X5"""

df_drop_x3 = df_log.drop(
    columns=['x3_team_forwards_goals_per_cap']
)

x = df_drop_x3.drop(columns=['goals_scored'])
y = df_drop_x3['goals_scored']

x = sm.add_constant(x)
model_drop_x3 = sm.OLS(y, x).fit()

print("\n========== DROP X3 (KEEP X5) ==========")
print(model_drop_x3.summary())


"""COMPARE ALL FOUR MODELS"""

results = pd.DataFrame({
    'Model': [
        'Original (8 variables)',
        'LOG_X4 (8 variables)',
        'Drop X5 (Keep X3)',
        'Drop X3 (Keep X5)'
    ],
    'R2': [
        model_original.rsquared,
        model_log.rsquared,
        model_drop_x5.rsquared,
        model_drop_x3.rsquared
    ],
    'Adjusted R2': [
        model_original.rsquared_adj,
        model_log.rsquared_adj,
        model_drop_x5.rsquared_adj,
        model_drop_x3.rsquared_adj
    ],
    'AIC': [
        model_original.aic,
        model_log.aic,
        model_drop_x5.aic,
        model_drop_x3.aic
    ],
    'BIC': [
        model_original.bic,
        model_log.bic,
        model_drop_x5.bic,
        model_drop_x3.bic
    ]
})

print("\n========== MODEL COMPARISON ==========")
print(results.round(4).to_string(index=False))
