import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer

# Read dataset into a DataFrame
df = pd.read_csv("dataset_2_2_final.csv")

# Apply log transformation to X4
df["LOG_X4"] = np.log(df["x4_team_attacking_avg_caps"])

# All variables to check
x = df[['x1_team_goals_scored', 'x2_opponent_goals_conceded', 'x3_team_forwards_goals_per_cap',
        'x4_team_attacking_avg_caps', 'LOG_X4', 'x5_team_forwards_npg_per90',
        'x6_team_midfielders_assists_per90', 'x7_team_big5_players', 'x8_opponent_elo']]

# Apply Yeo-Johnson transformation to all variables
yj_x = pd.DataFrame(PowerTransformer().fit_transform(x), columns=x.columns)

# Draw before and after histograms (3 variables per window, 3 windows)
for start in [0, 3, 6]:
    plt.figure(figsize=(14, 10))
    for i, col in enumerate(x.columns[start:start+3]):

        plt.subplot(3, 2, i*2 + 1)
        plt.hist(x[col], bins=12, density=True)
        plt.title(f"{col} (original) skew = {x[col].skew():.2f}")

        plt.subplot(3, 2, i*2 + 2)
        plt.hist(yj_x[col], bins=12, density=True)
        plt.title(f"{col} (Yeo-Johnson) skew = {yj_x[col].skew():.2f}")

    plt.tight_layout()
    plt.show()