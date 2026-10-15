import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer

# Read dataset into a DataFrame
df = pd.read_csv("dataset_2_1_final.csv")

# Apply signed log transformation to X5
df["LOG_X5"] = np.sign(df["x5_big5_players_diff"]) * np.log1p(np.abs(df["x5_big5_players_diff"]))

# All variables to check
x = df[['x1_goals_scored_diff', 'x2_goals_conceded_diff', 'x3_forwards_goals_per_cap_diff',
        'x4_attacking_avg_caps_diff', 'x5_big5_players_diff', 'LOG_X5',
        'x6_fifa_points_diff', 'x7_previous_world_cups_diff', 'x8_host_nation']]

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
