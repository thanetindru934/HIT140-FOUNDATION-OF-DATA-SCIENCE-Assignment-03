
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Read dataset into a DataFrame
BASE = Path(__file__).resolve().parent
df = pd.read_csv(BASE / "dataset_2_1_final.csv", encoding="utf-8-sig")

# Create folder to save scatterplots
folder = BASE / "Scatterplots"
folder.mkdir(parents=True, exist_ok=True)

""" FIGURE 1 - X1 TO X4 """

fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(14, 9))

ax1.scatter(x=df['x1_goals_scored_diff'], y=df['goal_difference'])
ax1.set_xlabel("X1 - Goals Scored Difference")
ax1.set_ylabel("Goal Difference (Y)")

ax2.scatter(x=df['x2_goals_conceded_diff'], y=df['goal_difference'])
ax2.set_xlabel("X2 - Goals Conceded Difference")
ax2.set_ylabel("Goal Difference (Y)")

ax3.scatter(x=df['x3_forwards_goals_per_cap_diff'], y=df['goal_difference'])
ax3.set_xlabel("X3 - Forwards Goals Per Cap Difference")
ax3.set_ylabel("Goal Difference (Y)")

ax4.scatter(x=df['x4_attacking_avg_caps_diff'], y=df['goal_difference'])
ax4.set_xlabel("X4 - Attacking Average Caps Difference")
ax4.set_ylabel("Goal Difference (Y)")

fig1.tight_layout()
fig1.savefig(folder / "Figure_1_X1_to_X4.png", dpi=300)


""" FIGURE 2 - X5 TO X8 """

fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(14, 9))

ax1.scatter(x=df['x5_big5_players_diff'], y=df['goal_difference'])
ax1.set_xlabel("X5 - Big Five League Players Difference")
ax1.set_ylabel("Goal Difference (Y)")

ax2.scatter(x=df['x6_fifa_points_diff'], y=df['goal_difference'])
ax2.set_xlabel("X6 - FIFA Points Difference")
ax2.set_ylabel("Goal Difference (Y)")

ax3.scatter(x=df['x7_previous_world_cups_diff'], y=df['goal_difference'])
ax3.set_xlabel("X7 - Previous World Cups Difference")
ax3.set_ylabel("Goal Difference (Y)")

ax4.scatter(x=df['x8_host_nation'], y=df['goal_difference'])
ax4.set_xlabel("X8 - Host Nation")
ax4.set_ylabel("Goal Difference (Y)")

fig2.tight_layout()
fig2.savefig(folder / "Figure_2_X5_to_X8.png", dpi=300)

plt.show()
