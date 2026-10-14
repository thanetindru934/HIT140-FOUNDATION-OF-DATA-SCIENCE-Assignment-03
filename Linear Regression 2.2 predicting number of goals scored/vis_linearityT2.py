
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Read dataset into a DataFrame
BASE = Path(__file__).resolve().parent
df = pd.read_csv(BASE / "dataset_2_2_final.csv", encoding="utf-8-sig")

# Create folder to save scatterplots
folder = BASE / "Scatterplots"
folder.mkdir(parents=True, exist_ok=True)

""" FIGURE 1 - X1 TO X4 """

fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(14, 9))

ax1.scatter(x=df['x1_team_goals_scored'], y=df['goals_scored'])
ax1.set_xlabel("X1 - Team Goals Scored")
ax1.set_ylabel("Goals Scored (Y)")

ax2.scatter(x=df['x2_opponent_goals_conceded'], y=df['goals_scored'])
ax2.set_xlabel("X2 - Opponent Goals Conceded")
ax2.set_ylabel("Goals Scored (Y)")

ax3.scatter(x=df['x3_team_forwards_goals_per_cap'], y=df['goals_scored'])
ax3.set_xlabel("X3 - Team Forwards Goals Per Cap")
ax3.set_ylabel("Goals Scored (Y)")

ax4.scatter(x=df['x4_team_attacking_avg_caps'], y=df['goals_scored'])
ax4.set_xlabel("X4 - Team Attacking Average Caps")
ax4.set_ylabel("Goals Scored (Y)")

fig1.tight_layout()
fig1.savefig(folder / "Figure_1_X1_to_X4.png", dpi=300)


""" FIGURE 2 - X5 TO X8 """

fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(14, 9))

ax1.scatter(x=df['x5_team_forwards_npg_per90'], y=df['goals_scored'])
ax1.set_xlabel("X5 - Team Forwards Non-Penalty Goals Per 90")
ax1.set_ylabel("Goals Scored (Y)")

ax2.scatter(x=df['x6_team_midfielders_assists_per90'], y=df['goals_scored'])
ax2.set_xlabel("X6 - Team Midfielders Assists Per 90")
ax2.set_ylabel("Goals Scored (Y)")

ax3.scatter(x=df['x7_team_big5_players'], y=df['goals_scored'])
ax3.set_xlabel("X7 - Team Big Five League Players")
ax3.set_ylabel("Goals Scored (Y)")

ax4.scatter(x=df['x8_opponent_elo'], y=df['goals_scored'])
ax4.set_xlabel("X8 - Opponent Elo")
ax4.set_ylabel("Goals Scored (Y)")

fig2.tight_layout()
fig2.savefig(folder / "Figure_2_X5_to_X8.png", dpi=300)

plt.show()