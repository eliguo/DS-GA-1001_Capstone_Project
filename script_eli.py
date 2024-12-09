import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import t, ttest_ind, levene

# set RNG
rng = np.random.default_rng(14420733)

# set significance level
ALPHA = 0.005

# Question 5
rmf = pd.read_csv('rmpCapstoneAdjusted_69706.csv')

# extract average difficulty for male and female professors
diff_male = rmf[(rmf['Male gender'] == 1) & (rmf["Female"] == 0)]['Average Difficulty (Adjusted)']
diff_female = rmf[(rmf['Female'] == 1) & (rmf["Male gender"] == 0)]['Average Difficulty (Adjusted)']

# calculate and print sample sizes
n_male = len(diff_male)
n_female = len(diff_female)

print(f"Number of male professors: {n_male}")
print(f"Number of female professors: {n_female}")

# prepare data for visualization
df_plot = pd.DataFrame({
    'Average Difficulty (Adjusted)': pd.concat([diff_male, diff_female], ignore_index=True),
    'Gender': ['Male'] * n_male + ['Female'] * n_female
})

# visualize the distribution of average difficulty by gender
sns.displot(data=df_plot, x='Average Difficulty (Adjusted)', hue='Gender', kind='kde', fill=True, height=6, aspect=1.5)
plt.title('Distribution of Average Difficulty (Adjusted) by Gender')
plt.xlabel('Average Difficulty (Adjusted)')
plt.ylabel('Density')
plt.savefig(os.path.join('fig', 'avg_diff_adj_dist.png'), bbox_inches='tight')
plt.show()

# calculate and print variances
var_male = diff_male.var(ddof=1)
var_female = diff_female.var(ddof=1)

print(f"Variance in Average Difficulty (Adjusted) of male professors: {var_male:.4f}")
print(f"Variance in Average Difficulty (Adjusted) of female professors: {var_female:.4f}")

# perform Levene's test to check for equal variances
levene_test = levene(diff_male, diff_female)
print(f"Levene's Test:\n  Statistic = {levene_test.statistic:.4f}\n  P-Value = {levene_test.pvalue:.4e}")
if levene_test.pvalue > ALPHA:
    print("  Result: Assume equal variances (p > 0.005).\n")
else:
    print("  Result: Variances are significantly different (p <= 0.005).\n")

# perform Welch's t-test
welch_t_test = ttest_ind(diff_male, diff_female, equal_var=False)
print(f"Welch's t-Test:\n  Statistic = {welch_t_test.statistic:.4f}\n  P-Value = {welch_t_test.pvalue:.4e}")
if welch_t_test.pvalue <= ALPHA:
    print("  Result: The difference in Average Difficulty between genders is statistically significant (p <= 0.005).")
else:
    print("  Result: The difference in Average Difficulty between genders is not statistically significant (p > 0.005).")

# Question 6

# calculate means
mean_male = diff_male.mean()
mean_female = diff_female.mean()

# calculate t-distribution confidence interval
se_diff = ((var_male / n_male) + (var_female / n_female)) ** 0.5
dof_welch = ((var_male / n_male + var_female / n_female) ** 2) / \
            (((var_male / n_male) ** 2) / (n_male - 1) + ((var_female / n_female) ** 2) / (n_female - 1))

mean_diff = mean_male - mean_female
t_critical = t.ppf(1 - 0.05 / 2, dof_welch)

ci_lower_t = mean_diff - t_critical * se_diff
ci_upper_t = mean_diff + t_critical * se_diff

print(f"Mean Difficulty Difference (Male - Female): {mean_diff:.4f}")
print(f"95% CI for Mean Difficulty Difference (t-distribution): ({ci_lower_t:.4f}, {ci_upper_t:.4f})")

# bootstrap confidence interval
boot_mean_diff = []
for _ in range(5000):
    boot_male_ind = rng.integers(low=0, high=n_male, size=n_male)
    boot_female_ind = rng.integers(low=0, high=n_female, size=n_female)

    boot_male_samp = diff_male.iloc[boot_male_ind].values
    boot_female_samp = diff_female.iloc[boot_female_ind].values

    each_mean_diff = boot_male_samp.mean() - boot_female_samp.mean()
    boot_mean_diff.append(each_mean_diff)

ci_mean_diff_bootstrap = np.percentile(boot_mean_diff, [2.5, 97.5])

print(f"Bootstrapped Mean Difficulty Difference (Male - Female): {np.mean(boot_mean_diff):.4f}")
print(f"95% CI for Mean Difficulty Difference (Bootstrap): ({ci_mean_diff_bootstrap[0]:.4f}, {ci_mean_diff_bootstrap[1]:.4f})")

# combined visualization for t-distribution and bootstrap
plt.figure(figsize=(10, 6))

# bootstrap visualization
sns.histplot(boot_mean_diff, kde=True, bins=30, color='blue', alpha=0.5, label='Bootstrap Distribution')
plt.axvline(ci_mean_diff_bootstrap[0], color='purple', linestyle='--', label=f'Bootstrap Lower CI: {ci_mean_diff_bootstrap[0]:.4f}')
plt.axvline(ci_mean_diff_bootstrap[1], color='orange', linestyle='--', label=f'Bootstrap Upper CI: {ci_mean_diff_bootstrap[1]:.4f}')
plt.axvline(np.mean(boot_mean_diff), color='black', linestyle='-', label=f'Bootstrap Mean: {np.mean(boot_mean_diff):.4f}')

# t-distribution visualization
plt.axvline(ci_lower_t, color='red', linestyle='--', label=f'T-dist Lower CI: {ci_lower_t:.4f}')
plt.axvline(ci_upper_t, color='green', linestyle='--', label=f'T-dist Upper CI: {ci_upper_t:.4f}')
plt.axvline(mean_diff, color='black', linestyle='-', label=f'T-dist Mean: {mean_diff:.4f}')

plt.xlim(-0.01, 0.01)  # adjust x-axis limits for clarity
plt.title('Confidence Intervals: T-Distribution vs Bootstrap')
plt.xlabel('Mean Difference (Male - Female)')
plt.ylabel('Density / Value')
plt.legend()
plt.savefig(os.path.join('fig', 'combined_ci_visualization.png'))
plt.show()

# Extra Credit
import geopandas as gpd
from mpl_toolkits.axes_grid1 import make_axes_locatable

# group data by us state and calculate the average rating
state_avg_rating = rmf.groupby("US State")["Average Rating (Adjusted)"].mean().reset_index()
state_avg_rating.columns = ["State", "Average Rating (Adjusted)"]

# define valid u.s. state abbreviations
us_state_abbreviations = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", 
    "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", 
    "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", 
    "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"
]

# filter the dataset to include only valid u.s. state abbreviations
state_avg_rating = state_avg_rating[state_avg_rating["State"].isin(us_state_abbreviations)]

# load the geojson file
us_states = gpd.read_file('us-states.json')

# rename the column to match the dataframe for merging
us_states = us_states.rename(columns={"id": "State"})

# merge the geojson data with the average rating data
us_states = us_states.merge(state_avg_rating, on="State", how="left")

# create the plot
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# adjust color bar height
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=-1)

# plot the map
us_states.plot(column="Average Rating (Adjusted)", cmap="viridis", legend=False, edgecolor="black", ax=ax)

# create a color bar and adjust its height
sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=us_states["Average Rating (Adjusted)"].min(), vmax=us_states["Average Rating (Adjusted)"].max()))
sm._A = []
cbar = fig.colorbar(sm, cax=cax)
cbar.ax.tick_params(labelsize=10)

# set map limits to show continental us and alaska
ax.set_xlim([-180, -50])  # longitude range
ax.set_ylim([20, 75])     # latitude range

# add title and layout
ax.set_title("Average Professor Ratings (Adjusted) by US State", fontsize=16, pad=20)
plt.tight_layout()

plt.savefig(os.path.join('fig', "us_state_avg_rating_map.png"))
plt.show()

import json
from scipy.stats import f_oneway

# # define state abbreviations to region mapping
# state_to_region_mapping = {
#     # northeast
#     "ME": "Northeast", "NH": "Northeast", "VT": "Northeast", "MA": "Northeast",
#     "RI": "Northeast", "CT": "Northeast", "NY": "Northeast", "NJ": "Northeast", "PA": "Northeast",
#     # midwest
#     "OH": "Midwest", "IN": "Midwest", "IL": "Midwest", "MI": "Midwest", "WI": "Midwest",
#     "MN": "Midwest", "IA": "Midwest", "MO": "Midwest", "ND": "Midwest", "SD": "Midwest",
#     "NE": "Midwest", "KS": "Midwest",
#     # south
#     "DE": "South", "MD": "South", "DC": "South", "VA": "South", "WV": "South",
#     "NC": "South", "SC": "South", "GA": "South", "FL": "South", "KY": "South",
#     "TN": "South", "AL": "South", "MS": "South", "AR": "South", "LA": "South",
#     "OK": "South", "TX": "South",
#     # west
#     "MT": "West", "ID": "West", "WY": "West", "CO": "West", "NM": "West", "AZ": "West",
#     "UT": "West", "NV": "West", "WA": "West", "OR": "West", "CA": "West",
#     "AK": "West", "HI": "West"
# }

# # save the mapping to a json file
# with open("state_to_region_mapping.json", "w") as file:
#     json.dump(state_to_region_mapping, file)

# load the mapping from the json file
with open("state_to_region_mapping.json", "r") as file:
    state_to_region_mapping = json.load(file)

# map the region to the dataframe
state_avg_rating["Region"] = state_avg_rating["State"].map(state_to_region_mapping)

# perform one-way ANOVA test
regions = state_avg_rating["Region"].unique()
data_by_region = [state_avg_rating[state_avg_rating["Region"] == region]["Average Rating (Adjusted)"] for region in regions]

anova_result = f_oneway(*data_by_region)

# print ANOVA result
print(f"ANOVA Result: F-statistic = {anova_result.statistic:.4f}, p-value = {anova_result.pvalue:.4e}")
if anova_result.pvalue <= ALPHA:
    print(f"result: the average rating is significantly different across regions (p <= {ALPHA}).")
else:
    print(f"result: the average rating is not significantly different across regions (p > {ALPHA}).")

# visualize with boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x="Region", y="Average Rating (Adjusted)", data=state_avg_rating)
plt.title("Average Rating (Adjusted) by Region", fontsize=16)
plt.xlabel("Region", fontsize=14)
plt.ylabel("Average Rating (Adjusted)", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig(os.path.join('fig', 'avg_rating_by_region_boxplot.png'))
plt.show()
