import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.impute import KNNImputer
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    classification_report
)
from collections import Counter
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import t, ttest_ind, levene, mannwhitneyu
import geopandas as gpd
from mpl_toolkits.axes_grid1 import make_axes_locatable

"""
###########################################################################
Beginning of Data Preprocessing
###########################################################################
"""
# load datasets
rmp_num = pd.read_csv('rmpCapstoneNum.csv', header=None)
rmp_qual = pd.read_csv('rmpCapstoneQual.csv', header=None)
rmp_tags = pd.read_csv('rmpCapstoneTags.csv', header=None)

# assign column names
rmp_num.columns = [
    "Average Rating",
    "Average Difficulty",
    "Number of ratings",
    "Received a “pepper”?",
    "The proportion of students that said they would take the class again",
    "The number of ratings coming from online classes",
    "Male gender",
    "Female"
]

rmp_qual.columns = [
    "Major/Field",
    "University",
    "US State"
]

rmp_tags.columns = [
    "Tough grader",
    "Good feedback",
    "Respected",
    "Lots to read",
    "Participation matters",
    "Don’t skip class or you will not pass",
    "Lots of homework",
    "Inspirational",
    "Pop quizzes!",
    "Accessible",
    "So many papers",
    "Clear grading",
    "Hilarious",
    "Test heavy",
    "Graded by few things",
    "Amazing lectures",
    "Caring",
    "Extra credit",
    "Group projects",
    "Lecture heavy"
]

# merge and clean data
rmp = pd.concat([rmp_num, rmp_qual, rmp_tags], axis=1)
print(f"original length: {len(rmp)}")

rmp = rmp.dropna(subset=["Average Rating"]).drop_duplicates()
print(f"after dropping nan in Average Rating : {len(rmp)}")

tag_columns = rmp_tags.columns
rmp = rmp[rmp.loc[:, "Number of ratings"] >= rmp.loc[:, rmp_tags.columns].max(axis=1)]
print(f"then dropped bad tag data: {len(rmp)}")

# Bayesian adjustments for average rating and difficulty
# calculate prior means
prior_mean_rating = rmp["Average Rating"].mean()
prior_mean_difficulty = rmp["Average Difficulty"].mean()

# use median of 'Number of ratings' as strength of prior
strength_of_prior = rmp["Number of ratings"].median()

# apply Bayesian adjustment for ‘Average Rating’
rmp["Average Rating (Adjusted)"] = (
    (strength_of_prior * prior_mean_rating + 
     rmp["Number of ratings"] * rmp["Average Rating"]) /
    (strength_of_prior + rmp["Number of ratings"])
)

# apply Bayesian adjustment for ‘Average Difficulty’
rmp["Average Difficulty (Adjusted)"] = (
    (strength_of_prior * prior_mean_difficulty + 
     rmp["Number of ratings"] * rmp["Average Difficulty"]) /
    (strength_of_prior + rmp["Number of ratings"])
)

# normalize tag columns by 'Number of ratings'
tag_columns = rmp_tags.columns
for tag in tag_columns:
    rmp[f"{tag} (Normalized)"] = rmp[tag] / rmp["Number of ratings"]

# calculate total number of tags
rmp["Total Number of Tags"] = rmp[tag_columns].sum(axis=1)

# calculate average number of tags per rating
rmp["Average Tags per Rating"] = rmp["Total Number of Tags"] / rmp["Number of ratings"]

# reorder columns
column_order = [
    # numerical columns
    "Average Rating",
    "Average Rating (Adjusted)",
    "Average Difficulty",
    "Average Difficulty (Adjusted)",
    "Number of ratings",
    "Received a “pepper”?",
    "The proportion of students that said they would take the class again",
    "The number of ratings coming from online classes",
    "Male gender",
    "Female",
    
    # qualitative columns
    "Major/Field",
    "University",
    "US State",

    # tag columns
    *tag_columns,  # original tag columns
    *[f"{tag} (Normalized)" for tag in tag_columns],  # normalized tag columns
    "Total Number of Tags",
    "Average Tags per Rating"
]

# apply the column order
rmp = rmp[column_order]

# save the cleaned and adjusted data 
# The output file should be named as "rmpCapstoneAdjusted_69706.csv"
rmp.to_csv(f'rmpCapstoneAdjusted_{len(rmp)}.csv', index=False)

"""
###########################################################################
End of Data Preprocessing
###########################################################################
"""





# extract data
all_data = pd.read_csv("rmpCapstoneAdjusted_69706.csv")

"""
###########################################################################
Beginning of Question 1
###########################################################################
For Question 1, we firstly apply Levene's test to check whether the two datasets have equal variances, 
and then use the proper t-test to test whether there is a pro-male effect.
BE CAREFUL that we are testing a pro-male effect, so it should be ONE-SIDED test.

H_0: There is no difference between students' ratings for male and female professors.
H_1: Male professors get higher ratings than female professors
"""
# We are choosing Tim Zhou's N number to be the seed.
rng = np.random.default_rng(14420733)

ALPHA = 0.005
# extract average difficulty for male and female professors
rating_male = all_data.query("`Male gender` == 1 & `Female` == 0")["Average Rating (Adjusted)"]
rating_female = all_data.query("`Male gender` == 0 & `Female` == 1")["Average Rating (Adjusted)"]

# We must reindex here. It is crucial for bootstrapping in Question 3
rating_male = rating_male.reset_index(drop=True)
rating_female = rating_female.reset_index(drop=True)

# calculate and print sample sizes
n_male, n_female = len(rating_male), len(rating_female)
print(f"Number of male professors: {n_male}")
print(f"Number of female professors: {n_female}")

# calculate and print variances
var_male, var_female  = rating_male.var(ddof=1), rating_female.var(ddof=1)
print(f"Variance in Average Difficulty (Adjusted) of male professors: {var_male:.4f}")
print(f"Variance in Average Difficulty (Adjusted) of female professors: {var_female:.4f}")

# perform Levene's test to check for equal variances, and then decide the proper test.
# BE CAREFUL that we are testing a pro-male effect, so it should be ONE-SIDED test.
levene_test = levene(rating_male, rating_female)
print(f"Levene's Test:\n  Statistic = {levene_test.statistic:.4f}\n  P-Value = {levene_test.pvalue:.4e}")
if levene_test.pvalue > ALPHA:
    print("We can assume equal vairance --> We shall use independent samples t-test\n")
    inde_t_test = ttest_ind(rating_male, rating_female, equal_var=True, alternative="greater")
    stats, pval = inde_t_test.statistic, inde_t_test.pvalue
    print(f"Independent Samples t-test:\n  Statistic = {stats:.4f}\n  P-Value = {pval:.4e}")
else:
    print("We cannot assume equal vairance --> We shall use Welch t-test\n")
    welch_t_test = ttest_ind(rating_male, rating_female, equal_var=False, alternative="greater")
    stats, pval = welch_t_test.statistic, welch_t_test.pvalue
    print(f"Welch t-Test:\n  Statistic = {welch_t_test.statistic:.4f}\n  P-Value = {welch_t_test.pvalue:.4e}")

whether_significant = 'significant!' if pval < ALPHA else 'not significant!'
print(f"The p-value for the one-sided t-test is {pval:.4e}, which means that the result is {whether_significant}")


# A plot that visualizes their distributions
bins = np.arange(0.75, 5.25, 0.1)
plt.hist(rating_male, bins=bins, label='male', alpha=0.5, color="blue", edgecolor='black')
plt.hist(rating_female, bins=bins, label='female', alpha=0.5, color="red", edgecolor='black')
plt.xlabel('Average Ratings')
plt.ylabel('Frequency')
plt.title('Comparison between the ratings of male / female professors')
plt.legend()
plt.show()
"""
The result for Question 1 is:
Levene's test indicates a statistically significant difference in the variance between the ratings for male and female
Thus, we decided to use Welch's t-test. The p-value is 8.8601e-13 < 0.005.
Hence, we reject the null hypothesis
###########################################################################
End of Question 1
###########################################################################
"""



"""
###########################################################################
Beginning of Question 2
###########################################################################
For Quesion 2, we use Levene's test to check whether there is a gender difference in spread

H_0: There is no difference between the variance of students' ratings for male and female professors.
H_1: There is a difference between the variance of students' ratings for male and female professors.
"""
# calculate and print variances
var_male, var_female  = rating_male.var(ddof=1), rating_female.var(ddof=1)
print(f"Variance in Average Difficulty (Adjusted) of male professors: {var_male:.4f}")
print(f"Variance in Average Difficulty (Adjusted) of female professors: {var_female:.4f}")

# perform Levene's test to check for equal variances, and then decide the proper test.
levene_test = levene(rating_male, rating_female)
print(f"Levene's Test:\n  Statistic = {levene_test.statistic:.4f}\n  P-Value = {levene_test.pvalue:.4e}")
if levene_test.pvalue > ALPHA:
    print("There is not a gender difference in the spread \n")
else:
    print("There is a gender difference in the spread \n")

# Visualization: 
plt.boxplot([rating_male, rating_female], 
            labels=["male professors", "female professors"])
plt.title("comparison between the average ratings for male and female professors")
plt.ylabel("Average ratings")
plt.legend()
plt.show()
"""
The result for Question 2 is:
Levene's test gives a p-value to be 1.1839e-06 < 0.005.
Hence, we reject the null hypothesis
###########################################################################
End of Question 2
###########################################################################
"""



"""
###########################################################################
Beginning of Question 3
###########################################################################
For Question 3, we use boostrap to determine the confidence interval of the effect size of mean and variance
"""
# Double check the RNG
rng = np.random.default_rng(14420733)
num_samples = 10000
n_exper = 5000
boot_male_mean, boot_female_mean, boot_male_var, boot_female_var = [], [], [], []
boot_cohen_d, boot_var_ratio = [], []

for i in range(n_exper):
    # Generate my bootstrap samples
    boot_male_ind = rng.integers(low=0, high=len(rating_male), size=num_samples)
    boot_female_ind = rng.integers(low=0, high=len(rating_female), size=num_samples)

    boot_male_samp, boot_female_samp = rating_male[boot_male_ind], rating_female[boot_female_ind]

    # Calculate and append bootstrapped means and variances
    each_mean_male, each_mean_female = boot_male_samp.mean(), boot_female_samp.mean()
    each_var_male, each_var_female = boot_male_samp.var(ddof=1), boot_female_samp.var(ddof=1)

    boot_male_mean.append(each_mean_male)
    boot_male_var.append(each_var_male)
    boot_female_mean.append(each_mean_female)
    boot_female_var.append(each_var_female)

    # Calculate Cohen's d and variance ratio
    # I am doing this inside the loop because the values are stored in lists rather than ndarray
    each_mean_diff = each_mean_male - each_mean_female
    boot_cohen_d.append(each_mean_diff / np.sqrt((each_var_male + each_var_female) / 2) )
    boot_var_ratio.append(each_var_male / each_var_female)

# Calculate 95% confidence intervals for Cohen's d and variance ratio
ci_cohen_d = np.percentile(boot_cohen_d, [2.5, 97.5])
ci_var_ratio = np.percentile(boot_var_ratio, [2.5, 97.5])

print(f"Bootstrapped Cohen's d: {np.mean(boot_cohen_d):.4f}")
print(f"95% CI for Cohen's d: {ci_cohen_d}")
print(f"Bootstrapped Variance Ratio: {np.mean(ci_var_ratio):.4f}")
print(f"95% CI for Variance Ratio: {ci_var_ratio}")


# Visualizations:
# Since the plots are histograms, it is better to show them separately rather than in one row
plt.figure(figsize=(10, 6))
plt.hist(boot_cohen_d, bins=50, color='yellow', edgecolor='black')

# plot the confidence interval
plt.axvline(np.mean(boot_cohen_d), color='red', linestyle='--', label=f"Mean Cohen's d")
plt.axvline(ci_cohen_d[0], color='green', linestyle='--', label=f"left bound for 95% CI")
plt.axvline(ci_cohen_d[1], color='green', linestyle='--', label=f"right bound for 95% CI")
plt.title("Bootstrapped Cohen's d with 95% CI")
plt.xlabel("Cohen's d")
plt.ylabel("Frequency")
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(boot_var_ratio, bins=50, color='orange', edgecolor='black')

# plot the confidence interval
plt.axvline(np.mean(boot_var_ratio), color='blue', linestyle='--', label=f"Mean Variance Ratio")
plt.axvline(ci_var_ratio[0], color='gray', linestyle='--', label=f"left bound for 95% CI")
plt.axvline(ci_var_ratio[1], color='gray', linestyle='--', label=f"right bound for 95% CI")
plt.title("Bootstrapped Variance Ratio with 95% CI")
plt.xlabel("Variance Ratio")
plt.ylabel("Frequency")
plt.legend()
plt.show()
"""
The result for Question 3 is:

Bootstrapped Cohen's d: 0.0633
95% CI for Cohen's d: [0.03573299 0.09076494]
Bootstrapped Variance Ratio: 0.9519
95% CI for Variance Ratio: [0.91048425 0.99339719]
###########################################################################
End of Question 3
###########################################################################
"""




"""
###########################################################################
Beginning of Question 4
###########################################################################
For Question 4, we use Mann-Whitney U test to determine whether each tag is gendered.

In this question, the normalized number of tags sometimes appear to be 1.0 (100%) for those who received few ratings (1 or 2 ratings).
Given that the normalized tag columns only have values between 0 and 1, those extreme values are not representative.
Therefore, to reduce these extreme values caused few number of ratings, we choose to only consider those who received 3 or more ratings.
After this filtering, we still have around 40,000 rows, which is still large.
"""
# First extract columns that are tages
normalized_tag_columns = [
    'Tough grader (Normalized)', 'Good feedback (Normalized)', 'Respected (Normalized)',
    'Lots to read (Normalized)', 'Participation matters (Normalized)', 
    "Don’t skip class or you will not pass (Normalized)", 'Lots of homework (Normalized)',
    'Inspirational (Normalized)', 'Pop quizzes! (Normalized)', 'Accessible (Normalized)', 
    'So many papers (Normalized)', 'Clear grading (Normalized)', 'Hilarious (Normalized)',
    'Test heavy (Normalized)', 'Graded by few things (Normalized)', 'Amazing lectures (Normalized)',
    'Caring (Normalized)', 'Extra credit (Normalized)', 'Group projects (Normalized)',
    'Lecture heavy (Normalized)'
]

# This result list will be used to generate a dataframe to match tag and p-values
tag_results = []

for tag in normalized_tag_columns:
    # Split normalized tag values by gender
    # Select those who received 3 or more ratings
    tag_male = all_data.query("`Male gender` == 1 & `Female` == 0 & `Number of ratings` >= 3")
    tag_female = all_data.query("`Male gender` == 0 & `Female` == 1 & `Number of ratings` >= 3")

    # Sometimes there are unrealistic data, such as "received 17 ratings but received 100 tags"
    # Thus, drop those abnormal data. Then we can begin our test
    tag_male = tag_male.loc[tag_male[tag] <= 1.01, tag]
    tag_female = tag_female.loc[tag_female[tag] <= 1.01, tag]
    
    each_result = mannwhitneyu(tag_male, tag_female)
    
    tag_results.append({'Tag': tag, 'P-Value': each_result.pvalue})



# Convert results to DataFrame, such that tags are paired with p-values, then sort
results_df = pd.DataFrame(tag_results)
results_df = results_df.sort_values('P-Value')

# extract significantly gendered tags
gendered_tags = results_df[results_df["P-Value"] <= ALPHA]["Tag"].to_list()
gendered_tags = [each_tag.replace(" (Normalized)", "") for each_tag in gendered_tags]
print(f"The following {len(gendered_tags)} tags exhibit a statistically significant different in gender: {gendered_tags}")

most_sig_3, least_sig_3 = results_df["Tag"].to_list()[:3], results_df["Tag"].to_list()[-3:]
most_sig_3_pv, least_sig_3_pv = results_df["P-Value"].to_list()[:3], results_df["P-Value"].to_list()[-3:]
most_sig_3_original_word = [each_tag.replace(' (Normalized)', '') for each_tag in most_sig_3]
least_sig_3_original_word = [each_tag.replace(' (Normalized)', '') for each_tag in least_sig_3]
print(f"The most gendered 3 tags are {most_sig_3_original_word} with pvalues {most_sig_3_pv},\n\
The least gendered 3 tags are {least_sig_3_original_word} with pvalues {least_sig_3_pv}.")

# Visualization:
# For plotting, plotting all 20 tags are too many, so I just plot the 3 most and 3 least significant tags.
bins = np.arange(-0.05, 1.15, 0.1)
place = 1
plt.figure(figsize=(20, 4))
for most_tag in most_sig_3:
    # Still need to extract data again
    tag_male = all_data.query("`Male gender` == 1 & `Female` == 0 & `Number of ratings` >= 3")
    tag_female = all_data.query("`Male gender` == 0 & `Female` == 1 & `Number of ratings` >= 3")
    tag_male = tag_male.loc[tag_male[most_tag] <= 1.01, most_tag]
    tag_female = tag_female.loc[tag_female[most_tag] <= 1.01, most_tag]

    # determine subplot place
    plt.subplot(1, 3, place)
    plt.hist(tag_male, bins=bins, alpha=0.5, color="blue", edgecolor="black", label="Male")
    plt.hist(tag_female, bins=bins, alpha=0.5, color="red", edgecolor="black", label="Female")
    plt.title(f"Comparison on the tag \"{most_tag.replace(' (Normalized)', '')}\"")
    plt.ylabel("frequency")
    plt.xlabel("normalized value")
    plt.legend()
    place += 1
plt.show()

# For least significant 3:
place = 1
plt.figure(figsize=(20, 4))
for least_tag in least_sig_3:
    # Still need to extract data again
    tag_male = all_data.query("`Male gender` == 1 & `Female` == 0 & `Number of ratings` >= 3")
    tag_female = all_data.query("`Male gender` == 0 & `Female` == 1 & `Number of ratings` >= 3")
    tag_male = tag_male.loc[tag_male[least_tag] <= 1.01, least_tag]
    tag_female = tag_female.loc[tag_female[least_tag] <= 1.01, least_tag]

    # determine subplot place
    plt.subplot(1, 3, place)
    plt.hist(tag_male, bins=bins, alpha=0.5, color="blue", edgecolor="black", label="Male")
    plt.hist(tag_female, bins=bins, alpha=0.5, color="red", edgecolor="black", label="Female")
    plt.title(f"Comparison on the tag \"{least_tag.replace(' (Normalized)', '')}\"")
    plt.ylabel("frequency")
    plt.xlabel("normalized value")
    plt.legend()
    place += 1
plt.show()
"""
The result for Question 4 is:

The following 19 tags exhibit a statistically significant different in gender: 
['Hilarious', 'Amazing lectures', 'Lecture heavy', 'Caring', 'Respected', 'Participation matters', 
'Good feedback', 'Graded by few things', 'Group projects', 'Lots of homework', 'So many papers', 
'Extra credit', 'Test heavy', 'Lots to read', 'Inspirational', 'Accessible', 'Clear grading', 
'Tough grader', 'Don’t skip class or you will not pass']

The most gendered 3 tags are ['Hilarious', 'Amazing lectures', 'Lecture heavy'] 
with pvalues [1.225e-227, 1.482e-53, 3.384e-39];

The least gendered 3 tags are ['Tough grader', 'Don’t skip class or you will not pass', 'Pop quizzes!'] 
with pvalues [3.091e-05, 1.515e-4, 2.921e-2].
###########################################################################
End of Question 4
###########################################################################
"""



"""
###########################################################################
Beginning of Question 5
###########################################################################
"""
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
"""
###########################################################################
End of Question 5
###########################################################################
"""



"""
###########################################################################
Beginning of Question 6
###########################################################################
"""
# Question 6

# calculate means and pooled standard deviation
mean_male = diff_male.mean()
mean_female = diff_female.mean()

var_male = diff_male.var(ddof=1)
var_female = diff_female.var(ddof=1)
n_male = len(diff_male)
n_female = len(diff_female)

pooled_sd = np.sqrt(((n_male - 1) * var_male + (n_female - 1) * var_female) / (n_male + n_female - 2))
cohen_d = (mean_male - mean_female) / pooled_sd

# calculate standard error for Cohen's d
se_cohen_d = np.sqrt((1 / n_male) + (1 / n_female))

dof_welch = ((var_male / n_male + var_female / n_female) ** 2) / \
            (((var_male / n_male) ** 2) / (n_male - 1) + ((var_female / n_female) ** 2) / (n_female - 1))

# t critical value
t_critical = t.ppf(1 - 0.05 / 2, dof_welch)

# confidence interval for Cohen's d
ci_lower_d = cohen_d - t_critical * se_cohen_d
ci_upper_d = cohen_d + t_critical * se_cohen_d

print(f"Cohen's d: {cohen_d:.4f}")
print(f"95% CI for Cohen's d (t-distribution): ({ci_lower_d:.4f}, {ci_upper_d:.4f})")

# bootstrap confidence interval for Cohen's d
boot_cohen_d = []
for _ in range(5000):
    boot_male_ind = rng.integers(low=0, high=n_male, size=n_male)
    boot_female_ind = rng.integers(low=0, high=n_female, size=n_female)

    boot_male_samp = diff_male.iloc[boot_male_ind].values
    boot_female_samp = diff_female.iloc[boot_female_ind].values

    boot_mean_diff = boot_male_samp.mean() - boot_female_samp.mean()
    boot_pooled_sd = np.sqrt(((len(boot_male_samp) - 1) * boot_male_samp.var(ddof=1) + \
                              (len(boot_female_samp) - 1) * boot_female_samp.var(ddof=1)) / \
                             (len(boot_male_samp) + len(boot_female_samp) - 2))

    boot_cohen_d.append(boot_mean_diff / boot_pooled_sd)

ci_cohen_d_bootstrap = np.percentile(boot_cohen_d, [2.5, 97.5])

print(f"Bootstrapped Cohen's d: {np.mean(boot_cohen_d):.4f}")
print(f"95% CI for Cohen's d (Bootstrap): ({ci_cohen_d_bootstrap[0]:.4f}, {ci_cohen_d_bootstrap[1]:.4f})")

# combined visualization for t-distribution and bootstrap
plt.figure(figsize=(10, 6))

# bootstrap visualization
sns.histplot(boot_cohen_d, kde=True, bins=30, color='blue', alpha=0.5, label='Bootstrap Distribution')
plt.axvline(ci_cohen_d_bootstrap[0], color='purple', linestyle='--', label=f'Bootstrap Lower CI: {ci_cohen_d_bootstrap[0]:.4f}')
plt.axvline(ci_cohen_d_bootstrap[1], color='orange', linestyle='--', label=f'Bootstrap Upper CI: {ci_cohen_d_bootstrap[1]:.4f}')
plt.axvline(np.mean(boot_cohen_d), color='black', linestyle='-', label=f'Bootstrap Mean: {np.mean(boot_cohen_d):.4f}')

# t-distribution visualization
plt.axvline(ci_lower_d, color='red', linestyle='--', label=f'T-dist Lower CI: {ci_lower_d:.4f}')
plt.axvline(ci_upper_d, color='green', linestyle='--', label=f'T-dist Upper CI: {ci_upper_d:.4f}')
plt.axvline(cohen_d, color='black', linestyle='-', label=f'T-dist Mean: {cohen_d:.4f}')

plt.title("Confidence Intervals for Cohen's d: T-Distribution vs Bootstrap")
plt.xlabel("Cohen's d")
plt.ylabel("Density / Value")
plt.legend()
plt.savefig(os.path.join('fig', "cohens_d_confidence_intervals.png"))
plt.show()
"""
###########################################################################
End of Question 6
###########################################################################
"""



"""
###########################################################################
Data Preparation for Q7-Q10
########################################################################### 
"""
# Here we return preprocessinf code with an additional line: Dropped rows with missing values in "The proportion of students that said they would take the class again".
# This reduces the dataset size. However, after testing various fillna methods, we observed noticeably worse results compared to dropping rows.
# Therefore, for the regression task, we prioritize data quality over quantity, as the remaining dataset is still sufficiently large for analysis.

# # load datasets

rmp_num = pd.read_csv('rmpCapstoneNum.csv', header=None)
rmp_qual = pd.read_csv('rmpCapstoneQual.csv', header=None)
rmp_tags = pd.read_csv('rmpCapstoneTags.csv', header=None)

# assign column names
rmp_num.columns = [
    "Average Rating",
    "Average Difficulty",
    "Number of ratings",
    "Received a “pepper”?",
    "The proportion of students that said they would take the class again",
    "The number of ratings coming from online classes",
    "Male gender",
    "Female"
]

rmp_qual.columns = [
    "Major/Field",
    "University",
    "US State"
]

rmp_tags.columns = [
    "Tough grader",
    "Good feedback",
    "Respected",
    "Lots to read",
    "Participation matters",
    "Don’t skip class or you will not pass",
    "Lots of homework",
    "Inspirational",
    "Pop quizzes!",
    "Accessible",
    "So many papers",
    "Clear grading",
    "Hilarious",
    "Test heavy",
    "Graded by few things",
    "Amazing lectures",
    "Caring",
    "Extra credit",
    "Group projects",
    "Lecture heavy"
]

# merge and clean data
rmp = pd.concat([rmp_num, rmp_qual, rmp_tags], axis=1)
rmp = rmp.dropna(subset=["Average Rating"]).drop_duplicates()
rmp = rmp.dropna(subset=["The proportion of students that said they would take the class again"])

rmp = rmp[rmp.loc[:, "Number of ratings"] >= rmp.loc[:, rmp_tags.columns].max(axis=1)]


# Bayesian adjustments for average rating and difficulty
# calculate prior means
prior_mean_rating = rmp["Average Rating"].mean()
prior_mean_difficulty = rmp["Average Difficulty"].mean()

# use median of 'Number of ratings' as strength of prior
strength_of_prior = rmp["Number of ratings"].median()

# apply Bayesian adjustment for ‘Average Rating’
rmp["Average Rating (Adjusted)"] = (
    (strength_of_prior * prior_mean_rating + 
     rmp["Number of ratings"] * rmp["Average Rating"]) /
    (strength_of_prior + rmp["Number of ratings"])
)

# apply Bayesian adjustment for ‘Average Difficulty’
rmp["Average Difficulty (Adjusted)"] = (
    (strength_of_prior * prior_mean_difficulty + 
     rmp["Number of ratings"] * rmp["Average Difficulty"]) /
    (strength_of_prior + rmp["Number of ratings"])
)



# normalize tag columns by 'Number of ratings'
tag_columns = rmp_tags.columns
for tag in tag_columns:
    rmp[f"{tag} (Normalized)"] = rmp[tag] / rmp["Number of ratings"]

# calculate total number of tags
rmp["Total Number of Tags"] = rmp[tag_columns].sum(axis=1)

# calculate average number of tags per rating
rmp["Average Tags per Rating"] = rmp["Total Number of Tags"] / rmp["Number of ratings"]

# reorder columns
column_order = [
    # numerical columns
    "Average Rating",
    "Average Rating (Adjusted)",
    "Average Difficulty",
    "Average Difficulty (Adjusted)",
    "Number of ratings",
    "Received a “pepper”?",
    "The proportion of students that said they would take the class again",
    "The number of ratings coming from online classes",
    "Male gender",
    "Female",
    
    # qualitative columns
    "Major/Field",
    "University",
    "US State",

    # tag columns
    *tag_columns,  # original tag columns
    *[f"{tag} (Normalized)" for tag in tag_columns],  # normalized tag columns
    "Total Number of Tags",
    "Average Tags per Rating"
]

# apply the column order
rmp = rmp[column_order]

# save the cleaned and adjusted data
rmp.to_csv(f'rmpCapstoneAdjusted_regression.csv', index=False)


# ##### random number seed
rng = np.random.default_rng(14420733) # Uncomment if rng is not defined elsewhere
r_number = rng.integers(1, 100)


# ##### import adjusted data
df_cleaned = pd.read_csv('rmpCapstoneAdjusted_regression.csv')
"""
###########################################################################
End of Data Preparation for Q7-Q10
########################################################################### 
"""



"""
###########################################################################
Beginning of Question 7
###########################################################################
# ### Q7 - Build a regression model predicting average rating from all numerical predictors 
# (the ones in the rmpCapstoneNum.csv) file.Make sure to include the R2and RMSE of this model. Which of these factors is most strongly predictive of average rating? Hint: Make sure to address collinearity concern
"""
# Step 1: specify X and y
y =  df_cleaned['Average Rating (Adjusted)']
# we keep all predictors because it is required in the question
X = df_cleaned[['Average Difficulty (Adjusted)', 
       'Number of ratings',
       'The proportion of students that said they would take the class again',
       'The number of ratings coming from online classes', 
       'Received a “pepper”?',
       'Male gender',
       'Female'
      ]]

X_with_constant = X.copy()
X_with_constant.insert(0, 'Intercept', 1)

# Calculate VIF for each feature
vif_data = pd.DataFrame()
vif_data["Feature"] = X_with_constant.columns
vif_data["VIF"] = [variance_inflation_factor(X_with_constant.values, i) 
                   for i in range(X_with_constant.shape[1])]
print(vif_data)

# check correlation
correlation_matrix = X.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="RdBu_r", center=0, xticklabels=True, yticklabels=True)
plt.title('Correlation Matrix Heatmap')
plt.show()


# check missing values
missing_values = X.isna().sum()
print(missing_values)


# Step 2: Train-Test Split (80-20)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=r_number)

# Step 3: Standardize the features (scale to zero mean and unit variance) - only scale continous variable
X_train_continuous = X_train.iloc[:, :4]  
X_train_others = X_train.iloc[:, 4:]     

X_val_continuous = X_val.iloc[:, :4]     
X_val_others = X_val.iloc[:, 4:]        

scaler = StandardScaler()
X_train_continuous_scaled = scaler.fit_transform(X_train_continuous)
X_val_continuous_scaled = scaler.transform(X_val_continuous)

# Combine the scaled continuous columns with the rest (dummy variables)
X_train_scaled = np.hstack([X_train_continuous_scaled, X_train_others.values])
X_val_scaled = np.hstack([X_val_continuous_scaled, X_val_others.values])

# Step 4: Train a Ridge Regression model with different alpha values
# alphas = np.logspace(0, 3, 100)
alphas = np.arange(0, 100, 1)
ridge_train_rmse = []
ridge_val_rmse = []
ridge_train_r2 = []  
ridge_val_r2 = []    

for alpha in alphas:
    ridge_model = Ridge(alpha=alpha)  # Use Ridge Regression 
    ridge_model.fit(X_train_scaled, y_train)
    
    # Predict on training and validation data
    y_train_pred_ridge = ridge_model.predict(X_train_scaled)
    y_val_pred_ridge = ridge_model.predict(X_val_scaled)
    
    # Compute RMSE for training and validation
    ridge_train_rmse.append(np.sqrt(mean_squared_error(y_train, y_train_pred_ridge)))
    ridge_val_rmse.append(np.sqrt(mean_squared_error(y_val, y_val_pred_ridge)))
    ridge_train_r2.append(r2_score(y_train, y_train_pred_ridge))
    ridge_val_r2.append(r2_score(y_val, y_val_pred_ridge))

# Find the alpha with the lowest validation RMSE
best_alpha_ridge = alphas[np.argmin(ridge_val_rmse)]
print(f"Best Alpha (Ridge): {best_alpha_ridge}")
print(f"Lowest Validation RMSE: {min(ridge_val_rmse)}")

# Step 4: Plot Training and Validation RMSE
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(alphas, ridge_train_rmse, label='Training RMSE', marker='o')
plt.plot(alphas, ridge_val_rmse, label='Validation RMSE', marker='o')
plt.xlabel('Alpha (Regularization Strength)')
plt.ylabel('Root Mean Squared Error (RMSE)')
plt.title('Effect of Ridge Regularization on RMSE')
plt.xscale('log')
plt.legend()
plt.grid(True)

# Plot R²
plt.subplot(1, 2, 2)
plt.plot(alphas, ridge_val_r2, label='Validation R²', marker='o')
plt.xlabel('Alpha (Regularization Strength)')
plt.ylabel('R² (Coefficient of Determination)')
plt.title('Effect of Ridge Regularization on R²')
plt.xscale('log')

plt.legend()
plt.grid(True)
plt.show()




# Step 7: Visualize the betas (coefficients) from models
ridge_model = Ridge(alpha=best_alpha_ridge)
ridge_model.fit(X_train_scaled, y_train)
betas_ridge = ridge_model.coef_

# Plot the coefficients (betas) for all models
plt.figure(figsize=(18, 6))

feature_name = ['Difficulty', 
       'Number of ratings',
       'would take again',
       'ratings number from online', 
       'Received a “pepper”?',
       'Male gender',
       'Female'
      ]

# Ridge Regression Coefficients (for best alpha)
plt.subplot(1, 2, 1)
plt.bar(feature_name, betas_ridge)
plt.title(f'Coefficients Ridge (alpha={best_alpha_ridge})')
plt.xlabel('Feature')
plt.ylabel('Coefficient Value')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

final_train_r2 = r2_score(y_train, ridge_model.predict(X_train_scaled))
final_val_r2 = r2_score(y_val, ridge_model.predict(X_val_scaled))

print(f"Final Validation R²: {final_val_r2}")
final_val_rmse = np.sqrt(mean_squared_error(y_val, ridge_model.predict(X_val_scaled)))
print(f"Final Validation RMSE: {final_val_rmse}")

highest_beta_index = np.argmax(np.abs(betas_ridge))
highest_beta_feature = feature_name[highest_beta_index]
highest_beta_value = betas_ridge[highest_beta_index]

print(f"Feature with the highest coefficient (absolute value): {highest_beta_feature}")
print(f"Value of the highest coefficient: {highest_beta_value}")
"""
###########################################################################
End of Question 7
###########################################################################
"""



"""
###########################################################################
Beginning of Question 8
###########################################################################
# ### Q8 - Build a regression model predicting average ratings from all tags
# (the ones in the rmpCapstoneTags.csv) file. Make sure to include the R2and RMSE of this model. Which of these tags is most strongly predictive of average rating? Hint: Make sure to address collinearity concerns. Also comment on how this model compares to the previous one
"""
y2 =  df_cleaned['Average Rating (Adjusted)']
X2 = df_cleaned[[ 'Tough grader (Normalized)',
       'Good feedback (Normalized)', 'Respected (Normalized)',
       'Lots to read (Normalized)', 'Participation matters (Normalized)',
       'Don’t skip class or you will not pass (Normalized)',
       'Lots of homework (Normalized)', 'Inspirational (Normalized)',
       'Pop quizzes! (Normalized)', 'Accessible (Normalized)',
       'So many papers (Normalized)', 'Clear grading (Normalized)',
       'Hilarious (Normalized)', 'Test heavy (Normalized)',
       'Graded by few things (Normalized)', 'Amazing lectures (Normalized)',
       'Caring (Normalized)', 'Extra credit (Normalized)',
       'Group projects (Normalized)', 'Lecture heavy (Normalized)'
      ]]

X_with_constant = X2.copy()
X_with_constant.insert(0, 'Intercept', 1)

# Calculate VIF for each feature
vif_data = pd.DataFrame()
vif_data["Feature"] = X_with_constant.columns
vif_data["VIF"] = [variance_inflation_factor(X_with_constant.values, i) 
                   for i in range(X_with_constant.shape[1])]
print(vif_data)


#check correlation
correlation_matrix2 = X2.corr()

plt.figure(figsize=(40, 28))
sns.heatmap(correlation_matrix2, annot=True, cmap="RdBu_r", center=0, xticklabels=True, yticklabels=True)
plt.title('Correlation Matrix Heatmap')
plt.show()

# check missing values
missing_values2 = X2.isna().sum()
print(missing_values2)
len(X2)


# Step 2: Train-Test Split (80-20)
X2_train, X2_val, y2_train, y2_val = train_test_split(X2, y2, test_size=0.2, random_state=r_number)

# Step 3: Standardize the features (scale to zero mean and unit variance)  
scaler = StandardScaler()
X2_train_scaled = scaler.fit_transform(X2_train)
X2_val_scaled = scaler.transform(X2_val)

# Step 4: Train a Ridge Regression model with different alpha values
# alphas = np.logspace(0, 3, 100)
alphas = np.arange(0, 100, 1)
ridge_train_rmse_2 = []  # RMSE for training
ridge_val_rmse_2 = []    # RMSE for validation
ridge_train_r2_2 = []    # Store R² for training
ridge_val_r2_2 = []      # Store R² for validation

for alpha in alphas:
    ridge_model = Ridge(alpha=alpha)  # Use Ridge Regression instead of Lasso
    ridge_model.fit(X2_train_scaled, y2_train)
    
    # Predict on training and validation data
    y_train_pred_ridge_2 = ridge_model.predict(X2_train_scaled)
    y_val_pred_ridge_2 = ridge_model.predict(X2_val_scaled)
    
    # Compute RMSE for training and validation
    ridge_train_rmse_2.append(np.sqrt(mean_squared_error(y2_train, y_train_pred_ridge_2)))
    ridge_val_rmse_2.append(np.sqrt(mean_squared_error(y2_val, y_val_pred_ridge_2)))
    ridge_train_r2_2.append(r2_score(y2_train, y_train_pred_ridge_2))
    ridge_val_r2_2.append(r2_score(y2_val, y_val_pred_ridge_2))

# Find the alpha with the lowest validation RMSE
best_alpha_ridge_2 = alphas[np.argmin(ridge_val_rmse_2)]
print(f"Best Alpha (Ridge): {best_alpha_ridge_2}")
print(f"Lowest Validation RMSE: {min(ridge_val_rmse_2)}")

# Step 4: Plot Training and Validation RMSE
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(alphas, ridge_train_rmse_2, label='Training RMSE', marker='o')
plt.plot(alphas, ridge_val_rmse_2, label='Validation RMSE', marker='o')
plt.xlabel('Alpha (Regularization Strength)')
plt.ylabel('Root Mean Squared Error')
plt.xscale('log')
plt.title('Effect of Ridge Regularization on RMSE')
plt.legend()
plt.grid(True)

# Plot R²
plt.subplot(1, 2, 2)
plt.plot(alphas, ridge_val_r2_2, label='Validation R²', marker='o')
plt.xlabel('Alpha (Regularization Strength)')
plt.ylabel('R² (Coefficient of Determination)')
plt.title('Effect of Ridge Regularization on R²')
plt.xscale('log')

plt.legend()
plt.grid(True)
plt.show()


# Step 7: Visualize the betas (coefficients) 

ridge_model = Ridge(alpha=best_alpha_ridge_2)
ridge_model.fit(X2_train_scaled, y2_train)
betas_ridge = ridge_model.coef_

# Plot the coefficients (betas) for all models
plt.figure(figsize=(28, 8))

feature_name = [ 'Tough grader (Normalized)',
       'Good feedback (Normalized)', 'Respected (Normalized)',
       'Lots to read (Normalized)', 'Participation matters (Normalized)',
       'Don’t skip class or you will not pass (Normalized)',
       'Lots of homework (Normalized)', 'Inspirational (Normalized)',
       'Pop quizzes! (Normalized)', 'Accessible (Normalized)',
       'So many papers (Normalized)', 'Clear grading (Normalized)',
       'Hilarious (Normalized)', 'Test heavy (Normalized)',
       'Graded by few things (Normalized)', 'Amazing lectures (Normalized)',
       'Caring (Normalized)', 'Extra credit (Normalized)',
       'Group projects (Normalized)', 'Lecture heavy (Normalized)'
      ]

# Ridge Regression Coefficients (for best alpha)
plt.subplot(1, 2, 1)
plt.bar(feature_name, betas_ridge)
plt.title(f'Coefficients Ridge (alpha={best_alpha_ridge_2})')
plt.xlabel('Feature')
plt.ylabel('Coefficient Value')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

final_train_r2 = r2_score(y2_train, ridge_model.predict(X2_train_scaled))
final_val_r2 = r2_score(y2_val, ridge_model.predict(X2_val_scaled))

print(f"Final Validation R²: {final_val_r2}")

highest_beta_index = np.argmax(np.abs(betas_ridge))
highest_beta_feature = feature_name[highest_beta_index]
highest_beta_value = betas_ridge[highest_beta_index]

print(f"Feature with the highest coefficient (absolute value): {highest_beta_feature}")
print(f"Value of the highest coefficient: {highest_beta_value}")
"""
###########################################################################
End of Question 8
###########################################################################
"""




"""
###########################################################################
Beginning of Question 9
###########################################################################
# ### Q9 - Build a regression model predicting average difficulty from all tags
# (the ones in the rmpCapstoneTags.csv) file. Make sure to include the R2and RMSE of this model. Which of these tags is most strongly predictive of average rating? Hint: Make sure to address collinearity concerns. Also comment on how this model compares to the previous one
"""

y3 =  df_cleaned['Average Difficulty (Adjusted)']
X3 = df_cleaned[[ 'Tough grader (Normalized)',
       'Good feedback (Normalized)', 'Respected (Normalized)',
       'Lots to read (Normalized)', 'Participation matters (Normalized)',
       'Don’t skip class or you will not pass (Normalized)',
       'Lots of homework (Normalized)', 'Inspirational (Normalized)',
       'Pop quizzes! (Normalized)', 'Accessible (Normalized)',
       'So many papers (Normalized)', 'Clear grading (Normalized)',
       'Hilarious (Normalized)', 'Test heavy (Normalized)',
       'Graded by few things (Normalized)', 'Amazing lectures (Normalized)',
       'Caring (Normalized)', 'Extra credit (Normalized)',
       'Group projects (Normalized)', 'Lecture heavy (Normalized)'
      ]]

# check correlation
correlation_matrix3 = X3.corr()

plt.figure(figsize=(40, 28))
sns.heatmap(correlation_matrix3, annot=True, cmap="RdBu_r", center=0, xticklabels=True, yticklabels=True)
plt.title('Correlation Matrix Heatmap')
plt.show()


# Step 2: Train-Test Split (80-20)
X3_train, X3_val, y3_train, y3_val = train_test_split(X3, y3, test_size=0.2, random_state=r_number)

# Step 3: Standardize the features (scale to zero mean and unit variance)
scaler = StandardScaler()
X3_train_scaled = scaler.fit_transform(X3_train)
X3_val_scaled = scaler.transform(X3_val)

# Step 4: Train a Ridge Regression model with different alpha values
alphas = np.logspace(0, 3, 200)
ridge_train_rmse_3 = []  # RMSE for training
ridge_val_rmse_3 = []    # RMSE for validation
ridge_train_r2_3 = []    # Store R² for training
ridge_val_r2_3 = []      # Store R² for validation

for alpha in alphas:
    ridge_model = Ridge(alpha=alpha)  # Use Ridge Regression instead of Lasso
    ridge_model.fit(X3_train_scaled, y3_train)
    
    # Predict on training and validation data
    y_train_pred_ridge_3 = ridge_model.predict(X3_train_scaled)
    y_val_pred_ridge_3 = ridge_model.predict(X3_val_scaled)
    
    # Compute RMSE for training and validation
    ridge_train_rmse_3.append(np.sqrt(mean_squared_error(y3_train, y_train_pred_ridge_3)))
    ridge_val_rmse_3.append(np.sqrt(mean_squared_error(y3_val, y_val_pred_ridge_3)))
    ridge_train_r2_3.append(r2_score(y3_train, y_train_pred_ridge_3))
    ridge_val_r2_3.append(r2_score(y3_val, y_val_pred_ridge_3))

# Find the alpha with the lowest validation RMSE
best_alpha_ridge_3 = alphas[np.argmin(ridge_val_rmse_3)]
print(f"Best Alpha (Ridge): {best_alpha_ridge_3}")
print(f"Lowest Validation RMSE: {min(ridge_val_rmse_3)}")

# Step 4: Plot Training and Validation RMSE
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(alphas, ridge_train_rmse_3, label='Training RMSE', marker='o')
plt.plot(alphas, ridge_val_rmse_3, label='Validation RMSE', marker='o')
plt.xlabel('Alpha (Regularization Strength)')
plt.ylabel('Root Mean Squared Error')
plt.title('Effect of Ridge Regularization on RMSE')
plt.xscale('log')
plt.legend()
plt.grid(True)

# Plot R²
plt.subplot(1, 2, 2)
plt.plot(alphas, ridge_val_r2_3, label='Validation R²', marker='o')
plt.xlabel('Alpha (Regularization Strength)')
plt.ylabel('R² (Coefficient of Determination)')
plt.title('Effect of Ridge Regularization on R²')
plt.xscale('log')
plt.legend()
plt.grid(True)
plt.show()


# Step 7: Visualize the betas (coefficients) from all models
# Ridge model coefficients (for the best alpha, e.g., alpha=1.0)
ridge_model = Ridge(alpha=best_alpha_ridge_3)
ridge_model.fit(X3_train_scaled, y3_train)
betas_ridge = ridge_model.coef_

# Plot the coefficients (betas) for all models
plt.figure(figsize=(18, 8))

feature_name = [ 'Tough grader (Normalized)',
       'Good feedback (Normalized)', 'Respected (Normalized)',
       'Lots to read (Normalized)', 'Participation matters (Normalized)',
       'Don’t skip class or you will not pass (Normalized)',
       'Lots of homework (Normalized)', 'Inspirational (Normalized)',
       'Pop quizzes! (Normalized)', 'Accessible (Normalized)',
       'So many papers (Normalized)', 'Clear grading (Normalized)',
       'Hilarious (Normalized)', 'Test heavy (Normalized)',
       'Graded by few things (Normalized)', 'Amazing lectures (Normalized)',
       'Caring (Normalized)', 'Extra credit (Normalized)',
       'Group projects (Normalized)', 'Lecture heavy (Normalized)'
      ]

# Ridge Regression Coefficients (for best alpha)
plt.subplot(1, 2, 1)
plt.bar(feature_name, betas_ridge)
plt.title(f'Coefficients Ridge (alpha={best_alpha_ridge_3})')
plt.xlabel('Feature')
plt.ylabel('Coefficient Value')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

final_train_r2 = r2_score(y3_train, ridge_model.predict(X3_train_scaled))
final_val_r2 = r2_score(y3_val, ridge_model.predict(X3_val_scaled))

print(f"Final Validation R²: {final_val_r2}")


highest_beta_index = np.argmax(np.abs(betas_ridge))
highest_beta_feature = feature_name[highest_beta_index]
highest_beta_value = betas_ridge[highest_beta_index]

print(f"Feature with the highest coefficient (absolute value): {highest_beta_feature}")
print(f"Value of the highest coefficient: {highest_beta_value}")
"""
###########################################################################
End of Question 9
###########################################################################
"""



"""
###########################################################################
Beginning of Question 10
###########################################################################
# ### Q10 - Build a classification model that predicts whether a professor receives a “pepper” from all available factors(both tags and numerical). 
# Make sure to include model quality metrics such as AU(RO)C and also address class imbalanceconcerns.
"""
y4 =  df_cleaned['Received a “pepper”?']
X4 = df_cleaned[[ 'Tough grader (Normalized)',
       'Good feedback (Normalized)', 'Respected (Normalized)',
       'Lots to read (Normalized)', 'Participation matters (Normalized)',
       'Don’t skip class or you will not pass (Normalized)',
       'Lots of homework (Normalized)', 'Inspirational (Normalized)',
       'Pop quizzes! (Normalized)', 'Accessible (Normalized)',
       'So many papers (Normalized)', 'Clear grading (Normalized)',
       'Hilarious (Normalized)', 'Test heavy (Normalized)',
       'Graded by few things (Normalized)', 'Amazing lectures (Normalized)',
       'Caring (Normalized)', 'Extra credit (Normalized)',
       'Group projects (Normalized)', 'Lecture heavy (Normalized)','Average Difficulty (Adjusted)', 
       'Number of ratings', 'Average Rating (Adjusted)',
       'The proportion of students that said they would take the class again',
       'The number of ratings coming from online classes', 
       'Male gender',
       'Female'
      ]]

# check correlation
correlation_matrix4 = X4.corr()

plt.figure(figsize=(40, 28))
sns.heatmap(correlation_matrix4, annot=True, cmap="RdBu_r", center=0, xticklabels=True, yticklabels=True)
plt.title('Correlation Matrix Heatmap')
plt.show()


missing_values4 = X4.isna().sum()
print(missing_values4)
len(X4)


# Train-test split
X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4, test_size=0.2, random_state=r_number)

X4_train_continuous = X4_train.iloc[:, :25]  
X4_train_others = X4_train.iloc[:, 25:]     

X4_test_continuous = X4_test.iloc[:, :25]     
X4_test_others = X4_test.iloc[:, 25:]        

scaler = StandardScaler()
X4_train_continuous_scaled = scaler.fit_transform(X4_train_continuous)
X4_test_continuous_scaled = scaler.transform(X4_test_continuous)

# Combine the scaled continuous columns with the rest (dummy variables)
X4_train_scaled = np.hstack([X4_train_continuous_scaled, X4_train_others.values])
X4_test_scaled = np.hstack([X4_test_continuous_scaled, X4_test_others.values])

# Fit logistic regression
log_reg = LogisticRegression()
log_reg.fit(X4_train_scaled, y4_train)

print(f'Check imbalance: {Counter(y4_train)}')

# Predictions
y4_pred = log_reg.predict(X4_test_scaled)
y4_prob = log_reg.predict_proba(X4_test_scaled)[:, 1]

# Efficiently create and display the DataFrame
results = pd.DataFrame({'Predictions': y4_pred, 'Probabilities': y4_prob})

THRESHOLD = 0.5 # Revisit later!
y4_pred_new = (y4_prob > THRESHOLD).astype(int)

class_report = classification_report(y4_test, y4_pred_new) #y_pred_new
# print(class_report)

# ROC Curve
fpr, tpr, thresholds = roc_curve(y4_test, y4_prob)
roc_auc = auc(fpr, tpr)

optimal_threshold_index = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_threshold_index]
print(f"optimal_threshold is {optimal_threshold}")


##-------------------------------------------
THRESHOLD = optimal_threshold 

y4_pred_new = (y4_prob > THRESHOLD).astype(int)

class_report = classification_report(y4_test, y4_pred_new) #y_pred_new
print(class_report)

plt.figure(figsize=(18, 6))

# Confusion Matrix
plt.subplot(1, 2, 1)
conf_matrix = confusion_matrix(y4_test, y4_pred_new)  # y_pred_new
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="RdBu_r",
            xticklabels=["0 (not received)", "1 (received)"],
            yticklabels=["0 (not received)", "1 (received)"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

# ROC Curve
plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.legend()

# Adjust layout and display
plt.tight_layout()
plt.show()

# Print Precision and Recall at Optimal Threshold
print(f"Precision at optimal threshold: {precision_score(y4_test, y4_pred_new)}")
print(f"Recall at optimal threshold: {recall_score(y4_test, y4_pred_new)}")
"""
###########################################################################
End of Question 10
###########################################################################
"""




"""
###########################################################################
Beginning of Extra Credit
###########################################################################
"""

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
"""
###########################################################################
End of Extra Credit
###########################################################################
"""
