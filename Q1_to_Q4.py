from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t, ttest_ind, levene

all_data = pd.read_csv("rmpCapstoneAdjusted_69989.csv")

"""
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
rating_male = all_data.query("`Male gender` == 1")["Average Rating (Adjusted)"]
rating_female = all_data.query("`Female` == 1")["Average Rating (Adjusted)"]

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
bins = np.arange(0.75, 5.25, 0.5)
plt.hist(rating_male, bins=bins, label='male', alpha=0.5, color="blue")
plt.hist(rating_female, bins=bins, label='female', alpha=0.5, color="red")
plt.xlabel('Rating from 1 to 5')
plt.ylabel('Frequency')
plt.title('Comparison between the ratings of male / female professore')
plt.legend()
plt.show()
"""
The result for Question 1 is:
Levene's test indicates a statistically significant difference in the variance between the ratings for male and female
Thus, we decided to use Welch's t-test. The p-value is 2.3259e-12 < 0.005.
Hence, we reject the null hypothesis
"""



"""
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
"""
The result for Question 2 is:
Levene's test gives a p-value to be 3.3177e-06 < 0.005.
Hence, we reject the null hypothesis
"""



"""
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
"""
The result for Question 3 is:

Bootstrapped Cohen's d: 0.0582
95% CI for Cohen's d: [0.03010664 0.08580679]
Bootstrapped Variance Ratio: 0.9565
95% CI for Variance Ratio: [0.91368686 0.99934664]
"""




"""
For Question 4, we use t test to determine whether each tag is gendered.
"""
# First extrace columns that are tages
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
    tag_male = all_data.query("`Male gender` == 1")[tag]
    tag_female = all_data.query("`Female` == 1")[tag]
    
    # Perform Levene first, and then t-test
    levene_test = levene(tag_male, tag_female)
    if_eq_var = False if levene_test.pvalue > ALPHA else True
    each_result = ttest_ind(tag_male, tag_female, equal_var=if_eq_var)
    
    tag_results.append({'Tag': tag, 'P-Value': each_result.pvalue})

# Convert results to DataFrame, such that tags are paired with p-values, then sort
results_df = pd.DataFrame(tag_results)
results_df = results_df.sort_values('P-Value')

# extract significantly gendered tags
gendered_tags = results_df[results_df["P-Value"] <= ALPHA]["Tag"].to_list()
gendered_tags = [each_tag.replace(" (Normalized)", "") for each_tag in gendered_tags]
print(f"The following {len(gendered_tags)} tags exhibit a statistically significant different in gender: {gendered_tags}")

most_sig_3, least_sig_3 = results_df["Tag"].to_list()[:3], results_df["Tag"].to_list()[-3:]
most_sig_3 = [each_tag.replace(" (Normalized)", "") for each_tag in most_sig_3]
least_sig_3 = [each_tag.replace(" (Normalized)", "") for each_tag in least_sig_3]
print(f"The most gendered 3 tags are {most_sig_3}, and the least gendered 3 tags are {least_sig_3}")
"""
The result for Question 4 is:

The following 19 tags exhibit a statistically significant different in gender: 
['Hilarious', 'Amazing lectures', 'Caring', 'Respected', 'Lecture heavy', 'Participation matters', 
'Good feedback', 'Graded by few things', 'Lots of homework', 'Group projects', 'So many papers', 
'Extra credit', 'Tough grader', 'Test heavy', 'Clear grading', 'Accessible', 'Inspirational', 
'Lots to read', 'Don’t skip class or you will not pass']

The most gendered 3 tags are ['Hilarious', 'Amazing lectures', 'Caring'],
and the least gendered 3 tags are ['Lots to read', 'Don’t skip class or you will not pass', 'Pop quizzes!']
"""

