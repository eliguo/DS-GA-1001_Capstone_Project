#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
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


# In[2]:


# Re-running preprocessing with an additional step: Dropped rows with missing values in "The proportion of students that said they would take the class again," which appears to be an important feature
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

# In[3]:


rng = np.random.default_rng(14420733) # Uncomment if rng is not defined elsewhere

r_number = rng.integers(1, 100)
r_number


# ##### import adjusted data

# In[4]:


df_cleaned = pd.read_csv('rmpCapstoneAdjusted_regression.csv')
df_cleaned


# ### Q7 - Build a regression model predicting average rating from all numerical predictors 
# (the ones in the rmpCapstoneNum.csv) file.Make sure to include the R2and RMSE of this model. Which of these factors is most strongly predictive of average rating? Hint: Make sure to address collinearity concern

# In[5]:


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
X


# In[6]:


# check missing values
missing_values = X.isna().sum()
print(missing_values)


# In[7]:


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


# In[8]:


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


# ### Q8 - Build a regression model predicting average ratings from all tags
# (the ones in the rmpCapstoneTags.csv) file. Make sure to include the R2and RMSE of this model. Which of these tags is most strongly predictive of average rating? Hint: Make sure to address collinearity concerns. Also comment on how this model compares to the previous one

# In[9]:


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
X2


# In[10]:


# check missing values
missing_values2 = X2.isna().sum()
print(missing_values2)
len(X2)


# In[11]:


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


# In[12]:


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


# ### Q9 - Build a regression model predicting average difficulty from all tags
# (the ones in the rmpCapstoneTags.csv) file. Make sure to include the R2and RMSE of this model. Which of these tags is most strongly predictive of average rating? Hint: Make sure to address collinearity concerns. Also comment on how this model compares to the previous one

# In[13]:


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
X3


# In[14]:


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


# In[15]:


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


# ### Q10 - Build a classification model that predicts whether a professor receives a “pepper” from all available factors(both tags and numerical). 
# Make sure to include model quality metrics such as AU(RO)C and also address class imbalanceconcerns.

# In[16]:


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
X4


# In[17]:


missing_values4 = X4.isna().sum()
print(missing_values4)
len(X4)


# In[18]:


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

