import pandas as pd

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
rmp = rmp.dropna(subset=["Average Rating"])

# Count the number of duplicated rows in the DataFrame
num_duplicates = rmp.duplicated().sum()

print(f"Number of duplicated rows: {num_duplicates}")


# rmp = rmp.dropna(subset=["Average Rating"]).drop_duplicates()

# # Bayesian adjustments for average rating and difficulty
# # calculate prior means
# prior_mean_rating = rmp["Average Rating"].mean()
# prior_mean_difficulty = rmp["Average Difficulty"].mean()

# # use median of 'Number of ratings' as strength of prior
# strength_of_prior = rmp["Number of ratings"].median()

# # apply Bayesian adjustment for ‘Average Rating’
# rmp["Average Rating (Adjusted)"] = (
#     (strength_of_prior * prior_mean_rating + 
#      rmp["Number of ratings"] * rmp["Average Rating"]) /
#     (strength_of_prior + rmp["Number of ratings"])
# )

# # apply Bayesian adjustment for ‘Average Difficulty’
# rmp["Average Difficulty (Adjusted)"] = (
#     (strength_of_prior * prior_mean_difficulty + 
#      rmp["Number of ratings"] * rmp["Average Difficulty"]) /
#     (strength_of_prior + rmp["Number of ratings"])
# )

# # normalize tag columns by 'Number of ratings'
# tag_columns = rmp_tags.columns
# for tag in tag_columns:
#     rmp[f"{tag} (Normalized)"] = rmp[tag] / rmp["Number of ratings"]

# # calculate total number of tags
# rmp["Total Number of Tags"] = rmp[tag_columns].sum(axis=1)

# # calculate average number of tags per rating
# rmp["Average Tags per Rating"] = rmp["Total Number of Tags"] / rmp["Number of ratings"]

# # reorder columns
# column_order = [
#     # numerical columns
#     "Average Rating",
#     "Average Rating (Adjusted)",
#     "Average Difficulty",
#     "Average Difficulty (Adjusted)",
#     "Number of ratings",
#     "Received a “pepper”?",
#     "The proportion of students that said they would take the class again",
#     "The number of ratings coming from online classes",
#     "Male gender",
#     "Female",
    
#     # qualitative columns
#     "Major/Field",
#     "University",
#     "US State",

#     # tag columns
#     *tag_columns,  # original tag columns
#     *[f"{tag} (Normalized)" for tag in tag_columns],  # normalized tag columns
#     "Total Number of Tags",
#     "Average Tags per Rating"
# ]

# # apply the column order
# rmp = rmp[column_order]

# # save the cleaned and adjusted data
# rmp.to_csv(f'rmpCapstoneAdjusted_{len(rmp)}.csv', index=False)