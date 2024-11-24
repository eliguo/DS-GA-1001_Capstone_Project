import pandas as pd

rmp_num = pd.read_csv('rmpCapstoneNum.csv', header=None)
rmp_qual = pd.read_csv('rmpCapstoneQual.csv', header=None)
rmp_tags = pd.read_csv('rmpCapstoneTags.csv', header=None)

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

merged = pd.concat([rmp_num, rmp_qual, rmp_tags], axis=1)

# print(len(merged))
# print("-" * 50)
# print(merged.duplicated().sum())
# print("-" * 50)
# print(merged.isnull().sum())

cleaned_data = merged.dropna(subset=["Average Rating"]).drop_duplicates()

cleaned_data.to_csv(f'rmpCapstoneCleaned_{len(cleaned_data)}.csv', index=False)