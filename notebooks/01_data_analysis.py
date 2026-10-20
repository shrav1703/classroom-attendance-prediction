import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------
# 1. Load Dataset
# -----------------------------------

file_path = "data/attendance_dataset.xlsx"

df = pd.read_excel(file_path)

print("Dataset Loaded Successfully!")
print("--------------------------------")

# Dataset shape
print("Number of Rows:", df.shape[0])
print("Number of Columns:", df.shape[1])

# First 5 rows
print("\nFirst 5 Rows:")
print(df.head())

# Column names
print("\nColumn Names:")
print(df.columns.tolist())

# Dataset information
print("\nDataset Information:")
print(df.info())

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicate rows
print("\nDuplicate Rows:", df.duplicated().sum())

# Statistical summary
print("\nStatistical Summary:")
print(df.describe(include="all"))


# -----------------------------------
# 2. Data Cleaning
# -----------------------------------

# Convert Date to datetime
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# Convert Start Time to string
df["Start Time"] = df["Start Time"].astype(str)

# Remove completely empty rows
df = df.dropna(how="all")

# Remove duplicate records
df = df.drop_duplicates()

# Clean text columns
text_columns = [
    "Day",
    "Subject",
    "Faculty ID",
    "Branch",
    "Section",
    "Classroom",
    "Practical/Theory",
    "Test Week",
    "Assignment Due",
    "Holiday",
    "Weather",
    "Special Event",
    "Lecture Occurred",
    "Faculty Experience"
]

for col in text_columns:
    df[col] = df[col].astype("string").str.strip()

# Fill missing Faculty ID
df["Faculty ID"] = df["Faculty ID"].fillna("Unknown")

# Fill missing Faculty Experience
df["Faculty Experience"] = df["Faculty Experience"].fillna("Unknown")

# Previous Lecture Attendance:
# Missing values are expected for lectures where previous attendance
# is not available.
df["Previous Lecture Attendance"] = (
    df["Previous Lecture Attendance"]
    .fillna(df["Attendance %"].median())
)

print("\nAfter Cleaning:")
print("Rows:", len(df))
print("Missing Values:")
print(df.isnull().sum())

# -----------------------------------
# 3. Feature Engineering
# -----------------------------------

# Sort chronologically
df = df.sort_values(["Date", "Start Time"]).reset_index(drop=True)

# Day number
day_mapping = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}

df["Day Number"] = df["Day"].map(day_mapping)

# Week number
df["Week Number"] = (
    (df["Date"] - df["Date"].min()).dt.days // 7
) + 1

# Day of semester
df["Day of Semester"] = (
    df["Date"] - df["Date"].min()
).dt.days + 1

# Month
df["Month"] = df["Date"].dt.month

# Day of month
df["Day of Month"] = df["Date"].dt.day

# Extract hour from Start Time
df["Hour"] = pd.to_datetime(
    df["Start Time"],
    format="%H:%M",
    errors="coerce"
).dt.hour

# Time of day category
def get_time_category(hour):

    if pd.isna(hour):
        return "Unknown"

    if hour < 12:
        return "Morning"

    elif hour < 16:
        return "Afternoon"

    else:
        return "Evening"


df["Time of Day"] = df["Hour"].apply(get_time_category)

# -----------------------------------
# Previous Attendance Features
# -----------------------------------

# Previous 3 lecture rolling average
df["Previous 3 Lecture Average"] = (
    df["Attendance %"]
    .shift(1)
    .rolling(window=3, min_periods=1)
    .mean()
)

# Previous lecture attendance
df["Previous Lecture Attendance"] = (
    df["Attendance %"].shift(1)
)

# Fill first lecture missing value
df["Previous Lecture Attendance"] = (
    df["Previous Lecture Attendance"]
    .fillna(df["Attendance %"].median())
)

df["Previous 3 Lecture Average"] = (
    df["Previous 3 Lecture Average"]
    .fillna(df["Attendance %"].median())
)

plt.figure(figsize=(8, 5))

sns.histplot(
    df["Attendance %"],
    bins=20,
    kde=True
)

plt.title("Attendance Percentage Distribution")
plt.xlabel("Attendance Percentage")
plt.ylabel("Number of Lectures")

plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 5))

sns.boxplot(
    data=df,
    x="Day",
    y="Attendance %"
)

plt.title("Attendance Distribution by Day")
plt.xlabel("Day")
plt.ylabel("Attendance %")

plt.xticks(rotation=30)

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 7))

subject_avg = (
    df.groupby("Subject")["Attendance %"]
    .mean()
    .sort_values()
)

subject_avg.plot(kind="barh")

plt.title("Average Attendance by Subject")
plt.xlabel("Average Attendance %")
plt.ylabel("Subject")

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))

time_avg = (
    df.groupby("Time of Day")["Attendance %"]
    .mean()
)

time_avg.plot(kind="bar")

plt.title("Average Attendance by Time of Day")
plt.xlabel("Time of Day")
plt.ylabel("Average Attendance %")

plt.xticks(rotation=0)

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 7))

numeric_columns = [
    "Lecture Number",
    "Semester",
    "Total Enrolled Students",
    "Previous Lecture Attendance",
    "Attendance %",
    "Day Number",
    "Week Number",
    "Day of Semester",
    "Month",
    "Day of Month",
    "Hour",
    "Previous 3 Lecture Average"
]

corr = df[numeric_columns].corr()

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.show()

# -----------------------------------
# Save Cleaned Dataset
# -----------------------------------

df.to_csv(
    "outputs/cleaned_attendance_dataset.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")