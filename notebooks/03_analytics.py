import pandas as pd
import matplotlib.pyplot as plt
import os


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\nLoading cleaned attendance dataset...")

df = pd.read_csv(
    "outputs/cleaned_attendance_dataset.csv"
)

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

print("Dataset loaded successfully!")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# ============================================================
# 2. CREATE ANALYTICS FOLDER
# ============================================================

os.makedirs(
    "outputs/analytics",
    exist_ok=True
)


# ============================================================
# 3. BASIC ATTENDANCE STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("ATTENDANCE STATISTICS")
print("=" * 60)

print(
    "Average Attendance:",
    round(df["Attendance %"].mean(), 2),
    "%"
)

print(
    "Highest Attendance:",
    round(df["Attendance %"].max(), 2),
    "%"
)

print(
    "Lowest Attendance:",
    round(df["Attendance %"].min(), 2),
    "%"
)

print(
    "Median Attendance:",
    round(df["Attendance %"].median(), 2),
    "%"
)


# ============================================================
# 4. ATTENDANCE DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 4))

plt.hist(
    df["Attendance %"],
    bins=10
)

plt.title(
    "Attendance Percentage Distribution"
)

plt.xlabel(
    "Attendance Percentage"
)

plt.ylabel(
    "Number of Lectures"
)

plt.tight_layout()

plt.savefig(
    "outputs/analytics/attendance_distribution.png"
)

plt.close()

print(
    "\nSaved: attendance_distribution.png"
)


# ============================================================
# 5. SUBJECT-WISE ATTENDANCE
# ============================================================

subject_attendance = (
    df.groupby("Subject")["Attendance %"]
    .mean()
    .sort_values(ascending=False)
)


print("\n" + "=" * 60)
print("SUBJECT-WISE AVERAGE ATTENDANCE")
print("=" * 60)

print(
    subject_attendance
)


plt.figure(figsize=(8, 4))

subject_attendance.plot(
    kind="bar"
)

plt.title(
    "Average Attendance by Subject"
)

plt.xlabel(
    "Subject"
)

plt.ylabel(
    "Average Attendance (%)"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    "outputs/analytics/subject_wise_attendance.png"
)

plt.close()

print(
    "\nSaved: subject_wise_attendance.png"
)


# ============================================================
# 6. DAY-WISE ATTENDANCE
# ============================================================

day_attendance = (
    df.groupby("Day")["Attendance %"]
    .mean()
)


day_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]


day_attendance = (
    day_attendance
    .reindex(day_order)
    .dropna()
)


print("\n" + "=" * 60)
print("DAY-WISE AVERAGE ATTENDANCE")
print("=" * 60)

print(
    day_attendance
)


plt.figure(figsize=(8, 4))

day_attendance.plot(
    kind="bar"
)

plt.title(
    "Average Attendance by Day"
)

plt.xlabel(
    "Day"
)

plt.ylabel(
    "Average Attendance (%)"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    "outputs/analytics/day_wise_attendance.png"
)

plt.close()

print(
    "\nSaved: day_wise_attendance.png"
)


# ============================================================
# 7. TIME-OF-DAY ATTENDANCE
# ============================================================

time_attendance = (
    df.groupby("Time of Day")["Attendance %"]
    .mean()
)


print("\n" + "=" * 60)
print("TIME-OF-DAY ATTENDANCE")
print("=" * 60)

print(
    time_attendance
)


plt.figure(figsize=(8, 4))

time_attendance.plot(
    kind="bar"
)

plt.title(
    "Average Attendance by Time of Day"
)

plt.xlabel(
    "Time of Day"
)

plt.ylabel(
    "Average Attendance (%)"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    "outputs/analytics/time_of_day_attendance.png"
)

plt.close()

print(
    "\nSaved: time_of_day_attendance.png"
)


# ============================================================
# 8. FACULTY-WISE ATTENDANCE
# ============================================================

faculty_attendance = (
    df.groupby("Faculty ID")["Attendance %"]
    .mean()
    .sort_values(ascending=False)
)


print("\n" + "=" * 60)
print("FACULTY-WISE AVERAGE ATTENDANCE")
print("=" * 60)

print(
    faculty_attendance
)


plt.figure(figsize=(8, 4))

faculty_attendance.plot(
    kind="bar"
)

plt.title(
    "Average Attendance by Faculty"
)

plt.xlabel(
    "Faculty ID"
)

plt.ylabel(
    "Average Attendance (%)"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    "outputs/analytics/faculty_wise_attendance.png"
)

plt.close()

print(
    "\nSaved: faculty_wise_attendance.png"
)


# ============================================================
# 9. ATTENDANCE VS PREVIOUS LECTURE
# ============================================================

plt.figure(figsize=(8, 4))

plt.scatter(
    df["Previous Lecture Attendance"],
    df["Attendance %"]
)

plt.title(
    "Previous Lecture Attendance vs Current Attendance"
)

plt.xlabel(
    "Previous Lecture Attendance (%)"
)

plt.ylabel(
    "Current Attendance (%)"
)

plt.tight_layout()

plt.savefig(
    "outputs/analytics/previous_vs_current_attendance.png"
)

plt.close()

print(
    "\nSaved: previous_vs_current_attendance.png"
)


# ============================================================
# 10. ATTENDANCE BY PRACTICAL / THEORY
# ============================================================

type_attendance = (
    df.groupby("Practical/Theory")["Attendance %"]
    .mean()
    .sort_values(ascending=False)
)


print("\n" + "=" * 60)
print("THEORY / PRACTICAL ATTENDANCE")
print("=" * 60)

print(
    type_attendance
)


plt.figure(figsize=(8, 4))

type_attendance.plot(
    kind="bar"
)

plt.title(
    "Average Attendance: Theory vs Practical"
)

plt.xlabel(
    "Lecture Type"
)

plt.ylabel(
    "Average Attendance (%)"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    "outputs/analytics/theory_practical_attendance.png"
)

plt.close()

print(
    "\nSaved: theory_practical_attendance.png"
)


# ============================================================
# 11. ATTENDANCE CATEGORY
# ============================================================

def attendance_category(value):

    if value < 50:
        return "Low"

    elif value <= 75:
        return "Medium"

    else:
        return "High"


df["Attendance Category"] = (
    df["Attendance %"]
    .apply(attendance_category)
)


category_counts = (
    df["Attendance Category"]
    .value_counts()
)


print("\n" + "=" * 60)
print("ATTENDANCE CATEGORY")
print("=" * 60)

print(
    category_counts
)


plt.figure(figsize=(8, 4))

category_counts.plot(
    kind="bar"
)

plt.title(
    "Attendance Category Distribution"
)

plt.xlabel(
    "Attendance Category"
)

plt.ylabel(
    "Number of Lectures"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    "outputs/analytics/attendance_category.png"
)

plt.close()

print(
    "\nSaved: attendance_category.png"
)


# ============================================================
# 12. SAVE ANALYTICS SUMMARY
# ============================================================

summary = pd.DataFrame({

    "Metric": [
        "Total Records",
        "Average Attendance",
        "Highest Attendance",
        "Lowest Attendance",
        "Median Attendance"
    ],

    "Value": [
        len(df),
        round(df["Attendance %"].mean(), 2),
        round(df["Attendance %"].max(), 2),
        round(df["Attendance %"].min(), 2),
        round(df["Attendance %"].median(), 2)
    ]
})


summary.to_csv(
    "outputs/analytics/analytics_summary.csv",
    index=False
)


# ============================================================
# 13. FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 60)
print("ANALYTICS COMPLETED SUCCESSFULLY!")
print("=" * 60)

print(
    "\nAll analytics files are available in:"
)

print(
    "outputs/analytics/"
)