import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from datetime import date


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Attendance Prediction System",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# LOAD DATA AND MODEL
# ============================================================

model = joblib.load(
    "models/attendance_prediction_model.pkl"
)

dataset = pd.read_csv(
    "outputs/cleaned_attendance_dataset.csv"
)


# ============================================================
# TITLE
# ============================================================

st.title(
    "📊 Classroom Attendance Prediction System"
)

st.write(
    """
    A Machine Learning based system for predicting the expected
    attendance percentage of an upcoming lecture using historical
    attendance and academic schedule information.
    """
)

st.divider()


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.header("🤖 Model Performance")

results = pd.read_csv(
    "outputs/model_comparison.csv"
)

best_model = results.iloc[0]

metric1, metric2, metric3, metric4 = st.columns(4)


with metric1:

    st.metric(
        "Best Model",
        best_model["Model"]
    )


with metric2:

    st.metric(
        "MAE",
        f"{best_model['MAE']:.2f}"
    )


with metric3:

    st.metric(
        "RMSE",
        f"{best_model['RMSE']:.2f}"
    )


with metric4:

    st.metric(
        "R² Score",
        f"{best_model['R2']:.2f}"
    )


st.divider()


# ============================================================
# LECTURE INFORMATION
# ============================================================

st.header("📚 Lecture Information")

col1, col2, col3 = st.columns(3)


with col1:

    lecture_date = st.date_input(
        "Lecture Date",
        value=date.today()
    )

    day = lecture_date.strftime("%A")

    st.info(
        f"Selected Day: {day}"
    )

    lecture_number = st.number_input(
        "Lecture Number",
        min_value=1,
        max_value=50,
        value=1
    )


with col2:

    start_time = st.selectbox(
        "Start Time",
        [
            "08:00",
            "08:30",
            "09:00",
            "09:15",
            "09:30",
            "10:00",
            "10:15",
            "10:30",
            "11:00",
            "11:15",
            "11:30",
            "12:00",
            "13:00",
            "13:30",
            "14:00",
            "14:30",
            "15:00",
            "16:00"
        ]
    )


with col3:

    semester = st.number_input(
        "Semester",
        min_value=1,
        max_value=8,
        value=3
    )

    total_students = st.number_input(
        "Total Enrolled Students",
        min_value=1,
        max_value=500,
        value=200
    )


# ============================================================
# DAY VALIDATION
# ============================================================

if day == "Sunday":

    st.warning(
        "⚠️ Sunday is not present in the regular lecture "
        "schedule of the dataset. Consider selecting a weekday."
    )


# ============================================================
# ACADEMIC INFORMATION
# ============================================================

st.header("🎓 Academic Information")

col4, col5, col6 = st.columns(3)


with col4:

    subject = st.selectbox(
        "Subject",
        sorted(
            dataset["Subject"]
            .dropna()
            .unique()
        )
    )

    faculty_id = st.selectbox(
        "Faculty ID",
        sorted(
            dataset["Faculty ID"]
            .dropna()
            .unique()
        )
    )


with col5:

    practical_theory = st.selectbox(
        "Practical / Theory",
        sorted(
            dataset["Practical/Theory"]
            .dropna()
            .unique()
        )
    )

    assignment_due = st.selectbox(
        "Assignment Due",
        sorted(
            dataset["Assignment Due"]
            .dropna()
            .unique()
        )
    )


with col6:

    weather_options = sorted(
        dataset["Weather"]
        .dropna()
        .unique()
        .tolist()
    )

    if "Sunny" not in weather_options:
        weather_options.append("Sunny")

    weather = st.selectbox(
        "Weather",
        weather_options
    )

    special_event = st.selectbox(
        "Special Event",
        sorted(
            dataset["Special Event"]
            .dropna()
            .unique()
        )
    )

# ============================================================
# HISTORICAL ATTENDANCE
# ============================================================

st.header("📈 Historical Attendance")

col7, col8, col9 = st.columns(3)


with col7:

    previous_attendance = st.number_input(
        "Previous Lecture Attendance %",
        min_value=0.0,
        max_value=100.0,
        value=70.0,
        step=0.1
    )


with col8:

    previous_three_avg = st.number_input(
        "Previous 3 Lecture Average %",
        min_value=0.0,
        max_value=100.0,
        value=70.0,
        step=0.1
    )


with col9:

    faculty_experience = st.selectbox(
        "Faculty Experience",
        sorted(
            dataset["Faculty Experience"]
            .dropna()
            .unique()
        )
    )


# ============================================================
# DATASET VALUES
# ============================================================

branch = dataset["Branch"].mode()[0]

section = dataset["Section"].mode()[0]

classroom = dataset["Classroom"].mode()[0]

test_week = dataset["Test Week"].mode()[0]

holiday = dataset["Holiday"].mode()[0]


# ============================================================
# FEATURE ENGINEERING
# ============================================================

day_mapping = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}

day_number = day_mapping[day]


hour = int(
    start_time.split(":")[0]
)


if hour < 12:

    time_of_day = "Morning"

elif hour < 16:

    time_of_day = "Afternoon"

else:

    time_of_day = "Evening"


# ============================================================
# DATE FEATURES
# ============================================================

dataset["Date"] = pd.to_datetime(
    dataset["Date"],
    errors="coerce"
)

minimum_date = dataset["Date"].min()


day_of_semester = (
    lecture_date - minimum_date.date()
).days + 1


if day_of_semester < 1:

    day_of_semester = 1


week_number = (
    (day_of_semester - 1) // 7
) + 1


month = lecture_date.month

day_of_month = lecture_date.day


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

predict_clicked = st.button(
    "🔮 Predict Attendance",
    type="primary",
    use_container_width=True
)


if predict_clicked:

    # ========================================================
    # INPUT DATA
    # ========================================================

    input_data = pd.DataFrame({

        "Day": [
            day
        ],

        "Lecture Number": [
            lecture_number
        ],

        "Start Time": [
            start_time
        ],

        "Subject": [
            subject
        ],

        "Faculty ID": [
            faculty_id
        ],

        "Semester": [
            semester
        ],

        "Branch": [
            branch
        ],

        "Section": [
            section
        ],

        "Classroom": [
            classroom
        ],

        "Total Enrolled Students": [
            total_students
        ],

        "Previous Lecture Attendance": [
            previous_attendance
        ],

        "Practical/Theory": [
            practical_theory
        ],

        "Test Week": [
            test_week
        ],

        "Assignment Due": [
            assignment_due
        ],

        "Holiday": [
            holiday
        ],

        "Weather": [
            weather
        ],

        "Special Event": [
            special_event
        ],

        "Faculty Experience": [
            faculty_experience
        ],

        "Day Number": [
            day_number
        ],

        "Week Number": [
            week_number
        ],

        "Day of Semester": [
            day_of_semester
        ],

        "Month": [
            month
        ],

        "Day of Month": [
            day_of_month
        ],

        "Hour": [
            hour
        ],

        "Time of Day": [
            time_of_day
        ],

        "Previous 3 Lecture Average": [
            previous_three_avg
        ]
    })


    # ========================================================
    # PREDICTION
    # ========================================================

    prediction = model.predict(
        input_data
    )[0]


    # Keep prediction between 0 and 100

    prediction = max(
        0,
        min(100, prediction)
    )


    # ========================================================
    # EXPECTED STUDENT COUNTS
    # ========================================================

    expected_present = round(
        total_students * prediction / 100
    )

    expected_absent = (
        total_students
        -
        expected_present
    )


    # ========================================================
    # ATTENDANCE CATEGORY
    # ========================================================

    if prediction < 50:

        category = "Low Attendance"
        emoji = "🔴"

    elif prediction < 75:

        category = "Medium Attendance"
        emoji = "🟡"

    else:

        category = "High Attendance"
        emoji = "🟢"


    # ========================================================
    # PREDICTION RESULT
    # ========================================================

    st.header(
        "🎯 Prediction Result"
    )


    # ========================================================
    # MAIN METRICS
    # ========================================================

    result1, result2, result3 = st.columns(3)


    with result1:

        st.metric(
            "Predicted Attendance",
            f"{prediction:.2f}%"
        )


    with result2:

        st.metric(
            "Expected Students Present",
            expected_present
        )


    with result3:

        st.metric(
            "Expected Students Absent",
            expected_absent
        )


    # ========================================================
    # ATTENDANCE STATUS
    # ========================================================

    if category == "Low Attendance":

        st.error(
            f"{emoji} {category}"
        )

        st.write(
            "The predicted attendance is below 50%. "
            "Student participation is expected to be low."
        )


    elif category == "Medium Attendance":

        st.warning(
            f"{emoji} {category}"
        )

        st.write(
            "The predicted attendance is between 50% and 75%. "
            "Moderate student participation is expected."
        )


    else:

        st.success(
            f"{emoji} {category}"
        )

        st.write(
            "The predicted attendance is 75% or above. "
            "Good student participation is expected."
        )


    # ========================================================
    # EXPECTED STUDENT DISTRIBUTION
    # ========================================================

    st.subheader(
        "📊 Expected Student Distribution"
    )


    chart_data = pd.DataFrame({

        "Status": [
            "Present",
            "Absent"
        ],

        "Students": [
            expected_present,
            expected_absent
        ]

    })


    fig, ax = plt.subplots(
        figsize=(8, 4)
    )


    bars = ax.bar(
        chart_data["Status"],
        chart_data["Students"]
    )


    ax.set_title(
        "Expected Students Present vs Absent"
    )

    ax.set_ylabel(
        "Number of Students"
    )


    # Add values on top of bars

    for bar in bars:

        height = bar.get_height()

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom"
        )


    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


    # ========================================================
    # RECOMMENDATION
    # ========================================================

    st.subheader(
        "💡 Recommendation"
    )


    if prediction < 50:

        st.error(
            "Attendance is expected to be low. "
            "Consider encouraging students to attend the lecture "
            "and reviewing factors affecting attendance."
        )


    elif prediction < 75:

        st.warning(
            "Attendance is expected to be moderate. "
            "Additional student engagement may help improve attendance."
        )


    else:

        st.success(
            "Attendance is expected to be good. "
            "The lecture is likely to have strong student participation."
        )


    # ========================================================
    # PREDICTION INPUT SUMMARY
    # ========================================================

    with st.expander(
        "📋 View Prediction Inputs"
    ):

        st.dataframe(
            input_data,
            use_container_width=True
        )


# ============================================================
# DATASET INFORMATION
# ============================================================

st.divider()

st.header(
    "📁 Dataset Information"
)


info1, info2, info3 = st.columns(3)


with info1:

    st.metric(
        "Total Records",
        len(dataset)
    )


with info2:

    st.metric(
        "Total Features",
        len(dataset.columns)
    )


with info3:

    average_attendance = dataset[
        "Attendance %"
    ].mean()

    st.metric(
        "Average Attendance",
        f"{average_attendance:.2f}%"
    )


# ============================================================
# MODEL COMPARISON
# ============================================================

st.subheader(
    "🤖 Model Comparison"
)


st.dataframe(
    results.style.format({
        "MAE": "{:.2f}",
        "RMSE": "{:.2f}",
        "MAPE": "{:.2f}%",
        "R2": "{:.2f}"
    }),
    use_container_width=True
)


# ============================================================
# ATTENDANCE ANALYTICS
# ============================================================

st.divider()

st.header(
    "📊 Attendance Analytics"
)

st.write(
    "Explore historical attendance patterns from the dataset."
)


# ============================================================
# ANALYTICS TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Distribution",
    "📚 Subject Analysis",
    "📅 Schedule Analysis",
    "👨‍🏫 Faculty Analysis"
])


# ============================================================
# TAB 1 - ATTENDANCE DISTRIBUTION
# ============================================================

with tab1:

    st.subheader(
        "Attendance Percentage Distribution"
    )

    image_path = (
        "outputs/analytics/"
        "attendance_distribution.png"
    )

    st.image(
        image_path,
        use_container_width=True
    )


# ============================================================
# TAB 2 - SUBJECT ANALYSIS
# ============================================================

with tab2:

    st.subheader(
        "Average Attendance by Subject"
    )

    image_path = (
        "outputs/analytics/"
        "subject_wise_attendance.png"
    )

    st.image(
        image_path,
        use_container_width=True
    )


# ============================================================
# TAB 3 - SCHEDULE ANALYSIS
# ============================================================

with tab3:

    st.subheader(
        "Attendance by Day"
    )

    image_path = (
        "outputs/analytics/"
        "day_wise_attendance.png"
    )

    st.image(
        image_path,
        use_container_width=True
    )


    st.subheader(
        "Attendance by Time of Day"
    )

    image_path = (
        "outputs/analytics/"
        "time_of_day_attendance.png"
    )

    st.image(
        image_path,
        use_container_width=True
    )


    st.subheader(
        "Theory vs Practical vs Assessment"
    )

    image_path = (
        "outputs/analytics/"
        "theory_practical_attendance.png"
    )

    st.image(
        image_path,
        use_container_width=True
    )


# ============================================================
# TAB 4 - FACULTY ANALYSIS
# ============================================================

with tab4:

    st.subheader(
        "Average Attendance by Faculty"
    )

    image_path = (
        "outputs/analytics/"
        "faculty_wise_attendance.png"
    )

    st.image(
        image_path,
        use_container_width=True
    )


    st.subheader(
        "Previous Lecture vs Current Attendance"
    )

    image_path = (
        "outputs/analytics/"
        "previous_vs_current_attendance.png"
    )

    st.image(
        image_path,
        use_container_width=True
    )


# ============================================================
# ATTENDANCE CATEGORY
# ============================================================

st.subheader(
    "📌 Attendance Category Distribution"
)

image_path = (
    "outputs/analytics/"
    "attendance_category.png"
)

st.image(
    image_path,
    use_container_width=True
)