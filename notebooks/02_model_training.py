import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 1. LOAD CLEANED DATASET
# ============================================================

print("\nLoading cleaned dataset...")

df = pd.read_csv(
    "outputs/cleaned_attendance_dataset.csv"
)

print("Dataset loaded successfully!")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# ============================================================
# 2. DEFINE TARGET VARIABLE
# ============================================================

target = "Attendance %"

print("\nTarget Variable:", target)


# ============================================================
# 3. REMOVE DATA LEAKAGE COLUMNS
# ============================================================

# Students Present directly determines Attendance %.
# Therefore, it must NOT be given to the model.

columns_to_remove = [
    "Attendance %",
    "Students Present",
    "Lecture Occurred",
    "Date"
]

X = df.drop(
    columns=columns_to_remove
)

y = df[target]


# ============================================================
# 4. SELECT FEATURES
# ============================================================

features = [
    "Day",
    "Lecture Number",
    "Start Time",
    "Subject",
    "Faculty ID",
    "Semester",
    "Branch",
    "Section",
    "Classroom",
    "Total Enrolled Students",
    "Previous Lecture Attendance",
    "Practical/Theory",
    "Test Week",
    "Assignment Due",
    "Holiday",
    "Weather",
    "Special Event",
    "Faculty Experience",
    "Day Number",
    "Week Number",
    "Day of Semester",
    "Month",
    "Day of Month",
    "Hour",
    "Time of Day",
    "Previous 3 Lecture Average"
]

X = X[features]

print("\nNumber of Features:", len(features))


# ============================================================
# 5. DEFINE CATEGORICAL FEATURES
# ============================================================

categorical_features = [
    "Day",
    "Start Time",
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
    "Faculty Experience",
    "Time of Day"
]


# ============================================================
# 6. DEFINE NUMERICAL FEATURES
# ============================================================

numeric_features = [
    "Lecture Number",
    "Semester",
    "Total Enrolled Students",
    "Previous Lecture Attendance",
    "Day Number",
    "Week Number",
    "Day of Semester",
    "Month",
    "Day of Month",
    "Hour",
    "Previous 3 Lecture Average"
]


# ============================================================
# 7. PREPROCESSING
# ============================================================

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)


categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)


# ============================================================
# 8. CHRONOLOGICAL TRAIN-TEST SPLIT
# ============================================================

# Since we are predicting future lectures,
# earlier records are used for training
# and later records are used for testing.

split_index = int(len(X) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


print("\nTraining Records:", len(X_train))
print("Testing Records :", len(X_test))


# ============================================================
# 9. DEFINE MACHINE LEARNING MODELS
# ============================================================

models = {

    "Linear Regression":
        LinearRegression(),

    "Decision Tree":
        DecisionTreeRegressor(
            random_state=42,
            max_depth=8
        ),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            max_depth=10
        ),

    "Gradient Boosting":
        GradientBoostingRegressor(
            n_estimators=150,
            random_state=42
        )
}


# ============================================================
# 10. TRAIN AND EVALUATE MODELS
# ============================================================

results = []

trained_models = {}


for name, model in models.items():

    print("\n" + "=" * 50)
    print("Training:", name)
    print("=" * 50)

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    # Train model
    pipeline.fit(
        X_train,
        y_train
    )

    # Predict
    predictions = pipeline.predict(
        X_test
    )

    # --------------------------------------------------------
    # MAE
    # --------------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    # --------------------------------------------------------
    # RMSE
    # --------------------------------------------------------

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    # --------------------------------------------------------
    # R2 Score
    # --------------------------------------------------------

    r2 = r2_score(
        y_test,
        predictions
    )

    # --------------------------------------------------------
    # MAPE
    # --------------------------------------------------------

    non_zero = y_test != 0

    mape = np.mean(
        np.abs(
            (
                y_test[non_zero]
                -
                predictions[non_zero]
            )
            /
            y_test[non_zero]
        )
    ) * 100

    # --------------------------------------------------------
    # Store Results
    # --------------------------------------------------------

    results.append({

        "Model": name,

        "MAE": mae,

        "RMSE": rmse,

        "MAPE": mape,

        "R2": r2

    })

    trained_models[name] = pipeline

    print("MAE  :", round(mae, 3))
    print("RMSE :", round(rmse, 3))
    print("MAPE :", round(mape, 3), "%")
    print("R2   :", round(r2, 3))


# ============================================================
# 11. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results
)


# Lower MAE is better
results_df = results_df.sort_values(
    by="MAE",
    ascending=True
)


print("\n")
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 12. SAVE MODEL COMPARISON
# ============================================================

results_df.to_csv(
    "outputs/model_comparison.csv",
    index=False
)

print(
    "\nModel comparison saved to:"
    " outputs/model_comparison.csv"
)


# ============================================================
# 13. SELECT BEST MODEL
# ============================================================

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[
    best_model_name
]


print("\n")
print("=" * 50)
print("BEST MODEL")
print("=" * 50)

print(
    "Best Model:",
    best_model_name
)


# ============================================================
# 14. SAVE BEST MODEL
# ============================================================

joblib.dump(
    best_model,
    "models/attendance_prediction_model.pkl"
)

print(
    "\nBest model saved successfully!"
)

print(
    "Location:"
    " models/attendance_prediction_model.pkl"
)


# ============================================================
# 15. FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 70)
print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 70)