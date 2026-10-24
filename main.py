"""Student Performance Analytics System
Reads student data from students.csv and analyses it with Pandas and NumPy.
"""

import numpy as np
import pandas as pd

SUBJECTS = ["Maths", "Physics", "Chemistry", "English", "Computer"]
PASS_MARK = 35  # a student must score at least this in EVERY subject to pass


# ---------- Reusable functions ----------

def load_data(path):
    """Read the CSV file into a DataFrame."""
    return pd.read_csv(path)


def calculate_totals(df):
    """Add Total and Average columns using NumPy."""
    marks = df[SUBJECTS].to_numpy()               # DataFrame -> NumPy array
    df["Total"] = np.sum(marks, axis=1)           # sum across each row
    df["Average"] = np.round(np.mean(marks, axis=1), 2)
    return df


def assign_grade(average):
    """Grading rule based on average marks."""
    if average >= 90:
        return "A+"
    elif average >= 80:
        return "A"
    elif average >= 70:
        return "B"
    elif average >= 60:
        return "C"
    elif average >= 50:
        return "D"
    elif average >= 40:
        return "E"
    else:
        return "F"


def check_result(row):
    """Pass only if every subject mark is >= PASS_MARK."""
    for subject in SUBJECTS:                      # loop over subjects
        if row[subject] < PASS_MARK:
            return "Fail"
    return "Pass"


def subject_analysis(df):
    """Print average, highest and lowest marks for each subject."""
    print(f"{'Subject':<12}{'Average':>9}{'Highest':>9}{'Lowest':>9}")
    for subject in SUBJECTS:
        col = df[subject].to_numpy()
        print(f"{subject:<12}{np.mean(col):>9.2f}{np.max(col):>9}{np.min(col):>9}")


def print_header(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ---------- Main program ----------

def main():
    df = load_data("students.csv")

    print_header("1. DATASET OVERVIEW")
    print("Number of records:", len(df))
    print("Columns:", list(df.columns))
    print(df.head())

    # Processing
    df = calculate_totals(df)
    df["Grade"] = df["Average"].apply(assign_grade)
    df["Result"] = df.apply(check_result, axis=1)

    print_header("2. STUDENT RESULTS")
    print(df[["StudentID", "Name", "Total", "Average", "Grade", "Result"]]
          .to_string(index=False))

    print_header("3. PASS / FAIL STATISTICS")
    passed = (df["Result"] == "Pass").sum()
    failed = (df["Result"] == "Fail").sum()
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Pass percentage: {passed / len(df) * 100:.1f}%")
    print("\nGrade distribution:")
    print(df["Grade"].value_counts().sort_index().to_string())

    print_header("4. HIGHEST, LOWEST AND AVERAGE PERFORMANCE")
    best = df.loc[df["Average"].idxmax()]
    worst = df.loc[df["Average"].idxmin()]
    print(f"Highest average: {best['Name']} ({best['Average']})")
    print(f"Lowest average : {worst['Name']} ({worst['Average']})")
    print(f"Overall class average: {np.mean(df['Average']):.2f}")

    print_header("5. SUBJECT-WISE ANALYSIS")
    subject_analysis(df)
    subject_means = df[SUBJECTS].mean()
    print(f"\nBest subject (highest average): {subject_means.idxmax()}")
    print(f"Weakest subject: {subject_means.idxmin()}")

    print_header("6. TOP 5 PERFORMERS")
    top5 = df.sort_values("Average", ascending=False).head(5)
    rank = 1
    for _, student in top5.iterrows():            # loop to print ranks
        print(f"{rank}. {student['Name']} ({student['Department']}) "
              f"- Average {student['Average']}, Grade {student['Grade']}")
        rank += 1

    print_header("7. DEPARTMENT-WISE AVERAGE")
    print(df.groupby("Department")["Average"].mean().round(2).to_string())

    print_header("8. LOW ATTENDANCE (below 75%)")
    low = df[df["Attendance"] < 75]
    print(low[["Name", "Attendance", "Result"]].to_string(index=False))


if __name__ == "__main__":
    main()
