from config import RAW_DATA_PATH
from data_utils import load_dataset, users_with_all_activities


def main():
    frame, feature_columns, resolved = load_dataset(
        RAW_DATA_PATH,
        ["accelerometer", "gyroscope", "magnetometer"],
    )

    print(f"Dataset: {RAW_DATA_PATH}")
    print(f"Usable rows: {len(frame):,}")
    print(f"Users: {frame['user'].nunique()}")
    print(f"Activities: {frame['activity'].nunique()}")
    print(f"Features: {feature_columns}")
    print(f"Resolved columns: {resolved}")

    print("\nRows per activity:")
    print(frame["activity"].value_counts().to_string())

    print("\nUsers per activity:")
    print(frame.groupby("activity")["user"].nunique().to_string())

    print("\nUsers containing every activity:")
    complete = users_with_all_activities(frame)
    print(complete if complete else "None")

    print("\nRows per user/activity:")
    print(
        frame.groupby(["user", "activity"]).size()
        .unstack(fill_value=0)
        .to_string()
    )


if __name__ == "__main__":
    main()
