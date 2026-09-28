import argparse
import json
import pickle

import numpy as np

from config import (
    DEFAULT_OVERLAP,
    DEFAULT_WINDOW_SIZE,
    PROCESSED_DIR,
    RAW_DATA_PATH,
    SAMPLING_RATE,
    SENSOR_PROFILES,
)
from data_utils import (
    add_magnitude_features,
    choose_test_user,
    load_dataset,
    make_windows,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sensor-profile",
        choices=sorted(SENSOR_PROFILES),
        default="ag",
        help="'ag' matches the deployment IMU; 'agm' uses all dataset sensors.",
    )
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP)
    parser.add_argument(
        "--test-user",
        default="auto",
        help="User ID to hold out, or 'auto' to choose a complete user.",
    )
    parser.add_argument(
        "--validation-user",
        default="auto",
        help="Second user held out for validation, or 'auto'.",
    )
    parser.add_argument(
        "--add-magnitude",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.window_size < 2:
        raise ValueError("--window-size must be at least 2")
    if not 0 <= args.overlap < 1:
        raise ValueError("--overlap must be in [0, 1)")

    sensor_names = SENSOR_PROFILES[args.sensor_profile]
    frame, raw_feature_columns, resolved = load_dataset(
        RAW_DATA_PATH, sensor_names
    )

    if args.add_magnitude:
        frame, feature_columns = add_magnitude_features(frame, sensor_names)
    else:
        feature_columns = raw_feature_columns

    test_user = choose_test_user(frame, args.test_user)
    validation_candidates = frame[frame["user"] != test_user]
    validation_user = choose_test_user(
        validation_candidates, args.validation_user
    )
    activities = sorted(frame["activity"].unique().tolist())
    activity_to_label = {
        activity: index for index, activity in enumerate(activities)
    }
    label_to_activity = {
        index: activity for activity, index in activity_to_label.items()
    }

    train_frame = frame[
        ~frame["user"].isin([test_user, validation_user])
    ].copy()
    validation_frame = frame[frame["user"] == validation_user].copy()
    test_frame = frame[frame["user"] == test_user].copy()
    for role, user, held_out_frame in [
        ("Validation", validation_user, validation_frame),
        ("Test", test_user, test_frame),
    ]:
        if held_out_frame["activity"].nunique() != len(activities):
            missing = sorted(
                set(activities) - set(held_out_frame["activity"])
            )
            raise ValueError(
                f"{role} user {user} is missing activities: {missing}. "
                "Choose a user listed by inspect_dataset.py."
            )

    mean = train_frame[feature_columns].mean().to_numpy(dtype=np.float64)
    std = train_frame[feature_columns].std().to_numpy(dtype=np.float64)
    std[std < 1e-8] = 1.0

    train_frame[feature_columns] = (
        train_frame[feature_columns] - mean
    ) / std
    test_frame[feature_columns] = (
        test_frame[feature_columns] - mean
    ) / std
    validation_frame[feature_columns] = (
        validation_frame[feature_columns] - mean
    ) / std

    step_size = max(1, int(round(args.window_size * (1 - args.overlap))))
    X_train, y_train, train_users = make_windows(
        train_frame,
        feature_columns,
        activity_to_label,
        args.window_size,
        step_size,
    )
    X_test, y_test, test_users = make_windows(
        test_frame,
        feature_columns,
        activity_to_label,
        args.window_size,
        step_size,
    )
    X_validation, y_validation, validation_users = make_windows(
        validation_frame,
        feature_columns,
        activity_to_label,
        args.window_size,
        step_size,
    )

    if not len(X_train) or not len(X_validation) or not len(X_test):
        raise ValueError(
            "No windows were generated. Check the window size and dataset."
        )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PROCESSED_DIR / "X_train.npy", X_train)
    np.save(PROCESSED_DIR / "y_train.npy", y_train)
    np.save(PROCESSED_DIR / "train_users.npy", train_users)
    np.save(PROCESSED_DIR / "X_validation.npy", X_validation)
    np.save(PROCESSED_DIR / "y_validation.npy", y_validation)
    np.save(PROCESSED_DIR / "validation_users.npy", validation_users)
    np.save(PROCESSED_DIR / "X_test.npy", X_test)
    np.save(PROCESSED_DIR / "y_test.npy", y_test)
    np.save(PROCESSED_DIR / "test_users.npy", test_users)

    config = {
        "dataset": "StresSense",
        "dataset_doi": "10.17632/2dn3hpbm5m.1",
        "sensor_profile": args.sensor_profile,
        "sensor_names": sensor_names,
        "feature_columns": feature_columns,
        "num_features": len(feature_columns),
        "sampling_rate": SAMPLING_RATE,
        "window_size": args.window_size,
        "overlap": args.overlap,
        "step_size": step_size,
        "test_user": test_user,
        "validation_user": validation_user,
        "activities": activities,
        "resolved_columns": resolved,
    }
    info = {
        "config": config,
        "mean": mean,
        "std": std,
        "activity_to_label": activity_to_label,
        "label_to_activity": label_to_activity,
        "train_shape": X_train.shape,
        "validation_shape": X_validation.shape,
        "test_shape": X_test.shape,
    }
    with open(PROCESSED_DIR / "preprocessing_info.pkl", "wb") as handle:
        pickle.dump(info, handle)
    with open(PROCESSED_DIR / "preprocessing_summary.json", "w") as handle:
        json.dump(
            {
                **config,
                "train_shape": list(X_train.shape),
                "validation_shape": list(X_validation.shape),
                "test_shape": list(X_test.shape),
                "train_windows_per_class": {
                    label_to_activity[label]: int(np.sum(y_train == label))
                    for label in label_to_activity
                },
                "test_windows_per_class": {
                    label_to_activity[label]: int(np.sum(y_test == label))
                    for label in label_to_activity
                },
                "validation_windows_per_class": {
                    label_to_activity[label]: int(
                        np.sum(y_validation == label)
                    )
                    for label in label_to_activity
                },
            },
            handle,
            indent=2,
        )

    print("=" * 70)
    print("STRESSSENSE PREPROCESSING COMPLETE")
    print("=" * 70)
    print(f"Sensor profile: {args.sensor_profile}")
    print(f"Features: {feature_columns}")
    print(f"Validation user: {validation_user}")
    print(f"Test user: {test_user}")
    print(f"Training: {X_train.shape}")
    print(f"Validation: {X_validation.shape}")
    print(f"Test: {X_test.shape}")
    print(f"Window: {args.window_size} samples")
    print(f"Overlap: {args.overlap:.0%}")


if __name__ == "__main__":
    main()
