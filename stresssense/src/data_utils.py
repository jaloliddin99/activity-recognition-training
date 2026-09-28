import re

import numpy as np
import pandas as pd

from config import AXIS_COLUMNS, COLUMN_ALIASES


def canonical_name(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def resolve_columns(frame, required):
    available = {canonical_name(column): column for column in frame.columns}
    resolved = {}

    for target in required:
        aliases = [target, *COLUMN_ALIASES.get(target, [])]
        source = next(
            (available[canonical_name(alias)] for alias in aliases
             if canonical_name(alias) in available),
            None,
        )
        if source is None:
            raise ValueError(
                f"Could not find column for {target!r}. "
                f"Available columns: {list(frame.columns)}"
            )
        resolved[target] = source

    return resolved


def load_dataset(path, sensor_names):
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Download StresSense.csv from "
            "https://data.mendeley.com/datasets/2dn3hpbm5m/1"
        )

    required = ["user", "activity"]
    for sensor_name in sensor_names:
        required.extend(AXIS_COLUMNS[sensor_name])

    frame = pd.read_csv(path)
    resolved = resolve_columns(frame, required)
    selected = frame[[resolved[name] for name in required]].copy()
    selected.columns = required

    selected["activity"] = (
        selected["activity"].astype(str).str.strip().str.lower()
        .str.replace(r"[\s-]+", "_", regex=True)
    )
    selected["user"] = selected["user"].astype(str).str.strip()

    feature_columns = required[2:]
    for column in feature_columns:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")

    # The paper fills interior gaps using nearby observations. Linear
    # interpolation within each user/activity segment provides the same intent
    # without allowing information to leak across segments.
    selected[feature_columns] = (
        selected.groupby(["user", "activity"], sort=False)[feature_columns]
        .transform(lambda values: values.interpolate(limit_direction="both"))
    )
    selected = selected.dropna(subset=feature_columns)
    return selected, feature_columns, resolved


def add_magnitude_features(frame, sensor_names):
    feature_columns = []
    for sensor_name in sensor_names:
        axes = AXIS_COLUMNS[sensor_name]
        feature_columns.extend(axes)
        magnitude_name = f"{sensor_name}_magnitude"
        frame[magnitude_name] = np.sqrt(
            sum(frame[axis].astype(np.float64) ** 2 for axis in axes)
        )
        feature_columns.append(magnitude_name)
    return frame, feature_columns


def users_with_all_activities(frame):
    activity_count = frame["activity"].nunique()
    counts = frame.groupby("user")["activity"].nunique()
    return sorted(counts[counts == activity_count].index.tolist())


def choose_test_user(frame, requested):
    users = sorted(frame["user"].unique().tolist())
    complete_users = users_with_all_activities(frame)

    if requested != "auto":
        requested = str(requested)
        if requested not in users:
            raise ValueError(f"Unknown test user {requested!r}. Users: {users}")
        return requested

    candidates = complete_users or users
    sizes = frame[frame["user"].isin(candidates)].groupby("user").size()
    return str(sizes.idxmax())


def make_windows(frame, feature_columns, activity_to_label, window_size, step_size):
    windows = []
    labels = []
    users = []

    for (user, activity), segment in frame.groupby(
        ["user", "activity"], sort=False
    ):
        values = segment[feature_columns].to_numpy(dtype=np.float32)
        if len(values) < window_size:
            continue

        label = activity_to_label[activity]
        for start in range(0, len(values) - window_size + 1, step_size):
            windows.append(values[start : start + window_size])
            labels.append(label)
            users.append(user)

    return (
        np.asarray(windows, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(users),
    )
