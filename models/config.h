#ifndef CONFIG_H
#define CONFIG_H

// Activity Recognition v5 — 10 activities
// No per-user normalization — global stats applied directly on raw sensor data

#define NUM_CLASSES 10
#define NUM_RAW_FEATURES 6
#define NUM_FEATURES 8  // 6 raw + 2 magnitude
#define WINDOW_SIZE 128
#define SAMPLING_RATE 50

// Feature order: Ax_w, Ay_w, Az_w, Gx_w, Gy_w, Gz_w, Acc_mag, Gyro_mag
// Acc_mag = sqrt(Ax^2 + Ay^2 + Az^2), Gyro_mag = sqrt(Gx^2 + Gy^2 + Gz^2)
// Normalize: (value - MEAN) / STD, then quantize to INT8
const float SENSOR_MEAN[8] = {0.196717f, 0.093949f, -0.451440f, -0.930565f, 1.677287f, 0.128417f, 0.996089f, 29.968271f};
const float SENSOR_STD[8] = {0.595617f, 0.485070f, 0.393003f, 42.673264f, 20.773724f, 19.017673f, 0.083101f, 41.498489f};

const char* ACTIVITY_LABELS[10] = {
    "ear_rubbing",
    "forehead_rubbing",
    "hair_pulling",
    "hand_tapping",
    "knuckles_cracking",
    "nail_biting",
    "nape_rubbing",
    "sitting",
    "smoking",
    "standing",
};

#endif
