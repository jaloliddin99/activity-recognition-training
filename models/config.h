#ifndef CONFIG_H
#define CONFIG_H

// Activity Recognition v4 — 10 activities (hand_scratching removed)

#define NUM_CLASSES 10
#define NUM_RAW_FEATURES 6
#define NUM_FEATURES 8  // 6 raw + 2 magnitude
#define WINDOW_SIZE 128
#define SAMPLING_RATE 50

// Feature order: Ax_w, Ay_w, Az_w, Gx_w, Gy_w, Gz_w, Acc_mag, Gyro_mag
const float SENSOR_MEAN[8] = {0.000878f, 0.001624f, 0.000808f, 0.000055f, -0.000163f, 0.000406f, 1.641183f, 1.070511f};
const float SENSOR_STD[8] = {1.000287f, 1.000545f, 1.000286f, 0.999537f, 0.997873f, 0.996105f, 0.554387f, 1.357964f};

const float INPUT_SCALE = 0.10124809294939041f;
const int8_t INPUT_ZERO_POINT = 10;
const float OUTPUT_SCALE = 0.00390625f;
const int8_t OUTPUT_ZERO_POINT = -128;

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
