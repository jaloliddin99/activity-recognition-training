// Auto-generated configuration file
// Generated from Python preprocessing

#ifndef CONFIG_H
#define CONFIG_H

// Model configuration
#define NUM_CLASSES 8
#define NUM_FEATURES 8
#define WINDOW_SIZE 600
#define SAMPLING_RATE 50

// Normalization parameters (from training data)
const float SENSOR_MEAN[NUM_FEATURES] = {
    0.108117f, 0.055653f, -0.473147f, -0.949495f, 1.627686f, 0.015654f, 0.993834f, 32.821249f
};

const float SENSOR_STD[NUM_FEATURES] = {
    0.598558f, 0.486906f, 0.409350f, 48.973817f, 24.061215f, 21.929622f, 0.117748f, 48.832386f
};

// Activity labels
const char* ACTIVITY_LABELS[NUM_CLASSES] = {
    "knuckles_cracking", "hand_tapping", "sitting", "standing", "smoking", "nail_biting", "hair_pulling", "nape_rubbing"
};

// Sensor column order (for reference)
// 0: Ax, 1: Ay, 2: Az, 3: Gx, 4: Gy, 5: Gz, 6: Acc_mag, 7: Gyro_mag

#endif // CONFIG_H
