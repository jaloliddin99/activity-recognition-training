// Auto-generated configuration file
// Generated from Python preprocessing

#ifndef CONFIG_H
#define CONFIG_H

// Model configuration
#define NUM_CLASSES 11
#define NUM_FEATURES 6
#define WINDOW_SIZE 128
#define SAMPLING_RATE 50

// Normalization parameters (from training data)
const float SENSOR_MEAN[NUM_FEATURES] = {
    0.224560f, 0.060412f, -0.458404f, -0.945975f, 1.756266f, 0.140541f
};

const float SENSOR_STD[NUM_FEATURES] = {
    0.580899f, 0.498843f, 0.391174f, 49.164834f, 23.811443f, 20.875010f
};

// Activity labels
const char* ACTIVITY_LABELS[NUM_CLASSES] = {
    "knuckles_cracking", "hand_tapping", "sitting", "standing", "smoking", "ear_rubbing", "hand_scratching", "nail_biting", "hair_pulling", "forehead_rubbing", "nape_rubbing"
};

// Sensor column order (for reference)
// 0: Ax, 1: Ay, 2: Az, 3: Gx, 4: Gy, 5: Gz

#endif // CONFIG_H
