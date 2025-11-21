// Auto-generated configuration file
// Generated from Python preprocessing

#ifndef CONFIG_H
#define CONFIG_H

// Model configuration
#define NUM_CLASSES 4
#define NUM_FEATURES 6
#define WINDOW_SIZE 128
#define SAMPLING_RATE 50

// Normalization parameters (from training data)
const float SENSOR_MEAN[NUM_FEATURES] = {
    0.293887f, -0.253967f, -0.588385f, -0.724846f, 1.659784f, 0.211650f
};

const float SENSOR_STD[NUM_FEATURES] = {
    0.424775f, 0.343340f, 0.428024f, 37.473249f, 16.065540f, 11.153075f
};

// Quantization parameters (from TFLite model)
const float INPUT_SCALE = 0.10651991f;
const int8_t INPUT_ZERO_POINT = 21;
const float OUTPUT_SCALE = 0.00390625f;
const int8_t OUTPUT_ZERO_POINT = -128;

// Activity labels
const char* ACTIVITY_LABELS[NUM_CLASSES] = {
    "nail_biting", "knuckles_cracking", "hand_tapping", "sitting"
};

// Sensor column order (for reference)
// 0: Ax_w, 1: Ay_w, 2: Az_w, 3: Gx_w, 4: Gy_w, 5: Gz_w

#endif // CONFIG_H
