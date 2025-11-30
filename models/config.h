// Auto-generated configuration file
// Generated from Python preprocessing

#ifndef CONFIG_H
#define CONFIG_H

// Model configuration
#define NUM_CLASSES 5
#define NUM_FEATURES 6
#define WINDOW_SIZE 128
#define SAMPLING_RATE 50

// Normalization parameters (from training data)
const float SENSOR_MEAN[NUM_FEATURES] = {
    -1.125685f, -2.306040f, 3.295036f, -0.000627f, -0.008586f, -0.031883f
};

const float SENSOR_STD[NUM_FEATURES] = {
    7.436702f, 3.355736f, 3.385428f, 0.445237f, 0.271540f, 0.503735f
};

// Quantization parameters (from TFLite model)
const float INPUT_SCALE = 0.12539911f;
const int8_t INPUT_ZERO_POINT = 8;
const float OUTPUT_SCALE = 0.00390625f;
const int8_t OUTPUT_ZERO_POINT = -128;

// Activity labels
const char* ACTIVITY_LABELS[NUM_CLASSES] = {
    "Eating", "Nail_Biting", "Face_Touch", "Smoking", "Staying_Still"
};

// Sensor column order (for reference)
// 0: Ax, 1: Ay, 2: Az, 3: Gx, 4: Gy, 5: Gz

#endif // CONFIG_H
