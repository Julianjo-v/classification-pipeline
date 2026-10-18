
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_TSL2591.h>
#include <math.h>
#include "model.h"

// ======================================
// PIN CONFIGURATION
// ======================================

#define SDA_PIN 21
#define SCL_PIN 22
#define UV_PIN 34

// ======================================
// SENSOR CONFIGURATION
// ======================================

Adafruit_TSL2591 tsl = Adafruit_TSL2591(2591);

// Match original data collection
const float ALPHA = 0.2f;
const unsigned long SAMPLE_INTERVAL_MS = 100;

// EMA filter variables
float uvFiltered = -1.0f;
float irFiltered = -1.0f;

unsigned long lastSampleTime = 0;

// ======================================
// TRAINING SCALING BOUNDS
// ======================================
// IMPORTANT:
// Replace NAN with the actual values
// printed by your Python training pipeline.
//
// UV is scaled directly with Min-Max.
// IR is log1p-transformed BEFORE Min-Max.

const float UV_TRAIN_MIN = 0.0;
const float UV_TRAIN_MAX = 3653.97;

const float LOG_IR_TRAIN_MIN = 4.88507207112091;
const float LOG_IR_TRAIN_MAX = 10.370462224117665;

// ======================================
// EMA FILTER
// ======================================

void updateFilters(int rawUV, uint16_t rawIR) {

  // Initialize using first measurements
  if (uvFiltered < 0.0f) {

    uvFiltered = (float)rawUV;
    irFiltered = (float)rawIR;

  } else {

    uvFiltered =
      ALPHA * rawUV +
      (1.0f - ALPHA) * uvFiltered;

    irFiltered =
      ALPHA * rawIR +
      (1.0f - ALPHA) * irFiltered;
  }
}

// ======================================
// MIN-MAX NORMALIZATION
// ======================================

float scaleFeature(float value,
                   float minimum,
                   float maximum) {

  float scaled =
    (value - minimum) / (maximum - minimum);

  return constrain(scaled, 0.0f, 1.0f);
}

// ======================================
// KDE CLASSIFICATION
// ======================================

// X1 = UV
// X2 = log(IR)

int classifyLight(float x1, float x2,
                  uint8_t &bestScore) {

  // Match Python cell_of():
  // int(value * GRID_N), clipped to 0-15

  int col = constrain(
    (int)(x1 * MODEL_GRID_SIZE),
    0, MODEL_GRID_SIZE - 1
  );

  int row = constrain(
    (int)(x2 * MODEL_GRID_SIZE),
    0, MODEL_GRID_SIZE - 1
  );

  int bestClass = -1;
  bestScore = 0;

  // Compare all KDE density maps
  for (int c = 0; c < MODEL_NUM_CLASSES; c++) {

    uint8_t score = modelRead(c, row, col);

    Serial.print(MODEL_CLASS_IDS[c]);
    Serial.print(" Score: ");
    Serial.println(score);

    if (score > bestScore) {

      bestScore = score;
      bestClass = c;
    }
  }

  return bestClass;
}

// ======================================
// SETUP
// ======================================

void setup() {

  Serial.begin(115200);
  Serial.println("=== MODEL TEST ===");

for (int c = 0; c < MODEL_NUM_CLASSES; c++) {
  Serial.print(MODEL_CLASS_IDS[c]);
  Serial.print(" peak score: ");

  uint8_t peak = 0;

  for (int row = 0; row < MODEL_GRID_SIZE; row++) {
    for (int col = 0; col < MODEL_GRID_SIZE; col++) {
      uint8_t score = modelRead(c, row, col);
      if (score > peak) peak = score;
    }
  }

  Serial.println(peak);
}
  delay(1000);

  // Configure UV ADC
  analogReadResolution(12);
  analogSetPinAttenuation(UV_PIN, ADC_11db);

  // Initialize I2C
  Wire.begin(SDA_PIN, SCL_PIN);

  if (!tsl.begin()) {

    Serial.println("TSL2591 NOT DETECTED!");

    while (true) {
      delay(1000);
    }
  }

  // Match data collection settings
  tsl.setGain(TSL2591_GAIN_MED);
  tsl.setTiming(TSL2591_INTEGRATIONTIME_100MS);

  Serial.println("IR Sensor Ready!");
  Serial.println("UV Sensor Ready!");
  Serial.println("KDE Model Ready!");

  // Prevent classification with incorrect bounds
  if (!isfinite(UV_TRAIN_MIN) ||
      !isfinite(UV_TRAIN_MAX) ||
      !isfinite(LOG_IR_TRAIN_MIN) ||
      !isfinite(LOG_IR_TRAIN_MAX) ||
      UV_TRAIN_MAX <= UV_TRAIN_MIN ||
      LOG_IR_TRAIN_MAX <= LOG_IR_TRAIN_MIN) {

    Serial.println();
    Serial.println("ERROR: Training bounds missing!");
    Serial.println("Enter the four Python bounds");
    Serial.println("before running classification.");

    while (true) {
      delay(1000);
    }
  }
}

// ======================================
// MAIN LOOP
// ======================================

void loop() {

  unsigned long now = millis();

  if (now - lastSampleTime < SAMPLE_INTERVAL_MS) {
    return;
  }

  lastSampleTime = now;

  // ====================================
  // READ UV SENSOR
  // ====================================

  int rawUV = analogRead(UV_PIN);

  // ====================================
  // READ IR SENSOR
  // ====================================

  uint32_t luminosity = tsl.getFullLuminosity();

  uint16_t full = luminosity & 0xFFFF;
  uint16_t rawIR = luminosity >> 16;

  // Check for saturation
  if (full == 0xFFFF || rawIR == 0xFFFF) {

    Serial.println("WARNING: TSL2591 saturated.");
    return;
  }

  // ====================================
  // APPLY EMA FILTERING
  // ====================================

  updateFilters(rawUV, rawIR);

  // ====================================
  // MATCH PYTHON PREPROCESSING
  // ====================================

  // X1: UV Min-Max normalization
  float x1 = scaleFeature(
    uvFiltered,
    UV_TRAIN_MIN,
    UV_TRAIN_MAX
  );

  // X2: Log transform IR first
  float logIR = log1pf(irFiltered);

  // Then Min-Max normalize log(IR)
  float x2 = scaleFeature(
    logIR,
    LOG_IR_TRAIN_MIN,
    LOG_IR_TRAIN_MAX
  );

  // ====================================
  // RUN KDE CLASSIFICATION
  // ====================================

  uint8_t bestScore = 0;

  int predicted = classifyLight(
    x1,
    x2,
    bestScore
  );


  // ====================================
  // CLEAN SINGLE-LINE SERIAL OUTPUT
  // ====================================

  Serial.print("UV: ");
  Serial.print(rawUV);

  Serial.print(" | IR: ");
  Serial.print(rawIR);

  Serial.print(" | EMA UV: ");
  Serial.print(uvFiltered, 2);

  Serial.print(" | EMA IR: ");
  Serial.print(irFiltered, 2);

  Serial.print(" | X1: ");
  Serial.print(x1, 3);

  Serial.print(" | X2: ");
  Serial.print(x2, 3);

  Serial.print(" | Class: ");

  if (predicted >= 0 &&
      predicted < MODEL_NUM_CLASSES) {
    Serial.print(MODEL_CLASS_IDS[predicted]);
  } else {
    Serial.print("UNKNOWN");
  }

  Serial.print(" | Score: ");
  Serial.println(bestScore);

}