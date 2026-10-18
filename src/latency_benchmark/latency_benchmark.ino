#include <Arduino.h>
#include "../model.h"

constexpr uint32_t QUERY_COUNT = 10000;

uint32_t measuredTotalUs = 0;
uint32_t measuredChecksum = 0;

uint8_t classifyCell(uint8_t row, uint8_t col) {
  uint8_t bestClass = 0;
  uint8_t bestScore = modelRead(0, row, col);
  for (uint8_t classIndex = 1; classIndex < MODEL_NUM_CLASSES; ++classIndex) {
    const uint8_t score = modelRead(classIndex, row, col);
    if (score > bestScore) {
      bestScore = score;
      bestClass = classIndex;
    }
  }
  return bestClass;
}

void setup() {
  Serial.begin(115200);
  delay(2000);

  volatile uint32_t checksum = 0;
  const uint32_t started = micros();
  for (uint32_t query = 0; query < QUERY_COUNT; ++query) {
    const uint8_t row = query % MODEL_GRID_SIZE;
    const uint8_t col = (query / MODEL_GRID_SIZE) % MODEL_GRID_SIZE;
    checksum += classifyCell(row, col);
  }
  measuredTotalUs = micros() - started;
  measuredChecksum = checksum;

  Serial.println("Classification latency benchmark ready");
}

void loop() {
  // Repeat the saved result so it remains visible even when Serial Monitor
  // opens after setup() has finished. The benchmark itself is not rerun.
  Serial.printf("queries=%u,total_us=%u,us_per_query=%.4f,checksum=%u\n",
                QUERY_COUNT, measuredTotalUs,
                static_cast<double>(measuredTotalUs) / QUERY_COUNT,
                measuredChecksum);
  delay(2000);
}
