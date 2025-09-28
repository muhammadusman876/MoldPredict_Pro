#include "DHT.h"

// === CONFIGURATION ===
#define DHTPIN 4       // GPIO pin where DHT22 DATA is connected
#define DHTTYPE DHT22  // DHT 22 (AM2302)
DHT dht(DHTPIN, DHTTYPE);

void setup() {
  Serial.begin(115200);
  delay(1000); // wait for serial
  Serial.println("timestamp,temperature,humidity"); // CSV header
  dht.begin();
}

void loop() {
  float h = dht.readHumidity();
  float t = dht.readTemperature(); // Celsius

  // Check if any reads failed
  if (isnan(h) || isnan(t)) {
    Serial.println("Failed to read from DHT sensor!");
  } else {
    // Print CSV-style: timestamp in  temp and  humidity
    Serial.print(t);
    Serial.print(",");
    Serial.println(h);
  }

  delay(2000); // Read every 2 seconds
}
