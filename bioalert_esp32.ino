#include "esp_camera.h"
#include <WiFi.h>
#include <ESPmDNS.h>
#include <HTTPClient.h>

const char* ssid      = "Galaxy S20 5G f003";
const char* password  = "nxwe58332";


const char* MDNS_SERVER_HOSTNAME = "DESKTOP-2H4S3PI";   // resolves "DESKTOP-2H4S3PI.local"
const int   SERVER_PORT          = 5000;
const char* SERVER_PATH          = "/classify";
const char* FALLBACK_SERVER_IP   = "192.168.65.240";     // Fallback static IP if mDNS fails (common on mobile hotspots)

// Cached after a successful mDNS lookup. Cleared (forcing a fresh
// lookup) whenever a request fails, in case the laptop's IP changed.
String resolvedServerIP = "";

#define PIR_PIN    13
#define BUZZER_PIN 12
#define LED_PIN    4

#define SIM800_TX  14
#define SIM800_RX  15
#define PHONE_NUMBER "+2347058660994"

HardwareSerial SIM800(2);

#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

bool cameraReady = false;

#define DELAY_AFTER_SUCCESS 10000
#define DELAY_AFTER_FAILURE 15000
#define WIFI_RETRY_INTERVAL 10000
#define MDNS_QUERY_TIMEOUT_MS 5000

// Local network, no cold-start concerns — 20s is plenty even with
// inference time included.
#define HTTP_TIMEOUT_MS 20000

// ── Blink flash twice to confirm WiFi connected ──────
void blinkWiFiReady() {
  for (int i = 0; i < 2; i++) {
    digitalWrite(LED_PIN, HIGH);
    delay(300);
    digitalWrite(LED_PIN, LOW);
    delay(300);
  }
}

void initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = Y2_GPIO_NUM;
  config.pin_d1       = Y3_GPIO_NUM;
  config.pin_d2       = Y4_GPIO_NUM;
  config.pin_d3       = Y5_GPIO_NUM;
  config.pin_d4       = Y6_GPIO_NUM;
  config.pin_d5       = Y7_GPIO_NUM;
  config.pin_d6       = Y8_GPIO_NUM;
  config.pin_d7       = Y9_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size   = FRAMESIZE_QVGA;
  config.jpeg_quality = 10;
  config.fb_count     = 1;
  config.grab_mode    = CAMERA_GRAB_WHEN_EMPTY;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    cameraReady = false;
    return;
  }
  sensor_t *s = esp_camera_sensor_get();
  if (s->id.PID == OV3660_PID) {
    s->set_vflip(s, 1);
    s->set_brightness(s, 1);
    s->set_saturation(s, -2);
  }
  cameraReady = true;
}

void ensureWiFi() {
  if (WiFi.status() == WL_CONNECTED) return;

  Serial.println("WiFi lost. Attempting reconnect...");
  WiFi.disconnect();
  WiFi.begin(ssid, password);

  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED) {
    if (millis() - start >= WIFI_RETRY_INTERVAL) {
      Serial.println("WiFi not available yet. Will retry on next cycle.");
      return;
    }
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi reconnected: " + WiFi.localIP().toString());
  blinkWiFiReady();

  // WiFi reconnecting can sometimes mean a new DHCP lease elsewhere on
  // the network too - force a fresh mDNS lookup just in case.
  resolvedServerIP = "";
}

// ── Looks up the Flask server's current IP via mDNS ──────
// Caches the result in resolvedServerIP on success. Falls back to static IP on failure.
bool resolveServerIP() {
  Serial.println("Resolving '" + String(MDNS_SERVER_HOSTNAME) + ".local' via mDNS...");
  IPAddress ip = MDNS.queryHost(MDNS_SERVER_HOSTNAME, MDNS_QUERY_TIMEOUT_MS);

  if (ip == IPAddress(0, 0, 0, 0)) {
    Serial.println("  -> mDNS lookup failed. Falling back to static IP: " + String(FALLBACK_SERVER_IP));
    
    // Blink LED once to indicate fallback is active
    digitalWrite(LED_PIN, HIGH);
    delay(500);
    digitalWrite(LED_PIN, LOW);
    
    resolvedServerIP = String(FALLBACK_SERVER_IP);
    return true;
  }

  resolvedServerIP = ip.toString();
  Serial.println("  -> Resolved to: " + resolvedServerIP);
  return true;
}

void sendSMS(String message) {
  SIM800.println("AT");
  delay(1000);
  SIM800.println("AT+CMGF=1");
  delay(1000);
  SIM800.print("AT+CMGS=\"");
  SIM800.print(PHONE_NUMBER);
  SIM800.println("\"");
  delay(1000);
  SIM800.print(message);
  delay(500);
  SIM800.write(26);
  delay(3000);
  Serial.println("SMS sent");
}

void beepHuman() {
  for (int i = 0; i < 3; i++) {
    digitalWrite(BUZZER_PIN, HIGH);
    delay(200);
    digitalWrite(BUZZER_PIN, LOW);
    delay(200);
  }
}

void beepAnimal() {
  digitalWrite(BUZZER_PIN, HIGH);
  delay(2000);
  digitalWrite(BUZZER_PIN, LOW);
}

bool captureAndClassify() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("No WiFi — skipping capture");
    return false;
  }

  if (!cameraReady) {
    Serial.println("Camera not ready");
    return false;
  }

  // Resolve via mDNS if we don't already have a cached IP (e.g. first
  // run, or a previous request failed and cleared the cache).
  if (resolvedServerIP == "") {
    if (!resolveServerIP()) {
      Serial.println("Server not found on network — skipping capture");
      return false;
    }
  }

  String targetURL = "http://" + resolvedServerIP + ":" + String(SERVER_PORT) + String(SERVER_PATH);

  digitalWrite(BUZZER_PIN, HIGH);
  delay(300);
  digitalWrite(BUZZER_PIN, LOW);

  // Flush the stale frame buffer to prevent lag/old images
  camera_fb_t *fb = esp_camera_fb_get();
  if (fb) esp_camera_fb_return(fb);

  // Capture the actual fresh frame
  fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Camera capture failed");
    return false;
  }

  HTTPClient http;
  http.begin(targetURL);
  http.addHeader("Content-Type", "image/jpeg");
  http.setTimeout(HTTP_TIMEOUT_MS);
  http.setConnectTimeout(HTTP_TIMEOUT_MS);

  Serial.println("Sending image to " + targetURL + " ...");
  int httpCode = http.POST(fb->buf, fb->len);
  esp_camera_fb_return(fb);

  if (httpCode != 200) {
    Serial.printf("Server error: %d\n", httpCode);
    http.end();
    // Could mean the server's IP changed since we last resolved it -
    // clear the cache so the next cycle re-resolves via mDNS.
    resolvedServerIP = "";
    return false;
  }

  String response = http.getString();
  http.end();
  Serial.println("Server response: " + response);

  if (response.indexOf("HUMAN") >= 0) {
    Serial.println("HUMAN detected");
    beepHuman();
    sendSMS("ALERT: Human intruder detected by your security system!");

  } else if (response.indexOf("ANIMAL") >= 0) {
    Serial.println("ANIMAL detected");
    beepAnimal();
    sendSMS("ALERT: Animal detected by your security system!");

  } else {
    Serial.println("UNKNOWN — no alert");
  }

  return true;
}

void setup() {
  Serial.begin(115200);

  pinMode(LED_PIN, OUTPUT);
  // Power-on flash
  digitalWrite(LED_PIN, HIGH);
  delay(1000);
  digitalWrite(LED_PIN, LOW);

  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);

  pinMode(PIR_PIN, INPUT);

  SIM800.begin(9600, SERIAL_8N1, SIM800_RX, SIM800_TX);
  delay(3000);

  initCamera();

  Serial.println("Connecting to WiFi...");
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    Serial.println("No WiFi yet — retrying in 10s...");
    delay(WIFI_RETRY_INTERVAL);
  }
  Serial.println("WiFi connected: " + WiFi.localIP().toString());
  blinkWiFiReady();

  // Start the ESP32's own mDNS responder - required before queryHost()
  // will work, even though we're only querying (not advertising) here.
  if (!MDNS.begin("esp32cam")) {
    Serial.println("Warning: failed to start mDNS responder on ESP32 (non-fatal)");
  }

  // Try to resolve the Flask server's hostname now, with a few retries
  // in case it hasn't fully started broadcasting yet.
  int attempts = 0;
  while (!resolveServerIP() && attempts < 5) {
    Serial.println("Retrying mDNS lookup in 2s...");
    delay(2000);
    attempts++;
  }

  Serial.println("System fully ready.");
}

void loop() {
  ensureWiFi();

  int motion = digitalRead(PIR_PIN);

  if (motion == HIGH) {
    Serial.println("Motion detected");

    bool success = captureAndClassify();

    if (success) {
      Serial.println("Cycle complete. Waiting 10s...");
      delay(DELAY_AFTER_SUCCESS);
    } else {
      Serial.println("Cycle failed. Waiting 15s...");
      delay(DELAY_AFTER_FAILURE);
    }
  }
}
