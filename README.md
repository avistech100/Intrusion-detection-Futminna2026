---

# BioAlert — Smart Human & Animal Intrusion Detection System

**BioAlert** is a smart security system designed to tell the difference between a human, an animal, or other moving objects. When someone (or something) enters a restricted area, the system captures a photo, analyzes it using Artificial Intelligence, and instantly sends an alert to your phone via SMS and Telegram.

---

## 💡 How It Works

The system operates in a few simple steps:
1. **Motion Detection:** A PIR motion sensor detects movement in the area.
2. **Image Capture:** The ESP32-CAM wakes up and takes a picture of whatever triggered the sensor.
3. **AI Analysis:** The picture is sent to our server where a trained AI model (YOLOv8) looks at it to figure out if it's a `human`, an `animal`, or `other`.
4. **Instant Alerts:** If a human or animal is detected, the SIM800L module sends an SMS text message, and the system sends a Telegram message with the alert details.

---

## 🛠️ Hardware Components

Here is what you need to build the physical device:

| Component | What it does |
| :--- | :--- |
| **ESP32-CAM** | The "eyes" and "brain" of the device. It takes pictures and connects to Wi-Fi. |
| **HC-SR501 PIR** | The motion sensor. It tells the camera when to wake up and look. |
| **SIM800L GSM** | The cellular module. It sends traditional SMS text messages to your phone. |
| **18650 Batteries** | Rechargeable batteries to power the system so it can work completely wireless. |
| **Voltage Regulators**| Keeps the power steady so the electronics don't get damaged. |

---

## 📂 What's in this Folder?

```text
BioAlert/
├── server.py            # The main program that receives images and runs the AI
├── server2.py           # An alternative version of the main program
├── settings.json        # Where you put your Wi-Fi and Telegram settings
├── Procfile             # Settings for putting the server on the internet (cloud)
├── requirements.txt     # A list of Python add-ons needed to run the server
│
├── firmware/
│   └── bioalert_esp32.ino   # The code that goes onto the physical ESP32 camera
│
├── training/
│   ├── setup.py             # Script to prepare data in Google Colab
│   └── train.py             # Script to teach the AI on Google Colab
│
├── model/
│   └── weights/
│       └── best.pt          # The "brain" of the AI after learning (99.7% accurate!)
│
└── docs/
    └── git_commit_info.txt  # History of changes made to the project
```

> **Note:** The thousands of images used to teach the AI (the dataset) are too large to keep here, so they are saved securely on a separate flash drive.

---

## 🚀 How to Run the Server

If you want to run the AI server on your computer, follow these simple steps:

1. **Install required software:** Open your terminal or command prompt and type:
   ```bash
   pip install -r requirements.txt
   ```
2. **Start the server:** 
   ```bash
   python server.py
   ```
   *Make sure you have your `settings.json` filled out correctly before starting!*

---

## 🧠 Teaching the AI (Training)

If you want to re-train the AI or see how it learned, we use **Google Colab** (a free online tool by Google).

1. Open Google Colab and upload the files from the `training/` folder.
2. Run `setup.py` to connect your Google Drive and extract the pictures.
3. Run `train.py` to start teaching the AI. It will learn from 7,050 pictures of humans, animals, and empty backgrounds.

**AI Performance:** After 50 rounds of learning, our AI correctly identifies humans and animals **99.7% of the time**.

---

## 👨‍🎓 About the Project

**BioAlert** was developed by **Hassan Adewale Abdulmalik** as a Final Year Project for the Computer Science department at the **Federal University of Technology, Minna** (2025/2026 Academic Session).

---
