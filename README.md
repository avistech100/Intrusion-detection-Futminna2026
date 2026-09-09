---

# BioAlert — Human/Animal Intrusion Detection System

An edge-based intrusion detection system that uses a custom-trained 
YOLOv8n-cls model to classify detected motion as human, animal, or other, 
triggering real-time alerts via Telegram and SMS.

---

## Project Overview

BioAlert is a final year project developed at the Federal University of 
Technology, Minna. It combines embedded systems hardware with a machine 
learning classification pipeline to deliver real-world intrusion detection.

**Achieved accuracy:** 99.7% top-1 classification accuracy

---

## Hardware Components

| Component       | Role                                      |
|-----------------|-------------------------------------------|
| ESP32-CAM       | Image capture and Wi-Fi communication     |
| HC-SR501 PIR    | Motion trigger (GPIO13)                   |
| SIM800L GSM     | Bulk SMS alerts (GPIO14/15 UART2)         |
| 18650 Batteries | Dual-cell power supply                    |
| MT3608 / LM2596 | Voltage regulation                        |

---

## Repository Structure

```
BioAlert/
├── server.py            # Main Flask classification server
├── server2.py           # Secondary server variant
├── settings.json        # Server configuration
├── Procfile             # Cloud deployment config (Render / Hugging Face)
├── requirements.txt     # Python dependencies
│
├── firmware/
│   └── bioalert_esp32.ino   # Arduino firmware for ESP32-CAM
│
├── training/
│   ├── setup.py             # Google Colab: mount Drive & extract dataset
│   └── train.py             # Google Colab: train / resume YOLOv8n-cls
│
├── model/
│   └── weights/
│       ├── best.pt          # Final deployed model weights (99.7% top-1)
│       └── last.pt          # Last checkpoint
│
├── models/                  # Supporting model assets
│
└── docs/
    └── git_commit_info.txt  # Development commit history
```

> **Note:** The training dataset (`human_animal_dataset_v2`, ~7,050 images) 
> and intermediate training runs are not included in this repository due to 
> size. They are archived separately on departmental storage media.

---

## How to Run the Server

```bash
pip install -r requirements.txt
python server.py
```

---

## Training (Google Colab)

Open Google Colab, upload `training/setup.py` and `training/train.py`, 
then run them in order:

1. **setup.py** — mounts Google Drive and extracts the dataset
2. **train.py** — starts or resumes YOLOv8n-cls training automatically

---

## Model Details

- **Architecture:** YOLOv8n-cls (classification head)
- **Classes:** human · animal · other
- **Training images:** 7,050 (2,350 per class)
- **Epochs:** 50
- **Final top-1 accuracy:** 99.7%
- **Training loss:** 0.006

---

## Author

**Hassan Adewale Abdulmalik**  
Computer Science — Federal University of Technology, Minna  
Final Year Project, 2025/2026 Academic Session

---
