import json
import os
import secrets
import threading
from functools import wraps
from collections import deque
from datetime import datetime
import cv2
import numpy as np
import requests
from flask import (
    Flask,
    Response,
    jsonify,
    redirect,
    render_template_string,
    request,
    session,
    url_for,
)
from ultralytics import YOLO

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(24))

# Persistent configuration path
SETTINGS_FILE = "settings.json"

# --- Admin Panel Credentials ---
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "password123"

# --- Model path ---
# Locally this was a hardcoded Windows path. On Render, the model file ships
# inside the repo (in /models) and is referenced relatively, or via MODEL_PATH
# env var if you want to point somewhere else.
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join("models", "best.pt"))

# --- Default Initialization Parameters ---
DEFAULT_SETTINGS = {
    "system_name": "BIOALERT",
    "bot_token": "8386965667:AAHrSpRiP8SmrfvtzItUlwhsiQlznJQ5H8w",
    "chat_id": "7228697381",
    "telegram_enabled": True,
    "sms_api_key": "CCBC84zax3A3Ch7i9CC26Fx52oJ03b6rlCcm5HxCAA1BgCksAnBvIeAy2wdb1781450669",
    "sms_recipients": "07058660994,09057289989",
    "sms_sender_name": "BIOALERT",
    "sms_enabled_animals": True,
    "sms_enabled_humans": False,
}


def load_settings():
    if not os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "w") as f:
            json.dump(DEFAULT_SETTINGS, f, indent=4)
        return DEFAULT_SETTINGS
    try:
        with open(SETTINGS_FILE, "r") as f:
            data = json.load(f)
        updated = False
        for key, value in DEFAULT_SETTINGS.items():
            if key not in data:
                data[key] = value
                updated = True
        if updated:
            save_settings(data)
        return data
    except Exception as e:
        print(f"Error loading settings file, fallback to default: {e}")
        return DEFAULT_SETTINGS


def save_settings(data):
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(data, f, indent=4)
        return True
    except Exception as e:
        print(f"Error persisting settings: {e}")
        return False


# Load initial properties
config = load_settings()

# Initialize ML Engine
model = YOLO(MODEL_PATH)

latest_frame = None
frame_lock = threading.Lock()

# --- Live Activity Log ---
MAX_LOG_ENTRIES = 200
activity_log = deque(maxlen=MAX_LOG_ENTRIES)
log_lock = threading.Lock()


def log_event(event_type, message):
    """event_type: ESP | INFO | ALERT | SUCCESS | ERROR"""
    entry = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "type": event_type,
        "message": message,
    }
    with log_lock:
        activity_log.appendleft(entry)
    print(f"[{entry['time']}] [{event_type}] {message}")


# --- Security Route Guard ---
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)

    return decorated_function


# --- UI Interface Web Templates ---
LOGIN_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ config.system_name }} - Portal Authentication</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-deep: #020617;
            --glass-bg: rgba(15, 23, 42, 0.45);
            --glass-border: rgba(255, 255, 255, 0.08);
            --primary: #38bdf8;
            --primary-glow: rgba(56, 189, 248, 0.2);
            --text: #f8fafc;
            --text-muted: #94a3b8;
            --error: #ef4444;
            --radius-lg: 16px;
            --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background: var(--bg-deep);
            color: var(--text);
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            margin: 0;
            overflow: hidden;
            position: relative;
        }
        /* Background Blobs */
        .bg-blobs {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            width: 100vw;
            height: 100vh;
            overflow: hidden;
            z-index: -1;
            pointer-events: none;
        }
        .blob {
            position: absolute;
            border-radius: 50%;
            filter: blur(120px);
            opacity: 0.15;
            animation: float 20s infinite alternate ease-in-out;
        }
        .blob-1 {
            width: 500px;
            height: 500px;
            background: radial-gradient(circle, #3b82f6 0%, transparent 80%);
            top: -10%;
            left: -10%;
        }
        .blob-2 {
            width: 600px;
            height: 600px;
            background: radial-gradient(circle, #8b5cf6 0%, transparent 80%);
            bottom: -15%;
            right: -10%;
            animation-delay: -5s;
        }
        @keyframes float {
            0% { transform: translate(0, 0) scale(1); }
            50% { transform: translate(50px, -40px) scale(1.1); }
            100% { transform: translate(-30px, 60px) scale(0.9); }
        }
        .card {
            background: var(--glass-bg);
            border: 1px solid var(--glass-border);
            backdrop-filter: blur(24px);
            -webkit-backdrop-filter: blur(24px);
            padding: 40px;
            border-radius: var(--radius-lg);
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4);
            width: 100%;
            max-width: 400px;
            box-sizing: border-box;
            text-align: center;
            z-index: 10;
            position: relative;
            animation: fadeIn 0.6s ease-out;
        }
        @keyframes fadeIn {
            from { transform: translateY(20px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }
        h2 {
            font-family: 'Outfit', sans-serif;
            font-size: 24px;
            font-weight: 800;
            margin: 0 0 8px 0;
            color: #fff;
            letter-spacing: 0.5px;
        }
        .subtitle {
            font-size: 11px;
            font-weight: 700;
            color: var(--primary);
            letter-spacing: 1.5px;
            text-transform: uppercase;
            margin-bottom: 32px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
        }
        .pulse-dot {
            width: 6px;
            height: 6px;
            background-color: var(--primary);
            border-radius: 50%;
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0% { opacity: 0.4; }
            50% { opacity: 1; }
            100% { opacity: 0.4; }
        }
        .input-group {
            position: relative;
            margin-bottom: 20px;
            text-align: left;
        }
        .input-group label {
            display: block;
            font-size: 11px;
            font-weight: 700;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.8px;
            margin-bottom: 6px;
        }
        input[type="text"], input[type="password"] {
            width: 100%;
            padding: 12px 16px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.05);
            color: #fff;
            box-sizing: border-box;
            font-family: inherit;
            font-size: 14px;
            transition: var(--transition);
        }
        input[type="text"]:focus, input[type="password"]:focus {
            outline: none;
            border-color: var(--primary);
            background: rgba(255, 255, 255, 0.08);
            box-shadow: 0 0 15px var(--primary-glow);
        }
        .btn {
            background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
            color: white;
            border: none;
            padding: 14px;
            width: 100%;
            border-radius: 8px;
            font-size: 15px;
            font-weight: 700;
            cursor: pointer;
            transition: var(--transition);
            box-shadow: 0 4px 15px rgba(2, 132, 199, 0.3);
            margin-top: 10px;
            letter-spacing: 0.5px;
        }
        .btn:hover {
            background: linear-gradient(135deg, #0369a1 0%, #0284c7 100%);
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(2, 132, 199, 0.45);
        }
        .btn:active {
            transform: translateY(0);
        }
        .error {
            background: rgba(239, 68, 68, 0.15);
            border: 1px solid rgba(239, 68, 68, 0.3);
            color: #f87171;
            padding: 12px;
            border-radius: 8px;
            font-weight: 600;
            margin-bottom: 20px;
            text-align: center;
            font-size: 13px;
            box-shadow: 0 0 15px rgba(239, 68, 68, 0.15);
            animation: shake 0.4s ease-in-out;
        }
        @keyframes shake {
            0%, 100% { transform: translateX(0); }
            25% { transform: translateX(-6px); }
            75% { transform: translateX(6px); }
        }
    </style>
</head>
<body>
    <div class="bg-blobs">
        <div class="blob blob-1"></div>
        <div class="blob blob-2"></div>
    </div>
    <div class="card">
        <h2>🛡️ {{ config.system_name }}</h2>
        <div class="subtitle">
            <span class="pulse-dot"></span>
            PORTAL AUTHENTICATION
        </div>
        {% if error %}<div class="error">{{ error }}</div>{% endif %}
        <form method="POST">
            <div class="input-group">
                <label>Username</label>
                <input type="text" name="username" placeholder="Enter username" required autofocus>
            </div>
            <div class="input-group">
                <label>Password</label>
                <input type="password" name="password" placeholder="Enter password" required>
            </div>
            <button type="submit" class="btn">ACCESS MANAGEMENT CONSOLE</button>
        </form>
    </div>
</body>
</html>
"""

DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ config.system_name }} - Management Console</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&family=Fira+Code:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-deep: #020617;
            --glass-bg: rgba(15, 23, 42, 0.45);
            --glass-border: rgba(255, 255, 255, 0.08);
            --glass-highlight: rgba(255, 255, 255, 0.03);
            
            --primary: #38bdf8;
            --primary-glow: rgba(56, 189, 248, 0.15);
            
            --success: #10b981;
            --success-glow: rgba(16, 185, 129, 0.2);
            
            --warning: #fbbf24;
            --warning-glow: rgba(251, 191, 36, 0.2);
            
            --error: #ef4444;
            --error-glow: rgba(239, 68, 68, 0.2);
            
            --text: #f8fafc;
            --text-muted: #94a3b8;
            
            --radius-lg: 16px;
            --radius-md: 10px;
            --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background: var(--bg-deep);
            color: var(--text);
            margin: 0;
            padding: 0;
            min-height: 100vh;
            overflow-x: hidden;
            position: relative;
        }
        /* Background Blobs */
        .bg-blobs {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            width: 100vw;
            height: 100vh;
            overflow: hidden;
            z-index: -1;
            pointer-events: none;
        }
        .blob {
            position: absolute;
            border-radius: 50%;
            filter: blur(120px);
            opacity: 0.12;
            animation: float 25s infinite alternate ease-in-out;
        }
        .blob-1 {
            width: 600px;
            height: 600px;
            background: radial-gradient(circle, #3b82f6 0%, transparent 80%);
            top: -15%;
            left: -10%;
        }
        .blob-2 {
            width: 700px;
            height: 700px;
            background: radial-gradient(circle, #8b5cf6 0%, transparent 80%);
            bottom: -20%;
            right: -10%;
            animation-delay: -6s;
        }
        .blob-3 {
            width: 500px;
            height: 500px;
            background: radial-gradient(circle, #06b6d4 0%, transparent 80%);
            top: 35%;
            left: 45%;
            animation-delay: -12s;
        }
        @keyframes float {
            0% { transform: translate(0, 0) scale(1); }
            50% { transform: translate(60px, -50px) scale(1.05); }
            100% { transform: translate(-40px, 70px) scale(0.95); }
        }
        /* Header Nav */
        .navbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 16px 40px;
            background: rgba(15, 23, 42, 0.4);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--glass-border);
            position: sticky;
            top: 0;
            z-index: 100;
        }
        .brand {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .brand-icon {
            font-size: 24px;
        }
        .brand-text h2 {
            font-family: 'Outfit', sans-serif;
            font-size: 18px;
            font-weight: 800;
            margin: 0;
            color: #fff;
            letter-spacing: 0.5px;
        }
        .brand-text .sub {
            font-size: 10px;
            font-weight: 700;
            color: var(--primary);
            letter-spacing: 1px;
            display: block;
        }
        .nav-links {
            display: flex;
            gap: 12px;
        }
        .nav-link {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            color: var(--text-muted);
            text-decoration: none;
            font-size: 13px;
            font-weight: 600;
            padding: 8px 16px;
            border-radius: 6px;
            transition: var(--transition);
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        .nav-link:hover {
            color: #fff;
            background: rgba(255, 255, 255, 0.06);
            border-color: rgba(255, 255, 255, 0.1);
        }
        .nav-link.sign-out:hover {
            color: var(--error);
            background: rgba(239, 68, 68, 0.1);
            border-color: rgba(239, 68, 68, 0.2);
        }
        /* Main Grid Layout */
        .dashboard-grid {
            display: grid;
            grid-template-columns: 1.1fr 0.9fr;
            gap: 32px;
            max-width: 1400px;
            margin: 40px auto;
            padding: 0 40px 60px 40px;
            box-sizing: border-box;
        }
        @media (max-width: 1024px) {
            .dashboard-grid {
                grid-template-columns: 1fr;
                padding: 0 20px 40px 20px;
                gap: 24px;
            }
            .navbar {
                padding: 16px 20px;
            }
        }
        /* Glass Card */
        .glass-card {
            background: var(--glass-bg);
            border: 1px solid var(--glass-border);
            backdrop-filter: blur(24px);
            -webkit-backdrop-filter: blur(24px);
            border-radius: var(--radius-lg);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
            padding: 28px;
            transition: var(--transition);
        }
        .glass-card:hover {
            border-color: rgba(255, 255, 255, 0.12);
        }
        /* Video Panel */
        .video-panel {
            display: flex;
            flex-direction: column;
            background: var(--glass-bg);
            border: 1px solid var(--glass-border);
            border-radius: var(--radius-lg);
            overflow: hidden;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        }
        .video-header {
            background: rgba(255, 255, 255, 0.01);
            border-bottom: 1px solid var(--glass-border);
            padding: 16px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .video-header h3 {
            margin: 0;
            font-family: 'Outfit', sans-serif;
            font-size: 14px;
            font-weight: 700;
            color: #fff;
            letter-spacing: 0.8px;
        }
        .status-indicator {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 11px;
            font-weight: 700;
            color: var(--success);
            letter-spacing: 0.5px;
        }
        .status-pulse {
            width: 8px;
            height: 8px;
            background-color: var(--success);
            border-radius: 50%;
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
            animation: pulse-dot 1.5s infinite;
        }
        @keyframes pulse-dot {
            0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
            70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
            100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
        }
        .video-viewport {
            position: relative;
            background: #000;
            aspect-ratio: 16/9;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }
        .video-feed {
            width: 100%;
            height: 100%;
            object-fit: cover;
            z-index: 1;
        }
        .video-overlay {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 2;
            background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.2) 50%);
            background-size: 100% 4px;
        }
        .video-hud {
            position: absolute;
            top: 20px;
            left: 20px;
            right: 20px;
            bottom: 20px;
            pointer-events: none;
            z-index: 3;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            font-family: 'Fira Code', monospace;
            font-size: 11px;
            color: rgba(255, 255, 255, 0.65);
            text-shadow: 0 1px 3px rgba(0,0,0,0.8);
        }
        .hud-top {
            display: flex;
            justify-content: space-between;
        }
        .hud-corner {
            border: 2px solid rgba(255, 255, 255, 0.25);
            width: 12px;
            height: 12px;
            position: absolute;
        }
        .corner-tl { top: 0; left: 0; border-right: none; border-bottom: none; }
        .corner-tr { top: 0; right: 0; border-left: none; border-bottom: none; }
        .corner-bl { bottom: 0; left: 0; border-right: none; border-top: none; }
        .corner-br { bottom: 0; right: 0; border-left: none; border-top: none; }
        
        /* Event Monitor / Log snippet */
        .log-viewport {
            height: 180px;
            overflow-y: auto;
            background: rgba(0, 0, 0, 0.25);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            padding: 12px;
            font-family: 'Fira Code', monospace;
            font-size: 11px;
            display: flex;
            flex-direction: column;
            gap: 8px;
            box-sizing: border-box;
        }
        .log-row {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 4px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.02);
        }
        .log-time {
            color: var(--text-muted);
            min-width: 65px;
        }
        .log-badge {
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 9px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            display: inline-block;
            min-width: 55px;
            text-align: center;
        }
        .type-ESP { background: rgba(56, 189, 248, 0.12); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.25); }
        .type-INFO { background: rgba(148, 163, 184, 0.12); color: #94a3b8; border: 1px solid rgba(148, 163, 184, 0.25); }
        .type-ALERT { background: rgba(251, 191, 36, 0.12); color: #fbbf24; border: 1px solid rgba(251, 191, 36, 0.25); }
        .type-SUCCESS { background: rgba(16, 185, 129, 0.12); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.25); }
        .type-ERROR { background: rgba(239, 68, 68, 0.12); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.25); }

        /* Wallet Balance Card */
        .balance-card {
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.6) 0%, rgba(30, 41, 59, 0.4) 100%);
            border: 1px solid var(--glass-border);
            border-radius: var(--radius-md);
            padding: 20px;
            margin-bottom: 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: relative;
            overflow: hidden;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05);
        }
        .balance-card::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(56, 189, 248, 0.03), transparent);
            transform: translateX(-100%);
            animation: shimmer 3s infinite;
        }
        @keyframes shimmer {
            100% { transform: translateX(100%); }
        }
        .balance-card.balance-error {
            border-color: rgba(239, 68, 68, 0.2);
            background: rgba(239, 68, 68, 0.05);
        }
        .balance-info .label {
            font-size: 11px;
            font-weight: 700;
            color: var(--text-muted);
            letter-spacing: 1px;
        }
        .balance-info .amount {
            font-size: 30px;
            font-weight: 800;
            color: var(--success);
            font-family: 'Fira Code', monospace;
            margin: 6px 0;
            text-shadow: 0 0 10px rgba(16, 185, 129, 0.25);
        }
        .balance-info .account-name {
            font-size: 12px;
            color: var(--text-muted);
        }
        .refresh-btn {
            display: flex;
            align-items: center;
            gap: 6px;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
            color: var(--primary);
            padding: 8px 14px;
            border-radius: 6px;
            text-decoration: none;
            font-size: 13px;
            font-weight: 600;
            transition: var(--transition);
        }
        .refresh-btn:hover {
            background: rgba(56, 189, 248, 0.1);
            border-color: rgba(56, 189, 248, 0.3);
            color: #fff;
        }

        /* Settings Card Structure */
        .settings-card {
            background: rgba(255, 255, 255, 0.01);
            border: 1px solid rgba(255, 255, 255, 0.03);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 24px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        }
        .section-title {
            font-family: 'Outfit', sans-serif;
            font-size: 15px;
            font-weight: 700;
            color: var(--primary);
            border-bottom: 1px solid rgba(255, 255, 255, 0.06);
            padding-bottom: 8px;
            margin-top: 5px;
            margin-bottom: 16px;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }
        .form-group {
            margin-bottom: 20px;
        }
        .form-group:last-child {
            margin-bottom: 0;
        }
        label {
            display: block;
            font-size: 12px;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.8px;
            margin-bottom: 8px;
        }
        input[type="text"], textarea {
            width: 100%;
            padding: 12px 14px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.04);
            color: #fff;
            box-sizing: border-box;
            font-family: inherit;
            font-size: 14px;
            transition: var(--transition);
        }
        input[type="text"]:focus, textarea:focus {
            outline: none;
            border-color: var(--primary);
            background: rgba(255, 255, 255, 0.08);
            box-shadow: 0 0 15px var(--primary-glow);
        }

        /* Checkbox as Switch */
        .toggle-group {
            display: flex;
            align-items: center;
            position: relative;
            padding: 12px;
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.04);
            border-radius: 8px;
            margin-bottom: 12px;
            transition: var(--transition);
        }
        .toggle-group:hover {
            background: rgba(255, 255, 255, 0.03);
        }
        .toggle-group input[type="checkbox"] {
            -webkit-appearance: none;
            appearance: none;
            width: 42px;
            height: 22px;
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 24px;
            position: relative;
            outline: none;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            flex-shrink: 0;
            margin-right: 12px;
        }
        .toggle-group input[type="checkbox"]::before {
            content: "";
            position: absolute;
            width: 14px;
            height: 14px;
            border-radius: 50%;
            top: 3px;
            left: 3px;
            background: #94a3b8;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .toggle-group input[type="checkbox"]:checked {
            background: rgba(16, 185, 129, 0.2);
            border-color: rgba(16, 185, 129, 0.4);
        }
        .toggle-group input[type="checkbox"]:checked::before {
            left: 23px;
            background: #10b981;
            box-shadow: 0 0 8px rgba(16, 185, 129, 0.5);
        }
        .toggle-group label {
            margin-bottom: 0;
            cursor: pointer;
            user-select: none;
            font-size: 13px;
            font-weight: 500;
            color: #cbd5e1;
            text-transform: none;
            letter-spacing: 0px;
        }

        /* Success alerts */
        .alert-success {
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid rgba(16, 185, 129, 0.3);
            color: #34d399;
            padding: 16px;
            border-radius: 8px;
            font-weight: 600;
            margin-bottom: 24px;
            text-align: center;
            font-size: 14px;
            box-shadow: 0 0 15px rgba(16, 185, 129, 0.1);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            animation: slideDown 0.4s cubic-bezier(0.16, 1, 0.3, 1);
        }
        @keyframes slideDown {
            from { transform: translateY(-10px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }

        /* Primary Action Buttons */
        .submit-btn {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
            border: none;
            padding: 14px;
            width: 100%;
            border-radius: 8px;
            font-size: 15px;
            font-weight: 700;
            cursor: pointer;
            transition: var(--transition);
            box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
            letter-spacing: 0.5px;
            text-transform: uppercase;
            margin-top: 10px;
        }
        .submit-btn:hover {
            background: linear-gradient(135deg, #059669 0%, #10b981 100%);
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(16, 185, 129, 0.45);
        }
        .submit-btn:active {
            transform: translateY(0);
        }
    </style>
</head>
<body>
    <div class="bg-blobs">
        <div class="blob blob-1"></div>
        <div class="blob blob-2"></div>
        <div class="blob blob-3"></div>
    </div>
    
    <header class="navbar">
        <div class="brand">
            <span class="brand-icon">🛡️</span>
            <div class="brand-text">
                <h2>{{ config.system_name }}</h2>
                <span class="sub">Edge Surveillance Config</span>
            </div>
        </div>
        <div class="nav-links">
            <a href="{{ url_for('logs_page') }}" class="nav-link"><span style="margin-right:2px;">📋</span> Live Log</a>
            <a href="{{ url_for('logout') }}" class="nav-link sign-out"><span style="margin-right:2px;">🚪</span> Sign Out</a>
        </div>
    </header>

    <div class="dashboard-grid">
        <!-- Left Column: Surveillance Monitor & Event log -->
        <div class="left-column" style="display:flex; flex-direction:column; gap:28px;">
            <div class="video-panel">
                <div class="video-header">
                    <h3>📷 LOCAL IMAGE CLASSIFICATION PIPELINE</h3>
                    <div class="status-indicator">
                        <span class="status-pulse" style="background-color: var(--primary); box-shadow: 0 0 0 0 rgba(56, 189, 248, 0.7); animation: pulse-dot-blue 1.5s infinite;"></span>
                        <style>
                            @keyframes pulse-dot-blue {
                                0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(56, 189, 248, 0.7); }
                                70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(56, 189, 248, 0); }
                                100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(56, 189, 248, 0); }
                            }
                        </style>
                        READY
                    </div>
                </div>
                <div class="video-viewport">
                    {% if latest_frame_exists %}
                    <img src="/latest_image?t={{ timestamp }}" alt="Classified Image" class="video-feed">
                    {% else %}
                    <div id="feed-offline" style="display:flex; flex-direction:column; align-items:center; gap:12px; z-index:1; color:var(--text-muted);">
                        <svg width="48" height="48" fill="currentColor" viewBox="0 0 256 256"><path d="M216,40H40A16,16,0,0,0,24,56V200a16,16,0,0,0,16,16H216a16,16,0,0,0,16-16V56A16,16,0,0,0,216,40ZM156,88a12,12,0,1,1-12,12A12,12,0,0,1,156,88Zm52,112H48V160l36-36a12,12,0,0,1,17,0l41,41,25-25a12,12,0,0,1,17,0l24,24Z"></path></svg>
                        <span style="font-family:'Fira Code'; font-size:12px; font-weight:600; letter-spacing:0.5px;">AWAITING LOCAL IMAGE UPLOAD</span>
                    </div>
                    {% endif %}
                    <div class="video-overlay"></div>
                    <div class="video-hud">
                        <div class="hud-top">
                            <div>SOURCE: BROWSER</div>
                            <div id="hud-time">UTC 00:00:00</div>
                        </div>
                        <div class="hud-bottom" style="display:flex; justify-content:space-between; align-items:flex-end;">
                            <div>CLASSIFIER ON</div>
                            <div>YOLOv8 MODEL</div>
                        </div>
                        <div class="hud-corner corner-tl"></div>
                        <div class="hud-corner corner-tr"></div>
                        <div class="hud-corner corner-bl"></div>
                        <div class="hud-corner corner-br"></div>
                    </div>
                </div>
                <!-- Browser Image Upload Form footer -->
                <div class="video-footer" style="padding: 16px; background: rgba(15, 23, 42, 0.35); border-top: 1px solid var(--glass-border); display: flex; align-items: center; justify-content: space-between; gap: 16px;">
                    <form action="/upload" method="POST" enctype="multipart/form-data" style="display: flex; align-items: center; width: 100%; gap: 12px; margin:0;">
                        <label for="image_file" style="margin-bottom: 0; background: rgba(255, 255, 255, 0.04); border: 1px solid rgba(255, 255, 255, 0.08); padding: 8px 16px; border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: 600; color: var(--primary); transition: var(--transition); display: inline-flex; align-items: center; gap: 8px; flex-shrink: 0;">
                            <svg width="16" height="16" fill="currentColor" viewBox="0 0 256 256"><path d="M216,40H40A16,16,0,0,0,24,56V200a16,16,0,0,0,16,16H216a16,16,0,0,0,16-16V56A16,16,0,0,0,216,40ZM156,88a12,12,0,1,1-12,12A12,12,0,0,1,156,88Zm52,112H48V160l36-36a12,12,0,0,1,17,0l41,41,25-25a12,12,0,0,1,17,0l24,24Z"></path></svg>
                            Select Image
                        </label>
                        <input type="file" id="image_file" name="image_file" accept="image/*" style="display: none;" onchange="document.getElementById('file-chosen').textContent = this.files[0].name; this.form.submit();">
                        <span id="file-chosen" style="font-size: 12px; color: var(--text-muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex-grow: 1;">No file chosen</span>
                        <button type="submit" class="refresh-btn" style="border-color: var(--success); color: var(--success); cursor: pointer; background: rgba(16, 185, 129, 0.05);">
                            <svg style="margin-right:2px;" width="14" height="14" fill="currentColor" viewBox="0 0 256 256"><path d="M208,80H176V48a16,16,0,0,0-16-16H96A16,16,0,0,0,80,48V80H48A16,16,0,0,0,32,96v96a16,16,0,0,0,16,16H208a16,16,0,0,0,16-16V96A16,16,0,0,0,208,80ZM96,48h64V80H96ZM208,192H48V96H80a16,16,0,0,0,16,16h64a16,16,0,0,0,16-16h32Z"></path></svg>
                            Classify
                        </button>
                    </form>
                </div>
            </div>

            <!-- Mini logs widget -->
            <div class="glass-card" style="display:flex; flex-direction:column; gap:16px; padding:24px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h3 style="margin:0; font-family:'Outfit'; font-size:14px; font-weight:700; letter-spacing:0.5px; color:#fff;">🛡️ SYSTEM EVENT MONITOR</h3>
                    <a href="{{ url_for('logs_page') }}" style="color:var(--primary); font-size:12px; text-decoration:none; font-weight:600;">Full Logs →</a>
                </div>
                <div id="mini-log-container" class="log-viewport">
                    <div class="empty-state" style="color:var(--text-muted); text-align:center; padding:20px; font-size:11px;">Awaiting event triggers...</div>
                </div>
            </div>
        </div>

        <!-- Right Column: Settings Form -->
        <div class="right-column">
            <div class="glass-card">
                {% if success %}<div class="alert-success">✓ Application metrics saved and re-cached successfully!</div>{% endif %}

                {% if balance_info.status == 'success' %}
                <div class="balance-card">
                    <div class="balance-info">
                        <div class="label">SMS WALLET BALANCE</div>
                        <div class="amount">₦{{ balance_info.balance }}</div>
                        <div class="account-name">Holder: {{ balance_info.name }}</div>
                    </div>
                    <a href="{{ url_for('dashboard') }}" class="refresh-btn">
                        <svg style="margin-right:2px;" width="14" height="14" fill="currentColor" viewBox="0 0 256 256"><path d="M240,128a112,112,0,1,1-21.28-65.73l-1.39,1.39a8,8,0,0,1-11.32-11.32l24-24a8,8,0,0,1,11.32,0l24,24a8,8,0,0,1-11.32,11.32l-1.39-1.39A111.44,111.44,0,0,1,240,128Z"></path></svg>
                        Refresh
                    </a>
                </div>
                {% else %}
                <div class="balance-card balance-error">
                    <div class="balance-info">
                        <div class="label">SMS WALLET BALANCE</div>
                        <div class="amount" style="color:var(--error); font-size:16px;">Error Loading Wallet</div>
                        <div class="account-name">{{ balance_info.message }}</div>
                    </div>
                    <a href="{{ url_for('dashboard') }}" class="refresh-btn">
                        <svg style="margin-right:2px;" width="14" height="14" fill="currentColor" viewBox="0 0 256 256"><path d="M240,128a112,112,0,1,1-21.28-65.73l-1.39,1.39a8,8,0,0,1-11.32-11.32l24-24a8,8,0,0,1,11.32,0l24,24a8,8,0,0,1-11.32,11.32l-1.39-1.39A111.44,111.44,0,0,1,240,128Z"></path></svg>
                        Retry
                    </a>
                </div>
                {% endif %}

                <form method="POST">
                    <!-- Branding Settings -->
                    <div class="settings-card">
                        <div class="section-title">🏷️ System Branding</div>
                        <div class="form-group">
                            <label for="system_name">System / Bot Display Name</label>
                            <input type="text" id="system_name" name="system_name" value="{{ config.system_name }}">
                        </div>
                    </div>

                    <!-- Telegram Settings -->
                    <div class="settings-card">
                        <div class="section-title">✈️ Telegram Dispatch Pipeline</div>
                        <div class="form-group">
                            <div class="toggle-group">
                                <input type="checkbox" id="telegram_enabled" name="telegram_enabled" {% if config.telegram_enabled %}checked{% endif %}>
                                <label for="telegram_enabled">Global Master Enable for Telegram Dispatch Pipeline</label>
                            </div>
                        </div>
                        <div class="form-group">
                            <label for="bot_token">Telegram Bot Token String</label>
                            <input type="text" id="bot_token" name="bot_token" value="{{ config.bot_token }}">
                        </div>
                        <div class="form-group">
                            <label for="chat_id">Target Chat ID / Channel Reference</label>
                            <input type="text" id="chat_id" name="chat_id" value="{{ config.chat_id }}">
                        </div>
                    </div>

                    <!-- CharlesClicksVTU SMS Settings -->
                    <div class="settings-card">
                        <div class="section-title">📱 CharlesClicksVTU Bulk SMS Configuration</div>
                        <div class="form-group">
                            <label for="sms_api_key">Authentication API Bearer Token</label>
                            <input type="text" id="sms_api_key" name="sms_api_key" value="{{ config.sms_api_key }}">
                        </div>
                        <div class="form-group">
                            <label for="sms_sender_name">Configured Sender ID Alpha Label (Max 11 Alpha Chars)</label>
                            <input type="text" id="sms_sender_name" name="sms_sender_name" value="{{ config.sms_sender_name }}">
                        </div>
                        <div class="form-group">
                            <label for="sms_recipients">Recipient Directory String (Strict Comma Separated Format)</label>
                            <textarea id="sms_recipients" name="sms_recipients" rows="2">{{ config.sms_recipients }}</textarea>
                        </div>
                    </div>

                    <!-- Automation Behavior Trigger Rules -->
                    <div class="settings-card">
                        <div class="section-title">⚙️ Automation Behavior Trigger Rules</div>
                        <div class="form-group">
                            <div class="toggle-group">
                                <input type="checkbox" id="sms_enabled_animals" name="sms_enabled_animals" {% if config.sms_enabled_animals %}checked{% endif %}>
                                <label for="sms_enabled_animals">Fire SMS Alerts immediately upon Animal Target Classification</label>
                            </div>
                            <div class="toggle-group" style="margin-bottom: 0;">
                                <input type="checkbox" id="sms_enabled_humans" name="sms_enabled_humans" {% if config.sms_enabled_humans %}checked{% endif %}>
                                <label for="sms_enabled_humans">Cross-route SMS Alerts for Human Targets (Disabled by Default)</label>
                            </div>
                        </div>
                    </div>

                    <button type="submit" class="submit-btn">Commit Changes & Reload Pipeline</button>
                </form>
            </div>
        </div>
    </div>

    <script>
        // Real-time HUD clock update
        function updateClock() {
            const now = new Date();
            const timeStr = now.toISOString().slice(0, 19).replace('T', ' ');
            document.getElementById('hud-time').textContent = 'UTC ' + timeStr;
        }
        updateClock();
        setInterval(updateClock, 1000);

        // Fetch logs data for the event monitor
        async function fetchMiniLogs() {
            try {
                const res = await fetch('/logs/data');
                const data = await res.json();
                const container = document.getElementById('mini-log-container');
                if (!data.logs.length) {
                    container.innerHTML = '<div class="empty-state" style="color:var(--text-muted); text-align:center; padding:20px; font-size:11px;">Awaiting event triggers...</div>';
                    return;
                }
                // Display latest 8 events
                const recentLogs = data.logs.slice(0, 8);
                container.innerHTML = recentLogs.map(entry => `
                    <div class="log-row">
                        <span class="log-time">${entry.time}</span>
                        <span class="log-badge type-${entry.type}">${entry.type}</span>
                        <span style="color:#e2e8f0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${entry.message}</span>
                    </div>
                `).join('');
            } catch (e) {
                console.error('Log fetch failed', e);
            }
        }
        fetchMiniLogs();
        setInterval(fetchMiniLogs, 2000);
    </script>
</body>
</html>
"""

LOG_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ config.system_name }} - Live Log</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&family=Fira+Code:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-deep: #020617;
            --glass-bg: rgba(15, 23, 42, 0.45);
            --glass-border: rgba(255, 255, 255, 0.08);
            --primary: #38bdf8;
            --success: #10b981;
            --warning: #fbbf24;
            --error: #ef4444;
            --text: #f8fafc;
            --text-muted: #94a3b8;
            --radius-lg: 16px;
            --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background: var(--bg-deep);
            color: var(--text);
            margin: 0;
            padding: 0;
            min-height: 100vh;
            overflow-x: hidden;
            position: relative;
        }
        /* Background Blobs */
        .bg-blobs {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            width: 100vw;
            height: 100vh;
            overflow: hidden;
            z-index: -1;
            pointer-events: none;
        }
        .blob {
            position: absolute;
            border-radius: 50%;
            filter: blur(120px);
            opacity: 0.12;
            animation: float 25s infinite alternate ease-in-out;
        }
        .blob-1 {
            width: 600px;
            height: 600px;
            background: radial-gradient(circle, #3b82f6 0%, transparent 80%);
            top: -15%;
            left: -10%;
        }
        .blob-2 {
            width: 700px;
            height: 700px;
            background: radial-gradient(circle, #8b5cf6 0%, transparent 80%);
            bottom: -20%;
            right: -10%;
            animation-delay: -6s;
        }
        @keyframes float {
            0% { transform: translate(0, 0) scale(1); }
            50% { transform: translate(60px, -50px) scale(1.05); }
            100% { transform: translate(-40px, 70px) scale(0.95); }
        }
        /* Header Nav */
        .navbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 16px 40px;
            background: rgba(15, 23, 42, 0.4);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--glass-border);
            position: sticky;
            top: 0;
            z-index: 100;
        }
        .brand {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .brand-text h2 {
            font-family: 'Outfit', sans-serif;
            font-size: 18px;
            font-weight: 800;
            margin: 0;
            color: #fff;
            letter-spacing: 0.5px;
        }
        .brand-text .sub {
            font-size: 10px;
            font-weight: 700;
            color: var(--primary);
            letter-spacing: 1px;
            display: block;
        }
        .status-dot {
            width: 8px;
            height: 8px;
            background-color: var(--success);
            border-radius: 50%;
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
            animation: pulse-dot 1.5s infinite;
        }
        @keyframes pulse-dot {
            0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
            70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
            100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
        }
        .nav-links {
            display: flex;
            gap: 12px;
        }
        .nav-link {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            color: var(--text-muted);
            text-decoration: none;
            font-size: 13px;
            font-weight: 600;
            padding: 8px 16px;
            border-radius: 6px;
            transition: var(--transition);
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        .nav-link:hover {
            color: #fff;
            background: rgba(255, 255, 255, 0.06);
            border-color: rgba(255, 255, 255, 0.1);
        }
        /* Main Container */
        #log-page-container {
            max-width: 1000px;
            margin: 40px auto;
            padding: 0 40px;
            box-sizing: border-box;
        }
        @media (max-width: 768px) {
            #log-page-container {
                padding: 0 20px;
                margin: 24px auto;
            }
            .navbar {
                padding: 16px 20px;
            }
        }
        /* Glass Card */
        .glass-card {
            background: var(--glass-bg);
            border: 1px solid var(--glass-border);
            backdrop-filter: blur(24px);
            -webkit-backdrop-filter: blur(24px);
            border-radius: var(--radius-lg);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
            padding: 28px;
        }
        /* Filters */
        .log-filters {
            display: flex;
            gap: 8px;
            margin-bottom: 24px;
            flex-wrap: wrap;
        }
        .filter-btn {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.06);
            color: var(--text-muted);
            padding: 8px 16px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            transition: var(--transition);
        }
        .filter-btn:hover {
            background: rgba(255, 255, 255, 0.06);
            color: #fff;
        }
        .filter-btn.active {
            background: rgba(56, 189, 248, 0.15);
            border-color: var(--primary);
            color: #fff;
            box-shadow: 0 0 10px rgba(56, 189, 248, 0.2);
        }
        /* Log List Viewport */
        .log-list {
            background: rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            padding: 8px 0;
            font-family: 'Fira Code', monospace;
            font-size: 12px;
            max-height: 600px;
            overflow-y: auto;
        }
        .log-entry {
            display: flex;
            align-items: center;
            gap: 16px;
            padding: 10px 16px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.02);
            transition: var(--transition);
        }
        .log-entry:hover {
            background: rgba(255, 255, 255, 0.02);
        }
        .log-time {
            color: var(--text-muted);
            min-width: 75px;
        }
        .log-badge {
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 10px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            display: inline-block;
            min-width: 60px;
            text-align: center;
        }
        .type-ESP { background: rgba(56, 189, 248, 0.12); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.25); }
        .type-INFO { background: rgba(148, 163, 184, 0.12); color: #94a3b8; border: 1px solid rgba(148, 163, 184, 0.25); }
        .type-ALERT { background: rgba(251, 191, 36, 0.12); color: #fbbf24; border: 1px solid rgba(251, 191, 36, 0.25); }
        .type-SUCCESS { background: rgba(16, 185, 129, 0.12); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.25); }
        .type-ERROR { background: rgba(239, 68, 68, 0.12); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.25); }
        
        .empty-state {
            color: var(--text-muted);
            text-align: center;
            padding: 40px;
        }
    </style>
</head>
<body>
    <div class="bg-blobs">
        <div class="blob blob-1"></div>
        <div class="blob blob-2"></div>
    </div>
    
    <div class="navbar">
        <div class="brand">
            <span class="status-dot"></span>
            <div class="brand-text">
                <h2>{{ config.system_name }} Live Activity Log</h2>
                <span class="sub">REAL-TIME PIPELINE MONITORING</span>
            </div>
        </div>
        <div class="nav-links">
            <a href="{{ url_for('dashboard') }}" class="nav-link"><span style="margin-right:2px;">🏠</span> Dashboard</a>
            <a href="{{ url_for('logout') }}" class="nav-link sign-out"><span style="margin-right:2px;">🚪</span> Sign Out</a>
        </div>
    </div>
    
    <div id="log-page-container">
        <div class="glass-card">
            <!-- Filter Tabs -->
            <div class="log-filters">
                <button id="btn-ALL" class="filter-btn active" onclick="setFilter('ALL')">ALL EVENTS</button>
                <button id="btn-ALERT" class="filter-btn" onclick="setFilter('ALERT')">ALERTS</button>
                <button id="btn-ESP" class="filter-btn" onclick="setFilter('ESP')">ESP32</button>
                <button id="btn-SUCCESS" class="filter-btn" onclick="setFilter('SUCCESS')">SUCCESS</button>
                <button id="btn-ERROR" class="filter-btn" onclick="setFilter('ERROR')">ERRORS</button>
            </div>
            
            <div id="log-container" class="log-list">
                <div class="empty-state">Waiting for activity logs from ESP32 pipeline...</div>
            </div>
        </div>
    </div>
    
    <script>
        let allLogs = [];
        let currentFilter = 'ALL';

        async function fetchLogs() {
            try {
                const res = await fetch('/logs/data');
                const data = await res.json();
                allLogs = data.logs || [];
                renderLogs();
            } catch (e) {
                console.error('Log fetch failed', e);
            }
        }

        function setFilter(filterType) {
            currentFilter = filterType;
            
            // Update active button classes
            const buttons = document.querySelectorAll('.filter-btn');
            buttons.forEach(btn => btn.classList.remove('active'));
            document.getElementById('btn-' + filterType).classList.add('active');
            
            renderLogs();
        }

        function renderLogs() {
            const container = document.getElementById('log-container');
            const filtered = currentFilter === 'ALL' 
                ? allLogs 
                : allLogs.filter(entry => entry.type === currentFilter);
                
            if (!filtered.length) {
                container.innerHTML = '<div class="empty-state">No matching logs found.</div>';
                return;
            }
            
            container.innerHTML = filtered.map(entry => `
                <div class="log-entry">
                    <span class="log-time">${entry.time}</span>
                    <span class="log-badge type-${entry.type}">${entry.type}</span>
                    <span style="color:#e2e8f0; word-break:break-all;">${entry.message}</span>
                </div>
            `).join('');
        }

        // Initial fetch and auto-refresh
        fetchLogs();
        setInterval(fetchLogs, 2000);
    </script>
</body>
</html>
"""


# --- Dispatch Services Backend Workers ---
def send_telegram(message):
    global config
    if not config.get("telegram_enabled"):
        return
    try:
        url = f"https://api.telegram.org/bot{config['bot_token']}/sendMessage"
        payload = {"chat_id": config["chat_id"], "text": message, "parse_mode": "HTML"}
        res = requests.post(url, json=payload, timeout=5)
        if res.ok:
            log_event("SUCCESS", "Telegram text message sent")
        else:
            log_event("ERROR", f"Telegram message failed - HTTP {res.status_code}")
    except Exception as e:
        log_event("ERROR", f"Telegram message exception: {e}")


def send_telegram_photo(image_bytes):
    global config
    if not config.get("telegram_enabled"):
        return
    try:
        url = f"https://api.telegram.org/bot{config['bot_token']}/sendPhoto"
        files = {"photo": ("capture.jpg", image_bytes, "image/jpeg")}
        data = {"chat_id": config["chat_id"]}
        res = requests.post(url, files=files, data=data, timeout=10)
        if res.ok:
            log_event("SUCCESS", "Telegram photo sent")
        else:
            log_event("ERROR", f"Telegram photo failed - HTTP {res.status_code}")
    except Exception as e:
        log_event("ERROR", f"Telegram photo exception: {e}")


def send_bulk_sms(message_content):
    global config
    url = "https://charlesclicksvtu.com/api/bulksms/"
    headers = {
        "Authorization": f"Token {config['sms_api_key']}",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    }
    payload = {
        "senderName": config["sms_sender_name"],
        "message": message_content,
        "phoneNumbers": config["sms_recipients"],
        "ref": f"wd_{secrets.token_hex(4)}",
    }
    try:
        res = requests.post(url, json=payload, headers=headers, timeout=15)
        if res.ok:
            log_event("SUCCESS", f"SMS dispatched - Status {res.status_code}")
        else:
            log_event("ERROR", f"SMS failed - Status {res.status_code}: {res.text}")
    except Exception as e:
        log_event("ERROR", f"SMS dispatch exception: {e}")


def get_sms_balance():
    """
    Pulls the current wallet balance from the CharlesClicksVTU account endpoint.
    Expected response shape: {"name": "...", "balance": "...", "status": "success"}
    NOTE: assumes the same Authorization scheme used for /api/bulksms/ (Token <key>).
    If the provider uses a different auth header for this endpoint, adjust below.
    """
    global config
    url = "https://charlesclicksvtu.com/api/user/"
    headers = {
        "Authorization": f"Token {config['sms_api_key']}",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    }
    try:
        res = requests.get(url, headers=headers, timeout=10)
        data = res.json()
        if data.get("status") == "success":
            return {
                "status": "success",
                "name": data.get("name", "N/A"),
                "balance": data.get("balance", "0.00"),
            }
        return {"status": "error", "message": data.get("message", "Unrecognized response from provider")}
    except Exception as e:
        print(f"[Balance Fetch Error] {e}")
        return {"status": "error", "message": "Could not reach balance endpoint"}


# --- Portal Interface Routings ---
@app.route("/", methods=["GET", "POST"])
@login_required
def dashboard():
    global config, latest_frame
    success = False
    if request.method == "POST":
        config["system_name"] = request.form.get("system_name", "").strip() or config["system_name"]

        config["bot_token"] = request.form.get("bot_token", "").strip()
        config["chat_id"] = request.form.get("chat_id", "").strip()
        config["telegram_enabled"] = "telegram_enabled" in request.form

        config["sms_api_key"] = request.form.get("sms_api_key", "").strip()
        config["sms_sender_name"] = request.form.get("sms_sender_name", "").strip()
        config["sms_recipients"] = request.form.get("sms_recipients", "").strip()

        config["sms_enabled_animals"] = "sms_enabled_animals" in request.form
        config["sms_enabled_humans"] = "sms_enabled_humans" in request.form

        save_settings(config)
        success = True

    balance_info = get_sms_balance()
    latest_frame_exists = latest_frame is not None
    import time
    timestamp = int(time.time())

    return render_template_string(
        DASHBOARD_TEMPLATE, config=config, success=success, balance_info=balance_info,
        latest_frame_exists=latest_frame_exists, timestamp=timestamp
    )


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        if (
            request.form["username"] == ADMIN_USERNAME
            and request.form["password"] == ADMIN_PASSWORD
        ):
            session["logged_in"] = True
            return redirect(url_for("dashboard"))
        else:
            error = "Invalid credential strings. Access Denied."
    return render_template_string(LOGIN_TEMPLATE, error=error, config=config)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/logs")
@login_required
def logs_page():
    return render_template_string(LOG_TEMPLATE, config=config)


@app.route("/logs/data")
@login_required
def logs_data():
    with log_lock:
        return jsonify({"logs": list(activity_log)})


@app.route("/latest_image")
@login_required
def latest_image():
    global latest_frame
    with frame_lock:
        if latest_frame is None:
            return "No image uploaded yet", 404
        frame = latest_frame.copy()
    _, jpeg = cv2.imencode(".jpg", frame)
    return Response(jpeg.tobytes(), mimetype="image/jpeg")


@app.route("/upload", methods=["POST"])
@login_required
def upload():
    global latest_frame, config
    if "image_file" not in request.files:
        log_event("ERROR", "No image file part in request")
        return redirect(url_for("dashboard"))
    
    file = request.files["image_file"]
    if file.filename == "":
        log_event("ERROR", "No selected file")
        return redirect(url_for("dashboard"))
        
    if file:
        image_bytes = file.read()
        log_event("INFO", f"Browser uploaded image ({len(image_bytes)} bytes)")
        
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img is None:
            log_event("ERROR", "Could not decode uploaded image")
            return redirect(url_for("dashboard"))
            
        with frame_lock:
            latest_frame = img.copy()
            
        results = model(img)
        probs = results[0].probs
        top_idx = probs.top1
        confidence = probs.top1conf.item()
        label = results[0].names[top_idx]
        
        log_event("INFO", f"Uploaded image classified as '{label}' at {confidence*100:.1f}% confidence")
        
        # Draw classification overlay text on the image
        color = (0, 255, 0) if label == "human" else (0, 165, 255)
        text = f"{label.upper()} ({confidence*100:.1f}%)"
        if confidence < 0.55:
            text = f"UNKNOWN ({confidence*100:.1f}%)"
            color = (0, 0, 255)
            
        cv2.putText(img, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        with frame_lock:
            latest_frame = img.copy()
            
        if confidence >= 0.55:
            if label == "human":
                log_event("ALERT", "HUMAN detected - dispatching Telegram + SMS pipeline")
                send_telegram(f"🚨 <b>INTRUSION ALERT - {config['system_name']}</b>\nHuman detected in browser upload!")
                send_telegram_photo(image_bytes)
                if config.get("sms_enabled_humans"):
                    sms_text = f"CRITICAL: Human intrusion detected by {config['system_name']}. Confidence: {confidence*100:.1f}%."
                    send_bulk_sms(sms_text)
                    
            elif label == "animal":
                log_event("ALERT", "ANIMAL detected - dispatching Telegram + SMS pipeline")
                send_telegram(f"🐾 <b>Animal Alert - {config['system_name']}</b>\nAnimal detected in browser upload!")
                send_telegram_photo(image_bytes)
                if config.get("sms_enabled_animals"):
                    sms_text = f"ALERT: Animal detected by {config['system_name']}. Confidence: {confidence*100:.1f}%."
                    send_bulk_sms(sms_text)
        else:
            log_event("INFO", "Confidence below threshold (55%) - treated as UNKNOWN")
            
        return redirect(url_for("dashboard"))


@app.route("/test", methods=["GET"])
def test():
    log_event("INFO", "Health check ping received on /test")
    return jsonify({"status": "Server is running"})


if __name__ == "__main__":
    print("Starting classification server...")
    print("Dashboard config layer mapped to root page address '/'")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
