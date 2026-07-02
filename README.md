# BIOALERT — Render Deployment

## 1. Before you push anything

Copy your trained model weights into this folder:

```
bioalert-render/models/best.pt
```

(This is the file currently at `C:\Users\MALIK\Desktop\runs\classify\train-5\weights\best.pt`
on your laptop — just copy it into `models/` here.)

Check the file is under 100 MB (YOLOv8 classify weights almost always are):

```
ls -lh models/best.pt
```

If it's small, normal git handles it fine — no Git LFS needed.

## 2. Folder structure (what you should have)

```
bioalert-render/
├── server.py          ← Flask app (already adjusted for Render)
├── requirements.txt    ← Python dependencies
├── Procfile             ← tells Render how to start the app
├── .gitignore
├── README.md            ← this file
└── models/
    └── best.pt          ← YOU add this
```

## 3. Push to GitHub

From inside the `bioalert-render` folder:

```bash
git init
git add .
git commit -m "Initial BIOALERT deployment for Render"
```

Create a new (can be private) repo on GitHub, then:

```bash
git remote add origin https://github.com/<your-username>/<repo-name>.git
git branch -M main
git push -u origin main
```

## 4. Deploy on Render

1. Go to https://dashboard.render.com → **New** → **Web Service**.
2. Connect your GitHub account, select the repo you just pushed.
3. Render should auto-detect Python. Confirm:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** leave blank (Render will read the `Procfile`) —
     or, if asked, paste: `gunicorn server:app --workers 1 --threads 4 --timeout 120`
4. Under **Environment** → **Environment Variables**, add these (use your real values):

   | Key | Value |
   |---|---|
   | `ADMIN_USERNAME` | your admin login username |
   | `ADMIN_PASSWORD` | your admin login password |
   | `TELEGRAM_BOT_TOKEN` | your real bot token |
   | `TELEGRAM_CHAT_ID` | your real chat id |
   | `SMS_API_KEY` | your real CharlesClicksVTU key |
   | `SMS_RECIPIENTS` | e.g. `07058660994,09057289989` |
   | `SYSTEM_NAME` | `BIOALERT` |

   None of these are in the committed code — Render injects them at runtime.

5. Pick an instance type:
   - **Free** — fine for setup/testing, but spins down after 15 min idle (30–60s cold start on next request).
   - **Starter ($7/mo)** — recommended for the actual presentation day; no spin-down, more CPU. Downgrade/delete right after.
6. Click **Create Web Service** and wait for the build to finish (first build with `ultralytics`/`torch` can take several minutes).

## 5. After it's live

Your service gets a permanent URL like:

```
https://bioalert-xxxx.onrender.com
```

Point your ESP32 firmware's `LOCAL_SERVER_URL`-style fallback (or even your primary `serverURL`)
directly at:

```
https://bioalert-xxxx.onrender.com/classify
```

Since this URL never changes, you can actually **retire the Gist/ngrok auto-fetch workaround
entirely** for this deployment and hardcode it in firmware — simpler for presentation day.

## 6. Day-of-presentation checklist

- [ ] Visit the Render URL in a browser ~10 minutes before presenting (wakes it from sleep if on free tier)
- [ ] Confirm `/test` returns `{"status": "Server is running"}`
- [ ] Confirm login at `/login` works with your env-var admin credentials
- [ ] Do one real end-to-end capture from the ESP32 to confirm `/classify` responds correctly
