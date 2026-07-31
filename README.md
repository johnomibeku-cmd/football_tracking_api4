# Grassroots Football Player Tracking API

AI-powered football player tracking from match footage — built for African grassroots football.

## What it does
Upload a football video and automatically get back:
- Distance run (metres and km)
- Average and max speed (km/h)
- Sprint moments
- Zone coverage (defensive, midfield, attacking)
- Movement heatmap
- Tracking CSV data

## How to use the API

### Step 1 — Find player IDs
Go to `/detect` → upload your video → see all player ID numbers

### Step 2 — Analyse a player
Go to `/analyse` → upload video + enter player_id and player_name → get full stats back

### Step 3 — Download results
- CSV: `/results/{filename}`
- Heatmap: `/heatmap/{filename}`

## Run locally

```bash
git clone https://github.com/johnomibeku-cmd/football_tracking_api4.git
cd football_tracking_api4
pip install -r requirements.txt
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

Open: http://localhost:8000/docs

## Tech Stack
- FastAPI
- YOLOv8
- ByteTrack
- OpenCV
- Python
