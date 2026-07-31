# ============================================================
# GRASSROOTS PLAYER TRACKING — FASTAPI SERVER
# Football Video Analysis API
# ============================================================

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from ultralytics import YOLO
import supervision as sv

# ── App setup ────────────────────────────────────────────────
app = FastAPI(
    title="Grassroots Football Player Tracking API",
    description="Upload a football video and get back player performance stats automatically.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

print("Loading YOLOv8 model...")
model = YOLO("yolov8n.pt")
print("YOLOv8 ready!")

PITCH_L_M    = 105.0
PITCH_W_M    =  68.0
SAMPLE_EVERY =  5


# ── Helper functions ─────────────────────────────────────────

def build_homography(width, height):
    pixel_pts = np.float32([
        [20,       15        ],
        [width-20, 15        ],
        [width-20, height-15 ],
        [20,       height-15 ],
    ])
    world_pts = np.float32([
        [0,         0         ],
        [PITCH_L_M, 0         ],
        [PITCH_L_M, PITCH_W_M ],
        [0,         PITCH_W_M ],
    ])
    H, _ = cv2.findHomography(pixel_pts, world_pts)
    return H


def px_to_m(px, py, H):
    res = cv2.perspectiveTransform(
        np.array([[[float(px), float(py)]]], dtype=np.float32), H
    )
    return float(res[0][0][0]), float(res[0][0][1])


def calculate_stats(df):
    df = df.sort_values('frame').reset_index(drop=True)
    df['dx']   = df['x_m'].diff().fillna(0)
    df['dy']   = df['y_m'].diff().fillna(0)
    df['step'] = np.sqrt(df['dx']**2 + df['dy']**2)
    df = df[df['step'] < 5].reset_index(drop=True)

    if len(df) < 2:
        return None, df

    dist_m    = df['step'].sum()
    time_s    = df['time_s'].max() - df['time_s'].min()
    speed_kmh = (dist_m/1000) / (time_s/3600) if time_s > 0 else 0

    WIN = 8
    speeds = []
    pos    = list(zip(df['x_m'], df['y_m']))
    ts     = df['time_s'].values
    for i in range(WIN, len(pos)):
        dx = pos[i][0] - pos[i-WIN][0]
        dy = pos[i][1] - pos[i-WIN][1]
        dt = ts[i] - ts[i-WIN]
        if dt > 0:
            spd = np.sqrt(dx**2+dy**2)/dt*3.6
            if spd < 40:
                speeds.append(spd)

    max_speed    = round(max(speeds), 1) if speeds else 0
    sprint_count = sum(1 for s in speeds if s > 20)

    def_pct = (df['x_m'] <  PITCH_L_M/3).mean() * 100
    mid_pct = ((df['x_m'] >= PITCH_L_M/3) &
               (df['x_m'] <  2*PITCH_L_M/3)).mean() * 100
    att_pct = (df['x_m'] >= 2*PITCH_L_M/3).mean() * 100

    stats = {
        "distance_m":       round(dist_m, 1),
        "distance_km":      round(dist_m/1000, 3),
        "time_tracked_s":   round(time_s, 1),
        "time_tracked_min": round(time_s/60, 2),
        "avg_speed_kmh":    round(speed_kmh, 1),
        "max_speed_kmh":    max_speed,
        "sprint_moments":   sprint_count,
        "defensive_%":      round(def_pct, 1),
        "midfield_%":       round(mid_pct, 1),
        "attacking_%":      round(att_pct, 1),
    }
    return stats, df


def generate_heatmap(df, player_name, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor('#1a1a2e')

    ax = axes[0]
    ax.set_facecolor('#2e7d32')
    ax.add_patch(plt.Rectangle((0,0),PITCH_L_M,PITCH_W_M,fill=False,edgecolor='white',lw=2))
    ax.plot([PITCH_L_M/2]*2,[0,PITCH_W_M],'white',lw=1.5)
    ax.add_patch(plt.Circle((PITCH_L_M/2,PITCH_W_M/2),9.15,color='white',fill=False,lw=1.5))
    ax.add_patch(plt.Rectangle((0,13.85),16.5,40.3,fill=False,edgecolor='white',lw=1.5))
    ax.add_patch(plt.Rectangle((PITCH_L_M-16.5,13.85),16.5,40.3,fill=False,edgecolor='white',lw=1.5))

    xs = df['x_m'].clip(0, PITCH_L_M)
    ys = df['y_m'].clip(0, PITCH_W_M)
    ax.hist2d(xs,ys,bins=[52,34],range=[[0,PITCH_L_M],[0,PITCH_W_M]],
              cmap='hot',alpha=0.65,density=True)
    ax.plot(xs,ys,color='cyan',lw=0.5,alpha=0.3)
    ax.scatter(xs.iloc[0], ys.iloc[0], color='lime',s=120,zorder=5,label='Start')
    ax.scatter(xs.iloc[-1],ys.iloc[-1],color='red', s=120,zorder=5,label='End',marker='X')
    ax.axvline(PITCH_L_M/3,  color='white',lw=0.8,ls='--',alpha=0.5)
    ax.axvline(2*PITCH_L_M/3,color='white',lw=0.8,ls='--',alpha=0.5)
    ax.text(PITCH_L_M/6,  2,'DEF',color='white',fontsize=9,ha='center')
    ax.text(PITCH_L_M/2,  2,'MID',color='white',fontsize=9,ha='center')
    ax.text(5*PITCH_L_M/6,2,'ATT',color='white',fontsize=9,ha='center')
    ax.set_xlim(0,PITCH_L_M); ax.set_ylim(PITCH_W_M,0)
    ax.set_xlabel("Pitch length (m)",color='white')
    ax.set_ylabel("Pitch width (m)", color='white')
    ax.tick_params(colors='white')
    ax.set_title(f"{player_name} — Movement Heatmap",color='white',fontsize=12)
    ax.legend(fontsize=8,facecolor='#1b5e20',labelcolor='white')
    for sp in ax.spines.values(): sp.set_edgecolor('white')

    ax2 = axes[1]
    ax2.set_facecolor('#1a1a2e')
    zones  = ['Defensive','Midfield','Attacking']
    values = [
        (df['x_m'] < PITCH_L_M/3).mean()*100,
        ((df['x_m'] >= PITCH_L_M/3) & (df['x_m'] < 2*PITCH_L_M/3)).mean()*100,
        (df['x_m'] >= 2*PITCH_L_M/3).mean()*100,
    ]
    colors = ['#e53935','#fb8c00','#43a047']
    bars   = ax2.bar(zones,values,color=colors,edgecolor='white',lw=0.5,width=0.5)
    ax2.set_ylim(0,100)
    ax2.set_ylabel("% of match time",color='white')
    ax2.set_title(f"{player_name} — Zone Coverage",color='white',fontsize=12)
    ax2.tick_params(colors='white')
    for bar,val in zip(bars,values):
        ax2.text(bar.get_x()+bar.get_width()/2,val+1.5,
                 f'{val:.0f}%',ha='center',fontsize=12,fontweight='bold',color='white')
    for sp in ax2.spines.values(): sp.set_edgecolor('#555')

    plt.tight_layout()
    plt.savefig(output_path,dpi=120,bbox_inches='tight',facecolor='#1a1a2e')
    plt.close()


# ── API Routes ────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "status":  "running",
        "message": "Grassroots Football Tracking API is live!",
        "docs":    "Visit /docs to upload a video and test the API",
        "version": "1.0.0"
    }


@app.post("/detect")
async def detect_players(
    video: UploadFile = File(..., description="Football match video (.mp4)")
):
    """Upload a video to see ALL player IDs in the first frame. Use this first before /analyse."""
    video_path = f"uploads/detect_{video.filename}"
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise HTTPException(status_code=400, detail="Could not read video.")

    results    = model(frame, classes=[0], conf=0.35, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    tracker    = sv.ByteTrack()
    detections = tracker.update_with_detections(detections)

    players = []
    for i in range(len(detections)):
        x1,y1,x2,y2 = map(int, detections.xyxy[i])
        tid  = int(detections.tracker_id[i])
        conf = float(detections.confidence[i])
        players.append({
            "player_id":   tid,
            "confidence":  round(conf, 2),
            "position_px": {"x": (x1+x2)//2, "y": y2},
            "box_size":    {"w": x2-x1, "h": y2-y1},
        })

    return JSONResponse({
        "status":           "success",
        "video":            video.filename,
        "players_detected": len(players),
        "players":          players,
        "tip": "Use the player_id in the /analyse endpoint to track that player."
    })


@app.post("/analyse")
async def analyse_video(
    video:       UploadFile = File(..., description="Football match video (.mp4)"),
    player_id:   int        = 1,
    player_name: str        = "Player"
):
    """Upload a football video and get back full player performance stats."""

    if not video.filename.endswith(('.mp4','.avi','.mov','.mkv')):
        raise HTTPException(status_code=400, detail="Please upload a video file (.mp4, .avi, .mov, .mkv)")

    print(f"New request: {video.filename} | Player: {player_name} (ID #{player_id})")

    safe_name    = player_name.replace(' ','_')
    video_path   = f"uploads/{video.filename}"
    csv_path     = f"outputs/{safe_name}_tracking.csv"
    heatmap_path = f"outputs/{safe_name}_heatmap.png"

    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Could not open video file.")

    fps          = cap.get(cv2.CAP_PROP_FPS)
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s   = total_frames / fps

    H       = build_homography(width, height)
    tracker = sv.ByteTrack()

    positions = []
    prev_pos  = None
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num % SAMPLE_EVERY != 0:
            frame_num += 1
            continue

        results    = model(frame, classes=[0], conf=0.35, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)

        for i in range(len(detections)):
            if int(detections.tracker_id[i]) != player_id:
                continue
            x1,y1,x2,y2 = map(int, detections.xyxy[i])
            fx, fy = (x1+x2)//2, y2
            if prev_pos is None or \
               np.sqrt((fx-prev_pos[0])**2+(fy-prev_pos[1])**2) < 200:
                mx, my = px_to_m(fx, fy, H)
                positions.append({
                    'frame':  frame_num,
                    'time_s': round(frame_num/fps, 2),
                    'x_px':   fx, 'y_px': fy,
                    'x_m':    round(mx,2), 'y_m': round(my,2),
                })
                prev_pos = (fx, fy)

        frame_num += 1

    cap.release()

    if len(positions) < 10:
        raise HTTPException(
            status_code=404,
            detail=f"Player ID #{player_id} not found. Use /detect first to see all player IDs."
        )

    df = pd.DataFrame(positions)
    stats, df_clean = calculate_stats(df)

    if stats is None:
        raise HTTPException(status_code=500, detail="Not enough tracking data.")

    df_clean.to_csv(csv_path, index=False)
    generate_heatmap(df_clean, player_name, heatmap_path)

    return JSONResponse({
        "status":             "success",
        "player":             player_name,
        "player_id":          player_id,
        "video":              video.filename,
        "video_duration_s":   round(duration_s, 1),
        "video_duration_min": round(duration_s/60, 2),
        **stats,
        "files": {
            "csv":     f"/results/{safe_name}_tracking.csv",
            "heatmap": f"/heatmap/{safe_name}_heatmap.png",
        },
        "interpretation": {
            "role_suggestion": (
                "Defensive player" if stats["defensive_%"] > 50 else
                "Midfielder"       if stats["midfield_%"] > 45  else
                "Attacking player"
            ),
            "fitness_level": (
                "High"   if stats["distance_km"] > 9 else
                "Medium" if stats["distance_km"] > 6 else
                "Low"
            ),
            "pace_rating": (
                "Excellent" if stats["max_speed_kmh"] > 28 else
                "Good"      if stats["max_speed_kmh"] > 22 else
                "Average"
            ),
        }
    })


@app.get("/results/{filename}")
async def download_csv(filename: str):
    """Download a player tracking CSV file."""
    path = f"outputs/{filename}"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(path, media_type="text/csv", filename=filename)


@app.get("/heatmap/{filename}")
async def download_heatmap(filename: str):
    """Download a player heatmap image."""
    path = f"outputs/{filename}"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(path, media_type="image/png", filename=filename)


@app.get("/list-results")
async def list_results():
    """List all available result files."""
    files = os.listdir("outputs") if os.path.exists("outputs") else []
    return {"total_files": len(files), "files": files}
