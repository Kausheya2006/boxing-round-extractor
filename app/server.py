import asyncio
import json
import os
import threading
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
from typing import List, Optional

from app.pipeline import extract_rounds_from_video, verify_extracted_rounds
from app.find_moments.moments import find_key_moments_for_rounds
from app.find_moments.verify_moments import verify_key_moments
from app.config import get_device

from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

# Allow your local HTML frontend to communicate with FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
assets_dir = os.path.join(base_dir, "assets")
os.makedirs(assets_dir, exist_ok=True)
app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in list(self.active_connections):
            try:
                await connection.send_text(message)
            except Exception:
                self.disconnect(connection)

manager = ConnectionManager()
cancel_events = {}

@app.websocket("/ws/logs")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

def get_log_cb(video_path: str = ""):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return None

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assets_dir = os.path.join(base_dir, "assets")
    os.makedirs(assets_dir, exist_ok=True)

    prefix = ""
    log_file = os.path.join(assets_dir, "server_logs.txt")
    
    if video_path:
        from pathlib import Path
        video_name = Path(video_path).stem
        prefix = f"[{video_name}] "
        log_file = os.path.join(assets_dir, f"server_logs_{video_name}.txt")

    def log_cb(msg: str):
        formatted_msg = f"{prefix}{msg}"
        print(f"[LOG] {formatted_msg}")
        with open(log_file, "a") as f:
            f.write(f"{formatted_msg}\n")
        asyncio.run_coroutine_threadsafe(manager.broadcast(formatted_msg), loop)
        
    return log_cb


def get_cache_path(video_path: str, suffix: str) -> str:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assets_dir = os.path.join(base_dir, "assets")
    os.makedirs(assets_dir, exist_ok=True)
    video_name = Path(video_path).stem
    
    if os.getenv("MOCK_MODE") == "True":
        video_name = f"mock_{video_name}"
        
    return os.path.join(assets_dir, f"{video_name}_{suffix}.json")


def add_to_clipboard(video_path: str):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assets_dir = os.path.join(base_dir, "assets")
    os.makedirs(assets_dir, exist_ok=True)
    clipboard_file = os.path.join(assets_dir, "clipboard.txt")
    
    paths = []
    if os.path.exists(clipboard_file):
        with open(clipboard_file, "r") as f:
            paths = [line.strip() for line in f if line.strip()]
            
    if video_path not in paths:
        paths.append(video_path)
        with open(clipboard_file, "w") as f:
            for p in paths:
                f.write(f"{p}\n")


class VideoRequest(BaseModel):
    video_path: str
    force: bool = False

class ExtractRequest(BaseModel):
    video_path: str
    reuse_debug: bool = False
    force: bool = False

class VerifyRequest(BaseModel):
    video_path: str
    extrapolated_rounds: list
    force: bool = False

class CacheCheckRequest(BaseModel):
    video_path: str

class FindMomentsRequest(BaseModel):
    video_path: str
    rounds: list
    force: bool = False

class VerifyMomentsRequest(BaseModel):
    video_path: str
    moments: dict
    force: bool = False

class VerifyMomentsRequest(BaseModel):
    video_path: str
    moments: dict
    force: bool = False

class CancelRequest(BaseModel):
    video_path: str

@app.post("/cancel")
def cancel_task(request: CancelRequest):
    video_path_str = str(Path(request.video_path).expanduser())
    if video_path_str in cancel_events:
        cancel_events[video_path_str].set()
        return {"status": "cancelled"}
    return {"status": "not_running"}

@app.post("/check_cache")
def check_cache(request: CacheCheckRequest):
    video_stem = Path(request.video_path).stem
    extract_path = get_cache_path(request.video_path, "extracts")
    verified_path = get_cache_path(request.video_path, "verified")
    moments_path = get_cache_path(request.video_path, "moments")
    verified_moments_path = get_cache_path(request.video_path, "verified_moments")

    return {
        "extract_exists": os.path.exists(extract_path),
        "verify_exists": os.path.exists(verified_path),
        "moments_exists": os.path.exists(moments_path),
        "verified_moments_exists": os.path.exists(verified_moments_path)
    }

@app.get("/")
def root():
    return {"message": "Boxing Round Detector API is running"}


@app.get("/recent_videos")
def recent_videos():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assets_dir = os.path.join(base_dir, "assets")
    clipboard_file = os.path.join(assets_dir, "clipboard.txt")
    paths = []
    if os.path.exists(clipboard_file):
        with open(clipboard_file, "r") as f:
            paths = [line.strip() for line in f if line.strip()]
    return {"recent_videos": paths}


@app.post("/extract")
async def extract(request: ExtractRequest):
    video_path = Path(request.video_path).expanduser()

    if not video_path.exists() or not video_path.is_file():
        raise HTTPException(
            status_code=400,
            detail=f"Video file not found or invalid path: {video_path}"
        )

    add_to_clipboard(request.video_path)
    cache_path = get_cache_path(str(video_path), "extracts")
    log_cb = get_log_cb(str(video_path))

    if not request.force and os.path.exists(cache_path):
        if log_cb: log_cb("Loading extracted rounds from cache...")
        with open(cache_path, 'r') as f:
            return json.load(f)

    video_path_str = str(video_path)
    cancel_events[video_path_str] = threading.Event()
    check_cancel = lambda: cancel_events.get(video_path_str) and cancel_events[video_path_str].is_set()

    try:
        data = await asyncio.to_thread(extract_rounds_from_video, video_path_str, request.reuse_debug, log_cb, check_cancel)
        with open(cache_path, 'w') as f:
            json.dump(data, f)
        if log_cb: log_cb("Saved extraction results to cache.")
        return data
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    finally:
        if video_path_str in cancel_events:
            del cancel_events[video_path_str]

@app.post("/verify")
async def verify(request: VerifyRequest):
    video_path = Path(request.video_path).expanduser()

    if not video_path.exists() or not video_path.is_file():
        raise HTTPException(
            status_code=400,
            detail=f"Video file not found or invalid path: {video_path}"
        )

    cache_path = get_cache_path(str(video_path), "verified")
    log_cb = get_log_cb(str(video_path))

    if not request.force and os.path.exists(cache_path):
        if log_cb: log_cb("Loading verified rounds from cache...")
        with open(cache_path, 'r') as f:
            return json.load(f)

    video_path_str = str(video_path)
    cancel_events[video_path_str] = threading.Event()
    check_cancel = lambda: cancel_events.get(video_path_str) and cancel_events[video_path_str].is_set()

    try:
        results = await asyncio.to_thread(verify_extracted_rounds, video_path_str, request.extrapolated_rounds, log_cb, check_cancel)
        data = {"verified_rounds": results}
        with open(cache_path, 'w') as f:
            json.dump(data, f)
        if log_cb: log_cb("Saved verification results to cache.")
        return data
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    finally:
        if video_path_str in cancel_events:
            del cancel_events[video_path_str]

@app.post("/find_moments")
async def find_moments(request: FindMomentsRequest):
    video_path = Path(request.video_path).expanduser()

    if not video_path.exists() or not video_path.is_file():
        raise HTTPException(
            status_code=400,
            detail=f"Video file not found or invalid path: {video_path}"
        )

    cache_path = get_cache_path(str(video_path), "moments")
    log_cb = get_log_cb(str(video_path))

    if not request.force and os.path.exists(cache_path):
        if log_cb: log_cb("Loading key moments from cache...")
        with open(cache_path, 'r') as f:
            return json.load(f)

    video_path_str = str(video_path)
    cancel_events[video_path_str] = threading.Event()
    check_cancel = lambda: cancel_events.get(video_path_str) and cancel_events[video_path_str].is_set()

    try:
        results = await asyncio.to_thread(find_key_moments_for_rounds, video_path_str, request.rounds, log_cb, check_cancel)
        data = {"key_moments": results}
        with open(cache_path, 'w') as f:
            json.dump(data, f)
        if log_cb: log_cb("Saved key moments results to cache.")
        return data
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    finally:
        if video_path_str in cancel_events:
            del cancel_events[video_path_str]

@app.post("/verify_moments")
async def verify_moments_endpoint(request: VerifyMomentsRequest):
    video_path = Path(request.video_path).expanduser()
    
    if not video_path.exists() or not video_path.is_file():
        raise HTTPException(status_code=400, detail="Video file not found.")

    video_path_str = str(video_path)
    cache_path = get_cache_path(video_path_str, "verified_moments")

    log_cb = get_log_cb(video_path_str)
    
    if not request.force and os.path.exists(cache_path):
        if log_cb: log_cb("Using cached visually verified moments.")
        with open(cache_path, 'r') as f:
            return json.load(f)

    cancel_events[video_path_str] = threading.Event()
    check_cancel = lambda: cancel_events.get(video_path_str) and cancel_events[video_path_str].is_set()

    try:
        results = await asyncio.to_thread(verify_key_moments, video_path_str, request.moments, log_cb, check_cancel)
        data = {"verified_moments": results}
        
        # Save as JSON
        with open(cache_path, 'w') as f:
            json.dump(data, f)
            
        # Also save as a clean CSV database
        import csv
        csv_path = cache_path.replace('.json', '.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Round", "Timestamp", "Verified", "Category", "Landed_Status", "Attacker_vs_Defender", "Attack_Type", "Defender_Reaction"])
            for round_name, items in results.items():
                for item in items:
                    writer.writerow([
                        round_name,
                        item.get("timestamp", ""),
                        item.get("verified", False),
                        item.get("category", ""),
                        item.get("landed_status", ""),
                        item.get("attacker_vs_defender", ""),
                        item.get("attack_type", ""),
                        item.get("defender_reaction", "")
                    ])
                    
        if log_cb: log_cb("Saved verified key moments to JSON and CSV database.")
        return data
    except Exception as e:
        if str(e) == "Cancelled by user":
            raise HTTPException(status_code=499, detail="Task cancelled by user.")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if video_path_str in cancel_events:
            del cancel_events[video_path_str]