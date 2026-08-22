import asyncio
import json
import os
import threading
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from typing import List, Optional

from app.pipeline import extract_rounds_from_video, verify_extracted_rounds

app = FastAPI()

# Allow your local HTML frontend to communicate with FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    extract_path = get_cache_path(request.video_path, "extracts")
    verify_path = get_cache_path(request.video_path, "verified")
    return {
        "extract_exists": os.path.exists(extract_path),
        "verify_exists": os.path.exists(verify_path)
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