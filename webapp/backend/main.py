"""
FastAPI app: static frontend, REST API, and the telemetry WebSocket. Run
with:

    conda activate drones
    cd webapp/backend
    uvicorn main:app --port 8000

Then open http://localhost:8000 in a browser.
"""
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config import ALGORITHMS, FRONTEND_DIR, REACH_RADIUS, ROOM, RUNS_DIR, START_Z, XY_MARGIN, Z_RANGE
from drone_manager import drone_manager
from drone_ws import stream_flight
from run_manager import run_manager
from schemas import (CameraToggleRequest, DroneCameraToggleRequest, ReplayRequest,
                     StartFlightRequest, StartRunRequest)
from telemetry_ws import stream_run

app = FastAPI(title="Hoop-finding drone dashboard")


# --------------------------------------------------------------------- API

@app.get("/api/config")
def get_config():
    return {
        "room": ROOM,
        "xy_margin": XY_MARGIN,
        "z_range": Z_RANGE,
        "start_z": START_Z,
        "reach_radius": REACH_RADIUS,
        "algorithms": ALGORITHMS,
    }


@app.get("/api/runs")
def get_runs():
    return run_manager.list_runs()


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    run = run_manager.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"no such run '{run_id}'")
    return run


@app.post("/api/runs", status_code=201)
def start_run(req: StartRunRequest):
    try:
        run_id = run_manager.start_run(
            algo=req.algo, seed=req.seed,
            start_pos=(req.start_pos.as_list() if req.start_pos else None),
            target_pos=(req.target_pos.as_list() if req.target_pos else None),
            timesteps=req.timesteps, n_envs=req.n_envs,
            eval_freq=req.eval_freq, n_eval_episodes=req.n_eval_episodes,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"run_id": run_id, "status": "starting"}


@app.post("/api/runs/{run_id}/stop")
def stop_run(run_id: str):
    stopped = run_manager.stop_run(run_id)
    if not stopped:
        raise HTTPException(status_code=404, detail=f"no active process for run '{run_id}'")
    return {"run_id": run_id, "status": "stopping"}


@app.post("/api/runs/{run_id}/replay", status_code=202)
def start_replay(run_id: str, req: ReplayRequest):
    try:
        replay_id = run_manager.start_replay(run_id, req.variant, req.episodes, req.pov)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"replay_id": replay_id, "status": "rendering"}


@app.get("/api/runs/{run_id}/replay/{replay_id}")
def get_replay(run_id: str, replay_id: str, pov: bool = False):
    return run_manager.get_replay_status(run_id, replay_id, pov=pov)


@app.post("/api/runs/{run_id}/camera")
def set_camera(run_id: str, req: CameraToggleRequest):
    try:
        run_manager.set_camera(run_id, req.enabled)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"run_id": run_id, "enabled": req.enabled}


@app.get("/api/runs/{run_id}/camera/latest.jpg")
def get_camera_frame(run_id: str):
    path = RUNS_DIR / run_id / "camera_latest.jpg"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="camera has not been enabled for this run yet")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.websocket("/ws/runs/{run_id}")
async def ws_run(websocket: WebSocket, run_id: str):
    await stream_run(websocket, run_id)


# ------------------------------------------------------------- real drone
# The dashboard can watch a real flight and make it safer; it cannot start one.
# POST /api/drone/flights accepts dry runs only -- see drone_manager's docstring.

@app.get("/api/drone/status")
def get_drone_status():
    return drone_manager.status()


@app.get("/api/drone/flights")
def get_drone_flights(limit: int = 25):
    return drone_manager.list_flights(limit=limit)


@app.get("/api/drone/flights/{flight_id}")
def get_drone_flight(flight_id: str):
    flight = drone_manager.get_flight(flight_id)
    if flight is None:
        raise HTTPException(status_code=404, detail=f"no such flight '{flight_id}'")
    return flight


@app.post("/api/drone/flights", status_code=201)
def start_drone_flight(req: StartFlightRequest):
    try:
        job = drone_manager.start_flight(
            mode=req.mode, profile=req.profile, seed=req.seed, attempts=req.attempts,
            no_forward=req.no_forward, no_commit=req.no_commit,
            max_travel=req.max_travel, max_airborne=req.max_airborne,
            record=req.record, model=req.model, dry_run=req.dry_run)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"job": job, "status": "starting", "dry_run": True}


@app.post("/api/drone/flights/{flight_id}/land")
def land_drone(flight_id: str):
    try:
        drone_manager.request_land(flight_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"flight_id": flight_id, "status": "landing_requested"}


@app.post("/api/drone/flights/{flight_id}/kill")
def kill_drone(flight_id: str):
    try:
        drone_manager.request_kill(flight_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"flight_id": flight_id, "status": "kill_requested"}


@app.post("/api/drone/flights/{flight_id}/stop")
def stop_drone_flight(flight_id: str):
    try:
        stopped = drone_manager.stop_flight(flight_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"flight_id": flight_id, "status": "stopping" if stopped else "not_running"}


@app.post("/api/drone/flights/{flight_id}/camera")
def set_drone_camera(flight_id: str, req: DroneCameraToggleRequest):
    try:
        drone_manager.set_camera(flight_id, req.enabled)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"flight_id": flight_id, "enabled": req.enabled}


@app.get("/api/drone/flights/{flight_id}/camera/latest.jpg")
def get_drone_camera_frame(flight_id: str):
    try:
        path = drone_manager.camera_frame_path(flight_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    if not path.is_file():
        raise HTTPException(status_code=404,
                            detail="the drone camera has not been enabled for this flight yet")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.websocket("/ws/drone/{flight_id}")
async def ws_drone(websocket: WebSocket, flight_id: str):
    await stream_flight(websocket, flight_id)


# ----------------------------------------------------------------- static

app.mount("/media", StaticFiles(directory=str(RUNS_DIR)), name="media")
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
