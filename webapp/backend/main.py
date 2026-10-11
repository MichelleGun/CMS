"""
FastAPI app: static frontend, REST API, and the telemetry WebSocket. Run
with:

    conda activate drones
    cd webapp/backend
    uvicorn main:app --port 8000

Then open http://localhost:8000 in a browser.
"""
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from config import ALGORITHMS, REWARD_INFO, ASSETS_DIR, FRONTEND_DIR, HOOP_SPEC, REACH_RADIUS, ROOM, RUNS_DIR, START_MARGIN, START_Z, XY_MARGIN, Z_RANGE
from drone_manager import drone_manager
from drone_ws import stream_flight
from imitation_manager import imitation_manager
from jobs import job_manager
from run_manager import run_manager
from schemas import (CameraToggleRequest, DroneCameraToggleRequest, EvaluateRequest,
                     GenerateDemosRequest, LabelDemoRequest, ReplayRequest,
                     RealtimeToggleRequest, RunLabelRequest, RunStarRequest, SaveAnswerRequest, ShowcaseRequest, StartFlightRequest, StartRunRequest,
                     TeacherDemosRequest, TrainCloneRequest)
from telemetry_ws import stream_run

app = FastAPI(title="Hoop-finding drone dashboard")


def _jpeg_snapshot(path):
    """A camera frame read fully into memory first. The trainer replaces these files several
    times a second; streaming one with FileResponse can be cut short mid-replace, which the
    browser shows as a broken image."""
    for _ in range(3):
        try:
            data = path.read_bytes()
            if data:
                return Response(content=data, media_type="image/jpeg",
                                headers={"Cache-Control": "no-store"})
        except OSError:
            pass
    raise HTTPException(status_code=503, detail="frame busy, try again")


@app.middleware("http")
async def no_stale_frontend(request: Request, call_next):
    """Make browsers revalidate the dashboard's own html/js/css on every load, so a
    restarted backend never runs against an old cached script. Not /assets (the 34 MB scan
    may stay cached) and not /media (videos)."""
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path.endswith((".html", ".js", ".css")):
        if not path.startswith(("/assets/", "/media/")):
            response.headers["Cache-Control"] = "no-cache"
    return response

# Real takeoffs and keyboard flying are accepted only from the computer running the
# backend: someone at that keyboard can see the cage. uvicorn binds to 127.0.0.1 by
# default; this also holds if someone starts it with --host 0.0.0.0.
LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost"}


def _is_local(client) -> bool:
    return client is not None and client.host in LOCAL_HOSTS


# --------------------------------------------------------------------- API

@app.get("/api/config")
def get_config():
    return {
        "room": ROOM,
        "hoop": HOOP_SPEC,
        "xy_margin": XY_MARGIN,
        "start_margin": START_MARGIN,
        "z_range": Z_RANGE,
        "start_z": START_Z,
        "reach_radius": REACH_RADIUS,
        "algorithms": ALGORITHMS,
        "reward": REWARD_INFO,
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
            env=req.env, ring_jitter=req.ring_jitter, radius_start=req.radius_start,
            sway_deg=req.sway_deg, carry_start=req.carry_start,
            reverse_prob=req.reverse_prob, lean_flip_prob=req.lean_flip_prob,
            until_stopped=req.until_stopped, vec=req.vec,
            init_run=req.init_run, init_clone=req.init_clone, demo_ids=req.demo_ids,
            human_weight=req.human_weight, bc_coef=req.bc_coef, profile=req.profile,
            onboard_dr=req.onboard_dr, onboard_sway_deg=req.onboard_sway_deg,
            onboard_wash_deg=req.onboard_wash_deg, wash_deg=req.wash_deg, start_z=req.start_z,
            hard_start_p=req.hard_start_p, dr_steps=req.dr_steps, bc_augment=req.bc_augment,
            wind_steps=req.wind_steps, reward_mode=req.reward_mode, budget=req.budget,
            laps=req.laps, reward_weights=req.reward_weights, onboard_laps=req.onboard_laps,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"run_id": run_id, "status": "starting"}


@app.put("/api/runs/{run_id}/label")
def set_run_label(run_id: str, req: RunLabelRequest):
    try:
        label = run_manager.set_label(run_id, req.label)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"run_id": run_id, "label": label}


@app.put("/api/runs/{run_id}/star")
def set_run_star(run_id: str, req: RunStarRequest):
    try:
        starred = run_manager.set_starred(run_id, req.starred)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"run_id": run_id, "starred": starred}


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


@app.post("/api/runs/{run_id}/showcase", status_code=202)
def start_showcase(run_id: str, req: ShowcaseRequest):
    try:
        run_manager.start_showcase(run_id, req.variant, req.seed, req.realtime)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"run_id": run_id, "status": "playing"}


@app.post("/api/runs/{run_id}/realtime")
def set_realtime(run_id: str, req: RealtimeToggleRequest):
    try:
        run_manager.set_realtime(run_id, req.enabled)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"run_id": run_id, "realtime": req.enabled}


@app.get("/api/runs/{run_id}/showcase/cam.jpg")
def get_showcase_frame(run_id: str):
    path = RUNS_DIR / run_id / "showcase_cam.jpg"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="no showcase frame yet")
    return _jpeg_snapshot(path)


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
    return _jpeg_snapshot(path)


@app.websocket("/ws/runs/{run_id}")
async def ws_run(websocket: WebSocket, run_id: str):
    await stream_run(websocket, run_id)


# ------------------------------------------------------------- real drone
# Real flights start from here with confirm=true, from this computer only, one at a
# time -- see drone_manager's docstring. Land and kill are always accepted.

@app.get("/api/drone/status")
def get_drone_status():
    return drone_manager.status()


@app.get("/api/drone/flights")
def get_drone_flights(limit: int = 25):
    return drone_manager.list_flights(limit=limit if limit > 0 else None)


@app.get("/api/drone/flights/{flight_id}")
def get_drone_flight(flight_id: str):
    flight = drone_manager.get_flight(flight_id)
    if flight is None:
        raise HTTPException(status_code=404, detail=f"no such flight '{flight_id}'")
    return flight


@app.post("/api/drone/flights", status_code=201)
def start_drone_flight(req: StartFlightRequest, request: Request):
    if (not req.dry_run or req.mode == "human") and not _is_local(request.client):
        raise HTTPException(status_code=403,
                            detail="real and keyboard flights start only from the computer "
                                   "running the dashboard")
    try:
        job = drone_manager.start_flight(
            mode=req.mode, profile=req.profile, seed=req.seed, attempts=req.attempts,
            no_forward=req.no_forward, no_commit=req.no_commit,
            max_travel=req.max_travel, max_airborne=req.max_airborne,
            record=req.record, model=req.model, dry_run=req.dry_run,
            confirm=req.confirm, human_gain=req.human_gain)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"job": job, "status": "starting", "dry_run": req.dry_run}


@app.get("/api/drone/flights/{flight_id}/events")
def get_drone_events(flight_id: str):
    try:
        return {"events": drone_manager.events(flight_id)}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/drone/flights/{flight_id}/ticks")
def get_drone_ticks(flight_id: str):
    try:
        return {"ticks": drone_manager.ticks(flight_id)}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/drone/flights/{flight_id}/save")
def answer_drone_save(flight_id: str, req: SaveAnswerRequest):
    try:
        drone_manager.answer_save(flight_id, req.save, req.passed)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"flight_id": flight_id, "save": req.save, "passed": req.passed}


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


@app.websocket("/ws/drone/{flight_id}/keys")
async def ws_drone_keys(websocket: WebSocket, flight_id: str):
    """Held keys from the browser: {"held": ["fb+", ...]}. Disconnecting releases them."""
    if not _is_local(websocket.client):
        await websocket.close(code=1008)
        return
    await websocket.accept()
    try:
        while True:
            msg = await websocket.receive_json()
            held = msg.get("held", []) if isinstance(msg, dict) else []
            try:
                drone_manager.send_keys(flight_id, held if isinstance(held, list) else [])
            except (ValueError, OSError):
                pass
    except (WebSocketDisconnect, ValueError):
        pass
    finally:
        try:
            drone_manager.send_keys(flight_id, [])
        except (ValueError, OSError):
            pass


# ------------------------------------------------------- imitation learning

def _job_errors(fn):
    try:
        return {"job_id": fn()}
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/imitation/demos")
def list_demos():
    return {"demos": imitation_manager.list_demos()}


@app.post("/api/imitation/demos/generate", status_code=202)
def generate_demos(req: GenerateDemosRequest):
    return _job_errors(lambda: imitation_manager.generate_demos(
        req.name, req.episodes, req.profile, req.seed, req.render,
        dr=req.dr, sway_deg=req.sway_deg, hard_start_p=req.hard_start_p))


@app.post("/api/imitation/demos/teacher", status_code=202)
def generate_teacher_demos(req: TeacherDemosRequest):
    """A PyBullet run flies the onboard simulator as a teacher (make_teacher_demos.py)."""
    return _job_errors(lambda: imitation_manager.generate_teacher_demos(
        req.run_id, req.name, req.episodes, req.profile, req.seed))


@app.post("/api/imitation/demos/label")
def label_demo(req: LabelDemoRequest):
    try:
        return {"demo_id": imitation_manager.label_demo(req.demo_id, req.passed)}
    except (ValueError, OSError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/imitation/models")
def list_models():
    return {"models": imitation_manager.list_models()}


@app.post("/api/imitation/train", status_code=202)
def train_clone(req: TrainCloneRequest):
    return _job_errors(lambda: imitation_manager.train(
        req.name, req.demo_ids, req.epochs, req.lr, req.use, req.eval_episodes, req.seed))


@app.post("/api/imitation/models/{name}/evaluate", status_code=202)
def evaluate_clone(name: str, req: EvaluateRequest):
    return _job_errors(lambda: imitation_manager.evaluate(name, req.episodes, req.render))


# --------------------------------------------------------------------- jobs

@app.get("/api/jobs")
def list_jobs(limit: int = 50):
    return {"jobs": job_manager.list(limit), "active": job_manager.active_job()}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    try:
        job = job_manager.get(job_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    if job is None:
        raise HTTPException(status_code=404, detail=f"no such job '{job_id}'")
    return job


@app.get("/api/jobs/{job_id}/log")
def get_job_log(job_id: str, offset: int = 0):
    try:
        text, new_offset = job_manager.log(job_id, max(0, offset))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"text": text, "offset": new_offset}


@app.post("/api/jobs/{job_id}/stop")
def stop_job(job_id: str):
    try:
        return {"job_id": job_id, "stopped": job_manager.stop(job_id)}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ----------------------------------------------------------------- static

app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")
app.mount("/media", StaticFiles(directory=str(RUNS_DIR)), name="media")
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
