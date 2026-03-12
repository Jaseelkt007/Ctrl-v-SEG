"""
Ctrl-V-Seg Backend API

FastAPI server for running inference and evaluation on trained Stage 1 and Stage 2 models.
Run on a SLURM GPU node. Frontend connects from the login node.
"""

import os
import sys
import json
import uuid
import shutil
import asyncio
import logging
import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional, List
from collections import defaultdict

# Load .env manually
_env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith('#') and '=' in _line:
                _k, _v = _line.split('=', 1)
                os.environ.setdefault(_k.strip(), _v.strip())
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')

import numpy as np
from PIL import Image
import imageio

from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Ensure ctrlv is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

app = FastAPI(title="Ctrl-V-Seg", description="Semantic Video Generation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Job storage
JOBS_DIR = os.path.join(os.path.dirname(__file__), 'jobs')
os.makedirs(JOBS_DIR, exist_ok=True)

# WebSocket connections for log streaming
ws_connections: dict[str, list[WebSocket]] = defaultdict(list)

# Global model manager (lazy loaded)
model_mgr = None


def get_model_manager():
    global model_mgr
    if model_mgr is None:
        from model_manager import ModelManager
        model_mgr = ModelManager()
    return model_mgr


async def send_log(job_id: str, message: str):
    """Send log message to all connected WebSocket clients for a job."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"

    # Save to job log file
    job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    log_file = os.path.join(job_dir, 'logs.txt')
    with open(log_file, 'a') as f:
        f.write(log_entry + '\n')

    # Send to WebSocket clients
    for ws in ws_connections.get(job_id, []):
        try:
            await ws.send_text(json.dumps({'type': 'log', 'message': log_entry}))
        except Exception:
            pass


def sync_log_fn(job_id: str, loop):
    """Create a synchronous log function that queues WebSocket sends."""
    def log(msg):
        asyncio.run_coroutine_threadsafe(send_log(job_id, msg), loop)
    return log


# ============================================================================
# Health & Info
# ============================================================================

@app.get("/api/health")
async def health():
    import torch
    return {
        "status": "ok",
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


@app.get("/api/info")
async def info():
    """Return model checkpoint info."""
    from model_manager import find_latest_checkpoint, STAGE1_CHECKPOINT_DIR, STAGE2_CHECKPOINT_DIR
    try:
        s1_path, s1_step = find_latest_checkpoint(STAGE1_CHECKPOINT_DIR)
        s1_info = {"checkpoint": s1_path, "step": s1_step}
    except Exception:
        s1_info = {"error": "not found"}
    try:
        s2_path, s2_step = find_latest_checkpoint(STAGE2_CHECKPOINT_DIR)
        s2_info = {"checkpoint": s2_path, "step": s2_step}
    except Exception:
        s2_info = {"error": "not found"}

    return {"stage1": s1_info, "stage2": s2_info}


# ============================================================================
# Dataset Browsing
# ============================================================================

@app.get("/api/dataset/samples")
async def get_dataset_samples(count: int = 20):
    """Get sample info from the dataset for browsing."""
    mgr = get_model_manager()
    try:
        mgr.load_dataset()
        from ctrlv.utils import get_n_training_samples
        samples = get_n_training_samples(mgr.dataloader, min(count, 50), show_progress=False)

        sample_infos = []
        for i, sample in enumerate(samples):
            # Save first frame thumbnail
            thumb_dir = os.path.join(JOBS_DIR, '_thumbnails')
            os.makedirs(thumb_dir, exist_ok=True)

            # RGB init frame
            if isinstance(sample['image_init'], Image.Image):
                rgb_path = os.path.join(thumb_dir, f'sample_{i}_rgb.jpg')
                sample['image_init'].save(rgb_path, quality=80)
            else:
                rgb_path = None

            # Semantic first frame (colorized)
            if 'semantic_ids' in sample:
                from ctrlv.utils.semantic_preprocessing import semantic_ids_to_viz_rgb
                sem_viz = semantic_ids_to_viz_rgb(sample['semantic_ids'][0].numpy())
                sem_path = os.path.join(thumb_dir, f'sample_{i}_sem.jpg')
                Image.fromarray(sem_viz).save(sem_path, quality=80)
            else:
                sem_path = None

            sample_infos.append({
                'index': i,
                'num_frames': sample['semantic_ids'].shape[0] if 'semantic_ids' in sample else 0,
                'rgb_thumb': f'/api/thumbnails/sample_{i}_rgb.jpg' if rgb_path else None,
                'sem_thumb': f'/api/thumbnails/sample_{i}_sem.jpg' if sem_path else None,
            })

        return {"samples": sample_infos, "total": len(mgr.dataset)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/thumbnails/{filename}")
async def get_thumbnail(filename: str):
    path = os.path.join(JOBS_DIR, '_thumbnails', filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404)
    return FileResponse(path)


# ============================================================================
# Stage 1: Generate Semantic Controls
# ============================================================================

@app.post("/api/stage1/generate")
async def stage1_generate(
    sample_index: Optional[int] = Form(None),
    image: Optional[UploadFile] = File(None),
    first_frame: Optional[UploadFile] = File(None),
    last_frame: Optional[UploadFile] = File(None),
    num_frames: int = Form(25),
    num_inference_steps: int = Form(30),
    min_guidance_scale: float = Form(3.0),
    max_guidance_scale: float = Form(7.0),
    noise_aug_strength: float = Form(0.01),
    seed: int = Form(1234),
    num_cond_bbox_frames: int = Form(1),
):
    """Start Stage 1 generation job."""
    job_id = str(uuid.uuid4())[:8]
    job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    # Save job config
    config = {
        'stage': 1,
        'sample_index': sample_index,
        'num_frames': num_frames,
        'num_inference_steps': num_inference_steps,
        'min_guidance_scale': min_guidance_scale,
        'max_guidance_scale': max_guidance_scale,
        'noise_aug_strength': noise_aug_strength,
        'seed': seed,
        'status': 'queued',
        'created_at': datetime.now().isoformat(),
    }
    with open(os.path.join(job_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # Save uploaded image if provided
    if image is not None:
        img_bytes = await image.read()
        img_path = os.path.join(job_dir, 'input_image.png')
        with open(img_path, 'wb') as f:
            f.write(img_bytes)

    # Save first/last frames if provided
    if first_frame is not None:
        img_bytes = await first_frame.read()
        img_path = os.path.join(job_dir, 'first_frame.png')
        with open(img_path, 'wb') as f:
            f.write(img_bytes)

    if last_frame is not None:
        img_bytes = await last_frame.read()
        img_path = os.path.join(job_dir, 'last_frame.png')
        with open(img_path, 'wb') as f:
            f.write(img_bytes)

    # Start generation in background
    loop = asyncio.get_event_loop()
    asyncio.create_task(_run_stage1_job(job_id, config, loop))

    return {"job_id": job_id, "status": "queued"}


async def _run_stage1_job(job_id: str, config: dict, loop):
    """Run Stage 1 generation in background."""
    job_dir = os.path.join(JOBS_DIR, job_id)
    log_fn = sync_log_fn(job_id, loop)

    try:
        # Update status
        config['status'] = 'running'
        _save_config(job_dir, config)
        await send_log(job_id, "Job started - Stage 1: RGB -> Semantic")

        mgr = get_model_manager()

        # Get input sample
        if config.get('sample_index') is not None:
            await send_log(job_id, f"Loading dataset sample {config['sample_index']}...")
            mgr.load_dataset(log_fn)
            from ctrlv.utils import get_n_training_samples
            samples = get_n_training_samples(mgr.dataloader, config['sample_index'] + 1, show_progress=False)
            sample = samples[config['sample_index']]
            image_init = sample['image_init']
            semantic_ids = sample['semantic_ids']
            bbox_img = sample['bbox_img']
            gt_semantic_ids = sample['semantic_ids'].numpy()
        else:
            await send_log(job_id, "Using uploaded image...")
            img_path = os.path.join(job_dir, 'input_image.png')
            image_init = Image.open(img_path).convert('RGB').resize((704, 192))
            # For custom uploads, create dummy conditioning
            import torch
            semantic_ids = torch.zeros(config['num_frames'], 192, 704, dtype=torch.long)
            bbox_img = torch.zeros(config['num_frames'], 3, 192, 704, dtype=torch.float32)
            gt_semantic_ids = None

        await send_log(job_id, "Starting inference pipeline...")

        # Run inference in executor to not block event loop
        pred_np = await loop.run_in_executor(
            None,
            lambda: mgr.run_stage1_inference(
                image_init=image_init,
                semantic_ids=semantic_ids,
                bbox_img=bbox_img,
                num_frames=config['num_frames'],
                num_inference_steps=config['num_inference_steps'],
                min_guidance_scale=config['min_guidance_scale'],
                max_guidance_scale=config['max_guidance_scale'],
                noise_aug_strength=config['noise_aug_strength'],
                seed=config['seed'],
                num_cond_bbox_frames=config.get('num_cond_bbox_frames', 1),
                log_fn=log_fn,
            )
        )

        await send_log(job_id, "Saving generated frames...")

        # Save frames
        from ctrlv.utils.semantic_preprocessing import (
            semantic_ids_to_viz_rgb, KITTI360_LABEL_MAPPING
        )
        frames_dir = os.path.join(job_dir, 'frames')
        pred_dir = os.path.join(frames_dir, 'pred')
        pred_color_dir = os.path.join(frames_dir, 'pred_color')
        os.makedirs(pred_dir, exist_ok=True)
        os.makedirs(pred_color_dir, exist_ok=True)

        gif_frames = []
        trainid_to_original = {v: k for k, v in KITTI360_LABEL_MAPPING.items()}

        for t in range(pred_np.shape[0]):
            # Colorized
            viz = semantic_ids_to_viz_rgb(pred_np[t])
            Image.fromarray(viz).save(os.path.join(pred_color_dir, f'frame_{t:03d}.png'))
            gif_frames.append(viz)

            # Grayscale (original KITTI-360 IDs)
            gray = np.zeros_like(pred_np[t], dtype=np.uint8)
            for tid, oid in trainid_to_original.items():
                gray[pred_np[t] == tid] = oid
            Image.fromarray(gray, mode='L').save(os.path.join(pred_dir, f'frame_{t:03d}.png'))

        # Save GT frames if available
        if gt_semantic_ids is not None:
            gt_dir = os.path.join(frames_dir, 'gt_color')
            os.makedirs(gt_dir, exist_ok=True)
            for t in range(gt_semantic_ids.shape[0]):
                viz = semantic_ids_to_viz_rgb(gt_semantic_ids[t])
                Image.fromarray(viz).save(os.path.join(gt_dir, f'frame_{t:03d}.png'))

        # Save RGB init
        if isinstance(image_init, Image.Image):
            image_init.save(os.path.join(frames_dir, 'input_rgb.png'))

        # Save GIF
        gif_path = os.path.join(job_dir, 'output.gif')
        imageio.mimsave(gif_path, gif_frames, fps=7, loop=0)
        await send_log(job_id, f"Saved GIF: {gif_path}")

        # Save pred_np for later evaluation
        np.save(os.path.join(job_dir, 'pred_semantic_ids.npy'), pred_np)
        if gt_semantic_ids is not None:
            np.save(os.path.join(job_dir, 'gt_semantic_ids.npy'), gt_semantic_ids)

        config['status'] = 'completed'
        config['num_generated_frames'] = int(pred_np.shape[0])
        _save_config(job_dir, config)

        await send_log(job_id, "Stage 1 generation complete!")
        await _send_status(job_id, 'completed')

    except Exception as e:
        config['status'] = 'error'
        config['error'] = str(e)
        _save_config(job_dir, config)
        await send_log(job_id, f"ERROR: {str(e)}")
        await send_log(job_id, traceback.format_exc())
        await _send_status(job_id, 'error')


# ============================================================================
# Stage 2: Generate RGB Video
# ============================================================================

@app.post("/api/stage2/generate")
async def stage2_generate(
    sample_index: Optional[int] = Form(None),
    image: Optional[UploadFile] = File(None),
    first_frame: Optional[UploadFile] = File(None),
    last_frame: Optional[UploadFile] = File(None),
    control_job_id: Optional[str] = Form(None),
    num_frames: int = Form(25),
    num_inference_steps: int = Form(30),
    min_guidance_scale: float = Form(1.0),
    max_guidance_scale: float = Form(3.0),
    conditioning_scale: float = Form(1.0),
    noise_aug_strength: float = Form(0.01),
    seed: int = Form(1234),
):
    """Start Stage 2 generation job."""
    job_id = str(uuid.uuid4())[:8]
    job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    config = {
        'stage': 2,
        'sample_index': sample_index,
        'control_job_id': control_job_id,
        'num_frames': num_frames,
        'num_inference_steps': num_inference_steps,
        'min_guidance_scale': min_guidance_scale,
        'max_guidance_scale': max_guidance_scale,
        'conditioning_scale': conditioning_scale,
        'noise_aug_strength': noise_aug_strength,
        'seed': seed,
        'status': 'queued',
        'created_at': datetime.now().isoformat(),
    }
    with open(os.path.join(job_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    if image is not None:
        img_bytes = await image.read()
        with open(os.path.join(job_dir, 'input_image.png'), 'wb') as f:
            f.write(img_bytes)

    # Save first/last frames if provided
    if first_frame is not None:
        img_bytes = await first_frame.read()
        img_path = os.path.join(job_dir, 'first_frame.png')
        with open(img_path, 'wb') as f:
            f.write(img_bytes)

    if last_frame is not None:
        img_bytes = await last_frame.read()
        img_path = os.path.join(job_dir, 'last_frame.png')
        with open(img_path, 'wb') as f:
            f.write(img_bytes)

    loop = asyncio.get_event_loop()
    asyncio.create_task(_run_stage2_job(job_id, config, loop))

    return {"job_id": job_id, "status": "queued"}


async def _run_stage2_job(job_id: str, config: dict, loop):
    """Run Stage 2 generation in background."""
    job_dir = os.path.join(JOBS_DIR, job_id)
    log_fn = sync_log_fn(job_id, loop)

    try:
        config['status'] = 'running'
        _save_config(job_dir, config)
        await send_log(job_id, "Job started - Stage 2: Semantic -> RGB Video")

        mgr = get_model_manager()

        if config.get('sample_index') is not None:
            await send_log(job_id, f"Loading dataset sample {config['sample_index']}...")
            mgr.load_dataset(log_fn)
            from ctrlv.utils import get_n_training_samples
            samples = get_n_training_samples(mgr.dataloader, config['sample_index'] + 1, show_progress=False)
            sample = samples[config['sample_index']]
            image_init = sample['image_init']
            semantic_ids = sample['semantic_ids']
            bbox_img = sample['bbox_img']
            gt_clip_np = sample.get('gt_clip_np', None)
            gt_semantic_ids = sample['semantic_ids'].numpy()
        elif config.get('control_job_id'):
            await send_log(job_id, f"Using control frames from job {config['control_job_id']}...")
            ctrl_dir = os.path.join(JOBS_DIR, config['control_job_id'])
            pred_path = os.path.join(ctrl_dir, 'pred_semantic_ids.npy')
            if not os.path.exists(pred_path):
                raise ValueError(f"Control job {config['control_job_id']} has no predictions")

            import torch
            pred_semantic_ids = np.load(pred_path)
            semantic_ids = torch.from_numpy(pred_semantic_ids).long()

            # Create bbox_img from semantic IDs
            from ctrlv.utils.semantic_preprocessing import semantic_ids_to_viz_rgb
            bbox_frames = []
            for t in range(semantic_ids.shape[0]):
                viz = semantic_ids_to_viz_rgb(pred_semantic_ids[t])
                viz_t = torch.from_numpy(viz).permute(2, 0, 1).float() / 255.0
                bbox_frames.append(viz_t)
            bbox_img = torch.stack(bbox_frames)

            # Load init image
            init_path = os.path.join(ctrl_dir, 'frames', 'input_rgb.png')
            img_path = os.path.join(job_dir, 'input_image.png')
            if os.path.exists(init_path):
                image_init = Image.open(init_path).convert('RGB')
            elif os.path.exists(img_path):
                image_init = Image.open(img_path).convert('RGB').resize((704, 192))
            else:
                raise ValueError("No input RGB image found")

            gt_clip_np = None
            gt_semantic_ids = None
        else:
            raise ValueError("Must provide either sample_index or control_job_id")

        await send_log(job_id, "Starting inference pipeline...")

        gen_frames_hwc = await loop.run_in_executor(
            None,
            lambda: mgr.run_stage2_inference(
                image_init=image_init,
                semantic_ids=semantic_ids,
                bbox_img=bbox_img,
                num_frames=config['num_frames'],
                num_inference_steps=config['num_inference_steps'],
                min_guidance_scale=config['min_guidance_scale'],
                max_guidance_scale=config['max_guidance_scale'],
                conditioning_scale=config['conditioning_scale'],
                noise_aug_strength=config['noise_aug_strength'],
                seed=config['seed'],
                log_fn=log_fn,
            )
        )

        await send_log(job_id, "Saving generated frames...")

        frames_dir = os.path.join(job_dir, 'frames')
        gen_dir = os.path.join(frames_dir, 'gen_rgb')
        os.makedirs(gen_dir, exist_ok=True)

        gif_frames = []
        for t in range(gen_frames_hwc.shape[0]):
            frame = gen_frames_hwc[t]
            Image.fromarray(frame).save(os.path.join(gen_dir, f'frame_{t:03d}.png'))
            gif_frames.append(frame)

        # Save GT RGB if available
        if gt_clip_np is not None:
            gt_dir = os.path.join(frames_dir, 'gt_rgb')
            os.makedirs(gt_dir, exist_ok=True)
            gt_hwc = np.transpose(gt_clip_np, (0, 2, 3, 1))
            for t in range(gt_hwc.shape[0]):
                Image.fromarray(gt_hwc[t]).save(os.path.join(gt_dir, f'frame_{t:03d}.png'))
            np.save(os.path.join(job_dir, 'gt_frames.npy'), gt_hwc)

        # Save semantic conditioning frames
        if gt_semantic_ids is not None:
            np.save(os.path.join(job_dir, 'gt_semantic_ids.npy'), gt_semantic_ids)

        # Save RGB init
        if isinstance(image_init, Image.Image):
            image_init.save(os.path.join(frames_dir, 'input_rgb.png'))

        # Save GIF
        gif_path = os.path.join(job_dir, 'output.gif')
        imageio.mimsave(gif_path, gif_frames, fps=7, loop=0)

        # Save raw frames for evaluation
        np.save(os.path.join(job_dir, 'gen_frames.npy'), gen_frames_hwc)

        config['status'] = 'completed'
        config['num_generated_frames'] = int(gen_frames_hwc.shape[0])
        _save_config(job_dir, config)

        await send_log(job_id, "Stage 2 generation complete!")
        await _send_status(job_id, 'completed')

    except Exception as e:
        config['status'] = 'error'
        config['error'] = str(e)
        _save_config(job_dir, config)
        await send_log(job_id, f"ERROR: {str(e)}")
        await send_log(job_id, traceback.format_exc())
        await _send_status(job_id, 'error')


# ============================================================================
# Gemini AI Analysis
# ============================================================================

import urllib.request

def _call_gemini_sync(prompt: str) -> dict:
    """Call Gemini API synchronously (run in executor)."""
    if not GEMINI_API_KEY:
        return {"error": "GEMINI_API_KEY not set"}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.3}
    }).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    text = data['candidates'][0]['content']['parts'][0]['text']
    return json.loads(text)


@app.post("/api/jobs/{job_id}/analyse")
async def analyse_job(job_id: str):
    """Use Gemini to analyse evaluation results for a job."""
    job_dir = os.path.join(JOBS_DIR, job_id)
    config = _load_config(job_dir)
    if not config:
        raise HTTPException(status_code=404, detail="Job not found")

    eval_path = os.path.join(job_dir, 'eval_results.json')
    if not os.path.exists(eval_path):
        raise HTTPException(status_code=400, detail="No evaluation results. Run evaluation first.")
    with open(eval_path) as f:
        eval_results = json.load(f)

    stage = config.get('stage', 1)
    # Build prompt
    if stage == 1:
        prompt = f"""You are a computer vision researcher analyzing semantic segmentation prediction results from a video diffusion model trained on KITTI-360 autonomous driving data.

Model: Stage 1 Semantic Predictor (SVD-based, predicts future semantic segmentation from RGB input)
Dataset: KITTI-360 (19 semantic classes: road, sidewalk, building, wall, fence, pole, traffic light, traffic sign, vegetation, terrain, sky, person, rider, car, truck, bus, train, motorcycle, bicycle)
Training step: {config.get('num_inference_steps', 'unknown')} inference steps
Config: sample_index={config.get('sample_index')}, seed={config.get('seed')}

Evaluation Results (comparing predicted vs ground-truth semantic IDs):
{json.dumps(eval_results, indent=2)}

Respond ONLY with valid JSON in this exact schema:
{{
  "summary": "2-3 sentence overall assessment of these results",
  "score": <integer 1-10>,
  "score_label": "Poor/Fair/Good/Very Good/Excellent",
  "observations": ["observation 1", "observation 2", "observation 3"],
  "issues": ["issue or weakness 1", "issue 2"],
  "class_insights": ["insight about specific class performance 1", "insight 2"],
  "recommendations": ["concrete recommendation 1", "recommendation 2"],
  "next_steps": ["specific next step 1", "step 2"]
}}"""
    else:
        prompt = f"""You are a computer vision researcher analyzing video generation quality from a ControlNet-based diffusion model trained on KITTI-360 autonomous driving data.

Model: Stage 2 Sem2Video (SVD ControlNet, generates RGB video from semantic maps)
Dataset: KITTI-360
Config: sample_index={config.get('sample_index')}, conditioning_scale={config.get('conditioning_scale')}, seed={config.get('seed')}

Evaluation Results (DRN semantic accuracy on generated RGB + image quality metrics):
{json.dumps(eval_results, indent=2)}

Metrics guide: DRN mIoU measures semantic consistency of generated frames. SSIM (0-1, higher=better), PSNR (higher=better, >25dB good), LPIPS (0-1, lower=better, <0.15 good).

Respond ONLY with valid JSON in this exact schema:
{{
  "summary": "2-3 sentence overall assessment",
  "score": <integer 1-10>,
  "score_label": "Poor/Fair/Good/Very Good/Excellent",
  "observations": ["observation 1", "observation 2", "observation 3"],
  "issues": ["issue 1", "issue 2"],
  "class_insights": ["insight about generation quality 1", "insight 2"],
  "recommendations": ["recommendation 1", "recommendation 2"],
  "next_steps": ["next step 1", "step 2"]
}}"""

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, lambda: _call_gemini_sync(prompt))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")


@app.post("/api/compare")
async def compare_jobs(body: dict):
    """Use Gemini to compare two jobs."""
    job_id1 = body.get('job_id1')
    job_id2 = body.get('job_id2')
    if not job_id1 or not job_id2:
        raise HTTPException(status_code=400, detail="Provide job_id1 and job_id2")

    def load_job(jid):
        d = os.path.join(JOBS_DIR, jid)
        cfg = _load_config(d)
        eval_p = os.path.join(d, 'eval_results.json')
        ev = json.load(open(eval_p)) if os.path.exists(eval_p) else None
        return cfg, ev

    cfg1, ev1 = load_job(job_id1)
    cfg2, ev2 = load_job(job_id2)
    cfg1 = cfg1 or {}
    cfg2 = cfg2 or {}
    stage = cfg1.get('stage', 1)

    # Build config diff table programmatically
    if stage == 1:
        param_keys = [
            'num_frames', 'num_inference_steps', 'min_guidance_scale', 'max_guidance_scale',
            'noise_aug_strength', 'num_cond_bbox_frames', 'seed', 'sample_index',
        ]
    else:
        param_keys = [
            'num_frames', 'num_inference_steps', 'min_guidance_scale', 'max_guidance_scale',
            'conditioning_scale', 'noise_aug_strength', 'seed', 'sample_index', 'control_job_id',
        ]

    config_diff = []
    for key in param_keys:
        v1 = cfg1.get(key)
        v2 = cfg2.get(key)
        config_diff.append({'param': key, 'job_a': v1, 'job_b': v2, 'changed': v1 != v2})

    # Format config table for Gemini prompt
    header = "| Parameter | Job A | Job B | Changed |"
    sep    = "|-----------|-------|-------|---------|"
    rows   = [f"| {r['param']} | {r['job_a']} | {r['job_b']} | {'YES' if r['changed'] else '-'} |"
              for r in config_diff]
    config_table_str = "\n".join([header, sep] + rows)

    prompt = f"""You are a computer vision researcher comparing two experiment runs of a {'semantic segmentation predictor (Stage 1: RGB→Semantic)' if stage == 1 else 'semantic-to-video ControlNet (Stage 2: Semantic→RGB)'} trained on KITTI-360 autonomous driving data.

=== CONFIGURATION COMPARISON ===
{config_table_str}

=== JOB A ({job_id1}) EVALUATION RESULTS ===
{json.dumps(ev1, indent=2) if ev1 else 'No evaluation available'}

=== JOB B ({job_id2}) EVALUATION RESULTS ===
{json.dumps(ev2, indent=2) if ev2 else 'No evaluation available'}

{'Metrics guide (Stage 1): mIoU (higher=better, >50% good), pixel_accuracy (higher=better).' if stage == 1 else 'Metrics guide (Stage 2): DRN mIoU measures semantic consistency of generated RGB (higher=better). SSIM (0-1, higher=better), PSNR (higher=better, >25dB good), LPIPS (0-1, lower=better, <0.15 good).'}

Focus especially on HOW the configuration differences (rows marked YES in the table) explain the metric differences between the two jobs.

Respond ONLY with valid JSON in this exact schema:
{{
  "winner": "A or B or tie",
  "winner_reason": "1 sentence citing specific metrics",
  "summary": "2-3 sentence comparison summary mentioning key config changes and their effect",
  "improvements": ["specific metric or quality improvement from A to B, citing numbers"],
  "regressions": ["specific metric or quality regression from A to B, citing numbers"],
  "config_impact": ["for each CHANGED config param: explain the likely effect on quality — e.g. 'Increasing num_inference_steps from 20 to 50 gave more refined predictions, boosting mIoU by X'"],
  "recommendation": "Concrete next experiment to run based on these two runs"
}}"""

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, lambda: _call_gemini_sync(prompt))
        result['job_id1'] = job_id1
        result['job_id2'] = job_id2
        result['config_diff'] = config_diff  # attach computed diff for frontend table
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")


# ============================================================================
# Evaluation
# ============================================================================

@app.post("/api/jobs/{job_id}/evaluate")
async def evaluate_job(job_id: str):
    """Run evaluation on a completed job."""
    job_dir = os.path.join(JOBS_DIR, job_id)
    config = _load_config(job_dir)
    if not config:
        raise HTTPException(status_code=404, detail="Job not found")

    if config['status'] not in ('completed', 'evaluated'):
        raise HTTPException(status_code=400, detail="Job not completed yet")

    loop = asyncio.get_event_loop()
    asyncio.create_task(_run_evaluation(job_id, config, loop))

    return {"status": "evaluating"}


async def _run_evaluation(job_id: str, config: dict, loop):
    """Run evaluation metrics."""
    job_dir = os.path.join(JOBS_DIR, job_id)
    log_fn = sync_log_fn(job_id, loop)

    try:
        await send_log(job_id, "Starting evaluation...")
        mgr = get_model_manager()

        stage = config['stage']

        if stage == 1:
            # Stage 1 evaluation: mIoU between predicted and GT semantic
            gt_path = os.path.join(job_dir, 'gt_semantic_ids.npy')
            pred_path = os.path.join(job_dir, 'pred_semantic_ids.npy')

            if not os.path.exists(gt_path):
                await send_log(job_id, "No ground truth available for evaluation. Upload GT or use a dataset sample.")
                return

            gt = np.load(gt_path)
            pred = np.load(pred_path)

            await send_log(job_id, "Computing semantic segmentation metrics...")
            metrics = await loop.run_in_executor(
                None, lambda: mgr.evaluate_stage1(pred, gt)
            )

        elif stage == 2:
            # Stage 2 evaluation: DRN mIoU + image quality metrics
            gen_path = os.path.join(job_dir, 'gen_frames.npy')
            gt_sem_path = os.path.join(job_dir, 'gt_semantic_ids.npy')
            gt_rgb_path = os.path.join(job_dir, 'gt_frames.npy')

            if not os.path.exists(gen_path):
                await send_log(job_id, "No generated frames found.")
                return

            gen_frames = np.load(gen_path)
            metrics = {}

            # DRN evaluation
            if os.path.exists(gt_sem_path):
                gt_sem = np.load(gt_sem_path)
                await send_log(job_id, "Running DRN segmentation on generated frames...")
                drn_metrics = await loop.run_in_executor(
                    None, lambda: mgr.evaluate_stage2_drn(gen_frames, gt_sem, log_fn)
                )
                metrics['drn'] = drn_metrics

            # Image quality metrics
            if os.path.exists(gt_rgb_path):
                gt_rgb = np.load(gt_rgb_path)
                await send_log(job_id, "Computing image quality metrics (SSIM, PSNR, LPIPS)...")
                img_metrics = await loop.run_in_executor(
                    None, lambda: mgr.compute_image_metrics(gt_rgb, gen_frames, log_fn)
                )
                metrics['image_quality'] = img_metrics

        # Save metrics
        metrics_path = os.path.join(job_dir, 'eval_results.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        # Update config status so HTTP polling can detect completion
        config['status'] = 'evaluated'
        _save_config(job_dir, config)

        await send_log(job_id, "Evaluation complete!")
        await _send_status(job_id, 'evaluated', metrics=metrics)

    except Exception as e:
        await send_log(job_id, f"Evaluation ERROR: {str(e)}")
        await send_log(job_id, traceback.format_exc())


# ============================================================================
# Job Management
# ============================================================================

@app.get("/api/jobs/{job_id}/status")
async def get_job_status(job_id: str):
    job_dir = os.path.join(JOBS_DIR, job_id)
    config = _load_config(job_dir)
    if not config:
        raise HTTPException(status_code=404, detail="Job not found")

    result = {
        'job_id': job_id,
        'status': config['status'],
        'stage': config.get('stage'),
        'config': config,
    }

    # Include frame list
    frames_dir = os.path.join(job_dir, 'frames')
    if os.path.exists(frames_dir):
        frame_dirs = {}
        for subdir in os.listdir(frames_dir):
            subpath = os.path.join(frames_dir, subdir)
            if os.path.isdir(subpath):
                files = sorted([f for f in os.listdir(subpath) if f.endswith('.png')])
                frame_dirs[subdir] = files
        result['frame_dirs'] = frame_dirs

    # Include eval results
    eval_path = os.path.join(job_dir, 'eval_results.json')
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            result['eval_results'] = json.load(f)

    # Check for GIF
    gif_path = os.path.join(job_dir, 'output.gif')
    result['has_gif'] = os.path.exists(gif_path)

    return result


@app.get("/api/jobs/{job_id}/frames/{subdir}/{filename}")
async def get_frame(job_id: str, subdir: str, filename: str):
    path = os.path.join(JOBS_DIR, job_id, 'frames', subdir, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404)
    return FileResponse(path, media_type='image/png')


@app.get("/api/jobs/{job_id}/gif")
async def get_gif(job_id: str):
    path = os.path.join(JOBS_DIR, job_id, 'output.gif')
    if not os.path.exists(path):
        raise HTTPException(status_code=404)
    return FileResponse(path, media_type='image/gif')


@app.get("/api/jobs/{job_id}/download")
async def download_job(job_id: str):
    """Download all job frames as a zip."""
    job_dir = os.path.join(JOBS_DIR, job_id)
    if not os.path.exists(job_dir):
        raise HTTPException(status_code=404)

    zip_path = os.path.join(job_dir, 'download')
    if not os.path.exists(zip_path + '.zip'):
        frames_dir = os.path.join(job_dir, 'frames')
        if os.path.exists(frames_dir):
            shutil.make_archive(zip_path, 'zip', frames_dir)
        else:
            raise HTTPException(status_code=404, detail="No frames to download")

    return FileResponse(zip_path + '.zip', filename=f'ctrlv_job_{job_id}.zip',
                        media_type='application/zip')


@app.get("/api/jobs")
async def list_jobs():
    """List all jobs sorted by creation time (newest first)."""
    jobs = []
    if os.path.exists(JOBS_DIR):
        for d in os.listdir(JOBS_DIR):
            if d.startswith('_'):
                continue
            config_path = os.path.join(JOBS_DIR, d, 'config.json')
            if os.path.exists(config_path):
                with open(config_path) as f:
                    config = json.load(f)
                has_eval = os.path.exists(os.path.join(JOBS_DIR, d, 'eval_results.json'))
                jobs.append({
                    'job_id': d,
                    'stage': config.get('stage'),
                    'status': config.get('status'),
                    'created_at': config.get('created_at', ''),
                    'sample_index': config.get('sample_index'),
                    'num_frames': config.get('num_frames'),
                    'num_inference_steps': config.get('num_inference_steps'),
                    'min_guidance_scale': config.get('min_guidance_scale'),
                    'max_guidance_scale': config.get('max_guidance_scale'),
                    'noise_aug_strength': config.get('noise_aug_strength'),
                    'seed': config.get('seed'),
                    'num_cond_bbox_frames': config.get('num_cond_bbox_frames'),
                    'conditioning_scale': config.get('conditioning_scale'),
                    'control_job_id': config.get('control_job_id'),
                    'has_eval': has_eval,
                })
    # Sort newest first
    jobs.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    return {"jobs": jobs}


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and all its associated files."""
    job_dir = os.path.join(JOBS_DIR, job_id)
    if not os.path.exists(job_dir):
        raise HTTPException(status_code=404, detail="Job not found")
    config = _load_config(job_dir)
    if config and config.get('status') == 'running':
        raise HTTPException(status_code=400, detail="Cannot delete a running job")
    shutil.rmtree(job_dir)
    return {"status": "deleted", "job_id": job_id}


# ============================================================================
# HTTP Log Polling (fallback when WebSocket proxy is unavailable)
# ============================================================================

@app.get("/api/jobs/{job_id}/logs")
async def get_job_logs(job_id: str, offset: int = 0):
    """Get logs for a job starting from a line offset. Used for polling."""
    job_dir = os.path.join(JOBS_DIR, job_id)
    log_file = os.path.join(job_dir, 'logs.txt')
    config_file = os.path.join(job_dir, 'config.json')

    lines = []
    if os.path.exists(log_file):
        with open(log_file) as f:
            all_lines = f.readlines()
            lines = [l.strip() for l in all_lines[offset:]]

    status = "unknown"
    if os.path.exists(config_file):
        with open(config_file) as f:
            config = json.load(f)
            status = config.get('status', 'unknown')

    return {
        "lines": lines,
        "offset": offset + len(lines),
        "status": status,
    }


# ============================================================================
# WebSocket for Log Streaming
# ============================================================================

@app.websocket("/ws/logs/{job_id}")
async def websocket_logs(websocket: WebSocket, job_id: str):
    await websocket.accept()
    ws_connections[job_id].append(websocket)

    try:
        # Send existing logs
        log_file = os.path.join(JOBS_DIR, job_id, 'logs.txt')
        if os.path.exists(log_file):
            with open(log_file) as f:
                for line in f:
                    await websocket.send_text(json.dumps({'type': 'log', 'message': line.strip()}))

        # Keep connection alive
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30)
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_text(json.dumps({'type': 'heartbeat'}))
    except WebSocketDisconnect:
        pass
    finally:
        ws_connections[job_id].remove(websocket)


# ============================================================================
# Upload Ground Truth
# ============================================================================

@app.post("/api/jobs/{job_id}/upload-gt")
async def upload_ground_truth(job_id: str, gt_frames: List[UploadFile] = File(...)):
    """Upload ground truth frames for evaluation."""
    job_dir = os.path.join(JOBS_DIR, job_id)
    config = _load_config(job_dir)
    if not config:
        raise HTTPException(status_code=404)

    gt_dir = os.path.join(job_dir, 'frames', 'gt_uploaded')
    os.makedirs(gt_dir, exist_ok=True)

    from ctrlv.utils.semantic_preprocessing import KITTI360_LABEL_MAPPING

    if config['stage'] == 1:
        # GT is semantic maps (grayscale)
        gt_ids_list = []
        for i, f in enumerate(sorted(gt_frames, key=lambda x: x.filename)):
            data = await f.read()
            img_path = os.path.join(gt_dir, f'frame_{i:03d}.png')
            with open(img_path, 'wb') as fp:
                fp.write(data)
            img = np.array(Image.open(img_path).convert('L'))
            # Remap to trainIDs
            remapped = np.full_like(img, 255, dtype=np.int64)
            for raw_id, train_id in KITTI360_LABEL_MAPPING.items():
                remapped[img == raw_id] = train_id
            gt_ids_list.append(remapped)

        gt_ids = np.stack(gt_ids_list)
        np.save(os.path.join(job_dir, 'gt_semantic_ids.npy'), gt_ids)
    else:
        # GT is RGB frames
        gt_frames_list = []
        for i, f in enumerate(sorted(gt_frames, key=lambda x: x.filename)):
            data = await f.read()
            img_path = os.path.join(gt_dir, f'frame_{i:03d}.png')
            with open(img_path, 'wb') as fp:
                fp.write(data)
            img = np.array(Image.open(img_path).convert('RGB'))
            gt_frames_list.append(img)

        gt_np = np.stack(gt_frames_list)
        np.save(os.path.join(job_dir, 'gt_frames.npy'), gt_np)

    return {"status": "uploaded", "num_frames": len(gt_frames)}


# ============================================================================
# Helpers
# ============================================================================

def _save_config(job_dir: str, config: dict):
    with open(os.path.join(job_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)


def _load_config(job_dir: str) -> Optional[dict]:
    config_path = os.path.join(job_dir, 'config.json')
    if not os.path.exists(config_path):
        return None
    with open(config_path) as f:
        return json.load(f)


async def _send_status(job_id: str, status: str, metrics: dict = None):
    """Send status update to WebSocket clients."""
    data = {'type': 'status', 'status': status}
    if metrics:
        data['metrics'] = metrics
    for ws in ws_connections.get(job_id, []):
        try:
            await ws.send_text(json.dumps(data))
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
