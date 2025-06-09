import os
import math
import json
import requests
import numpy as np
from PIL import Image
from io import BytesIO
from tifffile import TiffFile

from fastapi import FastAPI, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from functools import lru_cache

from utils.simpletiff import SimpleTiff
from utils.utils_image import Slide
from utils.utils_image import random_sampling_in_polygons

from app_worker import run_segmentation
from fastapi import FastAPI, HTTPException, Request, Response, Depends
from fastapi.responses import JSONResponse
from fastapi import HTTPException

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    SEGMENT_HOST = os.getenv('SEGMENT_HOST')
    if not SEGMENT_HOST:
        raise RuntimeError("SEGMENT_HOST must be defined!")
    
    if SEGMENT_HOST == "sam2":
        if not os.getenv('REDIS_HOST'):
            raise RuntimeError("Redis config required for local mode")
        
    yield

# from model_registry import AGENT_CONFIGS

DEFAULT_INPUT_SIZE = (512, 512)

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust the allowed origins as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=8)
def _get_slide(slide_path):
    try:
        print(f"Excute remote slide: {slide_path}")
        if slide_path.startswith('http'):
            print(f"Use SimpleTiff")
            osr = SimpleTiff(slide_path)
            engine = 'simpletiff'
        else:
            print(f"Use TiffFile")
            osr = TiffFile(slide_path)
            engine = 'tifffile'

        slide = Slide(osr)
        print(f"use engine={engine}")
        slide.attach_reader(osr, engine=engine)

        return slide
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Failed to load slide from {slide_path}: {str(e)}")


def generate_prompts(item, size=40):
    # Get bounding box, shape and poly
    shape = item.get('shape', 'rect')
    poly_x, poly_y = item.get('poly_x', ''), item.get('poly_y', '')
    x0, y0 = float(item['x0']), float(item['y0'])
    x1, y1 = float(item['x1']), float(item['y1'])
    xc, yc = float(item.get('xc', (x0 + x1)/2)), float(item.get('yc', (y0 + y1)/2))
    if shape == 'polygon' or shape == 'freehand':
        x_points = np.fromstring(poly_x, dtype=float, sep=',')
        y_points = np.fromstring(poly_y, dtype=float, sep=',')
        polygon = np.stack([x_points, y_points], axis=-1)
        points, _ = random_sampling_in_polygons([polygon], N=size, plot=False)
        labels = np.ones(len(points))
        prompts = {'points': points, 'labels': labels}
    elif shape == 'ellipse' or shape == 'circle':
        radius_x, radius_y = float(poly_x) * 0.8, float(poly_y) * 0.8
        # Generate random polar angles and radial distances
        angles = np.random.uniform(0, 2 * np.pi, size)
        radii = np.sqrt(np.random.uniform(0, 1, size))

        # Convert polar to Cartesian coordinates for an ellipse
        x_points = xc + radii * radius_x * np.cos(angles)
        y_points = yc + radii * radius_y * np.sin(angles)
        points = np.stack([x_points, y_points], axis=-1)
        labels = np.ones(len(points))
        prompts = {'points': points, 'labels': labels}
    elif shape == 'point' or (shape == 'rect' and (x1 - x0 <= 2 or y1 - y0 <= 2)):
        ## treat very small rectangle as points to avoid strange behavior (single point polygon)
        points = np.unique([
            [x0, y0], [x0, y1], [x1, y0], [x1, y1], [xc, yc],
        ], axis=0)
        labels = np.ones(len(points))
        prompts = {'points': points, 'labels': labels}
    elif shape == 'rect':
        prompts = {'bboxes': np.array([x0, y0, x1, y1])}

    return prompts, [x0, y0, x1, y1]


def get_patch_and_prompts(file, item, image_size=(512, 512)):
    """
        [x0_s, y0_s, x1_s, y1_s]: ROI absolute coordinates in slide level 0
        [x0_p, y0_p, x1_p, y1_p]: ROI relative coordinates in image patch at slide level 0
        [x0, y0, x1, y1, w, h]: image absolute coordinates/size in slide level 0
        [scale_w, scale_h]: absolute coordinates in level 0 * scale = absolute coordinates in level `page`
    """
    slide = _get_slide(file)
    max_w, max_h = slide.level_dims[0]

    prompts, [x0_s, y0_s, x1_s, y1_s] = generate_prompts(item)
    # pad roi with  of min(10, roi_h/w * 0.1) pixels
    pad_l = pad_r = max(30, (x1_s - x0_s) * 0.1)
    pad_u = pad_d = max(30, (y1_s - y0_s) * 0.1)
    # get padded roi coordinates in slide level 0
    x0, y0 = max(x0_s - pad_l, 0), max(y0_s - pad_u, 0)
    x1, y1 = min(x1_s + pad_r, max_w), min(y1_s + pad_d, max_h)
    w, h = x1 - x0, y1 - y0
    # # get relative roi coordinates in patch [x0, y0, x1, y1]
    # x0_p, y0_p = x0_s - x0, y0_s - y0
    # x1_p, y1_p = x1 - x1_s, y1 - y1_s

    # find the best level to extract ROI
    factor = max(w / image_size[0], h / image_size[1])
    if factor > 1.0:
        page = slide.get_resize_level(factor, downsample_only=True)
        scale_w, scale_h = slide.get_scales(page)[-1][0]
    else:
        page = 0
        scale_w, scale_h = 1.0, 1.0

    # Map all coordinates to the selected level
    x0_scaled, y0_scaled = int(x0 * scale_w), int(y0 * scale_h)
    w_scaled, h_scaled = int(math.ceil(w * scale_w)), int(math.ceil(h * scale_h))

    # get image patch and patch_info under level=page with scaled coordinates
    patch = slide.get_patch([x0_scaled, y0_scaled, w_scaled, h_scaled], level=page)
    # keep shrink to input_size if it's too big
    factor = max(patch.size[0] / image_size[0], patch.size[1] / image_size[1])
    if factor > 1.0:
        new_width = int(patch.size[0] / factor)
        new_height = int(patch.size[1] / factor)
        patch = patch.resize((new_width, new_height))
        scale_w, scale_h = scale_w / factor, scale_h / factor
    else:
        scale_w, scale_h = scale_w * 1.0, scale_h * 1.0

    # Map all coordinates to the selected level
    x0_scaled, y0_scaled = int(x0 * scale_w), int(y0 * scale_h)
    w_scaled, h_scaled = int(math.ceil(w * scale_w)), int(math.ceil(h * scale_h))

    patch_info = [x0_scaled, y0_scaled, w_scaled, h_scaled, 1./scale_w, 1./scale_h]

    # scale all coordinates in prompts
    if 'points' in prompts:
        prompts['points'] = (
            prompts['points'] * [scale_w, scale_h] - 
            [x0_scaled, y0_scaled]
        )
    if 'bboxes' in prompts:
        prompts['bboxes'] = (
            prompts['bboxes'] * [scale_w, scale_h, scale_w, scale_h] - 
            [x0_scaled, y0_scaled, x0_scaled, y0_scaled]
        )

    return patch, {k: v.tolist() for k, v in prompts.items()}, patch_info


def image_to_bytes(image: Image):
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr

@app.get("/health")
async def check_health(request: Request):
    # return JSONResponse(content={"status": "healthy", "message": "Service is running!"})
    try:
        health_check = "OK"  
       # Get environment variables from Docker build args
        status_response = {
            "health": health_check,
            "environment": {
                "http_proxy": os.getenv("http_proxy"),
                "https_proxy": os.getenv("https_proxy"),
                "no_proxy": os.getenv("no_proxy"),
            }
        }
        return JSONResponse(content=status_response)
    except Exception as e:
         return JSONResponse(
            content={
                "health": "unhealthy",
                "details": f"Error: {str(e)}"
            },
            status_code=500  # Internal Server Error
        )

SEGMENT_HOST = os.environ.get('SEGMENT_HOST')
SEGMENT_PORT = os.environ.get('SEGMENT_PORT', 8376)
USE_LOCAL = SEGMENT_HOST == 'sam2'

if USE_LOCAL:
    from app_worker import run_segmentation
    
@app.post("/segment")
async def segment(image_id: str, file: str, registry: str, item=Body(...)):
    key = f"{image_id}_{registry}"
    # if registry not in AGENT_CONFIGS:
    #     raise HTTPException(status_code=400, detail=f"Unknown model registry: {registry}")
    # config = AGENT_CONFIGS[registry]

    patch, prompts, patch_info = get_patch_and_prompts(file, item, DEFAULT_INPUT_SIZE)
    
    if USE_LOCAL:
    # Enqueue the task to Celery
        task = run_segmentation.apply_async(
            args=[image_to_bytes(patch), prompts, patch_info, {}], 
            queue=registry,
        )

        try:
            result = task.get(timeout=20)  # Set timeout to avoid long waits
            # {"task_id": task.id, "status": "Task submitted"}
            return result
        except Exception as e:
            print(e)
            raise HTTPException(status_code=500, detail="Task failed: " + str(e))
    else:
        # Remote processing via HTTP
        files = {'image':('image.png', image_to_bytes(patch), 'image/png')}
        data = {'params': json.dumps({'prompts': prompts, 'patch_info': patch_info, 'extra': {}})}
        try:
            response = requests.post(
                f'http://{SEGMENT_HOST}:{SEGMENT_PORT}/segment?registry={registry}',
                files=files,
                data=data,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=500, detail=f"Remote processing failed: {str(e)}")

if __name__ == "__main__":
    # asyncio.run(test_connection())
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9050)

