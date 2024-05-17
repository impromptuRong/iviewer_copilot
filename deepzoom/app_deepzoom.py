from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from aiocache import Cache
from aiocache.serializers import PickleSerializer, JsonSerializer

from typing import Optional
from async_lru import alru_cache
from collections.abc import Callable
from io import BytesIO

import os
import cv2
import json
import time
import requests
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
from tifffile import TiffFile

from utils.simpletiff import SimpleTiff
from utils.deepzoom import DeepZoomGenerator
from utils.utils_image import Slide


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust the allowed origins as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


DZI_SETTINGS = {
    'slide': {
        'format': 'jpeg', 
        'tile_size': 254, 
        'overlap': 1, 
        'limit_bounds': False, 
        'tile_quality': 75,
        'server': None,
    },
    'masks': {
        'format': 'png', 
        'tile_size': 254, 
        'overlap': 1, 
        'limit_bounds': False, 
        'tile_quality': 50,
        'server': None,
    },
    'default': {
        'format': 'jpeg', 
        'tile_size': 254, 
        'overlap': 1, 
        'limit_bounds': False, 
        'tile_quality': 75,
        'server': None,
    },
    'HDYolo': {
        'format': 'jpeg', 
        'tile_size': 512, 
        'overlap': 64, 
        'limit_bounds': False, 
        'tile_quality': 50,
        'server': 'HDYolo',
    },
    'SAM': {
        'format': 'png', 
        'tile_size': 512, 
        'overlap': 64, 
        'limit_bounds': False, 
        'tile_quality': 75,
        'server': 'SAM',
    },
}

# SERVER_REGISTRY = {
#     None: DeepZoomGenerator,
# }
    

class DeepZoomSettings(BaseModel):
    file: str = None
    format: str = 'jpeg'
    tile_size: int = 254
    overlap: int = 1
    limit_bounds: bool = True
    tile_quality: int = 75
    server: Optional[str] = None

# from cachetools import LRUCache
# SLIDE_CACHE_SIZE = 8
# DEEPZOOM_CACHE_SIZE = 16
# slide_cache = LRUCache(maxsize=SLIDE_CACHE_SIZE)
# deepzoom_cache = LRUCache(maxsize=DEEPZOOM_CACHE_SIZE)
# setting_cache = LRUCache(maxsize=DEEPZOOM_CACHE_SIZE)


redis_host = os.environ.get('REDIS_HOST', 'localhost')
redis_port = os.environ.get('REDIS_PORT', 6379)
setting_cache = Cache(
    Cache.REDIS,
    endpoint=redis_host,  # Replace with your Redis server address
    port=redis_port,            # Replace with your Redis server port
    namespace='setting',
    serializer=PickleSerializer(),
    timeout=5,           # Set the cache timeout (in seconds)
)

@alru_cache(maxsize=8)
async def _get_slide(slide_path):
# try:
    print(f"Excute remote slide: {slide_path}")
    slide = SimpleTiff(slide_path)
    return slide
# except:
#     raise HTTPException(status_code=404, detail="Slide not found")


@alru_cache(maxsize=16)
async def _get_generator(key):
# try:
    settings = await setting_cache.get(key)  # setting_cache[key]
    osr = await _get_slide(settings.file)

    # slide = osr
    slide = Slide(osr)
    slide.attach_reader(osr, engine='simpletiff')

    # SERVER_REGISTRY[settings.server]
    generator = DeepZoomGenerator(
        slide, 
        tile_size=settings.tile_size,
        overlap=settings.overlap,
        limit_bounds=settings.limit_bounds,
        format=settings.format,
    )

    return generator
# except:
#     raise HTTPException(status_code=404, detail="DeepZoomGenerator not found")


@app.get("/proxy/params")
async def proxy_params(request: Request):
# try:
    request_args = {k.split('amp;')[-1]: v for k, v in request.query_params.items()}
    image_id = request_args['image_id']
    registry = request_args.get('registry', 'slide')
    key = f"{image_id}_{registry}"

    generator = await _get_generator(key)
    w, h = generator.dz_dimensions[-1]
    info = generator._osr.info
    print(f"debug params: ({w}, {h}), {info}")

    # osr = await _get_slide(filepath)
    # slide = Slide(osr)
    # info = slide.info

    params = {
        'width': w,
        'height': h,
        'slide_mpp': info['mpp'],
        'magnitude': info['magnitude'],
        'description': info['description'],
    }
    return params
# except Exception:
#     raise HTTPException(status_code=404, detail="Slide not found")


@app.get("/proxy/thumbnail")
async def proxy_thumbnail(request: Request):
# try:
    request_args = {k.split('amp;')[-1]: v for k, v in request.query_params.items()}
    image_id = request_args['image_id']
    registry = request_args.get('registry', 'slide')
    key = f"{image_id}_{registry}"
    
    settings = await setting_cache.get(key)  # setting_cache[key]
    osr = await _get_slide(settings.file)

    slide = Slide(osr)
    return slide.thumbnail(image_size=512)

# except Exception:
#     raise HTTPException(status_code=404, detail="Slide not found")


@app.get("/proxy/dummy.dzi")
async def proxy_dzi(request: Request):
    """ Serve slides and different registries to openseadragon. 
        In the openseadragon `tileSources`, specify path and registry name.
        `file` for local files and `url` for remote files. 
        `registry` is the key for registry name: 'slide', 'nuclei', etc.

        e.g., to serve "abc/abc.svs" and its nuclei masks, specify the following:
        var viewer = new OpenSeadragon({
            id: "example-xmlhttprequest-for-dzi",
            tileSources: "http://localhost:9000/proxy/dummy.dzi?file=abc%2Fabc.svs&registry=slide",
        });
        viewer.addTiledImage({
            tileSource: "http://localhost:9000/dummy.dzi?file=abc%2Fabc.tiff&registry=masks",
            opacity: 0.0,  // start with 0, openseadragon won't pull tiles if it's 0
            x: 0,
            y: 0,
            width: 1,
        });
        or call HDYolo model to analyze the file, specify the following:
        var viewer = new OpenSeadragon({
            id: "example-xmlhttprequest-for-dzi",
            tileSources: "http://localhost:9000/proxy/dummy.dzi?file=abc%2Fabc.tiff&registry=HDYolo",
        });

        to serve a remote file, "http://localhost:8000/abc/abc.svs", specify the following:
        var viewer = new OpenSeadragon({
            id: "example-xmlhttprequest-for-dzi",
            tileSources: "http://localhost:9000/proxy/dummy.dzi?file=http%3A%2F%2Flocalhost%3A8000%2Fabc%2Fabc.svs&registry=slide",
        });

        to add a registry customize dzi format specify with request args:
        viewer.addTiledImage({
            tileSource: "http://localhost:9000/proxy/dummy.dzi?file=abc%2Fabc.svs&registry=new&format=png&tile_size=512&overlap=32",
            opacity: 0.0, x: 0, y: 0, width: 1,
        });

    Test in curl: 
        curl 'http://localhost:9000/proxy/dummy.dzi?file=abc%2Fabc.svs&registry=slide'
        curl 'http://localhost:9000/proxy/dummy.dzi?file=abc%2Fabc.tiff&registry=masks'

    Returns
        -------
        the content of .dzi file
    """
# try:
    request_args = {k.split('amp;')[-1]: v for k, v in request.query_params.items()}
    print(f"{request_args}")
    image_id, registry = request_args['image_id'], request_args['registry']
    key = f"{image_id}_{registry}"
    default_args = DZI_SETTINGS.get(registry, DZI_SETTINGS['default'])

    # cfgs = {k: request_args.get(k, v) for k, v in default_args.items()}  # 
    cfgs = {**default_args, **request_args}
    setting = DeepZoomSettings(**cfgs)

    await setting_cache.set(key, setting)  # , ttl=600)  # setting_cache[key] = setting
    print(f"*********************")
    print(f"proxy_dzi: {key} -> {setting}")
    print(default_args, request_args, cfgs)
    print(f"check if {key} is in cache: {await setting_cache.get(key)}")
    print(f"*********************")

    ## acquire a db, removed to frontend call
    # database_hostname = "http://localhost:9020"
    # create_db_url = f"{database_hostname}/annotation/create?image_id={image_id}"
    # response = requests.post(create_db_url)  # , json={'image_id': image_id})  # replace('/', '%2F').replace(':', '%3A')
    # print(response.text, response.status_code)  # 200

    generator = await _get_generator(key)
    dzi = generator.get_dzi()

    return Response(content=dzi, media_type="application/xml")
# except Exception:
#     raise HTTPException(status_code=404, detail="DZI not found")


@app.get("/proxy/dummy_files/{level}/{col}_{row}.{format}")
async def proxy_tile(
    level: int, col: int, row: int, format: str, request: Request,
):
# try:
    request_args = {k.split('amp;')[-1]: v for k, v in request.query_params.items()}
    image_id, registry = request_args['image_id'], request_args['registry']
    key = f"{image_id}_{registry}"

    generator = await _get_generator(key)
    settings = await setting_cache.get(key)

    tile = generator.get_tile(level, (col, row), format=format)
    buf = BytesIO()
    tile.save(buf, format, quality=settings.tile_quality)
    resp = Response(content=buf.getvalue(), media_type=f"image/{format}")

    return resp
# except ValueError:
#     raise HTTPException(status_code=404, detail="Invalid level or coordinates")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9010)
    # gunicorn app_deepzoom:app --workers 8 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:9010
