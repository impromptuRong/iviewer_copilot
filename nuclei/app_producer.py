from fastapi import FastAPI, HTTPException, Request, Response, Depends
from fastapi.middleware.cors import CORSMiddleware

import redis.asyncio as redis
from aiocache import Cache
from aiocache.serializers import PickleSerializer, JsonSerializer
from async_lru import alru_cache

import os
import pickle
import numpy as np

from utils.db import DeepZoomSettings
from utils.simpletiff import SimpleTiff
from tifffile import TiffFile
from utils.utils_image import Slide, pad_pil
from model_registry import MODEL_REGISTRY, AGENT_CONFIGS
from fastapi.responses import JSONResponse


# Redis connection
REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
REDIS_PORT = os.environ.get('REDIS_PORT', 6379)
pool = redis.ConnectionPool(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)
async def get_redis():
    return redis.Redis(connection_pool=pool)


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust the allowed origins as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


setting_cache = Cache(
    Cache.REDIS,
    endpoint=REDIS_HOST,  # Replace with your Redis server address
    port=REDIS_PORT,            # Replace with your Redis server port
    namespace="setting",
    serializer=PickleSerializer(),
    timeout=5,           # Set the cache timeout (in seconds)
)

@alru_cache(maxsize=8)
async def _get_slide(slide_path):
    try:
        print(f"Excute remote slide: {slide_path}")
        if slide_path.startswith('http'):
            print(f"Use SimpleTiff")
            slide = SimpleTiff(slide_path)
        else:
            print(f"Use TiffFile")
            slide = TiffFile(slide_path)
        return slide
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Failed to load slide from {slide_path}: {str(e)}")

@alru_cache(maxsize=16)
async def _get_generator(key):
    try:
        settings = await setting_cache.get(key)  # setting_cache[key]
        osr = await _get_slide(settings.file)

        slide = Slide(osr)
        engine = 'simpletiff' if settings.file.startswith('http') else 'tifffile'
        print(f"use engine={engine}")
        slide.attach_reader(osr, engine=engine)

        generator = MODEL_REGISTRY.get_generator(settings.server)(
            slide,
            tile_size=settings.tile_size,
            overlap=settings.overlap,
            # limit_bounds=settings.limit_bounds,
            # format=settings.format,
        )

        return generator
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Failed to get DeepZoomGenerator for {key}: {str(e)}")

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

@app.get("/proxy/dummy.dzi")
async def proxy_dzi(request: Request):
    """ Serve slides and different registries to openseadragon. """
    request_args = {k.split('amp;')[-1]: v for k, v in request.query_params.items()}
    print(f"{request_args}")
    image_id, registry = request_args['image_id'], request_args['registry']
    key = f"{image_id}_{registry}"
    try:
        cfg = AGENT_CONFIGS[registry]
        dzi_params = {'server': cfg.server, **cfg.dzi_settings, **request_args}
        setting = DeepZoomSettings(**dzi_params)

        await setting_cache.set(key, setting) # , ttl=600)  # setting_cache[key] = setting
        generator = await _get_generator(key)
        dzi = generator.get_dzi()
        print(f"*********************")
        print(f"proxy_dzi: {key} -> {setting}")
        print(f"check if {key} is in cache: {await setting_cache.get(key)}")
        print(f"generator parameters: {generator.tile_size, generator.overlap, generator.run_mpp, generator.scale}")
        print(f"*********************")

        return Response(content=dzi, media_type="application/xml")
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Failed to get dzi for {key}: {str(e)}")


@app.get("/proxy/dummy_files/{level}/{col}_{row}.{format}", status_code=201)
async def proxy_tile(
    level: int, col: int, row: int, format: str, request: Request,
    client=Depends(get_redis), 
):
    request_args = {k.split('amp;')[-1]: v for k, v in request.query_params.items()}
    image_id, registry = request_args['image_id'], request_args['registry']
    project_id = request_args.get('project_id', '')
    group_id = request_args.get('group_id', '')
    key = f"{image_id}_{registry}"
    try:
        generator = await _get_generator(key)
        settings = await setting_cache.get(key)
        assert settings.server is not None, f"settings.server({settings.server}) shouldn't be none."

        # Improper zoom level
        if level < generator.inference_dz_level:
            print(f"Not at the correct level: {level}/{generator.inference_dz_level}")
            return None
        else:
            factor = int(2 ** (generator.inference_dz_level - level))
            if factor > 1:
                dc, dr = np.meshgrid(np.arange(2 ** factor), np.arange(factor))
                tile_ids = [f"{generator.page}_{col+c}_{row+r}" 
                            for c, r in zip(dc.ravel(), dr.ravel())]
            else:
                tile_ids = [f"{generator.page}_{col}_{row}"]

        for tile_id in tile_ids:
            # Not in ROI (tissue region)
            if tile_id not in generator.indices:
                print(f"tile_id {tile_id} is out of ROI.")

            # Already analyzed by this service
            tag = await client.sismember(f"{key}_memo", tile_id)
            if tag:  # client.exists(f"{key}_memo") and 
                print(f"tile_id {tile_id} is detected in memo ({f'{key}_memo'}), already in queue. ")

            # Get patch and push into queue
            idx = generator.indices[tile_id]
            info = generator.images[idx]['data']

            patch = generator._osr.get_patch(info['coord'], generator.page)
            patch = pad_pil(patch.convert('RGB'), info['pad_width'], color=0)  # pad_l, pad_r, pad_u, pad_d
            # serialized_item = json.dumps({'image_id': image_id, 'registry': registry, 'info': info, 'img': pil2bytes(patch, format='jpeg')})
            extra = {'project_id': project_id, 'group_id': group_id}
            serialized_item = pickle.dumps({
                'image_id': image_id, 'registry': registry, 
                'tile_id': tile_id, 'info': info, 'img': patch, 
                'extra': extra,
            })
            # TODO: switch to redis stream
            tag = await client.lpush(registry, serialized_item)
            if not tag:
                raise HTTPException(status_code=404, detail="Failed to push image patch into job queue.")

            await client.sadd(f"{key}_memo", tile_id)
            print(f"Push tile_id {tile_id} into queue.")
            # resp = Response(content=buf.getvalue(), media_type=f"image/png")

        await client.aclose()

        return "Success."
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Failed to get image tile for {key}: {str(e)}")


async def test_connection():
    client = await get_redis()
    print(f"Test connection to redis ({REDIS_HOST}:{REDIS_PORT}). Ping successful: {await client.ping()}")
    await client.aclose()


if __name__ == "__main__":
    # asyncio.run(test_connection())
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9030)
