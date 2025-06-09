from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from aiocache import Cache
from aiocache.serializers import PickleSerializer, JsonSerializer
from async_lru import alru_cache

import os
from io import BytesIO
from tifffile import TiffFile

from utils.db import DeepZoomSettings
from utils.simpletiff import SimpleTiff
from utils.deepzoom import DeepZoomGenerator
from utils.utils_image import Slide
from fastapi.responses import JSONResponse


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


redis_host = os.environ.get('REDIS_HOST', 'localhost')
redis_port = os.environ.get('REDIS_PORT', 6379)
setting_cache = Cache(
    Cache.REDIS,
    endpoint=redis_host,
    port=redis_port,
    namespace='setting',
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

        generator = DeepZoomGenerator(
            slide, 
            tile_size=settings.tile_size,
            overlap=settings.overlap,
            limit_bounds=settings.limit_bounds,
            format=settings.format,
        )

        return generator
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Failed to get DeepZoomGenerator for {key}: {str(e)}")

@app.get("/health")
async def check_health(request: Request):
    # return JSONResponse(content={"status": "healthy", "message": "Service is running!"})
    try:
        health_check = "OK"  
        #proxies = {
            #"http": os.getenv("http_proxy"),
            #"https": os.getenv("https_proxy"),
        #}
        status_response = {
            "health": health_check,
            "environment": {
                "http_proxy": os.getenv("http_proxy"),
                "https_proxy": os.getenv("https_proxy"),
                "no_proxy": os.getenv("no_proxy"),
            }
            #"network": {
                #"ollama": check_service_status("http://ollama.example.com/health", proxies),
            #},
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

@app.get("/proxy/params")
async def proxy_params(request: Request):
    try:
        request_args = {k.split('amp;')[-1]: v for k, v in request.query_params.items()}
        image_id = request_args['image_id']
        registry = request_args.get('registry', 'slide')
        key = f"{image_id}_{registry}"

        generator = await _get_generator(key)
        w, h = generator.dz_dimensions[-1]
        info = generator._osr.info
        # print(f"debug params: ({w}, {h}), {info}")

        params = {
            'width': w,
            'height': h,
            'slide_mpp': info['mpp'],
            'magnitude': info['magnitude'],
            'description': info['description'],
        }
        return params
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Failed to get slide parameters for {key}: {str(e)}")


@app.get("/proxy/thumbnail")
async def proxy_thumbnail(request: Request):
    try:
        request_args = {k.split('amp;')[-1]: v for k, v in request.query_params.items()}
        image_id = request_args['image_id']
        registry = request_args.get('registry', 'slide')
        key = f"{image_id}_{registry}"

        settings = await setting_cache.get(key)  # setting_cache[key]
        osr = await _get_slide(settings.file)
        slide = Slide(osr)

        return slide.thumbnail(image_size=512)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Failed to get slide thumbnail for {key}: {str(e)}")


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
    request_args = {k.split('amp;')[-1]: v for k, v in request.query_params.items()}
    print(f"{request_args}")
    image_id, registry = request_args['image_id'], request_args['registry']
    key = f"{image_id}_{registry}"
    try:
        default_args = DZI_SETTINGS.get(registry, DZI_SETTINGS['default'])

        cfgs = {**default_args, **request_args}
        setting = DeepZoomSettings(**cfgs)

        await setting_cache.set(key, setting)  # , ttl=600)  # setting_cache[key] = setting
        print(f"*********************")
        print(f"proxy_dzi: {key} -> {setting}")
        print(default_args, request_args, cfgs)
        print(f"check if {key} is in cache: {await setting_cache.get(key)}")
        print(f"*********************")

        generator = await _get_generator(key)
        dzi = generator.get_dzi()

        return Response(content=dzi, media_type="application/xml")
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Failed to get dzi for {key}: {str(e)}")


@app.get("/proxy/dummy_files/{level}/{col}_{row}.{format}")
async def proxy_tile(
    level: int, col: int, row: int, format: str, request: Request,
):
    request_args = {k.split('amp;')[-1]: v for k, v in request.query_params.items()}
    image_id, registry = request_args['image_id'], request_args['registry']
    key = f"{image_id}_{registry}"
    try:
        generator = await _get_generator(key)
        settings = await setting_cache.get(key)

        tile = generator.get_tile(level, (col, row), format=format)
        buf = BytesIO()
        tile.save(buf, format, quality=settings.tile_quality)
        resp = Response(content=buf.getvalue(), media_type=f"image/{format}")

        return resp
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Failed to get image tile for {key}: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9010)
