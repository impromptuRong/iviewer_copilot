import json

from fastapi import FastAPI, HTTPException, File, Form, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware

from app_worker import run_segmentation


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust the allowed origins as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/segment")
async def segment(
    registry: str = Query(...),  
    image: UploadFile = File(...), 
    params: str = Form(...)
):
    image_bytes = await image.read()
    params = json.loads(params)

    # Enqueue the task to Celery
    task = run_segmentation.apply_async(
        args=[image_bytes, params['prompts'], params['patch_info'], params['extra']], 
        queue=registry,
    )
    # result = run_segmentation.delay(image_bytes, patch_info, prompts, extra)

    try:
        result = task.get(timeout=20)  # Set timeout to avoid long waits
        # {"task_id": task.id, "status": "Task submitted"}
        return result
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Task failed: " + str(e))


if __name__ == "__main__":
    # asyncio.run(test_connection())
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8376)
