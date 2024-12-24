import os
from PIL import Image
from io import BytesIO

from celery import Celery
from celery.signals import worker_process_init

from model_registry import MODEL_REGISTRY, AGENT_CONFIGS

# Get model registry
REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
REDIS_PORT = os.environ.get('REDIS_PORT', 6379)
celery = Celery(
    "segment",
    broker=f"redis://{REDIS_HOST}:{REDIS_PORT}/0",
    backend=f"redis://{REDIS_HOST}:{REDIS_PORT}/0",
)

celery.conf.update(
    worker_concurrency=1
)

service = None  # Global model variable
registry = os.getenv('MODEL_REGISTRY', 'sam2-b')
device = os.getenv('MODEL_DEVICE', 'cpu')

@worker_process_init.connect
def load_model(**kwargs):
    global service
    assert registry in AGENT_CONFIGS, f"AGENT_CONFIGS do not have registry: {registry}."

    config = AGENT_CONFIGS[registry]
    service = MODEL_REGISTRY.load_service(config, device=device, test=True)
    print(f"Load model {registry} and test run on {service.model.device} success!")

@celery.task(bind=True)
def run_segmentation(self, image_bytes, prompts, patch_info, extra):
    image = Image.open(BytesIO(image_bytes))
    # print("Worker received: ", image.size, prompts, patch_info, extra)
    segments = service(image, prompts=prompts, preprocess=False)
    generator = service.map_coords_to_records(
        segments, patch_info, annotator=registry, extra=extra
    )

    return next(generator)
