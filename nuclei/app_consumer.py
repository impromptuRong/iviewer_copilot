import os
import cv2
import time
import json
import redis
import pickle
import requests
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
from functools import lru_cache

from sqlalchemy import create_engine, insert, select
from sqlalchemy import or_, and_
from sqlalchemy.orm import sessionmaker

from utils.db import Base, Annotation, Cache

import config
from model_registry import MODEL_REGISTRY


# SQLAlchemy setup
DATABASE_DIR = os.environ.get('DATABASE_PATH', './databases/')
os.makedirs(DATABASE_DIR, exist_ok=True)

# Redis connection
REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
REDIS_PORT = os.environ.get('REDIS_PORT', 6379)
client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
print(f"Redis connection successful: {client.ping()}")


@lru_cache
def get_sessionmaker(image_id):
    db_name = f'{image_id}.db'
    database_url = f"sqlite:///{os.path.join(DATABASE_DIR, db_name)}"
    engine = create_engine(database_url, echo=False, future=True)
    session = sessionmaker(engine, expire_on_commit=False)

    return session


def test_run(service, image="http://images.cocodataset.org/val2017/000000039769.jpg"):
    print(f"Testing service for {image}")
    st = time.time()
    image = Image.open(requests.get(image, stream=True).raw)
    r = service([np.array(image)])

    print(f"{generated_text} ({time.time()-st}s)")


def entry_exists_in_cache(query, session):
    stmt = select(Cache).where(and_(Cache.registry == query['registry'], Cache.tile_id == query['tile_id']))
    result = session.execute(stmt)

    return result.one_or_none()


def export_to_db(entry):
    image_id, registry, tile_id, output = entry

    lock = client.lock(f"db_lock:{image_id}")
    acquired = lock.acquire(blocking=True, blocking_timeout=3)
    print(f"db_lock:{image_id} ({lock}) acquired={acquired}.")

    if acquired:
        session = get_sessionmaker(image_id)()
        query = {'registry': registry, 'tile_id': tile_id}
        try:
            print(f"db_lock:{image_id} ({lock}) got locked.")
            if not entry_exists_in_cache(query, session):
                session.execute(insert(Annotation), output)
                session.execute(insert(Cache), query)
                print(f"Insert into db:{image_id}: {len(output)} entries.")
            else:
                print(f"db_cache:{image_id} already analyzed query: {query}.")
            session.commit()
            return {"message": "Write successfully!", "status": 1}
        except:
            session.rollback()
            return {"message": "Failed to write!", "status": 0}
        finally:
            session.close()
            lock.release()
            print(f"db_lock:{image_id} ({lock}) got released.")
    else:
        return {"message": f"Failed to acquire `db_lock:{image_id}`.", "status": -1}


def run(service, max_halt=None, max_latency=0.5, max_write_attempts=5):
    global_running = True
    max_halt = max_halt or float('inf')
    batch_inputs = []
    batch_images = []
    st = time.time()

    while time.time() - st < max_halt and global_running:
        if client.exists(config.model_name):
            serialized_entry = client.rpop(config.model_name)
            entry = pickle.loads(serialized_entry)
            # entry = json.loads(serialized_entry)
            batch_inputs.append((
                entry['image_id'], entry['registry'], entry['project'], 
                entry['tile_id'], entry['info']['roi_slide'],
            ))
            batch_images.append(entry['img'])   # bytes2numpy(entry['img']) for json
            print(f"Retrieve entry from queue (size={client.llen(config.model_name)}): {len(batch_inputs)}")
        else:
            time.sleep(0.1)

        if len(batch_inputs) == config.batch_size or (len(batch_inputs) > 0 and time.time() - st > max_latency):
            pst = time.time()
            outputs = service(batch_images, preprocess=True)
            print(f"Inference batch (size={len(batch_images)}): {time.time() - pst}s.")

            pst = time.time()
            for (image_id, registry, project, tile_id, patch_info), output in zip(batch_inputs, outputs):
                output = service.convert_results_to_annotations(output, patch_info, annotator=registry, project=project)
                status, attempts = -1, 0
                while status <= 0 and attempts < max_write_attempts:
                    response = export_to_db([image_id, registry, tile_id, output])
                    status = response["status"]
                    if status <= 0:
                        time.sleep(0.1)
                        attempts += 1
                print(response)
            print(f"Write to db batch (size={len(batch_inputs)}): {time.time() - pst}s.")

            batch_inputs = []
            batch_images = []
            st = time.time()


if __name__ == "__main__":
    service = MODEL_REGISTRY.get_model(config.model_name)(
        config, device=config.device
    )
    # test_run(service)
    
    run(service)
