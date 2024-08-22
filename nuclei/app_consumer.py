import os
import time
import redis
import pickle
import argparse
from functools import lru_cache

from sqlalchemy import create_engine, insert, select
from sqlalchemy import and_
from sqlalchemy.orm import sessionmaker

from utils.db import Annotation, Cache
from model_registry import MODEL_REGISTRY, AGENT_CONFIGS


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
        except Exception as e:
            session.rollback()
            return {"message": f"Failed to write to db: {e}!", "status": 0}
        finally:
            session.close()
            lock.release()
            print(f"db_lock:{image_id} ({lock}) got released.")
    else:
        return {"message": f"Failed to acquire `db_lock:{image_id}`.", "status": -1}


def run(registry, max_halt=None, max_latency=0.5, max_write_attempts=5):
    config = AGENT_CONFIGS[registry]
    service = MODEL_REGISTRY.load_service(config)
    print(f"Successfully load model {registry}.")

    global_running = True
    max_halt = max_halt or float('inf')
    batch_inputs = []
    batch_images = []
    st = time.time()

    while time.time() - st < max_halt and global_running:
        if client.exists(registry):
            serialized_entry = client.rpop(registry)
            entry = pickle.loads(serialized_entry)
            # entry = json.loads(serialized_entry)
            batch_inputs.append((
                entry['image_id'], entry['registry'], 
                entry['tile_id'], entry['info']['roi_slide'],
                entry['extra'], 
            ))
            batch_images.append(entry['img'])   # bytes2numpy(entry['img']) for json
            print(f"Retrieve entry from queue (size={client.llen(registry)}): {len(batch_inputs)}")
        else:
            time.sleep(0.1)

        if len(batch_inputs) == config.batch_size or (len(batch_inputs) > 0 and time.time() - st > max_latency):
            pst = time.time()
            outputs = service(batch_images, preprocess=True)
            print(f"Inference batch (size={len(batch_images)}): {time.time() - pst}s.")

            pst = time.time()
            for (image_id, registry, tile_id, patch_info, extra), output in zip(batch_inputs, outputs):
                output = service.convert_results_to_annotations(
                    output, patch_info, annotator=registry, extra=extra,
                )
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
    parser = argparse.ArgumentParser(description="Script with a required positional argument.")
    parser.add_argument('registry', type=str, help='The registry name')
    args = parser.parse_args()

    registry = args.registry
    assert registry in AGENT_CONFIGS, f"AGENT_CONFIGS do not have registry: {registry}."
    run(args.registry)
