import os
import re
import json
import time
import asyncio
import redis.asyncio as redis
from datetime import datetime
from async_lru import alru_cache
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, Body, Response, Depends, WebSocket, WebSocketDisconnect

from sqlalchemy import or_, and_
from sqlalchemy import select, delete, update, insert
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from fastapi.middleware.cors import CORSMiddleware

from utils.db import Base, Annotation


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust the allowed origins as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# SQLAlchemy setup
DATABASE_DIR = os.environ.get('DATABASE_PATH', './databases/')
os.makedirs(DATABASE_DIR, exist_ok=True)


# Redis connection
REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
REDIS_PORT = os.environ.get('REDIS_PORT', 6379)
pool = redis.ConnectionPool(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)
async def get_redis():
    return redis.Redis(connection_pool=pool)


def encode_path(x):
    return x.replace('/', '%2F').replace(':', '%3A')


def format_labels(text):
    tags = set()
    for item in re.split(r'\s*[,;]\s*', text.strip()):
        item = re.sub(r'\W+', '_', item)
        if item:
            tags.add(item)

    return ','.join(tags)


## helpful codes: https://praciano.com.br/fastapi-and-async-sqlalchemy-20-with-pytest-done-right.html
# Dependency
@alru_cache
async def get_sessionmaker(image_id):
    db_name = f'{image_id}.db'
    db_path = os.path.join(DATABASE_DIR, db_name)
    database_url = f"sqlite+aiosqlite:///{db_path}"
    print(database_url)
    engine = create_async_engine(database_url)
    async_session = async_sessionmaker(engine, expire_on_commit=False)

    return async_session


async def get_session(image_id):
    async_session = await get_sessionmaker(image_id)
    session = async_session()
    try:
        yield session
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


@app.post("/annotation/create")
async def create_table(image_id: str, client=Depends(get_redis)):
    lock = client.lock(f"db_lock:{image_id}")
    acquired = await lock.acquire(blocking=True, blocking_timeout=3)
    print(f"(Create) db_lock:{image_id} ({lock}) acquired={acquired}.")

    if acquired:
        try:
            print(f"(Create) db_lock:{image_id} ({lock}) is locked.")
            db_name = f'{image_id}.db'
            db_path = os.path.join(DATABASE_DIR, db_name)
            if not os.path.exists(db_path):
                database_url = f"sqlite+aiosqlite:///{db_path}"
                engine = create_async_engine(database_url, echo=True)
                async with engine.begin() as conn:
                    await conn.run_sync(Base.metadata.create_all)
                msg = f"Create database {db_path}"
            else:
                msg = f"Database {db_path} exists."
            # return JSONResponse(content={"message": msg}, status_code=200)
            return Response(content=msg, status_code=200, media_type="text/plain")
        except Exception as e:
            msg = f"Failed to create database {db_path}: {str(e)}"
            # raise HTTPException(status_code=400, detail=str(e))
            return Response(content=msg, status_code=500, media_type="text/plain")
        finally:
            await lock.release()
            print(f"(Release) db_lock:{image_id} ({lock}) is released.")
    else:
        msg = f"Failed to acquire `db_lock:{image_id}`. Processed by another process/thread."
        # return JSONResponse(content={"message": msg}, status_code=400)
        return Response(content=msg, status_code=409, media_type="text/plain")


@app.get("/annotation/annotators")
async def get_all_annotators(image_id: str, session=Depends(get_session)):
    try:
        print(f"Find all annotators from `{image_id}.db/annotation`")
        stmt = select(Annotation.annotator).distinct()
        result = await session.execute(stmt)
        if not result:
            raise HTTPException(status_code=404, detail="Annotators not found. ")

        annotators = [obj for obj in result.scalars()]

        return annotators
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to get all annotators: {str(e)}")


@app.get("/annotation/groups")
async def get_all_groups(image_id: str, session=Depends(get_session)):
    try:
        print(f"Find all groups from `{image_id}.db/annotation`")
        stmt = select(Annotation.group_id).distinct()
        result = await session.execute(stmt)
        if not result:
            raise HTTPException(status_code=404, detail=str(e))

        groups = [obj for obj in result.scalars()]

        return groups
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/annotation/projects")
async def get_all_projects(image_id: str, session=Depends(get_session)):
    try:
        print(f"Find all projects from `{image_id}.db/annotation`")
        stmt = select(Annotation.project_id).distinct()
        result = await session.execute(stmt)
        if not result:
            raise HTTPException(status_code=404, detail="Projects not found. ")

        projects = [obj for obj in result.scalars()]

        return projects
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to get all projects: {str(e)}")


@app.get("/annotation/labels")
async def get_all_labels(image_id: str, session=Depends(get_session)):
    try:
        print(f"Find all labels from `{image_id}.db/annotation`")
        stmt = select(Annotation.label).distinct()
        result = await session.execute(stmt)
        if not result:
            raise HTTPException(status_code=404, detail="Labels not found. ")

        labels = set()
        for obj in result.scalars():
            if obj:
                for item in obj.split(','):
                    if item:
                        labels.add(item)
        labels = list(labels)

        return labels
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to get all labels: {str(e)}")


@app.post("/annotation/insert")
async def insert_data(image_id: str, item=Body(...), session=Depends(get_session), client=Depends(get_redis)):
    lock = client.lock(f"db_lock:{image_id}")
    acquired = await lock.acquire(blocking=True, blocking_timeout=3)
    print(f"(Insert) db_lock:{image_id} ({lock}) acquired={acquired}.")

    if acquired:
        msg = ''
        try:
            print(f"(Insert) db_lock:{image_id} ({lock}) is locked.")
            obj = {
                'x0': float(item['x0']),
                'y0': float(item['y0']),
                'x1': float(item['x1']),
                'y1': float(item['y1']),
                'xc': float(item['xc']) if 'xc' in item else None,
                'yc': float(item['yc']) if 'yc' in item else None,
                'poly_x': item.get('poly_x', ''),
                'poly_y': item.get('poly_y', ''),
                'label': format_labels(item.get('label', '')),
                'description': item.get('description', ''),
                'annotator': item.get('annotator', ''),
                'project_id': item.get('project_id', ''),
                'group_id': item.get('group_id', ''),
            }

            # Calculate xc and yc if not given
            if obj['xc'] is None:
                obj['xc'] = (obj['x0'] + obj['x1']) / 2
            if obj['yc'] is None:
                obj['yc'] = (obj['y0'] + obj['y1']) / 2

            # Parse the datetime string
            try:
                created_at = item['created_at'].replace('Z', '+00:00')
                created_at = datetime.fromisoformat(created_at)
            except:
                created_at = None
            if created_at:
                obj['created_at'] = created_at

            result = await session.execute(insert(Annotation).returning(Annotation), obj)
            await session.commit()
            obj = result.scalar().to_dict()
            msg = f"Insert item={obj} to `{image_id}/annotation` successfully. "
            print(msg)
            return obj
        except Exception as e:
            msg = f"Failed to add item={item} to database `{image_id}/annotation`. {str(e)}"
            raise HTTPException(status_code=500, detail=msg)
        finally:
            await lock.release()
            print(f"(Insert) db_lock:{image_id} ({lock}) is released.")
    else:
        msg = f"Failed to acquire `db_lock:{image_id}`. Processed by another process/thread."
        raise HTTPException(status_code=409, detail=msg)


@app.put("/annotation/update")
async def update_data(image_id: str, item_id: int, item=Body(...), session=Depends(get_session), 
                      client=Depends(get_redis)):
    lock = client.lock(f"db_lock:{image_id}")
    acquired = await lock.acquire(blocking=True, blocking_timeout=3)
    print(f"(Update) db_lock:{image_id} ({lock}) acquired={acquired}.")

    if acquired:
        try:
            print(f"(Update) db_lock:{image_id} ({lock}) is locked.")
            obj = {
                k: v for k, v in item.items()
                if k in Annotation.__table__.columns.keys() and k != 'id'
            }
            if 'x0' in obj:
                obj['x0'] = float(obj['x0'])
            if 'y0' in obj:
                obj['y0'] = float(obj['y0'])
            if 'x1' in obj:
                obj['x1'] = float(obj['x1'])
            if 'y1' in obj:
                obj['y1'] = float(obj['y1'])
            if 'xc' in obj:
                obj['xc'] = float(obj['xc'])
            if 'yc' in obj:
                obj['yc'] = float(obj['yc'])
            if 'x0' in obj and 'x1' in obj:
                obj['xc'] = obj.get('xc', (obj['x0'] + obj['x1']) / 2)
            if 'y0' in obj and 'y1' in obj:
                obj['yc'] = obj.get('yc', (obj['y0'] + obj['y1']) / 2)
            if 'label' in obj:
                obj['label'] = format_labels(obj['label'])

            # Parse the datetime string
            if obj['created_at']:
                try:
                    created_at = obj['created_at'].replace('Z', '+00:00')
                    created_at = datetime.fromisoformat(created_at)
                except:
                    created_at = None
                if created_at:
                    obj['created_at'] = created_at
                else:
                    del obj['created_at']

            stmt = update(Annotation).where(Annotation.id == item_id).values(**obj).returning(Annotation)
            result = await session.execute(stmt)
            await session.commit()
            obj = result.scalar().to_dict()
            msg = f"Updated item_id={item_id} to item={obj} in `{image_id}/annotation` successfully. "
            print(msg)
            # return Response(content=msg, status_code=200, media_type="text/plain")
            return obj
        except Exception as e:
            msg = f"Failed to update item_id={item_id} with item={item} in `{image_id}/annotation`. {str(e)}"
            # raise HTTPException(status_code=400, detail=str(e))
            return Response(content=msg, status_code=500, media_type="text/plain")
        finally:
            await lock.release()
            print(f"(Update) db_lock:{image_id} ({lock}) is released.")
    else:
        msg = f"Failed to acquire `db_lock:{image_id}`. Processed by another process/thread."
        return Response(content=msg, status_code=409, media_type="text/plain")


@app.delete("/annotation/delete")
async def delete_data(image_id: str, item_id: int, session=Depends(get_session), client=Depends(get_redis)):
    lock = client.lock(f"db_lock:{image_id}")
    acquired = await lock.acquire(blocking=True, blocking_timeout=3)
    print(f"(Delete) db_lock:{image_id} ({lock}) acquired={acquired}.")

    if acquired:
        try:
            print(f"(Delete) db_lock:{image_id} ({lock}) is locked.")
            stmt = delete(Annotation).where(Annotation.id == item_id).returning(Annotation)
            result = await session.execute(stmt)
            await session.commit()
            obj = result.scalar().to_dict()
            msg = f"Deleted item_id={obj} in `{image_id}/annotation` successfully. "
            return Response(content=msg, status_code=200, media_type="text/plain")
        except Exception as e:
            msg = f"Failed to delete item_id={item_id} in `{image_id}/annotation`. {str(e)}"
            # raise HTTPException(status_code=400, detail=str(e))
            return Response(content=msg, status_code=500, media_type="text/plain")
        finally:
            await lock.release()
            print(f"(Delete) db_lock:{image_id} ({lock}) is released.")
    else:
        msg = f"Failed to acquire `db_lock:{image_id}`. Processed by another process/thread."
        return Response(content=msg, status_code=409, media_type="text/plain")

        
@app.get("/annotation/read")
async def read_data(image_id: str, item_id: int, session=Depends(get_session)):
    try:
        stmt = select(Annotation).where(Annotation.id == item_id)
        result = await session.execute(stmt)
        result = result.one_or_none()
        if not result:
            raise HTTPException(status_code=404, detail=f"Item {item_id} not found. ")
        return result[0].to_dict()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read item_id={item_id}: {str(e)}")


@app.post("/annotation/v1/search")
async def search_data_v1(image_id: str, item=Body(...), session=Depends(get_session)):
    try:
        stmt = select(Annotation)

        filters = []
        print(f"Search `{image_id}.db/annotation` with query {item}")
        if 'x0' in item:
            filters.append(Annotation.x0 >= float(item['x0']))
        if 'y0' in item:
            filters.append(Annotation.y0 >= float(item['y0']))
        if 'x1' in item:
            filters.append(Annotation.x1 < float(item['x1']))
        if 'y1' in item:
            filters.append(Annotation.y1 < float(item['y1']))

        if 'label' in item:
            if isinstance(item['label'], str):
                filters.append(Annotation.label == item['label'])
            else:
                filters.append(Annotation.label.in_(item['label']))

        if 'annotator' in item:
            if isinstance(item['annotator'], str):
                filters.append(Annotation.annotator == item['annotator'])
            else:
                filters.append(Annotation.annotator.in_(item['annotator']))
        if filters:
            stmt = stmt.where(and_(*filters))

        keywords = []
        if 'description' in item:
            if isinstance(item['description'], str):
                keywords.append(Annotation.description.icontains(item['description']))
            else:
                for keyword in item['description']:
                    keywords.append(Annotation.description.icontains(keyword))
        if keywords:
            stmt = stmt.filter(or_(*keywords))

        # stmt = select(Annotation).where(and_(*filters)).filter(or_(True, *keywords))
        result = await session.execute(stmt)
        result = result.scalars()
        if not result:
            raise HTTPException(status_code=404, detail="Item not found")
        return [obj.to_dict() for obj in result][:100]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


async def search_iterator(query, session):
    stmt = select(Annotation)

    filters = []
    if 'x0' in query:
        filters.append(Annotation.x0 >= float(query['x0']))
    if 'y0' in query:
        filters.append(Annotation.y0 >= float(query['y0']))
    if 'x1' in query:
        filters.append(Annotation.x1 < float(query['x1']))
    if 'y1' in query:
        filters.append(Annotation.y1 < float(query['y1']))
    # if 'min_box_area' in query:
    #     filters.append((Annotation.x1 - Annotation.x0) * (Annotation.y1 - Annotation.y0) > float(query['min_box_area']))
    # if 'max_box_area' in query:
    #     filters.append((Annotation.x1 - Annotation.x0) * (Annotation.y1 - Annotation.y0) <= float(query['max_box_area']))

    if 'start_time' in query:
        try:
            start_time = datetime.fromisoformat(query['start_time'].replace('Z', '+00:00'))
        except:
            start_time = None
        if start_time:
            filters.append(Annotation.created_at >= start_time)
    if 'end_time' in query:
        try:
            end_time = datetime.fromisoformat(query['end_time'].replace('Z', '+00:00'))
        except:
            end_time = None
        if end_time:
            filters.append(Annotation.created_at < end_time)
    
    if 'label' in query:
        if isinstance(query['label'], str):
            filters.append(Annotation.label == query['label'])
        else:
            filters.append(Annotation.label.in_(query['label']))

    if 'annotator' in query:
        if isinstance(query['annotator'], str):
            filters.append(Annotation.annotator == query['annotator'])
        else:
            if query['annotator']:
                filters.append(Annotation.annotator.in_(query['annotator']))

    if 'project_id' in query:
        if isinstance(query['project_id'], str):
            filters.append(Annotation.project_id == query['project_id'])
        else:
            if query['project_id']:
                filters.append(Annotation.project_id.in_(query['project_id']))

    if 'group_id' in query:
        if isinstance(query['group_id'], str):
            filters.append(Annotation.group_id == query['group_id'])
        else:
            if query['group_id']:
                filters.append(Annotation.group_id.in_(query['group_id']))

    if filters:
        stmt = stmt.where(and_(*filters))

    keywords = []
    if 'description' in query:
        if isinstance(query['description'], str):
            keywords.append(Annotation.description.icontains(query['description']))
        else:
            for keyword in query['description']:
                keywords.append(Annotation.description.icontains(keyword))
    if keywords:
        stmt = stmt.filter(or_(*keywords))

    # stmt = select(Annotation).where(and_(*filters)).filter(or_(True, *keywords))
    result = await session.stream_scalars(stmt)
    async for scalar in result:
        if 'min_box_area' in query:
            if (scalar.x1 - scalar.x0) * (scalar.y1 - scalar.y0) < query['min_box_area']:
                continue
        if 'max_box_area' in query:
            if (scalar.x1 - scalar.x0) * (scalar.y1 - scalar.y0) > query['max_box_area']:
                continue
        yield scalar


@app.post("/annotation/search")
async def search_data(image_id: str, item=Body(...), session=Depends(get_session)):
    db_path = os.path.join(DATABASE_DIR, f'{image_id}.db')
    if not os.path.exists(db_path):
        return []
    try:
        print(f"Search `{image_id}.db/annotation` with query {item}")
        iterator = search_iterator(item, session=session)
        results = []
        async for obj in iterator:
            results.append(obj.to_dict())
        if not results:
            raise HTTPException(status_code=404, detail=f"Items not found with query: {item}")
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to search with query={item}. {str(e)}")


@app.post("/annotation/count")
async def count_data(image_id: str, item=Body(...), session=Depends(get_session)):
    db_path = os.path.join(DATABASE_DIR, f'{image_id}.db')
    if not os.path.exists(db_path):
        return 0
    try:
        print(f"Count no. of annotations in `{image_id}.db/annotation` with query {item}")
        iterator = search_iterator(item, session=session)
        N = 0
        async for obj in iterator:
            N += 1
        return N
    except Exception as e:
        # await session.rollback()
        raise HTTPException(status_code=400, detail=f"Failed to count annotations. {str(e)}")


## websockets should be able to stick to connected worker only. 
## So we should be safe to use local Dict to track connection without worry about multi-workers.
## Observed bugs: when search called too frequent, stream + search, will meet db close error.

# connected_websockets: Dict[WebSocket, asyncio.Task] = {}
@app.websocket("/annotation/stream")
async def search_data_stream(websocket: WebSocket, image_id: str):
    # we use one session for all websocket transaction to avoid close and recycle issue
    await websocket.accept()
    async_session = await get_sessionmaker(image_id)
    print(f"WebSocket connection established for {image_id}")

    async def get_query():
        data = await websocket.receive_text()
        return json.loads(data)
    
    async def stream_items(query: Dict):
        # sleep 0.001s every 0.5s, cannot receive new query if don't sleep.
        sleep_interval = 0.5
        sleep_time = 0.001
        session = async_session()
        try:
            iterator = search_iterator(query, session)
            st = time.time()
            async for obj in iterator:
                await websocket.send_json(obj.to_dict())
                if time.time() - st >= sleep_interval:
                    await asyncio.sleep(sleep_time)
                    st = time.time()
        except Exception:  # asyncio.CancelledError
            await session.rollback()
        finally:
            await session.close()

    _is_valid = lambda x: isinstance(x, dict)

    query, task = None, None
    try:
        fetch_interval = 1  # await for query no earlier than every 1s, too frequent can't fully rollback database connection.
        st = time.time()
        await asyncio.sleep(1)
        while True:
            query = await get_query()
            if time.time() - st > fetch_interval and _is_valid(query):
                print(f"Search `{image_id}.db/annotation` with query {query}")
                if task and not task.done():
                    task.cancel()
                    print(f"Cancelled {task}.")
                task = asyncio.create_task(stream_items(query))
                print(f"Create search job for: {query}")
                st = time.time()
            else:
                pass
    except WebSocketDisconnect:
        if task and not task.done():
            task.cancel()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9020)
