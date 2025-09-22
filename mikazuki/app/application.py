import asyncio
import mimetypes
import os
import sys
import webbrowser
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException
from prometheus_client import CONTENT_TYPE_LATEST

from mikazuki.app.config import app_config
from mikazuki.app.api import load_schemas, load_presets
from mikazuki.app.api import router as api_router
# from mikazuki.app.ipc import router as ipc_router
from mikazuki.app.proxy import router as proxy_router
from mikazuki.utils.devices import check_torch_gpu
from mikazuki.metrics import get_metrics

mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")


class SPAStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        try:
            return await super().get_response(path, scope)
        except HTTPException as ex:
            if ex.status_code == 404:
                return await super().get_response("index.html", scope)
            else:
                raise ex


async def app_startup():
    app_config.load_config()

    await load_schemas()
    await load_presets()
    await asyncio.to_thread(check_torch_gpu)
    

    if sys.platform == "win32" and os.environ.get("MIKAZUKI_DEV", "0") != "1":
        webbrowser.open(f'http://{os.environ["MIKAZUKI_HOST"]}:{os.environ["MIKAZUKI_PORT"]}')


@asynccontextmanager
async def lifespan(app: FastAPI):
    await app_startup()
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(proxy_router)


cors_config = os.environ.get("MIKAZUKI_APP_CORS", "")
if cors_config != "":
    if cors_config == "1":
        cors_config = ["http://localhost:8004", "*"]
    else:
        cors_config = cors_config.split(";")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_config,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.middleware("http")
async def add_cache_control_header(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "max-age=0"
    return response

app.include_router(api_router, prefix="/api")
# app.include_router(ipc_router, prefix="/ipc")


@app.get("/")
async def index():
    return FileResponse("./frontend/dist/index.html")


@app.get("/favicon.ico", response_class=FileResponse)
async def favicon():
    return FileResponse("assets/favicon.ico")


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(get_metrics(), media_type=CONTENT_TYPE_LATEST)

@app.post("/update_metrics")
async def update_metrics(data: dict):
    """接收训练脚本发送的metrics数据"""
    try:
        from mikazuki.metrics import (
            record_training_progress,
            record_training_steps, 
            record_training_loss,
            record_training_iteration_time
        )
        
        model_type = data.get('model_type', 'unknown')
        task_id = data.get('task_id', 'unknown')
        progress = data.get('progress', {})
        loss = data.get('loss', {})
        
        # 记录步数进度
        if 'current' in progress and 'total' in progress:
            record_training_steps(model_type, progress['current'], progress['total'])
            
        # 记录进度百分比
        if 'percent' in progress:
            record_training_progress(model_type, 'step_based', progress['percent'])
            
        # 记录损失值
        if 'average' in loss:
            record_training_loss(model_type, loss['average'])
            
        # 记录迭代时间
        if 'iteration_time' in data:
            record_training_iteration_time(model_type, data['iteration_time'])
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# 统一的 update_metrics 端点已处理所有类型的 metrics 数据

app.mount("/", SPAStaticFiles(directory="frontend/dist", html=True), name="static")
