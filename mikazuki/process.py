
import asyncio
import os
import sys
from typing import Optional

from mikazuki.app.models import APIResponse
from mikazuki.log import log
from mikazuki.tasks import tm
from mikazuki.launch_utils import base_dir_path
from mikazuki.metrics import record_training_start, record_training_completed, record_training_failed


def run_train(toml_path: str,
              trainer_file: str = "./scripts/train_network.py",
              gpu_ids: Optional[list] = None,
              cpu_threads: Optional[int] = 2):
    log.info(f"Training started with config file / 训练开始，使用配置文件: {toml_path}")
    
    # 确定模型类型
    model_type = "unknown"
    if "flux_train_network.py" in trainer_file:
        model_type = "flux-lora"
    elif "flux_train.py" in trainer_file:
        model_type = "flux-finetune"
    elif "sd3_train_network.py" in trainer_file:
        model_type = "sd3-lora"
    elif "sdxl_train_network.py" in trainer_file:
        model_type = "sdxl-lora"
    elif "train_network.py" in trainer_file:
        model_type = "sd-lora"
    
    args = [
        sys.executable, "-m", "accelerate.commands.launch",  # use -m to avoid python script executable error
        "--num_cpu_threads_per_process", str(cpu_threads),  # cpu threads
        "--quiet",  # silence accelerate error message
        trainer_file,
        "--config_file", toml_path,
    ]

    customize_env = os.environ.copy()
    customize_env["ACCELERATE_DISABLE_RICH"] = "1"
    customize_env["PYTHONUNBUFFERED"] = "1"
    customize_env["PYTHONWARNINGS"] = "ignore::FutureWarning,ignore::UserWarning"

    if gpu_ids:
        customize_env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
        log.info(f"Using GPU(s) / 使用 GPU: {gpu_ids}")

        if len(gpu_ids) > 1:
            args[3:3] = ["--multi_gpu", "--num_processes", str(len(gpu_ids))]
            if sys.platform == "win32":
                customize_env["USE_LIBUV"] = "0"
                args[3:3] = ["--rdzv_backend", "c10d"]

    if not (task := tm.create_task(args, customize_env, model_type)):
        return APIResponse(status="error", message="Failed to create task / 无法创建训练任务")

    # 设置task_id环境变量，供训练脚本使用
    customize_env["TRAINING_TASK_ID"] = task.task_id

    # 记录训练开始metrics
    record_training_start(model_type)

    def _run():
        try:
            task.execute()
            result = task.communicate()
            if result.returncode != 0:
                log.error(f"Training failed / 训练失败")
                # 记录训练失败metrics
                record_training_failed(model_type, "return_code_error")
            else:
                log.info(f"Training finished / 训练完成")
                # 记录训练完成metrics
                record_training_completed(model_type)
        except Exception as e:
            log.error(f"An error occurred when training / 训练出现致命错误: {e}")
            # 记录训练异常metrics
            record_training_failed(model_type, "exception")

    coro = asyncio.to_thread(_run)
    asyncio.create_task(coro)

    return APIResponse(status="success", message=f"Training started / 训练开始 ID: {task.task_id}")
