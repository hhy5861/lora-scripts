"""
Prometheus metrics for flux training monitoring
"""
from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry, generate_latest
from typing import Optional, Dict, Set
import time
import threading

# 清理配置常量
CLEANUP_DELAY_SECONDS = 300      # 训练完成后延迟清理时间（5分钟）
CLEANUP_INTERVAL_SECONDS = 3600  # 定期清理间隔（1小时）
MAX_AGE_HOURS = 24               # 最大保留时间（24小时）

# 创建自定义registry
registry = CollectorRegistry()

# 记录每个 task_id 的创建时间
task_creation_times: Dict[str, float] = {}
# 记录活跃的 task_id
active_task_ids: Set[str] = set()

# 1. 训练状态计数器 (Counter) - 移除，不需要
# flux_training_total = Counter(
#     'flux_training_total', 
#     'Total number of flux training operations',
#     ['status', 'model_type', 'task_id'],
#     registry=registry
# )

# 2. 当前训练状态 (Gauge)
flux_training_status = Gauge(
    'flux_training_status',
    'Current flux training status (0=idle, 1=running, 2=completed, 3=failed)',
    ['model_type', 'task_id'],
    registry=registry
)

# 3. 训练完成度百分比 (Gauge)
flux_training_progress_percent = Gauge(
    'flux_training_progress_percent',
    'Current training progress percentage (0-100)',
    ['model_type', 'task_id', 'progress_type'],
    registry=registry
)

# 4. 训练步数进度 (Gauge)
flux_training_steps = Gauge(
    'flux_training_steps',
    'Current training step progress',
    ['model_type', 'task_id', 'step_type'],
    registry=registry
)

# 5. 训练epoch进度已删除

# 6. 训练损失值 (Gauge)
flux_training_loss = Gauge(
    'flux_training_loss',
    'Current training loss value',
    ['model_type', 'task_id', 'loss_type'],
    registry=registry
)

flux_training_iteration_time = Gauge(
    'flux_training_iteration_time',
    'Current training iteration time in seconds',
    ['model_type', 'task_id'],
    registry=registry
)


def get_metrics() -> str:
    """获取Prometheus格式的metrics数据"""
    return generate_latest(registry)

def record_training_start(model_type: str, task_id: str):
    """记录训练开始"""
    # 记录创建时间
    task_creation_times[task_id] = time.time()
    active_task_ids.add(task_id)
    
    flux_training_status.labels(
        model_type=model_type, 
        task_id=task_id
    ).set(1)  # running

def record_training_running(model_type: str, task_id: str):
    """记录训练进行中"""
    flux_training_status.labels(
        model_type=model_type, 
        task_id=task_id
    ).set(1)  # running

def record_training_completed(model_type: str, task_id: str):
    """记录训练完成"""
    flux_training_status.labels(
        model_type=model_type, 
        task_id=task_id
    ).set(2)  # completed
    
    # 训练完成后立即清理
    schedule_cleanup(task_id, delay_seconds=CLEANUP_DELAY_SECONDS)

def record_training_failed(model_type: str, task_id: str, error_type: str = 'unknown'):
    """记录训练失败"""
    flux_training_status.labels(
        model_type=model_type, 
        task_id=task_id
    ).set(3)  # failed
    
    # 训练失败后立即清理
    schedule_cleanup(task_id, delay_seconds=CLEANUP_DELAY_SECONDS)

def record_training_progress(model_type: str, task_id: str, progress_type: str, percent: float):
    """记录训练进度"""
    flux_training_progress_percent.labels(
        model_type=model_type,
        task_id=task_id,
        progress_type=progress_type
    ).set(percent)

def record_training_steps(model_type: str, task_id: str, current_steps: int, total_steps: int):
    """记录训练步数"""
    flux_training_steps.labels(
        model_type=model_type,
        task_id=task_id,
        step_type='current'
    ).set(current_steps)
    
    flux_training_steps.labels(
        model_type=model_type,
        task_id=task_id,
        step_type='total'
    ).set(total_steps)

# record_training_epochs 函数已删除

def record_training_loss(model_type: str, task_id: str, current_loss: float, avg_loss: Optional[float] = None):
    """记录训练损失"""
    flux_training_loss.labels(
        model_type=model_type,
        task_id=task_id,
        loss_type='current'
    ).set(current_loss)
    
    if avg_loss is not None:
        flux_training_loss.labels(
            model_type=model_type,
            task_id=task_id,
            loss_type='average'
        ).set(avg_loss)

def record_training_iteration_time(model_type: str, task_id: str, iteration_time: float):
    """记录训练迭代时间"""
    flux_training_iteration_time.labels(
        model_type=model_type,
        task_id=task_id
    ).set(iteration_time)

def schedule_cleanup(task_id: str, delay_seconds: int):
    """延迟清理任务"""
    def cleanup():
        time.sleep(delay_seconds)
        cleanup_task_metrics(task_id)
        task_creation_times.pop(task_id, None)
        active_task_ids.discard(task_id)
    
    # 在后台线程中执行
    thread = threading.Thread(target=cleanup, daemon=True)
    thread.start()

def cleanup_task_metrics(task_id: str):
    """清理指定 task_id 的所有指标"""
    # 清理所有相关指标
    for metric in [flux_training_status, flux_training_progress_percent, 
                   flux_training_steps, flux_training_epochs, flux_training_loss]:
        try:
            # 获取所有标签组合
            for labels in list(metric._metrics.keys()):
                if 'task_id' in labels and labels['task_id'] == task_id:
                    # 构建标签值列表
                    label_values = [labels.get(label, '') for label in metric._labelnames]
                    metric.remove(*label_values)
        except (AttributeError, KeyError) as e:
            # 忽略清理错误，不影响主程序
            pass

def cleanup_old_metrics(max_age_hours: int = None):
    """清理超过指定时间的旧指标"""
    if max_age_hours is None:
        max_age_hours = MAX_AGE_HOURS
    
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    
    # 找出需要清理的 task_id
    to_remove = []
    for task_id, creation_time in task_creation_times.items():
        if current_time - creation_time > max_age_seconds:
            to_remove.append(task_id)
    
    # 清理指标
    for task_id in to_remove:
        cleanup_task_metrics(task_id)
        task_creation_times.pop(task_id, None)
        active_task_ids.discard(task_id)

def start_cleanup_scheduler():
    """启动定期清理任务"""
    def cleanup_loop():
        while True:
            time.sleep(CLEANUP_INTERVAL_SECONDS)
            cleanup_old_metrics()
    
    thread = threading.Thread(target=cleanup_loop, daemon=True)
    thread.start()
