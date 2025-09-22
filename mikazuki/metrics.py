"""
Prometheus metrics for flux training monitoring
"""
from prometheus_client import Gauge, CollectorRegistry, generate_latest

# 创建自定义registry
registry = CollectorRegistry()

# 1. 当前训练状态 (Gauge)
flux_training_status = Gauge(
    'flux_training_status',
    'Current flux training status (0=idle, 1=running, 2=completed, 3=failed)',
    ['model_type'],
    registry=registry
)

# 2. 训练完成度百分比 (Gauge)
flux_training_progress_percent = Gauge(
    'flux_training_progress_percent',
    'Current training progress percentage (0-100)',
    ['model_type', 'progress_type'],
    registry=registry
)

# 3. 训练步数进度 (Gauge)
flux_training_steps = Gauge(
    'flux_training_steps',
    'Current training step progress',
    ['model_type', 'step_type'],
    registry=registry
)

# 4. 训练损失值 (Gauge)
flux_training_loss = Gauge(
    'flux_training_loss',
    'Current training loss value',
    ['model_type'],
    registry=registry
)

# 5. 训练迭代时间 (Gauge)
flux_training_iteration_time = Gauge(
    'flux_training_iteration_time',
    'Current training iteration time in seconds',
    ['model_type'],
    registry=registry
)


def get_metrics() -> str:
    """获取Prometheus格式的metrics数据"""
    return generate_latest(registry)

def record_training_start(model_type: str, task_id: str):
    """记录训练开始"""
    flux_training_status.labels(
        model_type=model_type
    ).set(1)  # running

def record_training_running(model_type: str, task_id: str):
    """记录训练进行中"""
    flux_training_status.labels(
        model_type=model_type
    ).set(1)  # running

def record_training_completed(model_type: str, task_id: str):
    """记录训练完成"""
    flux_training_status.labels(
        model_type=model_type
    ).set(2)  # completed

def record_training_failed(model_type: str, task_id: str, error_type: str = 'unknown'):
    """记录训练失败"""
    flux_training_status.labels(
        model_type=model_type
    ).set(3)  # failed

def record_training_progress(model_type: str, task_id: str, progress_type: str, percent: float):
    """记录训练进度"""
    flux_training_progress_percent.labels(
        model_type=model_type,
        progress_type=progress_type
    ).set(percent)

def record_training_steps(model_type: str, task_id: str, current_steps: int, total_steps: int):
    """记录训练步数"""
    flux_training_steps.labels(
        model_type=model_type,
        step_type='current'
    ).set(current_steps)
    
    flux_training_steps.labels(
        model_type=model_type,
        step_type='total'
    ).set(total_steps)

# record_training_epochs 函数已删除

def record_training_loss(model_type: str, task_id: str, avg_loss: float):
    """记录训练损失"""
    flux_training_loss.labels(
        model_type=model_type
    ).set(avg_loss)

def record_training_iteration_time(model_type: str, task_id: str, iteration_time: float):
    """记录训练迭代时间"""
    flux_training_iteration_time.labels(
        model_type=model_type
    ).set(iteration_time)

