"""
Prometheus metrics for flux training monitoring
"""
from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry, generate_latest
from typing import Optional

# 创建自定义registry
registry = CollectorRegistry()

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

# 5. 训练epoch进度 (Gauge)
flux_training_epochs = Gauge(
    'flux_training_epochs',
    'Current training epoch progress',
    ['model_type', 'task_id', 'epoch_type'],
    registry=registry
)

# 6. 训练损失值 (Gauge)
flux_training_loss = Gauge(
    'flux_training_loss',
    'Current training loss value',
    ['model_type', 'task_id', 'loss_type'],
    registry=registry
)

# 7. 训练持续时间 (Histogram) - 恢复task_id标签
flux_training_duration_seconds = Histogram(
    'flux_training_duration_seconds',
    'Duration of flux training operations in seconds',
    ['model_type', 'status', 'task_id'],
    buckets=[60, 300, 600, 1800, 3600, 7200, 14400, 28800],  # 1min to 8hours
    registry=registry
)

# 8. 训练错误计数器 (Counter) - 恢复task_id标签
flux_training_errors_total = Counter(
    'flux_training_errors_total',
    'Total number of flux training errors',
    ['error_type', 'model_type', 'task_id'],
    registry=registry
)

# 9. 训练开始时间 (Gauge) - 用于计算持续时间
flux_training_start_time = Gauge(
    'flux_training_start_time',
    'Training start timestamp',
    ['model_type', 'task_id'],
    registry=registry
)

def get_metrics() -> str:
    """获取Prometheus格式的metrics数据"""
    return generate_latest(registry)

def record_training_start(model_type: str, task_id: str):
    """记录训练开始"""
    import time
    
    flux_training_status.labels(
        model_type=model_type, 
        task_id=task_id
    ).set(1)  # running
    
    flux_training_start_time.labels(
        model_type=model_type, 
        task_id=task_id
    ).set(time.time())

def record_training_running(model_type: str, task_id: str):
    """记录训练进行中"""
    flux_training_status.labels(
        model_type=model_type, 
        task_id=task_id
    ).set(1)  # running

def record_training_completed(model_type: str, task_id: str):
    """记录训练完成"""
    import time
    
    flux_training_status.labels(
        model_type=model_type, 
        task_id=task_id
    ).set(2)  # completed
    
    # 计算训练持续时间
    start_time = flux_training_start_time.labels(
        model_type=model_type, 
        task_id=task_id
    )._value._value
    
    if start_time > 0:
        duration = time.time() - start_time
        flux_training_duration_seconds.labels(
            model_type=model_type, 
            status='completed',
            task_id=task_id
        ).observe(duration)

def record_training_failed(model_type: str, task_id: str, error_type: str = 'unknown'):
    """记录训练失败"""
    import time
    
    flux_training_status.labels(
        model_type=model_type, 
        task_id=task_id
    ).set(3)  # failed
    
    flux_training_errors_total.labels(
        error_type=error_type, 
        model_type=model_type,
        task_id=task_id
    ).inc()
    
    # 计算训练持续时间
    start_time = flux_training_start_time.labels(
        model_type=model_type, 
        task_id=task_id
    )._value._value
    
    if start_time > 0:
        duration = time.time() - start_time
        flux_training_duration_seconds.labels(
            model_type=model_type, 
            status='failed',
            task_id=task_id
        ).observe(duration)

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

def record_training_epochs(model_type: str, task_id: str, current_epoch: int, total_epochs: int):
    """记录训练epoch"""
    flux_training_epochs.labels(
        model_type=model_type,
        task_id=task_id,
        epoch_type='current'
    ).set(current_epoch)
    
    flux_training_epochs.labels(
        model_type=model_type,
        task_id=task_id,
        epoch_type='total'
    ).set(total_epochs)

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
