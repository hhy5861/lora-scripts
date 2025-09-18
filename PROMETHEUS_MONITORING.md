# Prometheus监控功能说明

## 概述

本项目已集成Prometheus监控功能，可以实时监控flux训练的状态、进度和质量指标。

## 监控指标

### 1. 训练状态指标

- **flux_training_total**: 训练操作总数计数器
  - 标签: `status` (started/running/completed/failed), `model_type`, `task_id`
  
- **flux_training_status**: 当前训练状态
  - 标签: `model_type`, `task_id`
  - 值: 0=idle, 1=running, 2=completed, 3=failed

### 2. 训练进度指标

- **flux_training_progress_percent**: 训练完成度百分比 (0-100)
  - 标签: `model_type`, `task_id`, `progress_type` (step_based/epoch_based)

- **flux_training_steps**: 训练步数进度
  - 标签: `model_type`, `task_id`, `step_type` (current/total)

- **flux_training_epochs**: 训练epoch进度
  - 标签: `model_type`, `task_id`, `epoch_type` (current/total)

### 3. 训练质量指标

- **flux_training_loss**: 训练损失值
  - 标签: `model_type`, `task_id`, `loss_type` (current/average)

### 4. 训练性能指标

- **flux_training_duration_seconds**: 训练持续时间
  - 标签: `model_type`, `status`
  - 直方图: [60s, 5m, 10m, 30m, 1h, 2h, 4h, 8h]

- **flux_training_errors_total**: 训练错误总数
  - 标签: `error_type`, `model_type`

## 使用方法

### 1. 安装依赖

```bash
pip install prometheus_client==0.19.0
```

### 2. 启动服务

正常启动SD-Trainer服务：

```bash
python gui.py --host 0.0.0.0 --port 28000
```

### 3. 访问metrics端点

Prometheus metrics数据可通过以下端点获取：

```
http://localhost:28000/metrics
```

### 4. 配置Prometheus服务器

在Prometheus配置文件中添加以下配置：

```yaml
scrape_configs:
  - job_name: 'sd-trainer'
    static_configs:
      - targets: ['localhost:28000']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

### 5. 使用Grafana可视化

可以创建以下仪表板来监控训练状态：

#### 训练状态面板
- 当前运行中的训练任务数
- 训练成功率
- 训练失败率

#### 训练进度面板
- 当前训练完成度
- 训练步数进度
- 训练epoch进度

#### 训练质量面板
- 当前损失值
- 平均损失值
- 损失值趋势

## 监控示例

### 查看当前训练状态

```bash
curl http://localhost:28000/metrics | grep flux_training_status
```

### 查看训练进度

```bash
curl http://localhost:28000/metrics | grep flux_training_progress_percent
```

### 查看训练损失

```bash
curl http://localhost:28000/metrics | grep flux_training_loss
```

## 支持的训练类型

- **flux-lora**: Flux LoRA训练
- **flux-finetune**: Flux微调训练
- **sd3-lora**: SD3 LoRA训练
- **sdxl-lora**: SDXL LoRA训练
- **sd-lora**: SD LoRA训练

## 故障排除

### 1. metrics端点无法访问

检查服务是否正常启动：
```bash
curl http://localhost:28000/health
```

### 2. 没有metrics数据

确保训练任务正在运行，metrics只在训练过程中产生。

### 3. 依赖安装问题

确保安装了prometheus_client：
```bash
pip install prometheus_client==0.19.0
```

## 测试

运行测试脚本验证metrics功能：

```bash
python test_metrics.py
```

## 注意事项

1. metrics数据只在训练过程中更新
2. 每个训练任务都有唯一的task_id
3. 训练完成后，状态会保持为completed或failed
4. 建议设置合适的scrape_interval以避免过于频繁的请求
