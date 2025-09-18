#!/usr/bin/env python3
"""
测试Prometheus metrics功能
"""
import os
import sys
import time

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    from mikazuki.metrics import (
        record_training_start,
        record_training_running,
        record_training_completed,
        record_training_failed,
        record_training_progress,
        record_training_steps,
        record_training_epochs,
        record_training_loss,
        get_metrics
    )
    print("✅ Metrics模块导入成功")
except ImportError as e:
    print(f"❌ Metrics模块导入失败: {e}")
    sys.exit(1)

def test_metrics():
    """测试metrics功能"""
    print("\n🧪 开始测试Prometheus metrics...")
    
    # 测试数据
    model_type = "flux-lora"
    task_id = "test-task-123"
    
    try:
        # 1. 测试训练开始
        print("1. 测试训练开始...")
        record_training_start(model_type, task_id)
        print("   ✅ 训练开始metrics记录成功")
        
        # 2. 测试训练进行中
        print("2. 测试训练进行中...")
        record_training_running(model_type, task_id)
        print("   ✅ 训练进行中metrics记录成功")
        
        # 3. 测试训练进度
        print("3. 测试训练进度...")
        record_training_progress(model_type, task_id, "step_based", 25.5)
        record_training_progress(model_type, task_id, "epoch_based", 10.0)
        print("   ✅ 训练进度metrics记录成功")
        
        # 4. 测试训练步数
        print("4. 测试训练步数...")
        record_training_steps(model_type, task_id, 250, 1000)
        print("   ✅ 训练步数metrics记录成功")
        
        # 5. 测试训练epoch
        print("5. 测试训练epoch...")
        record_training_epochs(model_type, task_id, 2, 10)
        print("   ✅ 训练epoch metrics记录成功")
        
        # 6. 测试训练损失
        print("6. 测试训练损失...")
        record_training_loss(model_type, task_id, 0.0234, 0.0256)
        print("   ✅ 训练损失metrics记录成功")
        
        # 7. 测试训练完成
        print("7. 测试训练完成...")
        record_training_completed(model_type, task_id)
        print("   ✅ 训练完成metrics记录成功")
        
        # 8. 测试训练失败
        print("8. 测试训练失败...")
        record_training_failed(model_type, task_id, "test_error")
        print("   ✅ 训练失败metrics记录成功")
        
        # 9. 获取metrics数据
        print("9. 获取metrics数据...")
        metrics_data = get_metrics()
        print(f"   ✅ 获取到 {len(metrics_data)} 字节的metrics数据")
        
        print("\n🎉 所有metrics测试通过！")
        print("\n📊 生成的metrics数据预览:")
        print("-" * 50)
        print(metrics_data.decode('utf-8')[:500] + "..." if len(metrics_data) > 500 else metrics_data.decode('utf-8'))
        print("-" * 50)
        
        return True
        
    except Exception as e:
        print(f"❌ Metrics测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_metrics()
    sys.exit(0 if success else 1)
