#!/bin/bash
# 服务器环境配置脚本
echo "===== RLHF 实验环境配置 ====="
echo "[1/4] 更新 pip..."
pip install --upgrade pip -q

echo "[2/4] 安装核心依赖（含魔塔社区 ModelScope）..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 -q
pip install transformers==4.44.0 datasets peft accelerate bitsandbytes trl -q
pip install modelscope -q      # 魔塔社区下载库

echo "[3/4] 安装辅助工具..."
pip install numpy matplotlib jupyter ipywidgets -q


echo "[4/4] 验证安装..."
python -c "
import torch, transformers, peft, trl
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'PEFT: {peft.__version__}')
print(f'TRL: {trl.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
echo "✅ 环境配置完成！"
