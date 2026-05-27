#!/bin/bash
# BIM 户型图分割模型 - 服务器部署脚本
# 在 Ubuntu 服务器上执行

set -e

DEPLOY_DIR="/home/ubuntu/.gemini/antigravity/scratch/floorplan_deploy"
MODEL_DIR="$DEPLOY_DIR/models"

echo "=== BIM FloorPlan Segmentation - Server Deploy ==="

# 1. 创建目录
mkdir -p "$MODEL_DIR"
echo "[1/4] 目录创建完成"

# 2. 安装依赖 (仅需这3个包，不需要PyTorch!)
pip install onnxruntime opencv-python-headless numpy
echo "[2/4] 依赖安装完成"

# 3. 复制模型文件 (需要先从本地 scp 上传)
# scp models/M2_DA_best.onnx ubuntu@YOUR_SERVER:/home/ubuntu/.gemini/antigravity/scratch/floorplan_deploy/models/
if [ -f "$MODEL_DIR/M2_DA_best.onnx" ]; then
    echo "[3/4] 模型文件已就位 ($(du -h $MODEL_DIR/M2_DA_best.onnx | cut -f1))"
else
    echo "[3/4] ⚠️  请上传模型文件: scp M2_DA_best.onnx 到 $MODEL_DIR/"
fi

# 4. 验证
python3 -c "
import onnxruntime as ort
sess = ort.InferenceSession('$MODEL_DIR/M2_DA_best.onnx')
print('[4/4] ✅ 模型加载成功, provider:', sess.get_providers()[0])
"

echo ""
echo "=== 部署完成 ==="
echo "在 web_server.py 中添加:"
echo "  from floorplan_onnx import get_segmenter"
echo "  segmenter = get_segmenter('$MODEL_DIR/M2_DA_best.onnx')"
