# Floorplan Segmentation Project Status

**当前状态 (2026年4月11日) - M2 训练完成, M3 训练中**

## 已完成
1. 诊断出原 Ground Truth (SVG -> Mask) 生成粗糙，限制了 mIoU 上限。
2. 移植了官方 CubiCasa5K 精确解析代码 (`precompute_masks.py`)。
3. 把 5000 张图的精确 Mask 全部预计算完毕，无报错。
4. 编写了统一训练脚本 `train_all.py`，支持 `m1, m2, m3` 及自动 checkpoint 恢复。
5. 磁盘空间清理完成，C盘剩余 >25GB。
6. **M1 (LightUNet) 训练完成**: best mIoU = 0.685 (Epoch 32/57)
7. **训练稳定性修复 (重要!)**:
   - 发现 AMP (FP16) 在 batch_size=2 + 512x512 下反复导致梯度爆炸 → **禁用 AMP，改用 FP32**
   - Focal Loss 的 gamma 指数运算导致不稳定 → **改用标准 CE + Dice**
   - 降低学习率: M2 LR=3e-4/3e-5, M3 LR=3e-4/2e-5
   - 降低数据增强强度（ElasticTransform, GaussNoise 等）
   - 增加 checkpoint 保存频率 (每 5 epoch)
   - 增加训练崩溃检测和自动恢复机制
8. **M2 (UNet+ResNet34) 训练完成**: 
   - 完整跑完 100 Epoch，无 early stop。
   - best mIoU = **0.7861** (Epoch 78)
   - 最终 IoU 分项: background=0.970 | wall=0.739 | window=0.765 | door=0.661
   - 耗时约 675.5 分钟 (11个多小时)。

## 正在进行
- **M3 (DeepLabV3+ + EfficientNet-B4) 训练中**

## 下一步工作（下次启动时执行）
1. 监控 M3 训练：
   - 查看 `m3_train.log` 和 `output_paper/M3_DeepLabV3p_EffB4_history.json`

3. 对比所有模型：
```bash
python train_all.py --compare
```

## 注意事项
- AMP 已永久禁用（`use_amp = False`），FP32 训练更慢但完全稳定
- 每个模型训练约 6-10 小时（RTX 4050 6GB）
- M1 的历史结果使用了旧版 loss函数（Focal + Dice），与 M2/M3 的 CE+Dice 不同
