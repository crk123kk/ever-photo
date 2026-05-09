# Level 2 升级方案

基于 MVP 当前状态，整理需要升级的点及对应方案。

---

## 一、紧急：模型与运行时

### 1. PyTorch CUDA 兼容性

**现状**：RTX 5070（Blackwell, sm_120）不被 PyTorch cu126 支持，自动降级到 CPU，推理慢 10-50 倍。

**方案**：

```bash
# 安装支持 Blackwell 的 PyTorch（需 cu128+）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

> 注意：截至 2026-05，PyTorch 官方 cu128 nightly 可能仍不支持 sm_120。
> 关注 https://pytorch.org/get-started/locally 获取最新稳定版。
> 如果 cu128 稳定版仍未支持，可尝试 nightly：
> `pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128`

**验证**：

```bash
python -c "import torch; print(torch.cuda.is_available()); t=torch.zeros(1,device='cuda'); print('CUDA OK')"
```

---

### 2. Real-ESRGAN 权重

**现状**：`RealESRGAN_x2.pth` 未下载成功（网络问题），降级到 `cv2.resize`，放大效果无增强。

**方案**：

- 方案 A：网络畅通时运行 `python download_weights.py`，从 GitHub 下载 `RealESRGAN_x2plus.pth`
- 方案 B：手动下载后放到 `backend/weights/RealESRGAN_x2.pth`
  - 下载地址：https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth
  - 重命名为 `RealESRGAN_x2.pth`
- 方案 C：HuggingFace 镜像下载（需找到兼容架构的权重）

**验证**：重启后端后，日志应显示 `Step 3/3: Super resolution (2x)` 而非 `using cv2.resize`

---

### 3. LaMa 划痕修补权重

**现状**：`big-lama.pt` 下载不完整（GitHub 196MB 文件在当前网络下超时），降级到 OpenCV `inpaint()`，效果一般。

**方案**：

- 方案 A：网络畅通时运行 `python download_weights.py`
- 方案 B：手动下载后放到 `~/.cache/torch/hub/checkpoints/big-lama.pt`
  - 下载地址：https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt
- 方案 C：使用代理下载

**验证**：日志应显示 `LaMa` 而非 `using OpenCV inpainting`

---

## 二、体验优化

### 4. 异步任务队列

**现状**：同步 API，修复期间前端阻塞等待 10-30 秒，超时风险高。

**方案**：

```
POST /api/restore → 返回 task_id
GET /api/status/{task_id} → 返回进度/结果
```

- 使用 Celery + Redis 或简单的 `asyncio.Queue` + 后台线程
- 前端轮询状态，显示实际进度（划痕检测中 → 修补中 → 修脸中 → 放大中）

---

### 5. 前端交互增强

**现状**：简单上传 → 等待 → 展示结果。

**升级点**：

- [ ] 拖拽对比滑块（slider comparison），左右拖动查看修复前后
- [ ] 手动画掩码（canvas 画笔），用户标注需要修补的区域
- [ ] 修复参数面板（GFPGAN 保真度、放大倍数、划痕灵敏度）
- [ ] 批量上传与修复
- [ ] 修复历史记录

---

### 6. 修复效果调优

**现状**：划痕检测阈值固定，可能误检或漏检。

**升级点**：

- [ ] 划痕检测灵敏度可调（`SCRATCH_THRESHOLD`）
- [ ] GFPGAN 保真度可调（`fidelity_weight`：0=最保真 1=最增强）
- [ ] Real-ESRGAN 放大倍数可选（2x/4x）
- [ ] 各步骤可独立开关（只修脸不放大、只放大不修脸等）

---

## 三、架构升级

### 7. CodeFormer 替换 GFPGAN

**现状**：GFPGAN 人脸修复，不可控。

**方案**：引入 CodeFormer，支持 `fidelity_weight` 参数控制保真度。

```python
# CodeFormer 可通过 basicsr 集成
from basicsr.archs.codeformer_arch import CodeFormer
```

- 权重：https://github.com/sczhou/CodeFormer/releases
- 优势：保真度可调，效果更自然

---

### 8. 划痕检测升级

**现状**：OpenCV 形态学（顶帽+黑帽），对细小划痕和不规则损伤效果差。

**方案**：

- 方案 A：使用预训练的划痕检测模型（如 SCRDet、Mask R-CNN）
- 方案 B：使用 U-Net 语义分割做损伤区域检测
- 方案 C：前端手动画掩码 + 自动检测结合

---

### 9. 图片存储与清理

**现状**：上传/输出图片存在本地 `uploads/` / `outputs/`，无清理机制。

**方案**：

- [ ] 定时清理过期文件（如 1 小时后自动删除）
- [ ] 结果图存对象存储（S3/OSS）而非本地磁盘
- [ ] 上传大小限制与格式校验

---

## 四、部署与运维

### 10. Docker 化

**现状**：手动启动，环境依赖复杂。

**方案**：

```dockerfile
# 后端 Dockerfile
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
services:
  backend:
    build: ./backend
    ports: ["8000:8000"]
    volumes: ["./backend/weights:/app/weights"]
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
  frontend:
    build: ./frontend
    ports: ["3000:3000"]
```

---

### 11. 生产部署

**升级点**：

- [ ] Nginx 反向代理 + HTTPS
- [ ] GPU 云服务器部署（推荐 AutoDL / 矩池云，国内网络友好）
- [ ] 模型权重预加载（启动时加载，避免首次请求慢）
- [ ] 健康检查与自动重启
- [ ] 请求限流与认证

---

## 升级优先级

| 优先级 | 项目 | 影响 | 难度 |
|--------|------|------|------|
| P0 | PyTorch CUDA 兼容 | GPU 加速 10-50 倍 | 低 |
| P0 | Real-ESRGAN 权重 | 超分效果质变 | 低 |
| P1 | LaMa 权重 | 划痕修补效果提升 | 低 |
| P1 | 异步任务队列 | 用户体验根本改善 | 中 |
| P1 | 修复参数可调 | 灵活度提升 | 低 |
| P2 | CodeFormer 替换 | 人脸修复更可控 | 中 |
| P2 | 拖拽对比/画掩码 | 交互体验提升 | 中 |
| P2 | Docker 化 | 部署标准化 | 中 |
| P3 | 划痕检测升级 | 检测精度提升 | 高 |
| P3 | 生产部署 | 可对外服务 | 高 |
