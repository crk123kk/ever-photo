# AI 照片修复 MVP

# 一、技术选型

## 前端

- Next.js + React + TypeScript
- TailwindCSS

作用：

- 图片上传（拖拽/点击）
- 展示修复前后对比
- 下载修复结果

---

## AI 服务

- Python + FastAPI + PyTorch (CUDA)

作用：

- 接收上传图片
- 调用 AI 修复流水线
- 返回修复结果

---

## AI 模型

### 1. LaMa（划痕修补）

- 使用 `simple-lama-inpainting` pip 包
- OpenCV 自动划痕检测生成掩码
- 修复老照片划痕和缺损

### 2. GFPGAN（人脸修复）

- 使用 `gfpgan` pip 包
- 模糊/损坏人脸增强

### 3. Real-ESRGAN（超分辨率）

- 使用 `realesrgan` pip 包
- 2x 超分辨率放大

---

# 二、修复流水线

```text
原图
→ OpenCV 划痕检测（生成 mask）
→ LaMa 修补（划痕/缺损）
→ GFPGAN 修脸
→ Real-ESRGAN 放大（2x）
→ 输出结果
```

> 流水线顺序优化：先修结构性损伤，再增强人脸，最后放大。
> 若未检测到划痕，自动跳过 LaMa 步骤。

---

# 三、项目结构

```text
ever-photo/
├── backend/                # FastAPI 服务
│   ├── app/
│   │   ├── main.py         # FastAPI 入口 + CORS
│   │   ├── api/
│   │   │   └── restore.py  # POST /api/restore
│   │   ├── services/
│   │   │   └── pipeline.py # AI 修复流水线
│   │   └── core/
│   │       └── config.py   # 配置（设备/路径/参数）
│   ├── weights/            # 模型权重（自动下载）
│   ├── uploads/            # 上传图片
│   └── outputs/            # 输出图片
├── frontend/               # Next.js 应用
│   └── src/
│       ├── app/
│       │   └── page.tsx    # 主页面（上传/修复/结果）
│       └── components/
│           ├── Upload.tsx  # 图片上传组件
│           └── Result.tsx  # 修复结果展示
├── start.bat               # 一键启动脚本
└── .gitignore
```

---

# 四、API 设计

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/restore` | POST | 上传图片，返回修复后图片（同步，10-30秒） |
| `/api/health` | GET | 健康检查 |

前端通过 Next.js rewrites 代理到 FastAPI（`localhost:8000`）。

---

# 五、启动方式

```bash
# 后端
cd backend && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 120

# 前端
cd frontend && npm run dev
```

或双击 `start.bat` 一键启动。

---

# 六、后续优化方向

- [ ] CodeFormer 替换 GFPGAN（更可控的人脸保真度）
- [ ] 异步任务队列（长时间处理不阻塞）
- [ ] 手动标注修补区域（前端画 mask）
- [ ] 批量修复
- [ ] Docker 部署
