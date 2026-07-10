# LiveTalking-560 实时交互数字人系统

本项目基于开源的 [lipku/LiveTalking](https://github.com/lipku/LiveTalking) 实时数字人引擎进行二次开发，由 **五六零人工智能（560 AI）** 团队深度定制与重构，旨在提供极致低延迟、高稳定性、内置企业专属知识库的音视频同步实时交互数字人解决方案。

---

## 📥 部署前必读：需要手动下载和配置的文件

> 本仓库仅包含**源代码**（约 5 MB），以下文件因体积过大或涉及密钥安全，**未纳入 Git 版本管理**，部署时需按以下说明操作。

### 1️⃣ 需要手动下载的文件

#### 数字人形象包 — `data/avatars/`

| 形象 ID | 大小 | 说明 | 获取方式 |
|---------|:----:|------|----------|
| `zxw560-1` | 777 MB | 默认数字人形象（560 AI 定制） | **公司服务器下载**（联系运维） |
| 其他形象 | 300~1000 MB | 各测试/备用形象 | 同上，或从原版下载 |

#### 模型权重文件 — `models/`

| 文件 | 大小 | 说明 | 是否必须 | 获取方式 |
|------|:----:|------|:--------:|----------|
| `models/wav2lip.pth` | 205 MB | Wav2Lip 主模型 | **必须** | 从原版项目下载 |
| `models/asr/SenseVoiceSmall/model.pt` | 893 MB | 本地语音识别模型（ASR） | 可选 | 安装 funasr 后自动拉取 |
| `models/asr/speech_fsmn_vad_zh-cn-16k-common-pytorch/model.pt` | 1.7 MB | 语音活动检测（VAD） | 可选 | 安装 funasr 后自动拉取 |

#### 下载方式

**方式一：从原版 LiveTalking 仓库下载**
```bash
git clone https://github.com/lipku/LiveTalking.git
```
访问原版项目网盘获取模型和形象：
- 夸克云盘：https://pan.quark.cn/s/83a750323ef0
- Google Drive：https://drive.google.com/drive/folders/1FOC_MD6wdogyyX_7V1d4NDIO7P9NlSAJ?usp=sharing

放置说明：
1. 将 `wav2lip256.pth` 拷贝到 `models/` 目录，重命名为 `wav2lip.pth`
2. 将形象包解压后整个文件夹拷贝到 `data/avatars/` 目录

**方式二：从公司服务器下载（内网）**
联系运维团队获取形象包和模型文件的内部下载地址。

---

### 2️⃣ 需要自行配置的密钥（API Keys）

克隆仓库后，需要复制环境变量模板并填入真实 API 密钥：

```bash
cp .env.example .env
# 然后编辑 .env 文件
```

#### 最少需要配置的密钥（按需填写）

| 配置项 | 用途 | 获取方式 | 是否必须 |
|--------|------|----------|:--------:|
| `DASHSCOPE_API_KEY` | QwenTTS 语音合成 + DashScope LLM | 阿里云模型服务灵积 | **必须** |
| `SENSNOVA_API_KEY` | SenseNova LLM（如选用） | 商汤科技日日新 | 可选 |
| `TENCENT_APPID/SECRET_ID/SECRET_KEY` | 腾讯云 TTS | 腾讯云控制台 | 可选 |
| `DOUBAO_APPID/DOUBAO_TOKEN` | 豆包 TTS | 火山引擎 | 可选 |
| `AZURE_SPEECH_KEY/AZURE_TTS_REGION` | Azure TTS | Azure 门户 | 可选 |

#### 关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MODEL` | `wav2lip` | 数字人模型，可选 `wav2lip`/`musetalk`/`ultralight` |
| `AVATAR_ID` | `cc3` | 形象 ID，对应 `data/avatars/<ID>/` 目录名 |
| `TTS` | `qwentts` | TTS 引擎，可选 `qwentts`/`edgetts`/`dashscopetts`/`cosyvoice` 等 |
| `REF_FILE` | `Cherry` | TTS 音色，qwentts 可选 `Cherry`(女)/`Ethan`(男) |
| `LLM_PROVIDER` | `sensenova` | LLM 提供商，可选 `sensenova`/`dashscope` |
| `TRANSPORT` | `webrtc` | 传输方式，可选 `webrtc`/`rtmp`/`virtualcam` |
| `SSL_ENABLED` | `False` | 是否启用 HTTPS（远程麦克风需要） |
| `LISTEN_PORT` | `8010` | 服务监听端口 |

---

### 3️⃣ 560 二次开发新增/修改的文件说明

#### 新增文件（原版没有）

| 文件 | 功能说明 |
|------|----------|
| `prompt_config.yaml` | 🆕 **企业知识库配置文件**，热更新，修改无需重启 |
| `monitor.py` | 🆕 **实时监控脚本**，查看 FPS/GPU/会话/日志状态 |
| `mel.py` | 🆕 **GPU 加速 Mel 频谱**，降低 CPU 负载 |
| `audio.py` | 🆕 **音频处理**，GPU 加速 |
| `routes.py` | 🆕 **统一 API 路由**，自定义异常处理 |
| `rtc_manager.py` | 🆕 **WebRTC 连接管理器** |
| `session_manager.py` | 🆕 **全局会话管理器**，单例模式 |
| `server/tts_proxy.py` | 🆕 **TTS 代理**，兼容 DashScope OmniTTS |
| `tts/dashscopetts.py` | 🆕 **DashScope TTS 插件** |
| `wav2lip_avatar.py` | 🆕 **Wav2Lip 形象包装器** |
| `webrtc.py` | 🆕 **WebRTC 实现** |
| `ssl/cert.pem` + `ssl/key.pem` | 🆕 **自签名 SSL 证书**（用于 HTTPS） |
| `web/mic_test.html` | 🆕 **麦克风测试页面** |

#### 修改文件（改动较大）

| 文件 | 改动内容 |
|------|----------|
| `app.py` | 集成 SSL、CORS、SessionManager、更多路由注册 |
| `config.py` | 新增 SSL/QwenTTS/字幕/传输方式等配置项 |
| `llm.py` | 支持 `prompt_config.yaml` 热加载，支持 SenseNova/DashScope 双提供商 |
| `base_avatar.py` | **OpenCV 像素级字幕渲染**（替代 PIL，30ms→1ms），修复静音/说话节拍器 Bug，消除视频抖动 |
| `server/` | 路由重构，加入 ASR 端点、TTS 代理、会话管理 |
| `web/index.html` | 修复音色切换逻辑，优化 UI 布局 |
| `requirements.txt` | 启用 `funasr`/`modelscope`（本地 ASR）和 `dashscope`（QwenTTS） |
| `.env.example` | 完整的配置项说明和密钥模板 |

#### 删除的文件（相比原版）

| 文件 | 说明 |
|------|------|
| `Dockerfile` | 容器化部署暂未使用 |
| `config.yaml` | 配置已整合到 `config.py` + `.env` |
| `docs/virtualcam_guide.md` | 虚拟摄像头指南（未使用） |
| `web/virtualcam.html` | 虚拟摄像头页面（未使用） |

---

### 4️⃣ 部署完成后的目录结构

```
LiveTalking实时交互数字人/
├── app.py / config.py ...          ← 源码（已纳入 Git）
├── .env                            ← 密钥配置（自行创建，已 .gitignore）
├── prompt_config.yaml              ← 知识库配置（可自定义）
├── ssl/
│   ├── cert.pem                    ← SSL 证书（自行生成或替换）
│   └── key.pem                     ← SSL 私钥
├── data/
│   ├── avatars/
│   │   └── zxw560-1/               ← 形象包（需手动下载）
│   │       ├── face_imgs/          ← 面部裁剪图像
│   │       ├── full_imgs/          ← 完整画面帧
│   │       └── coords.pkl          ← 坐标映射文件
│   ├── record/                     ← 录制视频（运行时生成）
│   ├── tmp/                        ← 临时文件（运行时生成）
│   └── cloned_voices/              ← 声音克隆（可选）
├── models/
│   ├── wav2lip.pth                 ← 模型权重（需手动下载）
│   └── asr/                        ← ASR 模型（可选）
└── venv/                           ← Python 环境（自行创建）
```

---

## 🚦 快速开始

### 1. 准备 Python 环境

```bash
conda create -n livetalking python=3.12
conda activate livetalking
# 根据 CUDA 版本选择 PyTorch（运行 nvidia-smi 确认版本）
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

### 2. 下载模型与形象

参照上方 **"需要手动下载的文件"** 章节，将 `wav2lip.pth` 放入 `models/`，将 `zxw560-1/` 放入 `data/avatars/`。

### 3. 配置密钥

```bash
cp .env.example .env
# 编辑 .env，至少填入 DASHSCOPE_API_KEY（用于 QwenTTS）
```

### 4. 生成 SSL 证书（可选，但默认启用 HTTPS）

```bash
cd ssl
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes -subj "/CN=localhost"
```

### 5. 启动服务

```bash
python app.py
```

### 6. 客户端访问

**`https://<服务器IP>:8051`**（或根据 `LISTEN_PORT` 配置）

---

## 🌟 560 二次开发优化特性

### 1. 视频抖动深度消除
- 修复了原推理管线（Pacing）在静音与说话状态下的节拍器缩进 Bug，杜绝无限制帧堆积引发的画面卡顿与抖动。
- 采用 OpenCV 像素级矩阵运算代替 PIL 进行字幕渲染，单帧合成耗时从 30ms 降低至约 1ms，实现稳定 25fps 推流。

### 2. 专属企业知识库注入（热更新）
- 根目录 `prompt_config.yaml` 外部配置文件，内置 560 AI 公司知识体系。
- **修改无需重启服务**，下一次提问自动生效。

### 3. 本地 ASR 语音识别
- 补齐 FunASR 的 `torchaudio` 硬件驱动依赖，激活本地 SenseVoice 语音识别引擎。
- 内网/弱网环境下流畅使用"自由对话"与"按住说话"功能。

### 4. 内存泄漏修复
- 重构 WebRTC 连接生命周期，连接断开时强制回收渲染线程及视频队列资源。
- 内存常驻占用稳定在约 8.4 GB，无持续增长。

### 5. 预设男声 Ethan 默认音色
- 修复前端音色切换逻辑，自由切换男女声。

---

## 核心代码模块

| 模块 | 文件 | 说明 |
|------|------|------|
| 主入口 | `app.py` | 数字人服务启动，SSL/CORS/路由注册 |
| 配置 | `config.py` + `.env` | 命令行参数 + 环境变量 |
| LLM 引擎 | `llm.py` + `prompt_config.yaml` | 大模型对话 + 企业知识库热加载 |
| 数字人引擎 | `avatars/base_avatar.py` | 推理管线、字幕渲染、音视频同步 |
| WebRTC | `webrtc.py` + `rtc_manager.py` | 实时音视频推流、连接管理 |
| 会话管理 | `session_manager.py` | 多会话生命周期、资源回收 |
| API 路由 | `routes.py` | HTTP 接口（human/humanaudio/record 等） |
| 插件系统 | `registry.py` | TTS / Avatar / Output 可插拔注册 |
| 监控 | `monitor.py` | 实时运行状态查看 |
| 前端 | `web/index.html` | Web 客户端页面 |
