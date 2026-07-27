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
| `AVATAR_ID` | 自动检测 | 形象 ID，对应 `data/avatars/<ID>/` 目录名（不填则自动检测第一个可用形象） |
| `TTS` | `qwentts` | TTS 引擎，可选 `qwentts`/`edgetts`/`dashscopetts`/`cosyvoice` 等 |
| `REF_FILE` | `Ethan` | TTS 音色，qwentts 可选 `Ethan`(男)/`Cherry`(女) |
| `LLM_PROVIDER` | `sensenova` | LLM 提供商，可选 `sensenova`/`dashscope` |
| `TRANSPORT` | `webrtc` | 传输方式，可选 `webrtc`/`rtmp`/`virtualcam` |
| `SSL_ENABLED` | `True` | 是否启用 HTTPS（远程麦克风必须开启） |
| `LISTEN_PORT` | `5001` | 服务监听端口 |

---

### 3️⃣ 560 二次开发新增/修改的文件说明

#### 新增文件（原版没有）

| 文件 | 功能说明 |
|------|----------|
| `prompt_config.yaml` | 🆕 **企业知识库配置文件**，热更新，修改无需重启 |
| `monitor.sh` | 🆕 **资源监控脚本**，每 5 秒记录 RAM/GPU/CPU/Session/FPS |
| `monitor.py` | 🆕 **实时监控脚本**，查看 FPS/GPU/会话/日志状态 |
| `mel.py` | 🆕 **GPU 加速 Mel 频谱**，降低 CPU 负载 |
| `audio.py` | 🆕 **音频处理**，GPU 加速 |
| `server/` | 🆕 **服务端模块**（routes.py / rtc_manager.py / session_manager.py / webrtc.py） |
| `server/tts_proxy.py` | 🆕 **TTS 代理**，兼容 DashScope OmniTTS |
| `tts/qwentts.py` | 🆕 **QwenTTS 插件**（阿里云通义千问实时语音合成） |
| `tts/dashscopetts.py` | 🆕 **DashScope TTS 插件** |
| `streamout/` | 🆕 **流输出模块**（RTMP / VirtualCam） |
| `ssl/cert.pem` + `ssl/key.pem` | 🆕 **自签名 SSL 证书**（用于 HTTPS 远程访问） |
| `web/mic_test.html` | 🆕 **麦克风测试页面** |
| `.env.example` | 🆕 **环境变量模板**，完整的配置项说明和密钥模板 |

#### 修改文件（改动较大）

| 文件 | 改动内容 |
|------|----------|
| `app.py` | 集成 SSL/CORS/SessionManager、Session 缓存 LRU 淘汰（max=2）、avatar_id 自动检测、资源释放钩子 |
| `config.py` | 新增 SSL/QwenTTS/字幕/传输方式/fps 校验等配置项，avatar_id 自动从 `data/avatars/` 检测 |
| `llm.py` | 支持 `prompt_config.yaml` 热加载，SenseNova/DashScope 双提供商，API 超时与日志脱敏 |
| `avatars/base_avatar.py` | OpenCV 像素级字幕渲染（30ms→1ms）、Queue 全链超时死锁防护、静音帧 fallback、GPU 定期清理、`release_resources()` 资源释放 |
| `avatars/wav2lip_avatar.py` | face_gpu 缓存 maxsize=3 淘汰、`_face_count` 防 None 崩溃 |
| `avatars/audio_features/` | feat_queue / output_queue 有界化、feat 丢失时 pipeline 对齐 |
| `server/routes.py` | API 路由重构，新增 `/api/avatars`、`/api/admin/shutdown` |
| `server/rtc_manager.py` | WebRTC `disconnected` 状态延迟清理、双重 cleanup 防护 |
| `server/session_manager.py` | 单例线程安全、session 占位符失败清理、`release_resources` 集成 |
| `server/webrtc.py` | push_video/audio 超时防护、双线程启动锁、stop 时释放资源 |
| `tts/qwentts.py` | 音频归一化修正（32767）、重连失败重建、模型名对齐 |
| `tts/base_tts.py` | msgqueue maxsize=10 有界化 |
| `tts/edge.py` + `doubao.py` | asyncio 事件循环泄漏修复 |
| `web/index.html` | `<select>` 真下拉动态加载 avatar、音色链路修复（negotiate + sendVoiceText）、远程关闭服务器按钮 |

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

**`https://<服务器IP>:5001/index.html`**（或根据 `LISTEN_PORT` 配置）

---

## 🌟 560 二次开发优化特性

### 1. 运行时稳定性（修复 40+ 项 Bug）
- **死锁防护**：渲染管线全部 Queue 添加超时机制，防止满队列永久阻塞
- **资源释放**：连接断开时自动释放 TTS WebSocket、帧缓存 numpy 数组、渲染线程及队列资源，CPU 和内存实时回落
- **WebRTC 生命周期**：处理 `disconnected` 状态（用户关闭浏览器），15 秒延迟清理 + 双重 cleanup 防护
- **竞态条件修复**：SessionManager 单例线程安全、HumanPlayer 双线程启动锁、缓存命中 `thread.is_alive()` 检测
- **音频对齐**：Mel 特征队列满时同步丢弃 output_queue 保 pipeline 对齐、TTS 重连失败自动重建客户端

### 2. 内存优化
- Session 缓存 LRU 淘汰（max=2），防止多音色切换永久泄漏
- face_gpu 缓存 maxsize=3 淘汰，长期运行不膨胀
- full_imgs 帧数据断开后释放，GPU 定期清理碎片
- TTS msgqueue / ASR output_queue 全部有界化

### 3. 实测性能（2026-07-27 监控数据）

| 阶段 | RAM | GPU 显存 | CPU | fps |
|------|:---:|:-------:|:---:|:---:|
| 启动 warm_up | 369MB→4.1GB | 0.9→1.3GB | 300%→15% | — |
| 待机空闲 | **4.1GB** | **1.3GB** | 5~15% | — |
| 已连接（静默渲染） | 5.0GB | 1.4GB | 50~80% | 25 |
| 对话中（TTS+推理全开） | **8.2GB** | **3.2GB** | 160~220% | 25 |
| 断开后释放 | ↓ 回落 | ↓ 回落 | ↓ <10% | — |

**性能需求**：

| 资源 | 最低下限 | 推荐配置 |
|------|:-------:|:-------:|
| CPU 核心 | 2 核 | 4 核+ |
| RAM 内存 | 8 GB | 12 GB+ |
| GPU 显存 | 4 GB | 6 GB+ |

### 4. 视频渲染优化
- 采用 OpenCV 像素级矩阵运算代替 PIL 进行字幕渲染，单帧合成耗时从 30ms 降低至约 1ms
- 水印文字、字幕字号/Y轴位置可外部配置，水印不设时默认不显示

### 5. 专属企业知识库（热更新）
- 根目录 `prompt_config.yaml` 外部配置文件，内置 560 AI 公司知识体系
- **修改无需重启服务**，下一次提问自动生效

### 6. 本地 ASR 语音识别
- 补齐 FunASR 的 `torchaudio` 硬件驱动依赖，激活本地 SenseVoice 语音识别引擎
- 内网/弱网环境下流畅使用"自由对话"与"按住说话"功能

### 7. 音色链路修复
- 预设男声 **Ethan** 为默认音色，前端 TTS 下拉框 `<select>` 自由切换男女声
- 语音对话、文字输入、WebRTC 协商三个入口全部正确传递音色参数，消除 fallback 到女声的问题

### 8. 运维便利性
- Avatar ID 自动检测（扫描 `data/avatars/`），无需手动配置
- 前端 `<select>` 真下拉从 API 动态加载可用形象，无硬编码
- 前端一键关闭服务器进程按钮（含二次确认）

---

## 核心代码模块

| 模块 | 文件 | 说明 |
|------|------|------|
| 主入口 | `app.py` | 数字人服务启动，SSL/CORS/SessionCache |
| 配置 | `config.py` + `.env` | 命令行参数 + 环境变量 + avatar 自动检测 |
| LLM 引擎 | `llm.py` + `prompt_config.yaml` | 大模型对话 + 企业知识库热加载 |
| 数字人引擎 | `avatars/base_avatar.py` | 推理管线、字幕渲染、资源释放 |
| TTS 语音合成 | `tts/qwentts.py` + `tts/base_tts.py` | 阿里云通义千问实时 TTS |
| WebRTC | `server/webrtc.py` + `server/rtc_manager.py` | 实时音视频推流、连接管理、disconnected 处理 |
| 会话管理 | `server/session_manager.py` | 多会话生命周期、资源回收、缓存联动 |
| API 路由 | `server/routes.py` | HTTP 接口（avatars/human/shutdown 等） |
| 插件系统 | `registry.py` | TTS / Avatar / Output 可插拔注册 |
| 监控 | `monitor.sh` | 实时运行状态查看（每 5 秒） |
| 前端 | `web/index.html` | Web 客户端页面 |
