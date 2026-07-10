# LiveTalking-560 实时交互数字人系统

本项目基于开源的 `lipku/LiveTalking` （https://github.com/lipku/LiveTalking.git）实时数字人引擎进行二次开发，由 **五六零人工智能（560 AI）** 团队深度定制与重构，旨在提供极致低延迟、高稳定性、内置企业专属知识库的音视频同步实时交互数字人解决方案。

---

## 🌟 560 二次开发优化特性

相较于原版，本项目在生产环境落地层面进行了多项核心重构与底层 Bug 修复：

1. **视频抖动深度消除**
   - 修复了原推理管线（Pacing）在静音与说话状态下的节拍器缩进 Bug，杜绝无限制帧堆积引发的画面卡顿与抖动。
   - 采用 OpenCV 像素级矩阵运算代替 PIL 进行字幕渲染，大幅度减少单帧合成耗时（从 30ms 降低至 1ms 左右），实现流畅且极稳定的 25fps 视频推流。

2. **专属企业知识库注入＊支持热更新）**
   - 在项目根目录引入 `prompt_config.yaml` 外部配置文件。
   - 内置"五六零人工智能"的公司定位、三大业务方向、核心产品、OPC 创业基地孵化、产学研合作等知识体系。
   - **热更新设计**：修改提示词或新增公司知识，**无需重启服务或修改代码**，下一次提问自动生效。

3. **预设 Ethan（男声）默认音色**
   - 默认发音人改为男声 `Ethan`，并修复了前端音色切换逻辑。

4. **ASR 本地语音识别**
   - 补齐了 FunASR 的 `torchaudio` 硬件驱动依赖，激活本地 SenseVoice 语音识别引擎。

5. **消除内存溢出漏洞**
   - 重构 WebRTC 连接生命周期，连接断开时强制回收渲染线程及视频队列资源，内存常驻稳定在约 8.4 GB。

---

## 部署前必读：需要手动下载的文件

> 以下文件因体积过大，*)章节 Git 版本管理**，部署时需手动下载放置到对应目录。

### 1. 数字人形象文件（`data/avatars/`）

从 Git 克隆后，`data/avatars/` 目录为空，需放置数字人形象文件：

| 形象 ID | 说明 | 获取方式 |
|---------|------|----------|
| `zxw560-1` | 默认数字人形象（560 AI 定制） | **公司服务器下载**（联系运维获取） |
| 其他形象 | 各测试形象 | 同上，或从原版项目下载 |

### 2. 模型权重文件（`models/`）

从 Git 克隆后，`models/` 目录为空，需放置以下模型文件：

| 文件 | 大小 | 说明 | 获取方式 |
|------|:----:|------|----------|
| `models/wav2lip.pth` | 205 MB | Wav2Lip 主模型 | 从原版项目下载 |
| `models/asr/SenseVoiceSmall/model.pth | 893 MB | 语音识别模型（可选，用于本地 ASR） | 安装 funasr 后自动拉取 |

### 3. 下载方式

**方式一：从原版 LiveTalking 仓库下载**
```bash
git clone https://github.com/lipku/LiveTalking.git
```
访问原版项目网盘链接获取模型和形象：
- 夸克云盘：https://pan.quark.cn/s/83a750323ef0
- Google Drive：https://drive.google.com/drive/folders/1FOC_MD6wdogyyX_7V1d4NDIO7P9NlSAJ?usp=sharing

放置说明：
1. 将 `wav2lip256.pth` 拷贝到 `models/` 目录，重命名为 `wav2lip.pth`
2. 将形象包解压后整个文件夹拷贝到 `data/avatars/` 目录

**方式二：从公司服务器下载（内网）**
联系运维团队获取形象包和模型文件的内部下载地址。

### 4. 部署完成后的目录结构

```
data/
├── avatars/
│   └── zxw560-1/          ← 数字人形象包（需手动下载）
│       ├── face_imgs/      ← 面部裁剪图像
│       ├── full_imgs/      ← 完整画面帧
│       └── coords.pkl      ← 坐标映射文件
├── record/                 ← 录制视频（运行时生成）
├── tmp/                    ← 临时文件（运行时生成）
└── cloned_voices/          ← 声音克隆文件（可选）

models/
├── wav2lip.pth             ← 模型权重（需手动下载）
└── asr/                    ← 语音识别模型（可选）
```

---

## 快速开始

### 1. 准备环境

```bash
conda create -n livetalking python=3.12
conda activate livetalking
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

### 2. 下载模型与形象

参照上方 **"部署前必读"** 章节，下载 `models/wav2lip.pth` 和 `data/avatars/zxw560-1/` 放置到对应目录。

### 3. 配置提示词与知识库

`prompt_config.yaml` 可随时编辑修改，调整大模型的人设态度与公司信息。

### 4. 启动服务

```bash
python app.py
```

### 5. 客户端访问

由于默认启用了 SSL，请使用 HTTPS 在浏览器中打开：
**`https://<服务器IP>:8051`**

---

## 核心代码模块

- `app.py`：数字人主服务启动入口
- `llm.py`：大模型对接处理，已集成 `prompt_config.yaml` 动态读取
- `prompt_config.yaml`：大模型角色扮演与公司专属知识库
- `avatars/base_avatar.py`：数字人引擎基类
- `server/session_manager.py`：会话管理 和资源回收
- `web/index.html`:=Web 客户端
