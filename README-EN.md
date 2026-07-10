# 560 AI Digital Human Interactive Engine

Real-time interactive streaming digital human engine based on LiveTalking, with audio-video synchronized dialogue.

---

## Overview

This repository is a secondary development of [LiveTalking](https://github.com/lipku/LiveTalking) by **560 AI**, integrated with internal business scenarios and continuously iterated.

## Quick Start

### Requirements

- Ubuntu 22.04+ / CentOS 7+
- Python 3.10+
- CUDA 11.8+ (recommended 12.x)
- NVIDIA GPU (RTX 3060+)

### Installation

```bash
conda create -n livetalking python=3.12
conda activate livetalking
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
# Edit .env with your API keys
```

### Run

```bash
python app.py --transport webrtc --model wav2lip --avatar_id zxw560-1
```

Open `http://<serverip>:8010/index.html` in browser.

## License

Apache 2.0
