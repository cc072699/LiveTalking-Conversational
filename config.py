###############################################################################
#  配置解析 — CLI 参数 + YAML 配置
###############################################################################

import argparse
import json
import os


def str_or_int(value):
    """尝试转换为 int，失败则返回 str"""
    try:
        return int(value)
    except ValueError:
        return value


def str2bool(v):
    """将字符串转换为布尔值"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError(f'Boolean value expected, got {v}')


def parse_args():
    """解析命令行参数，支持环境变量作为默认值"""
    parser = argparse.ArgumentParser(description="LiveTalking Digital Human Server")

    # ─── 音频 ──────────────────────────────────────────────────────────
    parser.add_argument('--fps', type=int, default=int(os.environ.get('FPS', 25)),
                        help="video fps, must be 25")
    parser.add_argument('-l', type=int, default=int(os.environ.get('AUDIO_L', 10)))
    parser.add_argument('-m', type=int, default=int(os.environ.get('AUDIO_M', 8)))
    parser.add_argument('-r', type=int, default=int(os.environ.get('AUDIO_R', 10)))

    # ─── 数字人模型 ────────────────────────────────────────────────────
    parser.add_argument('--model', type=str,
                        default=os.environ.get('MODEL', 'wav2lip'),
                        help="avatar model: musetalk/wav2lip/ultralight")
    parser.add_argument('--avatar_id', type=str,
                        default=os.environ.get('AVATAR_ID', 'cc3'),
                        help="avatar id in data/avatars")
    parser.add_argument('--batch_size', type=int,
                        default=int(os.environ.get('BATCH_SIZE', 16)),
                        help="infer batch")
    parser.add_argument('--modelres', type=int,
                        default=int(os.environ.get('MODELRES', 192)))
    parser.add_argument('--modelfile', type=str,
                        default=os.environ.get('MODELFILE', ''))

    # ─── 自定义动作和多形象 ────────────────────────────────────────────
    parser.add_argument('--customvideo_config', type=str,
                        default=os.environ.get('CUSTOMVIDEO_CONFIG', ''),
                        help="custom action json")

    # ─── TTS ───────────────────────────────────────────────────────────
    parser.add_argument('--tts', type=str,
                        default=os.environ.get('TTS', 'qwentts'),
                        help="tts plugin: edgetts/gpt-sovits/cosyvoice/fishtts/tencent/doubao/indextts2/azuretts/qwentts/dashscopetts")
    parser.add_argument('--REF_FILE', type=str,
                        default=os.environ.get('REF_FILE', 'Cherry'),
                        help="参考文件名或语音模型ID (qwentts 音色名, 如 Cherry/Ethan)")
    parser.add_argument('--REF_TEXT', type=str,
                        default=os.environ.get('REF_TEXT', None))
    parser.add_argument('--TTS_SERVER', type=str,
                        default=os.environ.get('TTS_SERVER', 'http://127.0.0.1:9880'))
    parser.add_argument('--tts_speed', type=float,
                        default=float(os.environ.get('TTS_SPEED', 1.0)),
                        help="TTS 语速, 1.0 正常, <1 变慢, >1 变快")

    # ─── Qwen TTS ────────────────────────────────────────────────────
    parser.add_argument('--qwen_tts_model', type=str,
                        default=os.environ.get('QWEN_TTS_MODEL', 'qwen-tts-realtime-latest'),
                        help="Qwen TTS model name (for qwentts plugin)")

    # ─── 字幕 ─────────────────────────────────────────────────────────
    parser.add_argument('--subtitle', type=str2bool,
                        default=str2bool(os.environ.get('SUBTITLE', 'True')),
                        help="是否显示字幕 (默认: True)")
    parser.add_argument('--subtitle_size', type=float,
                        default=float(os.environ.get('SUBTITLE_SIZE', 1.0)),
                        help="字幕字号倍率 (默认: 1.0)")

    # ─── 传输 ─────────────────────────────────────────────────────────
    parser.add_argument('--transport', type=str,
                        default=os.environ.get('TRANSPORT', 'webrtc'),
                        help="output: rtcpush/webrtc/rtmp/virtualcam")
    parser.add_argument('--push_url', type=str,
                        default=os.environ.get('PUSH_URL',
                            'http://localhost:1985/rtc/v1/whip/?app=live&stream=livestream'))
    parser.add_argument('--max_session', type=int,
                        default=int(os.environ.get('MAX_SESSION', 1)))
    parser.add_argument('--listenport', type=int,
                        default=int(os.environ.get('LISTEN_PORT', 8010)),
                        help="web listen port")

    # ─── SSL/HTTPS ────────────────────────────────────────────────────
    parser.add_argument('--ssl', type=str2bool,
                        default=str2bool(os.environ.get('SSL_ENABLED', 'False')),
                        help="启用 HTTPS (让远程用户可使用麦克风)")
    parser.add_argument('--ssl_cert', type=str,
                        default=os.environ.get('SSL_CERT', ''),
                        help="SSL 证书路径 (.pem)")
    parser.add_argument('--ssl_key', type=str,
                        default=os.environ.get('SSL_KEY', ''),
                        help="SSL 私钥路径 (.key)")

    opt = parser.parse_args()

    # ─── 后处理 ────────────────────────────────────────────────────────
    opt.customopt = []
    if opt.customvideo_config:
        with open(opt.customvideo_config, 'r') as f:
            opt.customopt = json.load(f)

    return opt
