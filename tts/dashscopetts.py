"""
DashScope CosyVoice TTS — 使用阿里云 tts_v2 API

支持：
  - 预设音色: longxiaochun, longhua, longyi 等
  - 克隆音色: 通过 VoiceEnrollmentService 注册的 voice_id

用法:
  python app.py --tts dashscopetts --REF_FILE longxiaochun

环境变量:
  DASHSCOPE_API_KEY — 阿里云 DashScope API Key
"""

import os
import time
import threading
import json
import numpy as np
import resampy

from utils.logger import logger
from .base_tts import BaseTTS, State
from registry import register

# DashScope 输出采样率
SRC_SR = 16000
DST_SR = 16000

# 本地克隆音色元数据路径
META_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         "data", "cloned_voices_meta.json")


def _load_cloned_voices() -> dict:
    """加载本地克隆音色元数据"""
    try:
        if os.path.exists(META_FILE):
            with open(META_FILE, "r", encoding="utf-8") as f:
                meta = json.load(f)
            return meta.get("voices", {})
    except Exception as e:
        logger.warning(f"Failed to load cloned voices meta: {e}")
    return {}


@register("tts", "dashscopetts")
class DashScopeTTS(BaseTTS):
    """DashScope CosyVoice TTS — 基于 tts_v2 SpeechSynthesizer"""

    def __init__(self, opt, parent):
        super().__init__(opt, parent)

        self.api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        if not self.api_key:
            logger.error("DashScopeTTS: DASHSCOPE_API_KEY 未设置")

        # 音色：可以是预设名（longxiaochun）或克隆音色的 voice_id
        self.voice = opt.REF_FILE if opt.REF_FILE else "longxiaochun"
        # 预设音色列表，用于兜底判断
        self._preset_voices = {"longxiaochun", "longhua", "longyi", "longtong",
                               "longwan", "longrui", "longgang", "shanxuan",
                               "shanshan", "shuiyuan", "zhiyi", "aiqi"}
        self._fallback_voice = "longxiaochun"

        # 克隆音色映射表
        self.cloned_voices = _load_cloned_voices()

        # 默认语速，1.0 正常，<1 变慢，>1 变快
        self.speed = getattr(opt, 'tts_speed', None) or 1.0
        if isinstance(self.speed, str):
            try:
                self.speed = float(self.speed)
            except ValueError:
                self.speed = 1.0

        logger.info(f"DashScopeTTS init: voice={self.voice}, speed={self.speed}")

    def _resolve_voice(self, ref_file: str) -> str:
        """解析音色名称 → 实际的 voice_id"""
        if not ref_file:
            logger.warning(f"_resolve_voice: empty ref_file, fallback to '{self._fallback_voice}'")
            return self._fallback_voice

        # 检查是否是克隆音色名称
        if ref_file in self.cloned_voices:
            voice_id = self.cloned_voices[ref_file].get("voice_id", "")
            if voice_id:
                logger.info(f"Using cloned voice '{ref_file}' → voice_id='{voice_id}'")
                return voice_id
            else:
                logger.warning(f"Cloned voice '{ref_file}' has no voice_id (OSS not configured), "
                              f"falling back to preset '{self._fallback_voice}'")
                return self._fallback_voice

        logger.info(f"Using voice: '{ref_file}'")
        return ref_file

    def txt_to_audio(self, msg: tuple[str, dict]):
        text, textevent = msg
        t_start = time.perf_counter()

        # 获取本次请求的音色
        ref_file = textevent.get("tts", {}).get("ref_file", self.voice)
        voice = self._resolve_voice(ref_file)

        speed = float(textevent.get("tts", {}).get("speed", self.speed))

        try:
            # 动态导入 DashScope SDK
            import dashscope
            dashscope.api_key = self.api_key
            from dashscope.audio.tts_v2 import SpeechSynthesizer, AudioFormat

            logger.info(f"DashScopeTTS synthesizing: voice={voice}, speed={speed}, text='{text[:60]}...'")

            synthesizer = SpeechSynthesizer(
                model="cosyvoice-v1",
                voice=voice,
                format=AudioFormat.PCM_16000HZ_MONO_16BIT,
                speech_rate=speed,
            )
            audio_data = synthesizer.call(text)

            if not audio_data:
                logger.error("DashScopeTTS: empty audio returned")
                return

            # 已经返回 16kHz PCM 16-bit mono
            stream = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32767.0

            # 按 chunk (20ms) 切分推送到音频流水线
            total_len = stream.shape[0]
            idx = 0
            first = True

            while total_len - idx >= self.chunk and self.state == State.RUNNING:
                eventpoint = {}
                if first:
                    eventpoint = {"status": "start", "text": text}
                    first = False
                eventpoint.update(**textevent)
                self.parent.put_audio_frame(
                    stream[idx: idx + self.chunk], eventpoint
                )
                idx += self.chunk

            # 发送 end 事件
            eventpoint = {"status": "end", "text": text}
            eventpoint.update(**textevent)
            self.parent.put_audio_frame(
                np.zeros(self.chunk, dtype=np.float32), eventpoint
            )

            elapsed = time.perf_counter() - t_start
            audio_sec = len(audio_data) / 32000  # 16kHz * 2 bytes
            logger.info(f"DashScopeTTS done: {elapsed:.2f}s, audio={audio_sec:.1f}s")

        except Exception as e:
            logger.exception(f"DashScopeTTS synthesis error: {e}")

    def stop_tts(self):
        pass