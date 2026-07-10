import os
import base64
import time
import threading
import numpy as np
import resampy

from utils.logger import logger
from .base_tts import BaseTTS, State
from registry import register

try:
    import dashscope
    from dashscope.audio.qwen_tts_realtime import (
        QwenTtsRealtime,
        QwenTtsRealtimeCallback,
        AudioFormat,
    )
except ImportError:
    logger.error("QwenTTS 需要安装 dashscope SDK: pip install dashscope>=1.25.11")
    raise


SRC_SR = 24000   # Qwen TTS 只支持 24kHz 输出
DST_SR = 16000   # 项目标准采样率
TTS_RESPONSE_TIMEOUT = 10  # 等待 TTS 响应的超时秒数（原来 60s 会阻塞整条管线）


@register("tts", "qwentts")
class QwenTTS(BaseTTS):
    """
    阿里云通义千问实时语音合成 (Qwen TTS Realtime)
    基于 DashScope Python SDK (dashscope >= 1.25.11)
    使用 commit 模式：只建立一次 WebSocket 连接，每次合成通过 append_text + commit 触发。

    需要设置环境变量 DASHSCOPE_API_KEY。
    用法:
        python app.py --tts qwentts --REF_FILE Cherry
    其中 REF_FILE 用作音色名称 (voice)，如 Cherry / Ethan 等系统音色。
    """

    def __init__(self, opt, parent):
        super().__init__(opt, parent)

        # 音色名, 复用 REF_FILE 参数
        self.voice = opt.REF_FILE if opt.REF_FILE else 'Cherry'
        # 默认语速, 1.0 为正常速度, <1 变慢, >1 变快
        self.speech_rate = getattr(opt, 'tts_speed', None) or 1.0
        if isinstance(self.speech_rate, str):
            try:
                self.speech_rate = float(self.speech_rate)
            except ValueError:
                self.speech_rate = 1.0
        # 模型名
        self.model = getattr(opt, 'qwen_tts_model', 'qwen3-tts-flash-realtime')
        self._default_model = self.model
        # WebSocket URL
        self.ws_url = getattr(opt, 'qwen_tts_url',
                              'wss://dashscope.aliyuncs.com/api-ws/v1/realtime')

        # 设置 DashScope API Key
        api_key = getattr(opt, 'dashscope_api_key', None) or os.environ.get('DASHSCOPE_API_KEY')
        if api_key:
            dashscope.api_key = api_key
        else:
            logger.warning("QwenTTS: DASHSCOPE_API_KEY 未设置，请设置环境变量或通过参数传入")

        # ---------- 内部状态 ----------
        self._remainder = np.array([], dtype=np.float32)  # 上次重采样后不足一 chunk 的 16kHz 样本
        self._response_event = threading.Event()
        self._first_chunk = True          # 当前合成的一句话里的第一个音频包
        self._current_text = ''
        self._current_textevent = {}
        self._ws_connected = False        # WebSocket 连接状态

        # ---------- 建立唯一连接 ----------
        self._build_client()

    class _Callback(QwenTtsRealtimeCallback):
        def __init__(self, tts_ref):
            super().__init__()
            self._ref = tts_ref

        def on_open(self) -> None:
            self._ref._ws_connected = True
            logger.info("QwenTTS WebSocket 连接已建立")

        def on_close(self, close_status_code, close_msg) -> None:
            self._ref._ws_connected = False
            logger.info(f"QwenTTS WebSocket 关闭: code={close_status_code}, msg={close_msg}")
            self._ref._response_event.set()

        def on_error(self, error) -> None:
            """SDK 回调：WebSocket 错误（含 Idle timeout 等服务端断连）"""
            self._ref._ws_connected = False
            logger.warning(f"QwenTTS WebSocket 错误: {error}")
            self._ref._response_event.set()

        def on_event(self, response: dict) -> None:
            try:
                event_type = response.get('type', '')
                if event_type == 'session.created':
                    logger.info(f"QwenTTS session: {response.get('session', {}).get('id', '')}")
                elif event_type == 'response.audio.delta':
                    audio_b64 = response.get('delta', '')
                    if audio_b64:
                        pcm_data = base64.b64decode(audio_b64)
                        self._ref._on_audio_data(pcm_data)
                elif event_type == 'response.done':
                    logger.info("QwenTTS response done")
                    self._ref._flush_remainder()
                    self._ref._response_event.set()
                elif event_type == 'error':
                    logger.error(f"QwenTTS 错误: {response}")
                    self._ref._response_event.set()
            except Exception as e:
                logger.exception(f"QwenTTS 回调处理异常: {e}")

    def _build_client(self):
        """初始化 / 重建 QwenTtsRealtime 客户端（使用 self.model / self.voice / self.speech_rate）"""
        self._remainder = np.array([], dtype=np.float32)
        self._callback = self._Callback(self)
        try:
            self._tts_client = QwenTtsRealtime(
                model=self.model,
                callback=self._callback,
                url=self.ws_url,
            )
            self._tts_client.connect()
            self._tts_client.update_session(
                voice=self.voice,
                response_format=AudioFormat.PCM_24000HZ_MONO_16BIT,
                sample_rate=24000,
                mode='commit',
                speech_rate=self.speech_rate,
            )
            logger.info(f"QwenTTS 初始化完成: model={self.model}, voice={self.voice}, speech_rate={self.speech_rate}")
        except Exception as e:
            logger.error(f"QwenTTS 连接建立失败: {e}")
            self._ws_connected = False

    def _rebuild_client(self, model: str = None, voice: str = None, speed: float = None):
        """关闭旧连接，用新参数重建 QwenTtsRealtime 客户端"""
        try:
            self._tts_client.close()
        except Exception:
            pass
        if model:
            self.model = model
        if voice:
            self.voice = voice
        if speed is not None:
            self.speech_rate = speed
        self._remainder = np.array([], dtype=np.float32)
        self._callback = self.__class__._Callback(self)
        self._tts_client = QwenTtsRealtime(
            model=self.model,
            callback=self._callback,
            url=self.ws_url,
        )
        self._tts_client.connect()
        self._tts_client.update_session(
            voice=self.voice,
            response_format=AudioFormat.PCM_24000HZ_MONO_16BIT,
            sample_rate=24000,
            mode='commit',
            speech_rate=self.speech_rate,
        )
        logger.info(f"QwenTTS client rebuilt: model={self.model}, voice={self.voice}, speech_rate={self.speech_rate}")

    # ========================== 核心方法 ==========================

    def txt_to_audio(self, msg: tuple[str, dict]):
        text, textevent = msg
        t_start = time.perf_counter()

        ref_file = textevent.get('tts', {}).get('ref_file', self.opt.REF_FILE)

        # 重置状态
        self._remainder = np.array([], dtype=np.float32)
        self._first_chunk = True
        self._current_text = text
        self._current_textevent = textevent
        self._response_event.clear()

        try:
            speed = float(textevent.get('tts', {}).get('speed', self.speech_rate))
            new_model = textevent.get('tts', {}).get('model', self._default_model)

            need_reconnect = (ref_file != self.voice) or (speed != self.speech_rate) or (new_model != self.model)
            if need_reconnect:
                need_new_client = new_model != self.model
                self.voice = ref_file
                self.speech_rate = speed
                if need_new_client:
                    self.model = new_model
                    self._build_client()
                    logger.info(f"QwenTTS model changed, rebuilt client: model={self.model}")
                else:
                    self._tts_client.close()
                    self._tts_client.connect()
                    self._tts_client.update_session(
                        voice=self.voice,
                        response_format=AudioFormat.PCM_24000HZ_MONO_16BIT,
                        sample_rate=24000,
                        mode='commit',
                        speech_rate=self.speech_rate,
                    )
                    logger.info(f"QwenTTS reconnected: voice={self.voice}, speech_rate={self.speech_rate}")
            elif not self._ws_connected:
                # 连接已断开（Idle timeout / 服务端关闭），自动重建
                self._build_client()
                logger.info("QwenTTS auto-reconnected after server close")

            # 发送文本，失败时重连重试一次
            try:
                self._tts_client.append_text(text)
                self._tts_client.commit()
            except Exception as send_err:
                logger.warning(f"QwenTTS 发送失败({send_err})，重连重试")
                self._ws_connected = False
                self._build_client()
                self._response_event.clear()
                self._tts_client.append_text(text)
                self._tts_client.commit()

            # 等待 response.done（音频在回调中流式处理）
            # 使用较短超时避免阻塞整条管线；超时后尝试重连
            if not self._response_event.wait(timeout=TTS_RESPONSE_TIMEOUT):
                logger.warning(f"QwenTTS 响应超时({TTS_RESPONSE_TIMEOUT}s)，尝试重连")
                self._ws_connected = False
                try:
                    self._build_client()
                except Exception as rebuild_err:
                    logger.error(f"QwenTTS 重连失败: {rebuild_err}")

            t_end = time.perf_counter()
            logger.info(f"QwenTTS 合成完成，耗时: {t_end - t_start:.2f}s")

        except Exception as e:
            logger.exception(f"QwenTTS txt_to_audio 异常: {e}")
            # 异常后标记连接断开，下次自动重连
            self._ws_connected = False

    # ========================== 流式音频处理（回调中调用）==========================

    def _on_audio_data(self, pcm_data: bytes):
        """收到 PCM 24kHz 16bit mono 音频，一次性 resample 到 16kHz 后分块推送"""
        if self.state != State.RUNNING:
            self._remainder = np.array([], dtype=np.float32)
            return

        # 整段 24kHz PCM -> float32 -> 一次性 resample 到 16kHz
        samples_24k = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
        samples_16k = resampy.resample(x=samples_24k, sr_orig=SRC_SR, sr_new=DST_SR)

        # 拼接上次剩余
        if self._remainder.shape[0] > 0:
            samples_16k = np.concatenate([self._remainder, samples_16k])

        # 按 self.chunk (320 samples = 20ms @16kHz) 分块推送
        idx = 0
        total = samples_16k.shape[0]
        while total - idx >= self.chunk and self.state == State.RUNNING:
            frame = samples_16k[idx:idx + self.chunk]
            eventpoint = {}
            if self._first_chunk:
                eventpoint = {'status': 'start', 'text': self._current_text}
                self._first_chunk = False
            eventpoint.update(**self._current_textevent)
            self.parent.put_audio_frame(frame, eventpoint)
            idx += self.chunk

        # 不足一 chunk 的留到下次
        self._remainder = samples_16k[idx:] if idx < total else np.array([], dtype=np.float32)

    def _flush_remainder(self):
        """合成完毕，推送剩余样本并发送 end 事件"""
        if self.state != State.RUNNING:
            self._remainder = np.array([], dtype=np.float32)
            return

        # 推送剩余完整 chunk
        if self._remainder.shape[0] >= self.chunk:
            idx = 0
            total = self._remainder.shape[0]
            while total - idx >= self.chunk and self.state == State.RUNNING:
                frame = self._remainder[idx:idx + self.chunk]
                eventpoint = {}
                if self._first_chunk:
                    eventpoint = {'status': 'start', 'text': self._current_text}
                    self._first_chunk = False
                eventpoint.update(**self._current_textevent)
                self.parent.put_audio_frame(frame, eventpoint)
                idx += self.chunk

        self._remainder = np.array([], dtype=np.float32)

        # 发送 end 事件
        eventpoint = {'status': 'end', 'text': self._current_text}
        eventpoint.update(**self._current_textevent)
        self.parent.put_audio_frame(np.zeros(self.chunk, np.float32), eventpoint)

    def stop_tts(self):
        self._tts_client.close()
        logger.info("QwenTTS 已关闭")