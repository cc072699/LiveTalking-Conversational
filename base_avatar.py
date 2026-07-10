###############################################################################
#  Copyright (C) 2024 LiveTalking@lipku https://github.com/lipku/LiveTalking
#  email: lipku@foxmail.com
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  
#       http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
###############################################################################
#
#  Avatar 基类 — 合并自 basereal.py，集成到 Async Pipeline
#

import math
from numpy.typing import NDArray
import torch
import numpy as np
import subprocess
import os
import time
import cv2
import glob
import resampy
import queue
from queue import Queue
from threading import Thread, Event
from io import BytesIO
import soundfile as sf
import asyncio
from enum import Enum
import json
import importlib
import registry

import torch.multiprocessing as mp
from dataclasses import dataclass, field

from av import AudioFrame, VideoFrame
from fractions import Fraction

from utils.logger import logger
from utils.image import read_imgs,mirror_index

# ── 中文字幕渲染 ──
from PIL import Image, ImageDraw, ImageFont

# 在 macOS 上查找可用的中文字体
_SUBTITLE_FONT_PATH = None
_CANDIDATE_FONTS = [
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/STSong.ttf",
    "/System/Library/Fonts/Apple Symbols.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttf",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansSC-Regular.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
]
for _f in _CANDIDATE_FONTS:
    if os.path.exists(_f):
        _SUBTITLE_FONT_PATH = _f
        break
if not _SUBTITLE_FONT_PATH:
    try:
        _SUBTITLE_FONT_PATH = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
        if not os.path.exists(_SUBTITLE_FONT_PATH):
            _SUBTITLE_FONT_PATH = None
    except Exception:
        _SUBTITLE_FONT_PATH = None

import re
import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore', SyntaxWarning)
    _SUBTITLE_MD_CLEANER = re.compile(r'[*#_~`>]+|\[([^\]]*)\]\([^)]*\)|!\[([^\]]*)\]\([^)]*\)')


def _clean_subtitle_text(text: str) -> str:
    """去掉 markdown 标记和多余空白"""
    t = _SUBTITLE_MD_CLEANER.sub('', text)
    t = t.replace('\n', ' ').replace('\r', '')
    t = re.sub(r'\s+', ' ', t).strip()
    return t

# class State(Enum):
#     INIT=0
#     WAIT=1
#     QUESTION=2
#     ANSWER=3

@dataclass
class AudioFrameData:
    data: NDArray[np.float32]
    type: int = 0  # 默认值
    userdata: dict = field(default_factory=dict)

class BaseAvatar:
    def __init__(self, opt):
        self.opt = opt
        self.sample_rate = 16000
        self.chunk = self.sample_rate // (opt.fps*2) # 320 samples per chunk (20ms)
        self.sessionid = self.opt.sessionid

        self.speaking = False
        self.recording = False
        self._record_video_pipe = None
        self._record_audio_pipe = None
        self.width = self.height = 0

        self.custom_audiotype = 0 # 0: normal, 1: sinlence, >1: custom audio
        self.custom_img_cycle = {}
        self.custom_audio_cycle = {}
        self.custom_audio_index = {}
        self.custom_index = {}
        # self.custom_opt = {}
        self.__loadcustom()

        # ── 字幕配置 ──
        self._subtitle_enabled = getattr(opt, 'subtitle', True)
        self._subtitle_size = getattr(opt, 'subtitle_size', 1.0)
        self._current_subtitle = ""
        self._last_rendered_subtitle = ""  # 缓存上次渲染的字幕文本
        self._cached_subtitle_frame = None  # 缓存字幕叠加层（避免 PIL 每帧重绘）
        self._subtitle_font = None  # 缓存字体对象
        self._subtitle_font_size = 0

        self.batch_size = opt.batch_size
        self.res_frame_queue = Queue(self.batch_size*2)
        self.render_event = Event()

        _tts_modules = {
            'edgetts': 'tts.edge',
            'gpt-sovits': 'tts.sovits',
            'xtts': 'tts.xtts',
            'cosyvoice': 'tts.cosyvoice',
            'fishtts': 'tts.fish',
            'tencent': 'tts.tencent',
            'doubao': 'tts.doubao',
            'indextts2': 'tts.indextts2',
            'azuretts': 'tts.azure',
            'qwentts': 'tts.qwentts',
            'omnitts': 'tts.omnitts',
            'dashscopetts': 'tts.dashscopetts'
        }

        if opt.tts in _tts_modules:
            importlib.import_module(_tts_modules[opt.tts])
            self.tts = registry.create("tts", opt.tts, opt=opt, parent=self)
        else:
            logger.error(f"TTS module {opt.tts} not found.")

        _output_modules = {
            'webrtc': 'streamout.webrtc',
            'rtcpush': 'streamout.webrtc',
            'rtmp': 'streamout.rtmp',
            'virtualcam': 'streamout.virtualcam'
        }

        # 初始化 Output 模块
        if opt.transport in _output_modules:
            try:
                importlib.import_module(_output_modules[opt.transport])
                self.output = registry.create("streamout", opt.transport, opt=opt, parent=self)
            except ModuleNotFoundError:
                logger.error(f"Output transport module {_output_modules[opt.transport]} not found.")
        else:
            logger.error(f"Output transport {opt.transport} not found in map.")

    # 如果系统没有使用 pipeline，或者为了向后兼容原来的 ttsreal.py
    def put_msg_txt(self, msg, datainfo:dict={}):
        # 首次收到文本输入时自动启动渲染管线，无需等待 WebRTC 连接
        self._ensure_render_running()
        if hasattr(self, 'tts'):
            self.tts.put_msg_txt(msg, datainfo)

    def put_audio_frame(self, audio_chunk:NDArray[np.float32], datainfo:dict={}): # 16khz 20ms pcm
        self._ensure_render_running()
        if hasattr(self, 'asr'):
            self.asr.put_audio_frame(audio_chunk, datainfo)

    def put_audio_file(self, filebyte, datainfo:dict={}):
        self._ensure_render_running()
        input_stream = BytesIO(filebyte)
        stream = self.__create_bytes_stream(input_stream)
        streamlen = stream.shape[0]
        idx = 0
        first = True
        while streamlen >= self.chunk:
            eventpoint = {}
            if first:
                eventpoint = {'status': 'start'}
                first = False
            if streamlen - self.chunk < self.chunk:
                eventpoint = {'status': 'end'}
            eventpoint.update(**datainfo) 
            self.put_audio_frame(stream[idx:idx+self.chunk], eventpoint)
            streamlen -= self.chunk
            idx += self.chunk

    def put_audio_filepath(self, filepath, datainfo:dict={}):
        self._ensure_render_running()
        stream = self.__create_bytes_stream(filepath)
        streamlen = stream.shape[0]
        idx = 0
        first = True
        while streamlen >= self.chunk:
            eventpoint = {}
            if first:
                eventpoint = {'status': 'start'}
                first = False
            if streamlen - self.chunk < self.chunk:
                eventpoint = {'status': 'end'}
            eventpoint.update(**datainfo) 
            self.put_audio_frame(stream[idx:idx+self.chunk], eventpoint)
            streamlen -= self.chunk
            idx += self.chunk
    
    def __create_bytes_stream(self, byte_stream):
        stream, sample_rate = sf.read(byte_stream) # [T*sample_rate,] float64
        logger.info(f'[INFO]put audio stream {sample_rate}: {stream.shape}')
        stream = stream.astype(np.float32)

        if stream.ndim > 1:
            logger.info(f'[WARN] audio has {stream.shape[1]} channels, only use the first.')
            stream = stream[:, 0]
    
        if sample_rate != self.sample_rate and stream.shape[0] > 0:
            logger.info(f'[WARN] audio sample rate is {sample_rate}, resampling into {self.sample_rate}.')
            stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)

        return stream

    def flush_talk(self):
        if hasattr(self, 'tts') and hasattr(self.tts, 'flush_talk'):
            self.tts.flush_talk()
        if hasattr(self, 'asr') and hasattr(self.asr, 'flush_talk'):
            self.asr.flush_talk()
        # 清空推理结果队列
        while not self.res_frame_queue.empty():
            try:
                self.res_frame_queue.get_nowait()
            except queue.Empty:
                break
        # 清空输出队列（WebRTC HumanPlayer 等）
        if hasattr(self, 'output') and hasattr(self.output, 'flush'):
            self.output.flush()
        self.custom_audiotype = 0
        self.speaking = False
        self._current_subtitle = ""
        self._last_rendered_subtitle = ""  # 清除字幕缓存

    # def flush(self):
    #     self.flush_talk()

    def is_speaking(self) -> bool:
        return self.speaking
    
    def __loadcustom(self):
        if not hasattr(self.opt, 'customopt') or not self.opt.customopt:
            return
        for item in self.opt.customopt:
            logger.info(item)
            input_img_list = glob.glob(os.path.join(item['imgpath'], '*.[jpJP][pnPN]*[gG]'))
            input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            self.custom_img_cycle[item['audiotype']] = read_imgs(input_img_list)
            if item.get('audiopath'):
                self.custom_audio_cycle[item['audiotype']], sample_rate = sf.read(item['audiopath'], dtype='float32')
                self.custom_audio_index[item['audiotype']] = 0
            self.custom_index[item['audiotype']] = 0
            # self.custom_opt[item['audiotype']] = item

    def init_customindex(self):
        self.custom_audiotype = 0
        for key in self.custom_audio_index:
            self.custom_audio_index[key] = 0
        for key in self.custom_index:
            self.custom_index[key] = 0

    def notify(self, eventpoint:dict):
        if eventpoint and eventpoint.get('status'):
            logger.info("notify:%s", eventpoint)
            # 记录数字人最近说的话（供前端回声检测）
            if eventpoint.get('status') == 'start' and eventpoint.get('text'):
                if not hasattr(self, '_recent_spoken_texts'):
                    self._recent_spoken_texts = []
                self._recent_spoken_texts.append(eventpoint['text'])
                if len(self._recent_spoken_texts) > 10:
                    self._recent_spoken_texts = self._recent_spoken_texts[-10:]

    def start_recording(self):
        if self.recording:
            return
        command = ['ffmpeg',
                    '-y', '-an',
                    '-f', 'rawvideo',
                    '-vcodec','rawvideo',
                    '-pix_fmt', 'bgr24',
                    '-s', "{}x{}".format(self.width, self.height),
                    '-r', str(25),
                    '-i', '-',
                    '-pix_fmt', 'yuv420p', 
                    '-vcodec', "h264",
                    f'temp{self.opt.sessionid}.mp4']
        self._record_video_pipe = subprocess.Popen(command, shell=False, stdin=subprocess.PIPE)

        acommand = ['ffmpeg',
                    '-y', '-vn',
                    '-f', 's16le',
                    '-ac', '1',
                    '-ar', '16000',
                    '-i', '-',
                    '-acodec', 'aac',
                    f'temp{self.opt.sessionid}.aac']
        self._record_audio_pipe = subprocess.Popen(acommand, shell=False, stdin=subprocess.PIPE)

        self.recording = True
    
    def record_video_data(self, image):
        if self.width == 0:
            self.height, self.width, _ = image.shape
        if self.recording:
            self._record_video_pipe.stdin.write(image.tobytes()) #tostring()

    def record_audio_data(self, frame):
        if self.recording:
            self._record_audio_pipe.stdin.write(frame.tobytes())
		
    def stop_recording(self):
        if not self.recording:
            return
        self.recording = False 
        self._record_video_pipe.stdin.close()
        self._record_video_pipe.wait()
        self._record_audio_pipe.stdin.close()
        self._record_audio_pipe.wait()
        
        record_path = os.path.join('data', 'record')
        os.makedirs(record_path, exist_ok=True)
        output_file = os.path.join(record_path, f"{self.opt.sessionid}.mp4")
        
        temp_aac = f"temp{self.opt.sessionid}.aac"
        temp_mp4 = f"temp{self.opt.sessionid}.mp4"
        
        cmd_combine_audio = f"ffmpeg -y -i {temp_aac} -i {temp_mp4} -c:v copy -c:a copy {output_file}"
        os.system(cmd_combine_audio)
        
        # 删除临时文件
        try:
            os.remove(temp_aac)
            os.remove(temp_mp4)
        except Exception as e:
            logger.error(f"Error removing temp files: {e}")

    # def mirror_index(self, size, index):
    #     turn = index // size
    #     res = index % size
    #     if turn % 2 == 0:
    #         return res
    #     else:
    #         return size - res - 1 
    
    def get_custom_audio_stream(self, audiotype):
        idx = self.custom_audio_index[audiotype]
        stream = self.custom_audio_cycle[audiotype][idx:idx+self.chunk]
        self.custom_audio_index[audiotype] += self.chunk
        if self.custom_audio_index[audiotype] >= self.custom_audio_cycle[audiotype].shape[0]:
            self.custom_audiotype = 1
        return stream
    
    def set_custom_state(self, audiotype, reinit=True):
        print('set_custom_state:', audiotype)
        if self.custom_audio_index.get(audiotype) is None:
            return
        self.custom_audiotype = audiotype
        if reinit:
            self.custom_audio_index[audiotype] = 0
            self.custom_index[audiotype] = 0

    # ========================== 核心渲染及 Pipeline 桥接 ==========================
    def get_avatar_length(self):
        if hasattr(self, 'frame_list_cycle'):
            return len(self.frame_list_cycle)
        return 1
        
    def inference(self, quit_event):
        length = self.get_avatar_length()
        index = 0
        count = 0
        counttime = 0
        last_speaking = False

        # syncnet_T = 12  # 时间步
        # weight_dtype = torch.float16  # 数据类型
        # infernum = 0
        logger.info('start inference')
        while not quit_event.is_set():
            starttime = time.perf_counter()
            audiofeat_batch = []
            try:
                audiofeat_batch = self.asr.feat_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue

            is_all_silence = True
            audio_frames: list[AudioFrameData] = []
            try:
                for _ in range(self.batch_size * 2):
                    audioframe:AudioFrameData = self.asr.output_queue.get(timeout=5.0)
                    if audioframe.type == 0:
                        is_all_silence = False
                    audio_frames.append(audioframe)
            except queue.Empty:
                logger.warning('inference: output_queue.get() timed out, skipping batch')
                continue

             # 检测状态变化
            current_speaking = not is_all_silence

            if is_all_silence: #全为静音数据，只需要取fullimg，不需要推理
                for i in range(self.batch_size):
                    idx = mirror_index(length, index)
                    self.res_frame_queue.put((None, audio_frames[i*2:i*2+2], idx))
                    index = index + 1
            else:
                if current_speaking and not last_speaking and self.custom_index.get(1) is not None: #从静音到说话切换,并且有自定义静态视频
                    index = 0
                t = time.perf_counter()

                try:
                    pred = self.inference_batch(index, audiofeat_batch)
                except Exception as e:
                    logger.error(f'inference_batch failed: {e}')
                    # Put silence frames to keep pipeline alive
                    for i in range(self.batch_size):
                        idx = mirror_index(length, index)
                        self.res_frame_queue.put((None, audio_frames[i*2:i*2+2], idx))
                        index = index + 1
                    continue

                counttime += (time.perf_counter() - t)
                count += self.batch_size
                if count >= 100:
                    logger.info(f"------actual avg infer fps:{count/counttime:.4f}")
                    count = 0
                    counttime = 0
                for i, res_frame in enumerate(pred):
                    self.res_frame_queue.put((res_frame, audio_frames[i*2:i*2+2], mirror_index(length, index)))
                    index = index + 1
                    
            if current_speaking != last_speaking:
                logger.info(f"inference 状态切换：{'说话' if last_speaking else '静音'} → {'说话' if current_speaking else '静音'}")
                last_speaking = current_speaking         
        logger.info('baseavatar inference thread stop')

    def process_frames(self,quit_event):
        enable_transition = False  # 设置为False禁用过渡效果，True启用
        
        _last_speaking = False
        _transition_start = time.time()
        if enable_transition:
            _transition_duration = 0.1  # 过渡时间
            _last_silent_frame = None  # 静音帧缓存
            _last_speaking_frame = None  # 说话帧缓存

        self.output.start()
        
        while not quit_event.is_set():
            try:
                audio_frames: list[AudioFrameData]
                res_frame,audio_frames,idx = self.res_frame_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue
            
            # 检测状态变化
            current_speaking = not (audio_frames[0].type!=0 and audio_frames[1].type!=0)
            if current_speaking != _last_speaking:
                logger.info(f"状态切换：{'说话' if _last_speaking else '静音'} → {'说话' if current_speaking else '静音'}")
                _transition_start = time.time()
            _last_speaking = current_speaking

            if audio_frames[0].type!=0 and audio_frames[1].type!=0: #全为静音数据，只需要取fullimg
                self.speaking = False
                audiotype = audio_frames[0].type
                if self.custom_index.get(audiotype) is not None: #有自定义视频
                    mirindex = mirror_index(len(self.custom_img_cycle[audiotype]),self.custom_index[audiotype])
                    target_frame = self.custom_img_cycle[audiotype][mirindex]
                    self.custom_index[audiotype] += 1
                else:
                    target_frame = self.frame_list_cycle[idx]

                if enable_transition:
                    # 说话→静音过渡
                    if time.time() - _transition_start < _transition_duration and _last_speaking_frame is not None:
                        alpha = min(1.0, (time.time() - _transition_start) / _transition_duration)
                        combine_frame = cv2.addWeighted(_last_speaking_frame, 1-alpha, target_frame, alpha, 0)
                    else:
                        combine_frame = target_frame
                    # 缓存静音帧
                    _last_silent_frame = combine_frame.copy()
                else:
                    combine_frame = target_frame
            else:
                self.speaking = True
                try:
                    current_frame = self.paste_back_frame(res_frame,idx)
                except Exception as e:
                    logger.warning(f"paste_back_frame error: {e}")
                    continue
                if enable_transition:
                    # 静音→说话过渡
                    if time.time() - _transition_start < _transition_duration and _last_silent_frame is not None:
                        alpha = min(1.0, (time.time() - _transition_start) / _transition_duration)
                        combine_frame = cv2.addWeighted(_last_silent_frame, 1-alpha, current_frame, alpha, 0)
                    else:
                        combine_frame = current_frame
                    # 缓存说话帧
                    _last_speaking_frame = combine_frame.copy()
                else:
                    combine_frame = current_frame

            # ── 字幕缓存：仅当文本变化时才重绘 PIL（CPU 关键优化）──
            if self._subtitle_enabled:
                for af in audio_frames:
                    ud = af.userdata
                    if ud.get("status") == "start" and ud.get("text"):
                        self._current_subtitle = ud["text"]
                    elif ud.get("status") == "end":
                        self._current_subtitle = ""

            cv2.putText(combine_frame, "LiveTalking", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128,128,128), 1)

            # ── 叠加字幕（缓存优化：文本不变时不重绘 PIL，节省 5-10ms/帧）──
            if self._subtitle_enabled and self._current_subtitle:
                try:
                    txt = _clean_subtitle_text(self._current_subtitle)
                    if txt:
                        h, w = combine_frame.shape[:2]
                        if txt != self._last_rendered_subtitle:
                            # 重新生成缓存层
                            font_size = max(32, int(w * 0.065 * self._subtitle_size))
                            if self._subtitle_font is None or self._subtitle_font_size != font_size:
                                self._subtitle_font = ImageFont.truetype(_SUBTITLE_FONT_PATH, font_size)
                                self._subtitle_font_size = font_size
                            font = self._subtitle_font
                            # 在黑色背景上绘制字幕
                            overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
                            draw = ImageDraw.Draw(overlay)
                            max_w = w - 40
                            line_h = draw.textbbox((0, 0), "测", font=font)[3] - draw.textbbox((0, 0), "测", font=font)[1]
                            lines = []
                            for char in txt:
                                if not lines:
                                    lines.append(char)
                                else:
                                    candidate = lines[-1] + char
                                    tw = draw.textbbox((0, 0), candidate, font=font)[2] - draw.textbbox((0, 0), candidate, font=font)[0]
                                    if tw <= max_w:
                                        lines[-1] = candidate
                                    else:
                                        lines.append(char)
                            total_h = len(lines) * (line_h + 4)
                            y_start = h - int(h * 0.08) - total_h
                            for i, line in enumerate(lines):
                                tw = draw.textbbox((0, 0), line, font=font)[2] - draw.textbbox((0, 0), line, font=font)[0]
                                x = (w - tw) // 2
                                y = y_start + i * (line_h + 4) + line_h
                                draw.text((x, y - line_h), line, font=font, fill=(255, 255, 255, 255),
                                          stroke_width=max(2, int(font_size * 0.08)), stroke_fill=(0, 0, 0, 255))
                            self._cached_subtitle_frame = overlay
                            self._last_rendered_subtitle = txt
                        # 合成缓存层到视频帧
                        if self._cached_subtitle_frame is not None:
                            pil_img = Image.fromarray(cv2.cvtColor(combine_frame, cv2.COLOR_BGR2RGB))
                            pil_img = Image.alpha_composite(pil_img.convert('RGBA'), self._cached_subtitle_frame)
                            combine_frame = cv2.cvtColor(np.array(pil_img.convert('RGB')), cv2.COLOR_RGB2BGR)
                except Exception as e:
                    logger.warning(f"Subtitle render error: {e}")
            
            # 使用统一输出接口推送视频帧
            self.output.push_video_frame(combine_frame)
            self.record_video_data(combine_frame)

            for audio_frame in audio_frames:
                #frame,type,eventpoint = audio_frame
                frame = (audio_frame.data * 32767).astype(np.int16)

                # 使用统一输出接口推送音频帧
                self.output.push_audio_frame(frame, audio_frame.userdata)
                self.record_audio_data(frame)
                
            # if self.opt.transport == 'virtualcam' and hasattr(self.output, '_cam') and self.output._cam:
            #     self.output._cam.sleep_until_next_frame()

        self.output.stop()
        logger.info('baseavatar process_frames thread stop') 

    # ========================== 渲染管线生命周期管理 ==========================

    _render_started: bool = False
    _render_thread = None
    _render_quit = None

    def start_render(self):
        """启动渲染管线（如果尚未启动）。可在会话创建后立即调用，不必等待 WebRTC。"""
        if self._render_started:
            logger.info('Render already started for session %s, skipping', self.sessionid)
            return
        self._render_started = True
        self._render_quit = threading.Event()
        self._render_thread = Thread(
            target=self.render,
            args=(self._render_quit,),
            daemon=True,
        )
        self._render_thread.start()
        logger.info('Render pipeline started for session %s', self.sessionid)

    def stop_render(self):
        """停止渲染管线，清理子线程。"""
        if self._render_quit and not self._render_quit.is_set():
            logger.info('Stopping render pipeline for session %s', self.sessionid)
            self._render_quit.set()
        if self._render_thread and self._render_thread.is_alive():
            self._render_thread.join(timeout=5.0)
            if self._render_thread.is_alive():
                logger.warning('Render thread did not exit in time for session %s', self.sessionid)
        self._render_started = False

    def _ensure_render_running(self):
        """按需启动渲染管线（首次输入文本/音频时自动启动，不等待 WebRTC）"""
        if not self._render_started:
            logger.info('Render not yet started — triggering on-demand start for session %s', self.sessionid)
            self.start_render()

    def render(self,quit_event):
        self.quit_event = quit_event
        
        self.init_customindex()
        self.tts.render(quit_event)

        infer_quit_event = mp.Event()
        infer_thread = Thread(target=self.inference, args=(infer_quit_event,))
        infer_thread.start()
        
        process_quit_event = Event()
        process_thread = Thread(target=self.process_frames, args=(process_quit_event,))
        process_thread.start()

        count=0
        totaltime=0
        _starttime=time.perf_counter()
        _totalframe=0
        while not quit_event.is_set(): 
            t = time.perf_counter()
            self.asr.run_step()

            # ── 多层节流（关键：限制 asr.run_step 速度，避免音视频不同步）──
            # 1) WebRTC video buffer 限流（原版动态 sleep 机制）
            buffer_size = self.output.get_buffer_size() if hasattr(self.output, 'get_buffer_size') else 0
            if buffer_size >= 5:
                sleep_t = 0.04 * buffer_size * 0.8
                time.sleep(sleep_t)
            
            # 2) feat_queue 限流（避免 mel 丢弃）
            feat_queue_size = self.asr.feat_queue.qsize() if hasattr(self.asr, 'feat_queue') else 0
            if feat_queue_size >= 3:
                time.sleep(0.005)
            
            # 3) output_queue 限流（关键：避免音视频不同步）
            # asr.output_queue 持续累积 audio frame 时，inference 处理的 audio 是几秒前的旧音频
            # 导致 video frame idx 错位（人脸摇头速度异常）
            output_queue_size = self.asr.output_queue.qsize() if hasattr(self.asr, 'output_queue') else 0
            if output_queue_size > self.batch_size * 4:  # > 16 个 audio frame
                time.sleep(0.02)
            
            # 4) 全局节流：每 100ms 至多调一次 asr.run_step（10Hz），对应 40 fps 上限
            elapsed = time.perf_counter() - t
            if elapsed < 0.04:  # < 40ms，等满 40ms
                time.sleep(0.04 - elapsed)
        logger.info('baseavatar render thread stop')

        infer_quit_event.set()
        infer_thread.join(timeout=3.0)
        if infer_thread.is_alive():
            logger.warning('inference thread did not exit in time')

        process_quit_event.set()
        process_thread.join(timeout=3.0)
        if process_thread.is_alive():
            logger.warning('process_frames thread did not exit in time')

