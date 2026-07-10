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
#  Wav2Lip 数字人 — 迁移自 lipreal.py + lipasr.py
#

import math
import torch
import numpy as np

import os
import time
import cv2
import glob
import pickle
import copy
import gc

import queue
from queue import Queue
from threading import Thread, Event
import torch.multiprocessing as mp

from avatars.audio_features.mel import MelASR
import asyncio
from av import AudioFrame, VideoFrame
from avatars.wav2lip.models import Wav2Lip
from avatars.base_avatar import BaseAvatar

from tqdm import tqdm
from utils.logger import logger
from utils.image import read_imgs, mirror_index
from utils.device import initialize_device
from registry import register

device = initialize_device()
_use_gpu = str(device) == 'cuda'
logger.info('Using {} for inference. _use_gpu={}'.format(device, _use_gpu))

def _load(checkpoint_path):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_model(path):
    model = Wav2Lip()
    logger.info("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    return model.eval()

def load_avatar(avatar_id):
    avatar_path = f"./data/avatars/{avatar_id}"
    full_imgs_path = f"{avatar_path}/full_imgs" 
    face_imgs_path = f"{avatar_path}/face_imgs" 
    coords_path = f"{avatar_path}/coords.pkl"
    
    with open(coords_path, 'rb') as f:
        coord_list_cycle = pickle.load(f)
    frame_list_cycle = None
    input_img_list = glob.glob(os.path.join(full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
    input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    frame_list_cycle = read_imgs(input_img_list)
    input_face_list = glob.glob(os.path.join(face_imgs_path, '*.[jpJP][pnPN]*[gG]'))
    input_face_list = sorted(input_face_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    face_list_cycle = read_imgs(input_face_list)

    return frame_list_cycle,face_list_cycle,coord_list_cycle

@torch.no_grad()
def warm_up(batch_size,model,modelres):
    # 预热函数
    logger.info('warmup model...')
    img_batch = torch.ones(batch_size, 6, modelres, modelres).to(device)
    mel_batch = torch.ones(batch_size, 1, 80, 16).to(device)
    model(mel_batch, img_batch)

@register("avatar", "wav2lip")
class LipReal(BaseAvatar):
    @torch.no_grad()
    def __init__(self, opt, model, avatar):
        super().__init__(opt)

        self.model = model

        self.frame_list_cycle,self.face_list_cycle,self.coord_list_cycle = avatar

        # ── 预加载 GPU 张量（弱 CPU 强 GPU 场景关键优化）──
        if _use_gpu:
            # face_list_cycle → [N, 3, H, W] float32 归一化张量（避免每 batch 重复 CPU→GPU 传输）
            face_tensors = []
            for face_img in self.face_list_cycle:
                face_tensors.append(torch.from_numpy(face_img).permute(2, 0, 1).float().div(255.0))
            self.face_gpu = torch.stack(face_tensors).to(device)
            logger.info(f"Pre-loaded {len(face_tensors)} face frames to GPU: {self.face_gpu.shape}")

            # full frame list → [N, H, W, 3] uint8 张量（paste_back_frame 时 clone 避免 np.copy）
            full_tensors = []
            for frame in self.frame_list_cycle:
                full_tensors.append(torch.from_numpy(frame))
            self.full_gpu = torch.stack(full_tensors).to(device)
            logger.info(f"Pre-loaded {len(full_tensors)} full frames to GPU: {self.full_gpu.shape}")
            # release intermediate lists and trigger GC after GPU preload
            del face_tensors, full_tensors
            gc.collect()
            torch.cuda.empty_cache()
            logger.info("GPU pre-load done, cleaned up CPU temporary memory")

        else:
            self.face_gpu = None
            self.full_gpu = None

        # inference batch counter for periodic GC
        self._infer_count = 0

        self.asr = MelASR(opt,self)
        self.asr.warm_up()

        # 预分配 GPU 持久化张量 — 避免每 batch 重新分配
        self._img_buf = torch.zeros(self.batch_size, 6, opt.modelres, opt.modelres,
                                    dtype=torch.float32, device=device)
        self._audio_buf = torch.zeros(self.batch_size, 1, 80, 16,
                                      dtype=torch.float32, device=device)
        self._img_transposed = torch.zeros(1, dtype=torch.float32, device=device)
        self._audio_transposed = torch.zeros(1, dtype=torch.float32, device=device)

    def inference_batch(self, index, audiofeat_batch):
        length = len(self.face_list_cycle)

        if _use_gpu:
            # ── GPU 路径：从预加载张量切片，避免 numpy 操作和 CPU→GPU 传输 ──
            indices = [mirror_index(length, index + i) for i in range(self.batch_size)]
            img_batch = self.face_gpu[indices]  # [B, 3, H, W]

            # 音频处理（音频特征本身生成已在 GPU mel 中优化）
            audiofeat_batch = np.asarray(audiofeat_batch).astype(np.float32)
            if audiofeat_batch.size == 0 or len(audiofeat_batch) == 0:
                # 罕见的空数组情况：返回静音帧（保持 pipeline 运行）
                return np.zeros((self.batch_size, 96, 96, 3), dtype=np.uint8)
            if len(audiofeat_batch) > self.batch_size:
                audiofeat_batch = audiofeat_batch[:self.batch_size]
            elif len(audiofeat_batch) < self.batch_size:
                pad = np.zeros((self.batch_size - len(audiofeat_batch),) + audiofeat_batch.shape[1:], dtype=np.float32)
                audiofeat_batch = np.concatenate([audiofeat_batch, pad], axis=0)

            # 下框 mask
            img_masked = img_batch.clone()
            img_masked[:, :, img_batch.shape[2]//2:, :] = 0
            img_concat = torch.cat([img_masked, img_batch], dim=1)  # [B, 6, H, W]

            audio_t = torch.from_numpy(audiofeat_batch).to(device, non_blocking=True)
            audio_t = audio_t.unsqueeze(1)  # [B, 1, 80, 16] — Wav2Lip 需要 channel 维度在位置 1

            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    pred = self.model(audio_t, img_concat)
            # pred: [B, 3, 96, 96] float32 on GPU → 立即转 CPU numpy，
            # 避免跨线程 GPU 操作导致 CUDA 同步阻塞
            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
            self._infer_count += 1
            if self._infer_count % 200 == 0:
                gc.collect()
                torch.cuda.empty_cache()
            return pred
        else:
            # ── CPU 回退路径（原始逻辑）──
            img_batch = []
            for i in range(self.batch_size):
                idx = mirror_index(length, index + i)
                face = self.face_list_cycle[idx]
                img_batch.append(face)
            img_batch, audiofeat_batch = np.asarray(img_batch).astype(np.float32), np.asarray(audiofeat_batch).astype(np.float32)

            if len(audiofeat_batch) > self.batch_size:
                audiofeat_batch = audiofeat_batch[:self.batch_size]
            elif len(audiofeat_batch) < self.batch_size:
                pad = np.zeros((self.batch_size - len(audiofeat_batch),) + audiofeat_batch.shape[1:], dtype=np.float32)
                audiofeat_batch = np.concatenate([audiofeat_batch, pad], axis=0)

            img_masked = img_batch.copy()
            img_masked[:, face.shape[0]//2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            audiofeat_batch = np.reshape(audiofeat_batch, [len(audiofeat_batch), audiofeat_batch.shape[1], audiofeat_batch.shape[2], 1])

            img_t = torch.from_numpy(np.asarray(np.transpose(img_batch, (0, 3, 1, 2)), dtype=np.float32)).to(device, non_blocking=True)
            audio_t = torch.from_numpy(np.asarray(np.transpose(audiofeat_batch, (0, 3, 1, 2)), dtype=np.float32)).to(device, non_blocking=True)

            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    pred = self.model(audio_t, img_t)
            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
            # periodic cleanup every 200 batches to prevent memory leak
            self._infer_count += 1
            if self._infer_count % 200 == 0:
                gc.collect()
                torch.cuda.empty_cache()

            return pred

    def paste_back_frame(self, pred_frame, idx: int):
        bbox = self.coord_list_cycle[idx]

        # ── CPU 路径（推理结果已在推理线程中转 CPU，避免跨线程 GPU 争抢）──
        y1, y2, x1, x2 = bbox
        combine_frame = np.copy(self.frame_list_cycle[idx])
        res_frame = cv2.resize(pred_frame.astype(np.uint8, copy=False), (x2 - x1, y2 - y1))
        combine_frame[y1:y2, x1:x2] = res_frame
        return combine_frame

