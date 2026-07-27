import argparse
import glob
import json
import os
import pickle
import shutil

import cv2
import numpy as np
import torch
from tqdm import tqdm

from avatars.musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs
from avatars.musetalk.utils.blending import get_image_prepare_material
from avatars.musetalk.utils.utils import load_all_model

try:
    from utils.face_parsing import FaceParsing
except ModuleNotFoundError:
    from avatars.musetalk.utils.face_parsing import FaceParsing


def video2imgs(vid_path, save_path, ext='.png', cut_frame=10000000):
    cap = cv2.VideoCapture(vid_path)
    count = 0
    while True:
        if count > cut_frame:
            break
        ret, frame = cap.read()
        if ret:
            cv2.putText(frame, "LiveTalking", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128,128,128), 1)
            cv2.imwrite(f"{save_path}/{count:08d}.png", frame)
            count += 1
        else:
            break


def is_video_file(file_path):
    video_exts = ['.mp4', '.mkv', '.flv', '.avi', '.mov']
    file_ext = os.path.splitext(file_path)[1].lower()
    return file_ext in video_exts


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def generate_avatar(video_path, avatar_id, save_path='./data/avatars', bbox_shift=0, extra_margin=10, parsing_mode='jaw', version='v15', face_det_batch_size=8, progress_callback=None):
    """
    生成avatar的核心逻辑

    Args:
        video_path: 输入视频路径
        avatar_id: Avatar ID
        save_path: 保存根路径
        bbox_shift: 边界框偏移
        extra_margin: 额外边距
        parsing_mode: 解析模式
        version: 版本
        face_det_batch_size: 人脸检测批处理大小 (GPU 优化)
        progress_callback: 进度回调函数，接收 0-100 的整数
    """
    avatar_save_path = os.path.join(save_path, avatar_id)
    save_full_path = os.path.join(avatar_save_path, 'full_imgs')
    create_dir(avatar_save_path)
    create_dir(save_full_path)
    mask_out_path = os.path.join(avatar_save_path, 'mask')
    create_dir(mask_out_path)

    mask_coords_path = os.path.join(avatar_save_path, 'mask_coords.pkl')
    coords_path = os.path.join(avatar_save_path, 'coords.pkl')
    latents_out_path = os.path.join(avatar_save_path, 'latents.pt')

    if progress_callback: progress_callback(5)

    with open(os.path.join(avatar_save_path, 'avator_info.json'), "w") as f:
        json.dump({
            "avatar_id": avatar_id,
            "video_path": video_path,
            "bbox_shift": bbox_shift
        }, f)

    if os.path.isfile(video_path):
        if is_video_file(video_path):
            video2imgs(video_path, save_full_path, ext='png')
        else:
            shutil.copyfile(video_path, f"{save_full_path}/{os.path.basename(video_path)}")
    else:
        files = os.listdir(video_path)
        files.sort()
        files = [file for file in files if file.split(".")[-1] == "png"]
        for filename in files:
            shutil.copyfile(f"{video_path}/{filename}", f"{save_full_path}/{filename}")

    if progress_callback: progress_callback(20)

    input_img_list = sorted(glob.glob(os.path.join(save_full_path, '*.[jpJP][pnPN]*[gG]')))
    print("extracting landmarks...")
    coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift, face_det_batch_size)

    if progress_callback: progress_callback(50)

    input_latent_list = []
    idx = -1
    coord_placeholder = (0.0, 0.0, 0.0, 0.0)

    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    vae_local, unet_local, pe_local = load_all_model(device=device)
    vae_local.vae = vae_local.vae.half().to(device)

    # 释放不需要的模型以节省显存
    del unet_local, pe_local
    torch.cuda.empty_cache()

    if version == "v15":
        fp_local = FaceParsing(left_cheek_width=90, right_cheek_width=90)
    else:
        fp_local = FaceParsing()

    # ── 阶段 1: 收集所有需要 VAE 编码的帧 ──
    crop_frames = []
    valid_indices = []  # 有效帧在 coord_list 中的索引
    for idx, (bbox, frame) in enumerate(zip(coord_list, frame_list)):
        if bbox == coord_placeholder:
            continue
        x1, y1, x2, y2 = bbox
        if version == "v15":
            y2 = y2 + extra_margin
            y2 = min(y2, frame.shape[0])
            coord_list[idx] = [x1, y1, x2, y2]
        crop_frame = frame[y1:y2, x1:x2]
        resized_crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        crop_frames.append(resized_crop_frame)
        valid_indices.append(idx)

    # ── 阶段 2: GPU 批量 VAE 编码 ──
    vae_batch_size = 4  # RTX 3060 12GB 可轻松处理 4 帧 batch
    print(f'[GPU] Batch VAE encoding {len(crop_frames)} frames (batch_size={vae_batch_size}) ...')
    for batch_start in tqdm(range(0, len(crop_frames), vae_batch_size), desc='VAE encoding'):
        batch_imgs = crop_frames[batch_start:batch_start + vae_batch_size]
        batch_latents = vae_local.get_latents_for_unet_batch(batch_imgs)
        input_latent_list.extend(batch_latents)

        if progress_callback:
            progress = 50 + int((batch_start + len(batch_imgs)) / len(frame_list) * 25)
            progress_callback(min(progress, 75))

    # 释放 VAE 模型显存，后续不再需要
    del vae_local
    torch.cuda.empty_cache()

    mask_coords_list_cycle = []
    mask_list_cycle = []
    for i, frame in enumerate(frame_list):
        cv2.imwrite(f"{save_full_path}/{str(i).zfill(8)}.png", frame)

        x1, y1, x2, y2 = coord_list[i]
        mode = parsing_mode if version == "v15" else "raw"
        mask, crop_box = get_image_prepare_material(frame, [x1, y1, x2, y2], fp=fp_local, mode=mode)
        cv2.imwrite(f"{mask_out_path}/{str(i).zfill(8)}.png", mask)

        mask_coords_list_cycle += [crop_box]
        mask_list_cycle.append(mask)

        if progress_callback:
            progress = 75 + int((i + 1) / len(frame_list) * 20)
            progress_callback(progress)

    with open(mask_coords_path, 'wb') as f:
        pickle.dump(mask_coords_list_cycle, f)

    with open(coords_path, 'wb') as f:
        pickle.dump(coord_list, f)
    torch.save(input_latent_list, os.path.join(latents_out_path))

    if progress_callback: progress_callback(100)
    print("Avatar 生成完成！")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=r'D:\ok\00000000.png')
    parser.add_argument("--avatar_id", type=str, default='musetalk_avatar1')
    parser.add_argument('--save_path', default='data/avatars', type=str)
    parser.add_argument("--version", type=str, default="v15", choices=["v1", "v15"], help="Version of MuseTalk: v1 or v15")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--left_cheek_width", type=int, default=90, help="Width of left cheek region")
    parser.add_argument("--right_cheek_width", type=int, default=90, help="Width of right cheek region")
    parser.add_argument("--bbox_shift", type=int, default=0, help="Bounding box shift value")
    parser.add_argument("--extra_margin", type=int, default=10, help="Extra margin for face cropping")
    parser.add_argument("--parsing_mode", default='jaw', help="Face blending parsing mode")
    args = parser.parse_args()

    generate_avatar(
        video_path=args.file,
        avatar_id=args.avatar_id,
        save_path=args.save_path,
        bbox_shift=args.bbox_shift,
        extra_margin=args.extra_margin,
        parsing_mode=args.parsing_mode,
        version=args.version
    )
