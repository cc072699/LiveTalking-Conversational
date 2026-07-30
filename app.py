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

# server.py
import base64
import json
import re
import os
import numpy as np
from threading import Thread,Event
from concurrent.futures import ThreadPoolExecutor
#import multiprocessing
import torch.multiprocessing as mp

from aiohttp import web
import aiohttp
import aiohttp_cors
from aiortc import RTCPeerConnection, RTCSessionDescription,RTCIceServer,RTCConfiguration
from aiortc.rtcrtpsender import RTCRtpSender
from server.webrtc import HumanPlayer
from avatars.base_avatar import BaseAvatar
from llm import llm_response
import registry
from server.routes import setup_routes
from server.rtc_manager import RTCManager
from server.session_manager import session_manager

import argparse
import random
import shutil
import ssl
import asyncio
import torch
from io import BytesIO
from typing import Dict
from utils.logger import logger
import copy
import gc
from dotenv import load_dotenv


opt = None
model = None
global_avatars = {} # avatar_id: payload
        

#####webrtc###############################
# rtc_manager replaces the old pcs set and duplicate offer handlers.
rtc_manager = None

import collections

_session_cache: Dict[str, BaseAvatar] = collections.OrderedDict()
_SESSION_CACHE_MAX = 2  # 最多缓存 2 个不同配置，超出则淘汰最旧的

def _cache_evict_oldest():
    """淘汰最旧的一个缓存 session（停止渲染并释放资源）"""
    if len(_session_cache) > _SESSION_CACHE_MAX:
        old_key, old_session = _session_cache.popitem(last=False)
        try:
            old_session.release_resources()
        except Exception as e:
            logger.warning(f'Session cache evict failed: {e}')
        logger.info(f'Session cache evicted: avatar={old_key[0]} voice={old_key[1]}')

def randN(N)->int:
    '''生成长度为 N的随机数 '''
    min = pow(10, N - 1)
    max = pow(10, N)
    return random.randint(min, max - 1)

def build_avatar_session(sessionid:str, params:dict)->BaseAvatar:
    opt_this = copy.deepcopy(opt)
    opt_this.sessionid = sessionid

    avatar_id = params.get('avatar',opt.avatar_id) 
    opt_this.avatar_id = avatar_id
    ref_audio = params.get('refaudio','') #音色
    ref_text = params.get('reftext','')
    prompt_key = params.get('prompt','') or ''  # 数字人 Persona
    custom_config = params.get('custom_config','')

    # ── 缓存 key：(avatar_id, tts_voice, prompt, has_custom) ──
    cache_key = (avatar_id, ref_audio, prompt_key, bool(custom_config))
    if cache_key in _session_cache:
        cached = _session_cache[cache_key]
        if cached and not cached.is_speaking():
            cached.opt.sessionid = sessionid
            # 检测 render 是否需要重启：_render_started=False 或 render 线程已退出但标记未清理
            render_alive = cached._render_thread and cached._render_thread.is_alive()
            if not render_alive:
                cached._render_started = False  # 同步标记位
                cached.start_render()  # 重启渲染管线
            logger.info(f'Session cache HIT avatar={avatar_id}')
            return cached
        else:
            # talking session cannot be reused — evict and recreate
            if cached is not None:
                _stale = cached
                def _cleanup_stale():
                    try:
                        _stale.release_resources()
                    except Exception:
                        logger.exception('Error releasing stale session resources')
                Thread(target=_cleanup_stale, daemon=True).start()
            del _session_cache[cache_key]
            logger.info(f'Session cache INVALIDATED (busy): avatar={avatar_id}')

    if (avatar_id and avatar_id != opt.avatar_id):
        # Avoid reloading if already cached globally
        if avatar_id not in global_avatars:
            global_avatars[avatar_id] = load_avatar(avatar_id)
        avatar_this = global_avatars[avatar_id]
    elif opt.avatar_id and opt.avatar_id in global_avatars:
        # Default avatar loaded at startup
        avatar_this = global_avatars.get(opt.avatar_id)
    else:
        # No default avatar available — try the first available one
        avatar_this = next(iter(global_avatars.values()), None)
        if avatar_this:
            logger.warning(f"No specific avatar requested and default not available; falling back to first available avatar")
    if ref_audio: #请求参数配置了参考音频
        opt_this.REF_FILE = ref_audio
        opt_this.REF_TEXT = ref_text
    if prompt_key:
        opt_this.PROMPT_KEY = prompt_key
    if custom_config:
        opt_this.customopt = json.loads(custom_config)

    avatar_session = registry.create("avatar", opt.model, opt=opt_this, model=model, avatar=avatar_this)
    avatar_session._cache_key = cache_key  # 供 remove_session 定位缓存条目
    # 缓存前淘汰旧条目（LRU 策略，max 2）
    if cache_key in _session_cache:
        del _session_cache[cache_key]  # 移到末尾（刷新 LRU 顺序）
    _session_cache[cache_key] = avatar_session
    _cache_evict_oldest()
    logger.info(f'Session cache MISS avatar={avatar_id}')
    return avatar_session

async def offer(request):
    return await rtc_manager.handle_offer(request)

async def on_shutdown(app):
    await rtc_manager.shutdown()

async def download_record(request):
    sessionid = request.match_info.get('sessionid')
    if not sessionid:
        return web.Response(status=400, text="sessionid is required")
    
    record_file = os.path.join('data', 'record', f"{sessionid}.mp4")
    
    if os.path.exists(record_file):
        return web.FileResponse(record_file)
    else:
        return web.Response(status=404, text="Record not found")


def main():
    global rtc_manager, opt, model,load_avatar
    # 解析命令行参数
    from config import parse_args
    opt = parse_args()

    # ─── 加载 avatar 插件（触发 @register 注册）──────────────────────
    _avatar_modules = {
        'musetalk':   'avatars.musetalk_avatar',
        'wav2lip':    'avatars.wav2lip_avatar',
        'ultralight': 'avatars.ultralight_avatar',
    }
    import importlib
    avatar_mod = importlib.import_module(_avatar_modules[opt.model])
    load_model = avatar_mod.load_model
    load_avatar = avatar_mod.load_avatar
    warm_up = avatar_mod.warm_up
    logger.info(opt)

    if opt.model == 'musetalk':
        model = load_model()
        if opt.avatar_id:
            global_avatars[opt.avatar_id] = load_avatar(opt.avatar_id)
        else:
            logger.warning("No avatar_id configured — default avatar not loaded")
        warm_up(opt.batch_size,model)
    elif opt.model == 'wav2lip':
        model = load_model("./models/wav2lip.pth")
        if opt.avatar_id:
            global_avatars[opt.avatar_id] = load_avatar(opt.avatar_id)
        else:
            logger.warning("No avatar_id configured — default avatar not loaded")
        # 开启 cudnn benchmark: 输入形状固定时缓存最优卷积算法
        torch.backends.cudnn.benchmark = True
        warm_up(opt.batch_size,model,256)
        # ── 按需加载 avatar（不再预加载所有，节省内存 ~2.8GB/avatar）──
        gc.collect()
        torch.cuda.empty_cache()
    elif opt.model == 'ultralight':
        model = load_model(opt)
        if opt.avatar_id:
            global_avatars[opt.avatar_id] = load_avatar(opt.avatar_id)
            warm_up(opt.batch_size, global_avatars[opt.avatar_id], 160)
        else:
            logger.warning("No avatar_id configured — default avatar not loaded")

    # init rtc manager
    session_manager.init_builder(build_avatar_session)
    session_manager.set_cache(_session_cache)
    rtc_manager = RTCManager(opt)
    # share avatar_sessions (RTCManager handles it but routes.py expects it)
    
    if opt.transport=='virtualcam' or opt.transport=='rtmp':
        thread_quit = Event()
        params = {}
        # session 0 for virtualcam
        session_manager.add_session('0', build_avatar_session('0', params))
        rendthrd = Thread(target=session_manager.get_session('0').render,args=(thread_quit,))
        rendthrd.start()

    #############################################################################
    appasync = web.Application(client_max_size=1024**2*100)
    appasync["llm_response"] = llm_response
    appasync["opt"] = opt
    appasync["rtc_manager"] = rtc_manager

    appasync.on_shutdown.append(on_shutdown)
    appasync.router.add_post("/offer", offer)
    appasync.router.add_get("/record/{sessionid}", download_record)
    
    # 注册 server/routes.py 中的通用 API 路由
    setup_routes(appasync) 

    # Configure default CORS settings.
    cors = aiohttp_cors.setup(appasync, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
            )
        })
    # Configure CORS on all routes.
    for route in list(appasync.router.routes()):
        cors.add(route)

    pagename='index.html'
    if opt.transport=='rtmp':
        pagename='rtmpapi.html'
    elif opt.transport=='rtcpush':
        pagename='rtcpushapi.html'
    logger.info('start http server; http://<serverip>:'+str(opt.listenport)+'/'+pagename)
    if opt.ssl:
        logger.info('HTTPS enabled — remote users can use microphone')
    # logger.info('如果使用webrtc，推荐访问webrtc集成前端: http://<serverip>:'+str(opt.listenport)+'/dashboard.html')
    def run_server(runner):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        # 限制默认线程池大小，避免 futex 忙等撑高 CPU
        loop.set_default_executor(ThreadPoolExecutor(max_workers=3))
        loop.run_until_complete(runner.setup())
        if opt.ssl:
            cert = opt.ssl_cert or f'./ssl/cert.pem'
            key = opt.ssl_key or f'./ssl/key.pem'
            ssl_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            ssl_ctx.load_cert_chain(cert, key)
            site = web.TCPSite(runner, '0.0.0.0', opt.listenport, ssl_context=ssl_ctx)
            logger.info('HTTPS server ready at https://<serverip>:'+str(opt.listenport))
        else:
            site = web.TCPSite(runner, '0.0.0.0', opt.listenport)
        loop.run_until_complete(site.start())
        if opt.transport=='rtcpush':
            for k in range(opt.max_session):
                push_url = opt.push_url
                if k!=0:
                    push_url = opt.push_url+str(k)
                loop.run_until_complete(rtc_manager.handle_rtcpush(push_url, str(k)))
        
        # 定期内存回收：Python GC + glibc malloc_trim 归还 OS
        import gc as _gc
        async def _periodic_mem_cleanup():
            while True:
                await asyncio.sleep(45)
                _gc.collect()
                try:
                    import ctypes as _ct
                    _ct.CDLL('libc.so.6').malloc_trim(0)
                except Exception:
                    pass
        
        async def _delayed_start_mem_cleanup():
            await asyncio.sleep(30)
            asyncio.ensure_future(_periodic_mem_cleanup())
        asyncio.ensure_future(_delayed_start_mem_cleanup())
        
        loop.run_forever()    
    #Thread(target=run_server, args=(web.AppRunner(appasync),)).start()
    run_server(web.AppRunner(appasync))

    #app.on_shutdown.append(on_shutdown)
    #app.router.add_post("/offer", offer)

    # print('start websocket server')
    # server = pywsgi.WSGIServer(('0.0.0.0', 8000), app, handler_class=WebSocketHandler)
    # server.serve_forever()


# os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
# os.environ['MULTIPROCESSING_METHOD'] = 'forkserver'                                                    
if __name__ == '__main__':
    mp.set_start_method('spawn')
    load_dotenv()  # Load environment variables from .env file, if it exists
    main()
    
    
    
