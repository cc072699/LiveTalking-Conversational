"""
DashScope TTS 代理 — 兼容 OmniTTS API 协议

提供与 OmniTTS/vLLM 兼容的端点，后端使用阿里云 DashScope tts_v2 API：
  - GET  /v1/audio/voices      — 列出预设音色 + 克隆音色
  - POST /v1/audio/voices      — 上传音频克隆语音
  - DELETE /v1/audio/voices/{name} — 删除克隆语音
  - POST /v1/audio/speech      — 语音合成

克隆流程：
  1. 用户上传音频文件 → 保存到 data/cloned_voices/
  2. 上传到临时文件托管服务获取公网 URL
  3. 调用 DashScope VoiceEnrollmentService.create_voice()
  4. 将 voice_id 映射保存到本地 JSON
  5. 后续合成时用 voice_id 调用 DashScope tts_v2 SpeechSynthesizer
"""

import json
import os
import re
import time
import uuid
import io
import asyncio
import tempfile
from pathlib import Path
from aiohttp import web, multipart
from utils.logger import logger

# ─── 路径 ───────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CLONE_DIR = DATA_DIR / "cloned_voices"
META_FILE = DATA_DIR / "cloned_voices_meta.json"
CLONE_DIR.mkdir(parents=True, exist_ok=True)

# ─── 预设音色列表 (DashScope CosyVoice) ──────────────────────
PRESET_VOICES = [
    "longxiaochun",  # 中文女声
    "longhua",       # 中文女声
    "longyi",        # 中文男声
    "longya",        # 中文男声
    "longmei",       # 中文女声
    "longyu",        # 中文女声
    "stella",        # 英文女声
    "ava",           # 英文女声
    "clara",         # 英文女声
]

# ─── 元数据管理 ───────────────────────────────────────────────

def _load_meta() -> dict:
    if META_FILE.exists():
        try:
            return json.loads(META_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"voices": {}}


def _save_meta(meta: dict):
    META_FILE.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _get_dashscope_api_key(request) -> str:
    """从请求或环境变量获取 DashScope API Key"""
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        raise web.HTTPBadRequest(text=json.dumps({"error": "DASHSCOPE_API_KEY 未配置"}),
                                 content_type="application/json")
    return api_key


# ─── 临时文件上传服务 ─────────────────────────────────────────

async def _upload_to_temp_hosting(file_path: str) -> str:
    """将文件上传到临时文件托管服务，返回公网 URL"""
    # 使用 subprocess 调用 curl 上传
    import subprocess
    import shlex

    # 尝试多个免费文件托管服务
    upload_commands = [
        # 方案1: curl -F 到 tmpfiles.org
        ["curl", "-s", "-F", f"file=@{file_path}", "https://tmpfiles.org/api/v1/upload"],
        # 方案2: curl -F 到 file.io
        ["curl", "-s", "-F", f"file=@{file_path}", "https://file.io"],
        # 方案3: curl --upload-file 到 temp.sh (类似 transfer.sh)
        ["curl", "-s", "--upload-file", file_path, "https://temp.sh/upload"],
    ]

    last_error = None
    for cmd in upload_commands:
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            if proc.returncode != 0:
                last_error = f"{cmd[2]} exit={proc.returncode}: {stderr.decode()[:100]}"
                continue

            text = stdout.decode().strip()

            # 解析 tmpfiles.org 响应
            if "tmpfiles.org" in cmd[2]:
                import json
                try:
                    result = json.loads(text)
                    if result.get("status") == "success":
                        dl_url = result["data"]["url"]
                        direct_url = dl_url.replace("tmpfiles.org/", "tmpfiles.org/dl/")
                        logger.info(f"Temp hosting URL: {direct_url}")
                        return direct_url
                except (json.JSONDecodeError, KeyError):
                    pass

            # 解析 file.io 响应
            elif "file.io" in cmd[2]:
                import json
                try:
                    result = json.loads(text)
                    if result.get("success"):
                        logger.info(f"Temp hosting URL: {result['link']}")
                        return result["link"]
                except (json.JSONDecodeError, KeyError):
                    pass

            # temp.sh 直接返回 URL
            elif "temp.sh" in cmd[2]:
                url = text.strip()
                if url.startswith("http"):
                    logger.info(f"Temp hosting URL: {url}")
                    return url

        except asyncio.TimeoutError:
            last_error = f"{cmd[2]} timeout"
            continue
        except Exception as e:
            last_error = f"{cmd[2]}: {e}"
            continue

    raise RuntimeError(f"所有临时文件托管服务均失败: {last_error}")


# ─── DashScope API 调用 ──────────────────────────────────────

def _dashscope_create_voice(api_key: str, audio_url: str, prefix: str) -> str:
    """
    调用 DashScope VoiceEnrollmentService 创建克隆音色
    返回 voice_id
    """
    from dashscope.audio.tts_v2 import VoiceEnrollmentService
    import dashscope
    dashscope.api_key = api_key

    service = VoiceEnrollmentService(model="cosyvoice-v1")

    # 确保前缀只包含小写字母和数字
    import re
    clean_prefix = re.sub(r"[^a-z0-9]", "", prefix.lower())[:10] or f"voice{uuid.uuid4().hex[:8]}"

    voice_id = service.create_voice(
        target_model="cosyvoice-v1",
        prefix=clean_prefix,
        url=audio_url,
        language_hints=["zh"],
        max_prompt_audio_length=30.0,
    )
    return voice_id


def _dashscope_delete_voice(api_key: str, voice_id: str):
    """删除 DashScope 上的克隆音色"""
    import dashscope
    dashscope.api_key = api_key
    from dashscope.audio.tts_v2 import VoiceEnrollmentService
    service = VoiceEnrollmentService(model="cosyvoice-v1")
    service.delete_voice(voice_id)


def _dashscope_list_voices(api_key: str) -> list:
    """列出 DashScope 上注册的克隆音色"""
    import dashscope
    dashscope.api_key = api_key
    from dashscope.audio.tts_v2 import VoiceEnrollmentService
    try:
        service = VoiceEnrollmentService(model="cosyvoice-v1")
        return service.list_voices()
    except Exception as e:
        logger.warning(f"List cloned voices failed: {e}")
        return []


def _dashscope_synthesize(api_key: str, text: str, voice: str, speed: float = 1.0) -> bytes:
    """
    调用 DashScope tts_v2 SpeechSynthesizer 合成语音
    返回 PCM 16-bit 16kHz mono 音频 bytes
    """
    import dashscope
    dashscope.api_key = api_key

    from dashscope.audio.tts_v2 import SpeechSynthesizer, AudioFormat

    try:
        synthesizer = SpeechSynthesizer(
            model="cosyvoice-v1",
            voice=voice,
            format=AudioFormat.PCM_16000HZ_MONO_16BIT,
        )
        audio = synthesizer.call(text)
        if audio:
            return audio  # bytes
        else:
            raise RuntimeError("SpeechSynthesizer returned empty audio")
    except Exception as e:
        logger.exception(f"DashScope TTS synthesis failed: {e}")
        raise


# ─── API 路由处理函数 ────────────────────────────────────────

async def list_voices(request):
    """GET /v1/audio/voices — 列出所有可用音色"""
    try:
        api_key = _get_dashscope_api_key(request)
        meta = _load_meta()

        # 获取 DashScope 上注册的克隆音色
        dashscope_voices = _dashscope_list_voices(api_key)
        cloned_from_dashscope = {}
        for v in dashscope_voices:
            vid = v.get("voice_id", "")
            cloned_from_dashscope[vid] = v

        # 合并本地元数据中的克隆音色
        uploaded_voices = []
        for name, info in meta.get("voices", {}).items():
            vid = info.get("voice_id", "")
            status = "ready"
            if vid in cloned_from_dashscope:
                ds_status = cloned_from_dashscope[vid].get("status", "ready")
                if ds_status:
                    status = ds_status
            uploaded_voices.append({
                "name": name,
                "voice_id": vid,
                "status": status,
                "ref_text": info.get("ref_text", ""),
                "speaker_description": info.get("speaker_description", ""),
                "created_at": info.get("created_at", 0),
            })

        return web.json_response({
            "voices": PRESET_VOICES,
            "uploaded_voices": uploaded_voices,
        })
    except web.HTTPException:
        raise
    except Exception as e:
        logger.exception("list_voices error")
        return web.json_response({"error": str(e)}, status=500)


async def delete_voice(request):
    """DELETE /v1/audio/voices/{name} — 删除克隆音色"""
    try:
        api_key = _get_dashscope_api_key(request)
        name = request.match_info.get("name", "")

        meta = _load_meta()
        if name not in meta.get("voices", {}):
            return web.json_response({"error": f"Voice '{name}' not found"}, status=404)

        voice_id = meta["voices"][name].get("voice_id", "")
        if voice_id:
            try:
                _dashscope_delete_voice(api_key, voice_id)
            except Exception as e:
                logger.warning(f"Delete voice from DashScope failed: {e}")

        del meta["voices"][name]
        _save_meta(meta)

        return web.json_response({"message": f"Voice '{name}' deleted"})
    except web.HTTPException:
        raise
    except Exception as e:
        logger.exception("delete_voice error")
        return web.json_response({"error": str(e)}, status=500)


async def upload_voice(request):
    """POST /v1/audio/voices — 上传音频克隆语音"""
    try:
        api_key = _get_dashscope_api_key(request)

        # 解析 multipart 表单
        reader = await request.multipart()

        fields = {}
        audio_data = None
        audio_filename = None

        while True:
            part = await reader.next()
            if part is None:
                break

            if part.name == "audio_sample":
                audio_filename = part.filename or f"upload_{uuid.uuid4().hex}.wav"
                audio_data = await part.read()
            else:
                value = await part.text()
                fields[part.name] = value

        if audio_data is None:
            return web.json_response({"error": "缺少音频文件 (audio_sample)"}, status=400)

        name = fields.get("name", "").strip()
        if not name:
            return web.json_response({"error": "缺少语音名称 (name)"}, status=400)

        ref_text = fields.get("ref_text", "")
        speaker_description = fields.get("speaker_description", "")

        # 确保文件名有扩展名
        if not audio_filename:
            audio_filename = f"{name}.wav"
        if "." not in audio_filename:
            audio_filename += ".wav"

        # 保存到本地（确保路径无空格，便于后续上传）
        safe_name = re.sub(r'[^a-zA-Z0-9._-]', '_', audio_filename)
        local_path = CLONE_DIR / safe_name
        with open(local_path, "wb") as f:
            f.write(audio_data)
        logger.info(f"Saved uploaded audio: {local_path} ({len(audio_data)} bytes)")

        # 复制到无空格临时路径再上传
        import tempfile, shutil
        tmp_path = os.path.join(tempfile.gettempdir(), f"dashscope_clone_{int(time.time())}.wav")
        shutil.copy2(local_path, tmp_path)

        # 上传到临时托管获取公网 URL
        logger.info(f"Uploading to temp hosting for DashScope access...")
        public_url = None
        try:
            public_url = await _upload_to_temp_hosting(tmp_path)
        except Exception as e:
            logger.warning(f"Temp hosting upload failed: {e}")
        finally:
            # 清理临时文件
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        voice_id = ""
        if public_url:
            try:
                logger.info(f"Calling DashScope create_voice with URL: {public_url}")
                voice_id = _dashscope_create_voice(api_key, public_url, name)
                logger.info(f"DashScope voice created: voice_id={voice_id}")
            except Exception as e:
                logger.warning(f"DashScope create_voice failed: {e}")
                voice_id = ""

        if not voice_id:
            # 克隆失败，仅保存本地文件，返回提示
            meta = _load_meta()
            meta["voices"][name] = {
                "voice_id": "",
                "ref_text": ref_text,
                "speaker_description": speaker_description,
                "local_file": str(local_path),
                "created_at": int(time.time()),
                "status": "requires_oss_upload"
            }
            _save_meta(meta)

            return web.json_response({
                "message": f"语音 '{name}' 已保存到本地，但 DashScope 克隆需要将音频上传到阿里云 OSS 或可访问的公网 URL。",
                "name": name,
                "voice_id": "",
                "local_file": str(local_path),
                "hint": "请将音频上传到阿里云 OSS，然后调用 API 注册: POST /v1/audio/voices/register"
            }, status=202)

        # 克隆成功，保存元数据
        meta = _load_meta()
        meta["voices"][name] = {
            "voice_id": voice_id,
            "ref_text": ref_text,
            "speaker_description": speaker_description,
            "local_file": str(local_path),
            "created_at": int(time.time()),
            "status": "ready",
        }
        _save_meta(meta)

        return web.json_response({
            "message": f"Voice '{name}' created successfully",
            "name": name,
            "voice_id": voice_id,
        })

    except web.HTTPException:
        raise
    except Exception as e:
        logger.exception("upload_voice error")
        return web.json_response({"error": str(e)}, status=500)


async def speech_synthesize(request):
    """POST /v1/audio/speech — 语音合成"""
    try:
        api_key = _get_dashscope_api_key(request)
        params = await request.json()

        text = params.get("input", "")
        if not text:
            return web.json_response({"error": "缺少 input 文本"}, status=400)

        voice = params.get("voice", "longxiaochun")
        speed = float(params.get("speed", 1.0))
        response_format = params.get("response_format", "pcm")

        # 如果是克隆音色名称，查找 voice_id
        meta = _load_meta()
        if voice in meta.get("voices", {}):
            voice_id = meta["voices"][voice].get("voice_id", voice)
            logger.info(f"Using cloned voice: {voice} -> {voice_id}")
        else:
            voice_id = voice  # 直接使用预设音色名或 voice_id

        # 调用 DashScope 合成
        audio_data = _dashscope_synthesize(api_key, text, voice_id, speed)

        # 响应格式转换
        if response_format in ("wav", "pcm"):
            # DashScope 返回 PCM 16-bit 16kHz mono
            import wave
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(audio_data)
            wav_data = buf.getvalue()

            content_type = "audio/wav" if response_format == "wav" else "audio/pcm"
            return web.Response(body=wav_data, content_type=content_type)
        elif response_format == "mp3":
            # PCM → MP3 转换 (需要 ffmpeg)
            import subprocess
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-y", "-f", "s16le", "-ar", "16000", "-ac", "1",
                "-i", "pipe:0", "-f", "mp3", "pipe:1",
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            mp3_data, _ = await proc.communicate(audio_data)
            return web.Response(body=mp3_data, content_type="audio/mpeg")
        else:
            # 默认返回 WAV
            import wave
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(audio_data)
            return web.Response(body=buf.getvalue(), content_type="audio/wav")

    except web.HTTPException:
        raise
    except Exception as e:
        logger.exception("speech_synthesize error")
        return web.json_response({"error": str(e)}, status=500)


async def health_check(request):
    """GET /v1/audio/health — 健康检查"""
    api_key = _get_dashscope_api_key(request)
    return web.json_response({
        "status": "ok",
        "provider": "dashscope",
        "model": "cosyvoice-v1",
        "dashscope_configured": bool(api_key),
    })


# ─── 路由注册 ────────────────────────────────────────────────

def setup_tts_proxy_routes(app: web.Application):
    """注册 TTS 代理路由到 aiohttp app"""
    app.router.add_get("/v1/audio/voices", list_voices)
    app.router.add_post("/v1/audio/voices", upload_voice)
    app.router.add_delete("/v1/audio/voices/{name}", delete_voice)
    app.router.add_post("/v1/audio/speech", speech_synthesize)
    app.router.add_get("/v1/audio/health", health_check)

    logger.info("[TTS Proxy] ✅ DashScope TTS proxy routes registered")
    logger.info("[TTS Proxy]   GET  /v1/audio/voices       — 列出音色")
    logger.info("[TTS Proxy]   POST /v1/audio/voices       — 上传克隆语音")
    logger.info("[TTS Proxy]   DELETE /v1/audio/voices/{name} — 删除克隆语音")
    logger.info("[TTS Proxy]   POST /v1/audio/speech       — 语音合成")
    logger.info("[TTS Proxy]   GET  /v1/audio/health       — 健康检查")