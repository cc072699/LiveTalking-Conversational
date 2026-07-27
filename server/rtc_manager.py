###############################################################################
#  WebRTC 连接管理 + RTC 音频/视频接收
###############################################################################

import json
import asyncio
import random
import copy
from typing import Dict, Optional
import queue

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceServer, RTCConfiguration
from aiortc.rtcrtpsender import RTCRtpSender

from utils.logger import logger


# def _rand_session_id(n: int = 6) -> int:
#     """生成 N 位随机 session ID"""
#     return random.randint(10 ** (n - 1), 10 ** n - 1)


from server.session_manager import session_manager

class RTCManager:
    """
    WebRTC 连接管理器。
    
    管理 PeerConnection 生命周期、音视频轨道收发、DataChannel。
    """

    def __init__(self, opt):
        """
        Args:
            opt: 全局配置
        """
        self.opt = opt
        self.pcs: set = set()

    async def handle_offer(self, request):
        """处理 WebRTC offer 信令"""
        try:
            params = await request.json()
        except Exception as e:
            logger.error("offer JSON 解析失败: %s", e)
            return web.Response(
                content_type="application/json",
                text=json.dumps({"code": -1, "msg": f"invalid JSON: {e}"}),
                status=400,
            )

        if "sdp" not in params or "type" not in params:
            return web.Response(
                content_type="application/json",
                text=json.dumps({"code": -1, "msg": "missing sdp or type"}),
                status=400,
            )

        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        if False: # 不再由 RTCManager 控制 max_session，让业务逻辑或SessionManager 控制
            logger.info('reach max session')
            return web.Response(
                content_type="application/json",
                text=json.dumps({"code": -1, "msg": "reach max session"}),
            )

        #sessionid = _rand_session_id()

        # 通过 SessionManager 构建
        try:
            sessionid = await session_manager.create_session(params)
        except Exception as e:
            logger.error("session 创建失败: %s", e)
            return web.Response(
                content_type="application/json",
                text=json.dumps({"code": -1, "msg": f"session creation failed: {e}"}),
                status=500,
            )

        logger.info('offer sessionid=%s', sessionid)
        avatar_session = session_manager.get_session(sessionid)
        if avatar_session is None:
            logger.error("avatar session 为 None, sessionid=%s", sessionid)
            session_manager.remove_session(sessionid)
            return web.Response(
                content_type="application/json",
                text=json.dumps({"code": -1, "msg": "avatar session not ready"}),
                status=503,
            )

        # 创建 PeerConnection
        stun_url = getattr(self.opt, 'stun_server', 'stun:stun.qq.com:3478')
        ice_server = RTCIceServer(urls=stun_url)
        pc = RTCPeerConnection(
            configuration=RTCConfiguration(iceServers=[ice_server])
        )
        self.pcs.add(pc)

        _cleanup_done = False

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            nonlocal _cleanup_done
            logger.info("Connection state is %s, session=%s", pc.connectionState, sessionid)
            if _cleanup_done:
                return
            if pc.connectionState in ("failed", "closed"):
                _cleanup_done = True
                await pc.close()
                self.pcs.discard(pc)
                session_manager.remove_session(sessionid)
                logger.info("Session %s cleaned up (state=%s)", sessionid, pc.connectionState)
            elif pc.connectionState == "disconnected":
                # 用户关闭浏览器时 ICE 可能停在 disconnected，永不转 failed
                async def _delayed_cleanup():
                    await asyncio.sleep(15)
                    if not _cleanup_done:
                        _cleanup_done = True
                        await pc.close()
                        self.pcs.discard(pc)
                        session_manager.remove_session(sessionid)
                        logger.info("Session %s cleaned up (disconnected timeout)", sessionid)
                asyncio.create_task(_delayed_cleanup())

        # 添加发送轨道
        from server.webrtc import HumanPlayer
        player = HumanPlayer(avatar_session)
        pc.addTrack(player.audio)
        pc.addTrack(player.video)

        # 设置编解码器偏好
        capabilities = RTCRtpSender.getCapabilities("video")
        preferences = list(filter(lambda x: x.name == "H264", capabilities.codecs))
        preferences += list(filter(lambda x: x.name == "VP8", capabilities.codecs))
        preferences += list(filter(lambda x: x.name == "rtx", capabilities.codecs))
        transceiver = pc.getTransceivers()[1]
        transceiver.setCodecPreferences(preferences)

        try:
            await pc.setRemoteDescription(offer)
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
        except Exception as e:
            logger.error("WebRTC SDP 协商失败: %s", e)
            await pc.close()
            self.pcs.discard(pc)
            session_manager.remove_session(sessionid)
            return web.Response(
                content_type="application/json",
                text=json.dumps({"code": -1, "msg": f"SDP negotiation failed: {e}"}),
                status=500,
            )

        return web.Response(
            content_type="application/json",
            text=json.dumps({
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type,
                "sessionid": sessionid,
            }),
        )

    async def handle_rtcpush(self, push_url, sessionid: str):
        """RTCPush 模式：主动推流"""
        import aiohttp
        await session_manager.create_session({}, sessionid)
        avatar_session = session_manager.get_session(sessionid)

        pc = RTCPeerConnection()
        self.pcs.add(pc)

        _cleanup_done = False

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            nonlocal _cleanup_done
            logger.info("RTCPush Connection state is %s, session=%s", pc.connectionState, sessionid)
            if _cleanup_done:
                return
            if pc.connectionState in ("failed", "closed"):
                _cleanup_done = True
                await pc.close()
                self.pcs.discard(pc)
                session_manager.remove_session(sessionid)
                logger.info("RTCPush session %s cleaned up (state=%s)", sessionid, pc.connectionState)
            elif pc.connectionState == "disconnected":
                async def _delayed_cleanup():
                    await asyncio.sleep(15)
                    if not _cleanup_done:
                        _cleanup_done = True
                        await pc.close()
                        self.pcs.discard(pc)
                        session_manager.remove_session(sessionid)
                        logger.info("RTCPush session %s cleaned up (disconnected timeout)", sessionid)
                asyncio.create_task(_delayed_cleanup())

        from server.webrtc import HumanPlayer
        player = HumanPlayer(avatar_session)
        pc.addTrack(player.audio)
        pc.addTrack(player.video)

        await pc.setLocalDescription(await pc.createOffer())

        async with aiohttp.ClientSession() as session:
            async with session.post(push_url, data=pc.localDescription.sdp) as response:
                answer_sdp = await response.text()

        await pc.setRemoteDescription(
            RTCSessionDescription(sdp=answer_sdp, type='answer')
        )

    async def shutdown(self):
        """关闭所有 PeerConnection"""
        coros = [pc.close() for pc in self.pcs]
        await asyncio.gather(*coros, return_exceptions=True)
        self.pcs.clear()
