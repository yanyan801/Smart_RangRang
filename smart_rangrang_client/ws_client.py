"""
Pi 端 WebSocket 客户端
- 自动重连（指数退避）
- 双向数据管道：采集 → 服务器 | 服务器 → 播放
- 心跳监控
- 中断处理（stop_playback）
"""

import asyncio
import json
import logging
import signal
import time
from typing import Optional

import websockets
from websockets.exceptions import ConnectionClosed

logger = logging.getLogger("pi.client")

# 日志级别映射
_STATE_LABELS = {
    "IDLE": "待机",
    "LISTEN": "聆听",
    "THINK": "思考",
    "SPEAK": "播报",
}


class PiClient:
    """树莓派 WebSocket 客户端"""

    def __init__(self, ws_url: str,
                 capture: "AudioCapture",
                 player: "AudioPlayer",
                 heartbeat_interval: float = 3.0,
                 heartbeat_timeout: float = 10.0,
                 reconnect_base: float = 1.0,
                 reconnect_max: float = 30.0,
                 reconnect_backoff: float = 2.0):
        self.ws_url = ws_url
        self.capture = capture
        self.player = player
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_timeout = heartbeat_timeout
        self.reconnect_base = reconnect_base
        self.reconnect_max = reconnect_max
        self.reconnect_backoff = reconnect_backoff

        self._ws = None
        self._running = False
        self._shutdown: asyncio.Event = None  # type: ignore
        self._reconnect_count = 0
        self._last_heartbeat = 0.0
        self._tts_queue: Optional[asyncio.Queue] = None
        self._cap_queue: Optional[asyncio.Queue] = None

        # 状态展示
        self._server_state = "IDLE"
        self._last_asr = ""
        self._last_llm = ""

    # ==================== 主入口 ====================

    async def run(self):
        """启动客户端（阻塞）"""
        self._running = True
        self._shutdown = asyncio.Event()
        logger.info(f"PiClient starting, server={self.ws_url}")

        # 注册信号处理
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._signal_handler)
            except NotImplementedError:
                pass  # Windows 不支持 add_signal_handler

        # 启动采集和播放
        self._cap_queue = self.capture.start()
        self._tts_queue = self.player.start()

        while self._running:
            try:
                await self._connect_and_run()
            except Exception as e:
                if not self._running:
                    break
                delay = self._reconnect_delay()
                logger.warning(f"Connection lost: {e}. Reconnecting in {delay:.1f}s...")
                await asyncio.sleep(delay)

        await self._cleanup()

    def _signal_handler(self):
        """SIGINT/SIGTERM → 优雅退出"""
        logger.info("Shutdown signal received")
        self._running = False
        self._shutdown.set()

    async def shutdown(self):
        """外部调用关闭"""
        self._running = False
        self._shutdown.set()
        await self._cleanup()

    # ==================== 连接管理 ====================

    async def _connect_and_run(self):
        """建立 WebSocket 连接并运行消息循环"""
        async with websockets.connect(
            self.ws_url,
            max_size=10 * 1024 * 1024,
            ping_interval=None,        # 我们自己管理心跳
            close_timeout=3,
        ) as ws:
            self._ws = ws
            self._reconnect_count = 0
            self._last_heartbeat = time.time()
            logger.info(f"Connected to {self.ws_url}")

            # 并发运行四个任务
            sender = asyncio.create_task(self._sender())
            receiver = asyncio.create_task(self._receiver())
            heartbeat = asyncio.create_task(self._heartbeat_checker())
            shutdown_watcher = asyncio.create_task(self._shutdown_watcher())

            done, pending = await asyncio.wait(
                [sender, receiver, heartbeat, shutdown_watcher],
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            # 如果某个任务异常退出，传播异常
            for task in done:
                exc = task.exception()
                if exc and self._running:
                    raise exc

    def _reconnect_delay(self) -> float:
        """指数退避延迟"""
        self._reconnect_count += 1
        delay = self.reconnect_base * (self.reconnect_backoff ** (self._reconnect_count - 1))
        return min(delay, self.reconnect_max)

    # ==================== 数据管道 ====================

    async def _sender(self):
        """采集 → WebSocket"""
        while self._running:
            try:
                chunk = await asyncio.wait_for(self._cap_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue

            try:
                await self._ws.send(chunk)
            except ConnectionClosed:
                logger.info("Sender: connection closed")
                break
            except Exception as e:
                logger.error(f"Send error: {e}")
                break

    async def _receiver(self):
        """WebSocket → 分发（TTS 音频 / JSON 消息）"""
        while self._running:
            try:
                msg = await self._ws.recv()
            except ConnectionClosed as e:
                logger.info(f"Receiver: {e}")
                break

            if isinstance(msg, bytes):
                # TTS 音频帧 → 播放队列
                try:
                    self._tts_queue.put_nowait(msg)
                except asyncio.QueueFull:
                    pass  # 播放队列满，丢弃

            elif isinstance(msg, str):
                try:
                    data = json.loads(msg)
                except json.JSONDecodeError:
                    continue
                await self._handle_message(data)

    # ==================== 消息处理 ====================

    async def _handle_message(self, data: dict):
        msg_type = data.get("type", "")

        if msg_type == "asr_result":
            self._last_asr = data.get("text", "")
            logger.info(f"  ASR: {self._last_asr} ({data.get('time_s', 0):.1f}s)")

        elif msg_type == "llm_result":
            self._last_llm = data.get("text", "")
            ttft = data.get("ttft_ms", 0)
            logger.info(f"  LLM: {self._last_llm[:60]}... (TTFT={ttft}ms)")

        elif msg_type == "tts_end":
            logger.debug("  TTS done")

        elif msg_type == "stop_playback":
            logger.info("  >>> INTERRUPT <<<")
            await self.player.flush_and_restart()
            self._tts_queue = self.player._queue

        elif msg_type == "ping":
            # 服务器心跳 → 回复 pong
            self._last_heartbeat = time.time()
            try:
                await self._ws.send(json.dumps({"type": "ping"}))
            except ConnectionClosed:
                pass

        elif msg_type == "pong":
            self._last_heartbeat = time.time()

        elif msg_type == "state":
            state = data.get("state", "")
            label = _STATE_LABELS.get(state, state)
            if data.get("message"):
                logger.warning(f"  State: {label} — {data['message']}")
            else:
                logger.debug(f"  State: {label}")

        else:
            logger.debug(f"  msg: {msg_type}")

    # ==================== 心跳 / 关闭 ====================

    async def _heartbeat_checker(self):
        """检测心跳超时，超时则断开触发重连"""
        while self._running:
            await asyncio.sleep(self.heartbeat_interval)
            gap = time.time() - self._last_heartbeat
            if gap > self.heartbeat_timeout:
                logger.warning(f"Heartbeat lost ({gap:.0f}s), disconnecting...")
                try:
                    await self._ws.close()
                except Exception:
                    pass
                break

    async def _shutdown_watcher(self):
        """等待 shutdown 信号 → 关闭 WebSocket 连接"""
        await self._shutdown.wait()
        logger.debug("Shutdown watcher triggered")
        try:
            await self._ws.close()
        except Exception:
            pass

    # ==================== 清理 ====================

    async def _cleanup(self):
        self._running = False
        await self.capture.stop()
        await self.player.close()
        logger.info("PiClient stopped")
