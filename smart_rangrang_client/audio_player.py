"""
Pi 端音频播放模块
- 中断安全：收到 stop 信号立即清空缓冲并停止播放
- 异步安全：阻塞 PyAudio 写在线程池中执行
- 自适应缓冲：积累若干 chunk 再开始播放，减少 underflow
"""

import asyncio
import logging
from typing import Optional

logger = logging.getLogger("pi.player")


class AudioPlayer:
    """可立即中断的音频播放器"""

    def __init__(self, sample_rate: int = 16000, chunk_samples: int = 640,
                 buffer_chunks: int = 4):
        self.sample_rate = sample_rate
        self.chunk_samples = chunk_samples
        self.buffer_chunks = buffer_chunks
        self._p = None
        self._stream = None
        self._running = False
        self._flushed = asyncio.Event()
        self._flushed.set()

    # ---------- 生命周期 ----------

    def start(self) -> asyncio.Queue:
        """启动播放器，返回音频数据消费队列"""
        import pyaudio

        self._p = pyaudio.PyAudio()
        self._stream = self._p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            output=True,
            frames_per_buffer=self.chunk_samples,
        )
        self._running = True
        self._flushed.set()

        self._queue: asyncio.Queue = asyncio.Queue(maxsize=256)
        self._task = asyncio.create_task(self._run_loop())
        logger.info(f"Player started: rate={self.sample_rate}")
        return self._queue

    async def _run_loop(self):
        """在 executor 中写 PyAudio，从 asyncio.Queue 消费"""
        loop = asyncio.get_event_loop()
        buf: list[bytes] = []  # 播放缓冲

        while self._running or buf:
            if not self._running and not buf:
                break

            try:
                # 收集足够 chunk 再开始播放
                while len(buf) < self.buffer_chunks:
                    try:
                        chunk = await asyncio.wait_for(self._queue.get(), timeout=0.05)
                        buf.append(chunk)
                    except asyncio.TimeoutError:
                        break

                if buf:
                    data = buf.pop(0)
                elif self._running:
                    # 缓冲区空 → 播放静音防止 underflow
                    data = b'\x00' * (self.chunk_samples * 2)
                else:
                    break

                await loop.run_in_executor(
                    None,
                    lambda d=data: self._stream.write(d, exception_on_underflow=False),
                )
            except Exception as e:
                if self._running:
                    logger.error(f"Playback error: {e}")
                break

        self._flushed.set()
        logger.debug("Player loop ended")

    # ---------- 中断控制 ----------

    def stop(self):
        """立即停止播放（不等待缓冲排空）"""
        logger.info("Player stop requested")
        self._running = False
        # 排空队列
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def flush_and_restart(self):
        """排空当前播放后重新开始（用于打断后的快速恢复）"""
        self.stop()
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=0.5)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
        self._close_stream()
        self.start()
        logger.info("Player restarted after flush")

    # ---------- 清理 ----------

    def _close_stream(self):
        if self._stream:
            try:
                self._stream.stop_stream()
            except Exception:
                pass
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    async def close(self):
        """完全关闭播放器"""
        self.stop()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        self._close_stream()
        if self._p:
            self._p.terminate()
            self._p = None
        logger.info("Player closed")
