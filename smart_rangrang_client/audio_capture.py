"""
Pi 端音频采集模块
- 自动发现 XVF3800 / USB 麦克风
- 异步安全：阻塞 PyAudio 读在线程池中执行
- 输出 16kHz / 16bit / mono PCM chunks
"""

import asyncio
import logging
from typing import Optional

try:
    from .config import MIC_DEVICE_KEYWORDS
except ImportError:
    from config import MIC_DEVICE_KEYWORDS  # type: ignore

logger = logging.getLogger("pi.capture")


class AudioCapture:
    """异步麦克风采集器"""

    def __init__(self, sample_rate: int = 16000, chunk_samples: int = 640,
                 device_name: Optional[str] = None):
        self.sample_rate = sample_rate
        self.chunk_samples = chunk_samples
        self.device_name = device_name
        self._p = None
        self._stream = None
        self._running = False
        self._device_index: Optional[int] = None

    # ---------- 设备发现 ----------

    def _find_device(self, keywords: list[str]) -> int:
        """按关键词匹配设备名，返回设备索引"""
        import pyaudio
        p = pyaudio.PyAudio()
        count = p.get_device_count()
        for i in range(count):
            info = p.get_device_info_by_index(i)
            if info["maxInputChannels"] == 0:
                continue
            name = info["name"]
            for kw in keywords:
                if kw.lower() in name.lower():
                    p.terminate()
                    logger.info(f"Found mic device [{i}]: {name}")
                    return i
        p.terminate()
        # fallback：取第一个有输入通道的设备
        for i in range(count):
            info = p.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                p.terminate()
                logger.info(f"Fallback mic device [{i}]: {info['name']}")
                return i
        raise RuntimeError("No input device found")

    def find_device(self, keywords: Optional[list[str]] = None) -> int:
        """查找并缓存设备索引"""
        if keywords is None:
            keywords = MIC_DEVICE_KEYWORDS
        self._device_index = self._find_device(keywords)
        return self._device_index

    # ---------- 生命周期 ----------

    def start(self) -> asyncio.Queue:
        """启动采集，返回音频 chunk 队列"""
        import pyaudio

        if self._device_index is None:
            self.find_device()

        self._p = pyaudio.PyAudio()
        self._stream = self._p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            input_device_index=self._device_index,
            frames_per_buffer=self.chunk_samples,
        )
        self._running = True

        self._queue: asyncio.Queue = asyncio.Queue(maxsize=256)
        self._task = asyncio.create_task(self._run_loop())
        logger.info(
            f"Capture started: device={self._device_index}, "
            f"rate={self.sample_rate}, chunk={self.chunk_samples}"
        )
        return self._queue

    async def _run_loop(self):
        """在 executor 中读 PyAudio，chunk 放入 asyncio.Queue"""
        loop = asyncio.get_event_loop()
        while self._running:
            try:
                data = await loop.run_in_executor(
                    None,
                    self._stream.read,
                    self.chunk_samples,
                    False,  # exception_on_overflow
                )
                # 非阻塞入队；队列满则丢弃最旧 chunk
                try:
                    self._queue.put_nowait(data)
                except asyncio.QueueFull:
                    try:
                        self._queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                    self._queue.put_nowait(data)
            except Exception as e:
                if self._running:
                    logger.error(f"Capture error: {e}")
                await asyncio.sleep(0.01)

    async def stop(self):
        """停止采集"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._p:
            self._p.terminate()
            self._p = None
        logger.info("Capture stopped")

    # ---------- 设备信息 ----------

    @staticmethod
    def list_devices():
        """列出所有输入设备（调试用）"""
        import pyaudio
        p = pyaudio.PyAudio()
        count = p.get_device_count()
        for i in range(count):
            info = p.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                print(f"  [{i}] {info['name']} "
                      f"(in={info['maxInputChannels']}, "
                      f"rate={int(info['defaultSampleRate'])})")
        p.terminate()
