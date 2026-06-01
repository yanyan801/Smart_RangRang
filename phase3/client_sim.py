"""
Phase 2 - CLI 客户端模拟器
模拟树莓派端：从麦克风采集音频 → WebSocket 发送 → 接收 TTS 音频 → 本地播放

用法:
    python client_sim.py                          # 默认连接 localhost:8765
    python client_sim.py ws://192.168.1.100:8765  # 指定服务器地址
    python client_sim.py --wav test.wav           # 从 WAV 文件模拟（不需要麦克风）
"""

import asyncio
import json
import logging
import sys
import wave
from pathlib import Path

import numpy as np
import pyaudio
import websockets

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("client")

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_MS = 40
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_MS / 1000)  # 640


class AudioPlayer:
    """可中断的音频播放器"""

    def __init__(self):
        self._p = pyaudio.PyAudio()
        self._stream: pyaudio.Stream | None = None

    def start(self):
        self._stream = self._p.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            output=True,
            frames_per_buffer=CHUNK_SAMPLES,
        )

    def play_chunk(self, data: bytes):
        """播放一帧 PCM 音频"""
        if self._stream:
            try:
                self._stream.write(data, exception_on_underflow=False)
            except Exception:
                pass

    def stop(self):
        """立即停止播放"""
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None

    def close(self):
        self.stop()
        self._p.terminate()


async def run_microphone(ws_url: str):
    """麦克风模式：实时采集 → WebSocket → 播放 TTS"""
    p = pyaudio.PyAudio()

    # 列出设备
    logger.info("可用输入设备:")
    for i in range(p.get_device_count()):
        info = p.get_device_info(i)
        if info["maxInputChannels"] > 0:
            logger.info(f"  [{i}] {info['name']}")

    # 查找输入设备
    input_idx = None
    for i in range(p.get_device_count()):
        if p.get_device_info(i)["maxInputChannels"] > 0:
            input_idx = i
            break

    if input_idx is None:
        logger.error("未找到输入设备")
        return

    logger.info(f"使用输入设备 [{input_idx}]: {p.get_device_info(input_idx)['name']}")

    stream = p.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        input_device_index=input_idx,
        frames_per_buffer=CHUNK_SAMPLES,
    )

    player = AudioPlayer()
    player.start()

    logger.info(f"连接服务器: {ws_url}")
    async with websockets.connect(ws_url, max_size=10 * 1024 * 1024) as ws:
        logger.info("已连接。开始说话... (Ctrl+C 退出)")

        async def receiver():
            """接收 TTS 音频并播放"""
            while True:
                try:
                    msg = await ws.recv()
                except websockets.ConnectionClosed:
                    break

                if isinstance(msg, bytes):
                    # TTS 音频帧 → 播放
                    player.play_chunk(msg)
                else:
                    data = json.loads(msg)
                    msg_type = data.get("type", "")
                    if msg_type == "asr_result":
                        logger.info(f"  ASR: {data['text']} ({data.get('time_s', 0):.1f}s)")
                    elif msg_type == "llm_result":
                        logger.info(f"  LLM: {data['text']} (TTFT: {data.get('ttft_ms', 0)}ms)")
                    elif msg_type == "tts_end":
                        logger.info("  TTS 播放完成")
                    elif msg_type == "stop_playback":
                        logger.info("  打断! 停止播放")
                        player.stop()
                        player.start()
                    elif msg_type in ("state", "pong"):
                        pass
                    else:
                        logger.debug(f"  msg: {msg_type}")

        recv_task = asyncio.create_task(receiver())

        try:
            while True:
                # 读取麦克风 → 发送到服务器
                data = await asyncio.to_thread(stream.read, CHUNK_SAMPLES, exception_on_overflow=False)
                await ws.send(data)
                await asyncio.sleep(0)  # yield
        except KeyboardInterrupt:
            logger.info("中断")
        finally:
            recv_task.cancel()
            try:
                await recv_task
            except asyncio.CancelledError:
                pass

    stream.stop_stream()
    stream.close()
    player.close()
    p.terminate()


async def run_wav_file(ws_url: str, wav_path: str):
    """WAV 文件模拟模式：逐帧发送 WAV 文件"""
    wf = wave.open(wav_path, "rb")
    assert wf.getnchannels() == 1, "WAV 必须是单声道"
    assert wf.getframerate() == SAMPLE_RATE, f"WAV 采样率必须是 {SAMPLE_RATE}Hz"

    total_frames = wf.getnframes()
    logger.info(f"WAV 文件: {wav_path} ({total_frames/SAMPLE_RATE:.1f}s)")

    async with websockets.connect(ws_url, max_size=10 * 1024 * 1024) as ws:
        logger.info("已连接。开始发送...")

        async def receiver():
            while True:
                try:
                    msg = await ws.recv()
                except websockets.ConnectionClosed:
                    break
                if isinstance(msg, bytes):
                    pass  # TTS 音频，CLI 模式下不播放
                else:
                    data = json.loads(msg)
                    msg_type = data.get("type", "")
                    if msg_type == "asr_result":
                        logger.info(f"  ASR: {data['text']} ({data.get('time_s', 0):.1f}s)")
                    elif msg_type == "llm_result":
                        logger.info(f"  LLM: {data['text']} (TTFT: {data.get('ttft_ms', 0)}ms)")
                    elif msg_type == "tts_end":
                        logger.info("  TTS 完成")
                    elif msg_type == "stop_playback":
                        logger.info("  打断!")

        recv_task = asyncio.create_task(receiver())

        try:
            while True:
                frames = wf.readframes(CHUNK_SAMPLES)
                if not frames:
                    break
                await ws.send(frames)
                # 模拟实时速率: 40ms 间隔
                await asyncio.sleep(CHUNK_MS / 1000)
            # WAV发送完毕，发送结束标记
            await ws.send(json.dumps({"type": "wav_done"}))
            logger.info("WAV 发送完毕，等待服务器响应...")
            # 继续等待 TTS 结果（最多30秒）
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            pass
        finally:
            recv_task.cancel()
            try:
                await recv_task
            except asyncio.CancelledError:
                pass

    wf.close()
    logger.info("WAV 发送完毕")


async def main():
    ws_url = "ws://localhost:8765/ws"
    wav_file = None

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--wav" and i + 1 < len(args):
            wav_file = args[i + 1]
            i += 1
        elif args[i].startswith("ws://") or args[i].startswith("wss://"):
            ws_url = args[i]
        i += 1

    if wav_file:
        await run_wav_file(ws_url, wav_file)
    else:
        await run_microphone(ws_url)


if __name__ == "__main__":
    asyncio.run(main())
