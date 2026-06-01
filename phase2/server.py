"""
Phase 2 - WebSocket Server: 流式管道编排
接收 Pi 端音频流 → VAD 分句 → ASR → LLM(token流) → TTS(audio流) → 返回 Pi
"""

import asyncio
import json
import logging
import struct
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

from stt_engine import ASREngine
from llm_engine import LLMEngine
from tts_engine import TTSEngine
from vad_engine import VADEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("server")

# === 全局引擎实例 ===
asr = ASREngine(device="cuda:0")
llm = LLMEngine(model="qwen2.5:14b")
tts = TTSEngine()
vad = VADEngine(
    sample_rate=16000,
    energy_threshold=0.015,   # RMS能量阈值（环境噪音 < 0.01, 语音 > 0.02）
    min_speech_frames=4,      # ~160ms 确认说话
    min_silence_frames=12,    # ~480ms 确认停止
)

# 状态
IDLE = "IDLE"
LISTEN = "LISTEN"
THINK = "THINK"
SPEAK = "SPEAK"

app = FastAPI()

# === 测试页面 ===
@app.get("/")
async def test_page():
    return HTMLResponse(_TEST_PAGE)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    logger.info("Client connected")

    state = IDLE
    interrupt_event = asyncio.Event()
    tts_task: asyncio.Task | None = None
    llm_cancel: asyncio.Event | None = None

    # 音频缓冲
    audio_buffer: list[np.ndarray] = []
    CHUNK_SAMPLES = 640   # 40ms @ 16kHz

    try:
        while True:
            data = await ws.receive()

            if "bytes" in data:
                # === 音频帧 (int16 binary, 640 samples/40ms) ===
                raw = data["bytes"]
                if len(raw) != CHUNK_SAMPLES * 2:
                    continue  # 跳过不完整帧

                audio_chunk = np.frombuffer(raw, dtype=np.int16).copy()

                if state == SPEAK:
                    # 播报中 → 检测打断（简化版: 能量阈值）
                    # 正式版需加 Echo Judge (Phase 3)
                    energy = vad.process_frame(audio_chunk)
                    if energy > 0.04:  # 较高能量 → 疑似有人说话
                        logger.info(f"INTERRUPT detected! (energy={energy:.3f})")
                        state = IDLE  # 临时

                        # 取消 TTS
                        if tts_task and not tts_task.done():
                            tts_task.cancel()
                        # 通知客户端停止播放
                        await ws.send_json({"type": "stop_playback"})
                        # 清空音频缓冲
                        audio_buffer = []
                        vad.reset()
                        # 进入 LISTEN
                        state = LISTEN
                        continue

                if state in (LISTEN, IDLE):
                    energy = vad.process_frame(audio_chunk)
                    event, speech_audio = vad.update(energy, audio_chunk)

                    if event == "speech_start" and state == IDLE:
                        state = LISTEN
                        logger.info("State: IDLE → LISTEN")

                    if event == "speech_end" and state == LISTEN:
                        # VAD 检测到说话结束 → 出句
                        # speech_audio 已经是完整语音 (int16)
                        if speech_audio is None or len(speech_audio) < 16000 * 0.3:  # < 0.3s, 太短
                            continue

                        vad.reset()

                        # → ASR
                        state = THINK
                        logger.info(f"State: LISTEN → THINK (audio={len(speech_audio)/16000:.1f}s)")

                        t_asr = time.time()
                        text = await asyncio.to_thread(asr.transcribe, speech_audio.astype(np.float32))
                        t_asr = time.time() - t_asr
                        logger.info(f"ASR [{t_asr:.2f}s]: {text}")

                        if not text.strip():
                            state = LISTEN
                            logger.info("ASR empty → LISTEN")
                            continue

                        await ws.send_json({"type": "asr_result", "text": text, "time_s": round(t_asr, 2)})

                        # → LLM (streaming)
                        t_llm_start = time.time()
                        first_token = True
                        llm_text = ""
                        t_first_token = 0.0

                        llm_cancel = asyncio.Event()

                        async def stream_llm_to_tts():
                            nonlocal first_token, llm_text, t_first_token, state
                            try:
                                async for token in llm.chat_stream(text):
                                    if llm_cancel.is_set():
                                        break
                                    if first_token:
                                        t_first_token = time.time() - t_llm_start
                                        logger.info(f"LLM first token: {t_first_token*1000:.0f}ms")
                                        first_token = False
                                    llm_text += token

                                if llm_text.strip() and not llm_cancel.is_set():
                                    # → TTS (streaming)
                                    state = SPEAK
                                    logger.info(f"State: THINK → SPEAK | LLM: {llm_text[:50]}...")
                                    await ws.send_json({
                                        "type": "llm_result",
                                        "text": llm_text,
                                        "ttft_ms": round(t_first_token * 1000),
                                    })

                                    # 流式 TTS → 逐块发送音频
                                    chunk_count = 0
                                    async for audio_bytes in tts.synthesize_stream(llm_text):
                                        if llm_cancel.is_set():
                                            break
                                        # 发送 PCM 音频帧
                                        await ws.send_bytes(audio_bytes)
                                        chunk_count += 1
                                        await asyncio.sleep(0)  # yield control

                                    await ws.send_json({"type": "tts_end"})
                                    logger.info(f"TTS done: {chunk_count} chunks")
                                    state = LISTEN
                                    logger.info("State: SPEAK → LISTEN")
                            except asyncio.CancelledError:
                                logger.info("LLM/TTS cancelled")

                        tts_task = asyncio.create_task(stream_llm_to_tts())

            elif "text" in data:
                # === JSON 控制消息 ===
                msg = json.loads(data["text"])
                msg_type = msg.get("type", "")

                if msg_type == "reset":
                    vad.reset()
                    audio_buffer = []
                    llm.reset_history()
                    if tts_task and not tts_task.done():
                        tts_task.cancel()
                    state = IDLE
                    await ws.send_json({"type": "state", "state": state})

                elif msg_type == "ping":
                    await ws.send_json({"type": "pong"})

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        if tts_task and not tts_task.done():
            tts_task.cancel()
        logger.info(f"Session ended, final state={state}")


# === 简化的测试页面 ===
_TEST_PAGE = """<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Smart RangRang - Phase 2 Test</title></head>
<body>
<h2>Phase 2 流式管道测试</h2>
<button id="start">开始录音</button>
<button id="stop" disabled>停止</button>
<span id="status">就绪</span>
<hr>
<div><b>ASR:</b> <span id="asr"></span></div>
<div><b>LLM:</b> <span id="llm"></span></div>
<div><b>状态:</b> <span id="state"></span></div>
<hr>
<div id="log"></div>

<script>
let ws, mediaStream, audioCtx, processor, source;
let isRecording = false;

document.getElementById('start').onclick = async () => {
    ws = new WebSocket('ws://' + location.host + '/ws');
    ws.binaryType = 'arraybuffer';
    ws.onopen = () => log('WebSocket 已连接');

    ws.onmessage = (e) => {
        if (e.data instanceof ArrayBuffer) {
            // TTS 音频 - 后台播放
            return;
        }
        const msg = JSON.parse(e.data);
        if (msg.type === 'asr_result') {
            document.getElementById('asr').textContent = msg.text + ' (' + msg.time_s + 's)';
            log('ASR: ' + msg.text);
        } else if (msg.type === 'llm_result') {
            document.getElementById('llm').textContent = msg.text;
            document.getElementById('state').textContent = 'SPEAK';
            log('LLM: ' + msg.text);
        } else if (msg.type === 'tts_end') {
            document.getElementById('state').textContent = 'LISTEN';
            log('TTS 播放完毕');
        } else if (msg.type === 'stop_playback') {
            document.getElementById('state').textContent = 'LISTEN';
        }
    };

    // 开始录音
    mediaStream = await navigator.mediaDevices.getUserMedia({audio: {sampleRate: 16000, channelCount: 1}});
    audioCtx = new AudioContext({sampleRate: 16000});
    source = audioCtx.createMediaStreamSource(mediaStream);
    processor = audioCtx.createScriptProcessor(640, 1, 1); // 40ms chunks

    processor.onaudioprocess = (e) => {
        if (!isRecording || ws.readyState !== WebSocket.OPEN) return;
        const input = e.inputBuffer.getChannelData(0);
        const int16 = new Int16Array(input.length);
        for (let i = 0; i < input.length; i++) {
            int16[i] = Math.max(-32768, Math.min(32767, input[i] * 32768));
        }
        ws.send(int16.buffer);
    };

    source.connect(processor);
    processor.connect(audioCtx.destination);

    isRecording = true;
    document.getElementById('start').disabled = true;
    document.getElementById('stop').disabled = false;
    document.getElementById('status').textContent = '录音中...';

    // 发送 VAD 模拟: 静音 2s 后自动触发分句
    ws.send(JSON.stringify({type: 'state', state: 'LISTEN'}));
};

document.getElementById('stop').onclick = () => {
    isRecording = false;
    if (processor) processor.disconnect();
    if (source) source.disconnect();
    if (mediaStream) mediaStream.getTracks().forEach(t => t.stop());
    ws.send(JSON.stringify({type: 'reset'}));
    document.getElementById('start').disabled = false;
    document.getElementById('stop').disabled = true;
    document.getElementById('status').textContent = '已停止';
};

function log(msg) {
    const t = new Date().toLocaleTimeString();
    const div = document.getElementById('log');
    div.innerHTML = '<div>[' + t + '] ' + msg + '</div>' + div.innerHTML.slice(0, 5000);
}
</script>
</body>
</html>"""


if __name__ == "__main__":
    logger.info("Loading models...")
    asr.load()
    vad.load()
    logger.info("All models loaded. Starting server on :8765")
    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="info")
