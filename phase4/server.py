"""
Phase 4 - WebSocket Server: 打断 + 智能断句 + 聊天记忆
集成 EchoJudge、InterruptDetector、SentenceJudge、MemoryManager
"""

import asyncio
import json
import logging
import time

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

from stt_engine import ASREngine
from llm_engine import LLMEngine
from tts_engine import TTSEngine
from vad_engine import VADEngine
from echo_judge import EchoJudge, InterruptDetector
from sentence_judge import SentenceJudge
from memory_manager import MemoryManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("server")

SAMPLE_RATE = 16000
CHUNK_MS = 40
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_MS / 1000)

# === 全局引擎实例 ===
asr = ASREngine(device="cuda:0")
llm = LLMEngine(model="qwen2.5:14b")
tts = TTSEngine()
vad = VADEngine(
    sample_rate=SAMPLE_RATE,
    energy_threshold=0.015,
    min_speech_frames=4,
    min_silence_frames=12,
)

echo_judge = EchoJudge(
    sample_rate=SAMPLE_RATE,
    corr_threshold=0.5,
    corr_weak_threshold=0.3,
    env_threshold=0.6,
    ref_buffer_seconds=2.0,
    chunk_ms=CHUNK_MS,
)

interrupt_detector = InterruptDetector(
    echo_judge=echo_judge,
    vad_consecutive_frames=10,
    vad_energy_threshold=0.04,
)

sentence_judge = SentenceJudge(
    punct_silence_ms=200,
    force_silence_ms=800,
    long_text_chars=100,
    long_text_silence_ms=300,
    incomplete_extend_ms=1500,
    pending_timeout_ms=5000,
)

memory = MemoryManager()

IDLE = "IDLE"
LISTEN = "LISTEN"
THINK = "THINK"
SPEAK = "SPEAK"

app = FastAPI()


@app.get("/")
async def test_page():
    return HTMLResponse(_TEST_PAGE)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    logger.info("Client connected")

    # === 启动会话，加载记忆 ===
    session_id = memory.start_session()
    llm.reset_history()
    history = memory.get_history_for_llm()
    if history:
        llm.load_history_from_memory(history)
    memory_ctx = memory.get_memory_context()
    if memory_ctx:
        llm.set_memory_context(memory_ctx)
        logger.info(f"Memory context injected ({len(memory_ctx)} chars)")

    session = {"state": IDLE}
    tts_task: list[asyncio.Task] = []
    llm_cancel: list[asyncio.Event] = []

    try:
        while True:
            data = await ws.receive()

            if "bytes" in data:
                raw = data["bytes"]
                if len(raw) != CHUNK_SAMPLES * 2:
                    continue

                audio_chunk = np.frombuffer(raw, dtype=np.int16).copy()
                s = session["state"]

                # === SPEAK 状态: 检测打断 ===
                if s == SPEAK:
                    energy = vad.process_frame(audio_chunk)
                    if interrupt_detector.process(audio_chunk, energy):
                        logger.info("INTERRUPT triggered — destroy and recreate")
                        session["state"] = IDLE

                        if llm_cancel:
                            llm_cancel[0].set()
                        if tts_task and not tts_task[0].done():
                            tts_task[0].cancel()

                        await ws.send_json({"type": "stop_playback"})
                        echo_judge.flush()
                        vad.reset()
                        interrupt_detector.reset()
                        sentence_judge.reset()

                        session["state"] = LISTEN
                        logger.info("State: SPEAK → LISTEN (interrupted)")
                        continue

                # === LISTEN / IDLE 状态: VAD + 断句 ===
                if s in (LISTEN, IDLE):
                    energy = vad.process_frame(audio_chunk)
                    event, speech_audio = vad.update(energy, audio_chunk)

                    if event == "speech_start" and s == IDLE:
                        session["state"] = LISTEN
                        logger.info("State: IDLE → LISTEN")

                    if event == "speech_end" and s == LISTEN:
                        if speech_audio is None or len(speech_audio) < SAMPLE_RATE * 0.3:
                            continue

                        silence_ms = vad.last_silence_ms
                        vad.reset()

                        session["state"] = THINK
                        audio_len = len(speech_audio) / SAMPLE_RATE
                        logger.info(
                            f"State: LISTEN → THINK "
                            f"(audio={audio_len:.1f}s, silence={silence_ms:.0f}ms)"
                        )

                        t_asr = time.time()
                        text = await asyncio.to_thread(
                            asr.transcribe, speech_audio.astype(np.float32)
                        )
                        t_asr = time.time() - t_asr
                        logger.info(f"ASR [{t_asr:.2f}s]: {text}")

                        if not text.strip():
                            session["state"] = LISTEN
                            logger.info("ASR empty → LISTEN")
                            continue

                        await ws.send_json({
                            "type": "asr_result",
                            "text": text,
                            "time_s": round(t_asr, 2),
                        })

                        # Sentence Judge
                        is_complete, judged_text = sentence_judge.judge(
                            text, silence_ms
                        )

                        if not is_complete:
                            logger.info(
                                f"SentenceJudge: incomplete, buffering "
                                f"'{judged_text[-20:]}'"
                            )
                            session["state"] = LISTEN
                            continue

                        # === 保存用户消息到记忆 ===
                        memory.save_turn("user", judged_text)

                        # === 更新记忆上下文（语义检索） ===
                        memory_ctx = memory.get_memory_context(judged_text)
                        if memory_ctx:
                            llm.set_memory_context(memory_ctx)

                        await _start_llm_tts(
                            ws, judged_text, llm_cancel, tts_task, session, memory
                        )

                # === 检查断句器超时 ===
                if session["state"] == LISTEN and sentence_judge.has_pending:
                    pending = sentence_judge.check_timeout()
                    if pending:
                        logger.info(f"SentenceJudge: timeout release → '{pending}'")
                        await ws.send_json({
                            "type": "asr_result",
                            "text": pending,
                            "time_s": 0,
                        })
                        memory.save_turn("user", pending)
                        await _start_llm_tts(
                            ws, pending, llm_cancel, tts_task, session, memory
                        )

            elif "text" in data:
                msg = json.loads(data["text"])
                msg_type = msg.get("type", "")

                if msg_type == "reset":
                    vad.reset()
                    llm.reset_history()
                    echo_judge.flush()
                    interrupt_detector.reset()
                    sentence_judge.reset()
                    if tts_task and not tts_task[0].done():
                        tts_task[0].cancel()

                    # 结束当前会话（生成摘要）
                    await memory.end_session(llm)

                    # 开启新会话
                    session_id = memory.start_session()
                    history = memory.get_history_for_llm()
                    if history:
                        llm.load_history_from_memory(history)
                    memory_ctx = memory.get_memory_context()
                    if memory_ctx:
                        llm.set_memory_context(memory_ctx)

                    session["state"] = IDLE
                    await ws.send_json({"type": "state", "state": session["state"]})

                elif msg_type == "ping":
                    await ws.send_json({"type": "pong"})

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        if tts_task and not tts_task[0].done():
            tts_task[0].cancel()
        # 会话结束 → 生成摘要
        await memory.end_session(llm)
        logger.info(f"Session ended, final state={session['state']}")


async def _start_llm_tts(ws, text, llm_cancel, tts_task, session, memory):
    """启动 LLM 流式生成 + TTS 流式合成，完成后保存 assistant 回复到记忆"""
    t_llm_start = time.time()
    first_token = True
    llm_text = ""
    t_first_token = 0.0

    cancel_evt = asyncio.Event()
    llm_cancel.clear()
    llm_cancel.append(cancel_evt)

    async def stream_llm_to_tts():
        nonlocal first_token, llm_text, t_first_token
        try:
            async for token in llm.chat_stream(text):
                if cancel_evt.is_set():
                    break
                if first_token:
                    t_first_token = time.time() - t_llm_start
                    logger.info(f"LLM first token: {t_first_token*1000:.0f}ms")
                    first_token = False
                llm_text += token

            if llm_text.strip() and not cancel_evt.is_set():
                session["state"] = SPEAK
                logger.info(f"State: THINK → SPEAK | LLM: {llm_text[:50]}...")
                await ws.send_json({
                    "type": "llm_result",
                    "text": llm_text,
                    "ttft_ms": round(t_first_token * 1000),
                })

                chunk_count = 0
                async for audio_bytes in tts.synthesize_stream(llm_text):
                    if cancel_evt.is_set():
                        break
                    ref_chunk = np.frombuffer(audio_bytes, dtype=np.int16)
                    echo_judge.feed_reference(ref_chunk)
                    await ws.send_bytes(audio_bytes)
                    chunk_count += 1
                    await asyncio.sleep(0)

                await ws.send_json({"type": "tts_end"})
                logger.info(f"TTS done: {chunk_count} chunks")

                # === 保存助手回复到记忆 ===
                memory.save_turn("assistant", llm_text)

                session["state"] = LISTEN
                logger.info("State: SPEAK → LISTEN")
        except asyncio.CancelledError:
            # 被打断 → 保存不完整的回复
            if llm_text.strip():
                memory.save_turn("assistant", llm_text, interrupted=1)
            logger.info("LLM/TTS cancelled")

    task = asyncio.create_task(stream_llm_to_tts())
    tts_task.clear()
    tts_task.append(task)


_TEST_PAGE = """<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Smart RangRang - Phase 4</title></head>
<body>
<h2>Phase 4 聊天记忆测试</h2>
<button id="start">开始录音</button>
<button id="stop" disabled>停止</button>
<span id="status">就绪</span>
<hr>
<div><b>ASR:</b> <span id="asr"></span></div>
<div><b>LLM:</b> <span id="llm"></span></div>
<div><b>记忆:</b> <span id="memory"></span></div>
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
            return;
        }
        const msg = JSON.parse(e.data);
        if (msg.type === 'asr_result') {
            document.getElementById('asr').textContent = msg.text + ' (' + msg.time_s + 's)';
        } else if (msg.type === 'llm_result') {
            document.getElementById('llm').textContent = msg.text;
            document.getElementById('state').textContent = 'SPEAK';
        } else if (msg.type === 'tts_end') {
            document.getElementById('state').textContent = 'LISTEN';
        } else if (msg.type === 'stop_playback') {
            document.getElementById('state').textContent = 'LISTEN (打断)';
        }
        log(msg.type + ': ' + (msg.text || ''));
    };

    mediaStream = await navigator.mediaDevices.getUserMedia({audio: {sampleRate: 16000, channelCount: 1}});
    audioCtx = new AudioContext({sampleRate: 16000});
    source = audioCtx.createMediaStreamSource(mediaStream);
    processor = audioCtx.createScriptProcessor(640, 1, 1);

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
