"""
Phase 5 - WebSocket Server: 全链路稳定性打磨
修复断连崩溃 + 组件超时 + 会话看门狗 + 心跳 + 异常自愈
"""

import asyncio
import json
import logging
import time
import traceback

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
from watchdog import (
    SessionWatchdog, ComponentHealth, TimeoutGuard,
    ASR_TIMEOUT_MS, LLM_FIRST_TOKEN_MS, LLM_TOTAL_MS,
    TTS_CHUNK_MS, SESSION_IDLE_MS, HEARTBEAT_INTERVAL_MS,
    HEARTBEAT_TIMEOUT_MS, MAX_CONSECUTIVE_ERRORS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("server")

SAMPLE_RATE = 16000
CHUNK_MS = 40
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_MS / 1000)

# === 全局引擎 ===
asr = ASREngine(device="cuda:0")
llm = LLMEngine(model="qwen2.5:14b")
tts = TTSEngine()
vad = VADEngine(sample_rate=SAMPLE_RATE, energy_threshold=0.015,
                min_speech_frames=4, min_silence_frames=12)
echo_judge = EchoJudge(sample_rate=SAMPLE_RATE, corr_threshold=0.5,
                       corr_weak_threshold=0.3, env_threshold=0.6,
                       ref_buffer_seconds=2.0, chunk_ms=CHUNK_MS)
interrupt_detector = InterruptDetector(echo_judge=echo_judge,
                                       vad_consecutive_frames=10,
                                       vad_energy_threshold=0.04)
sentence_judge = SentenceJudge(punct_silence_ms=200, force_silence_ms=800,
                               long_text_chars=100, long_text_silence_ms=300,
                               incomplete_extend_ms=1500, pending_timeout_ms=5000)
memory = MemoryManager()
health = ComponentHealth()

IDLE = "IDLE"
LISTEN = "LISTEN"
THINK = "THINK"
SPEAK = "SPEAK"

app = FastAPI()


@app.get("/")
async def test_page():
    return HTMLResponse(_TEST_PAGE)


@app.get("/health")
async def health_check():
    return {
        "status": "ok" if health.all_healthy else "degraded",
        "components": {
            name: {
                "healthy": c.is_healthy,
                "ok_count": c.ok_count,
                "error_count": c.error_count,
                "consecutive_errors": c.consecutive_errors,
            }
            for name, c in health.components.items()
        },
    }


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    logger.info("Client connected")

    # === 会话初始化 ===
    session_id = memory.start_session()
    llm.reset_history()
    history = memory.get_history_for_llm()
    if history:
        llm.load_history_from_memory(history)
    memory_ctx = memory.get_memory_context()
    if memory_ctx:
        llm.set_memory_context(memory_ctx)

    session = {"state": IDLE}
    tts_task: list[asyncio.Task] = []
    llm_cancel: list[asyncio.Event] = []

    # === 看门狗 ===
    wd = SessionWatchdog(
        idle_timeout_ms=SESSION_IDLE_MS,
        heartbeat_timeout_ms=HEARTBEAT_TIMEOUT_MS,
        max_strikes=MAX_CONSECUTIVE_ERRORS,
        on_idle_timeout=lambda: logger.info("Watchdog: session idle timeout → IDLE"),
        on_heartbeat_lost=lambda: logger.warning("Watchdog: heartbeat lost"),
        on_strike_out=lambda: logger.error(
            f"Watchdog: {MAX_CONSECUTIVE_ERRORS} consecutive errors, service degraded"
        ),
    )

    # 心跳发送任务
    heartbeat_task = asyncio.create_task(_send_heartbeat(ws, wd))

    try:
        while True:
            # === 看门狗检查 ===
            wd_event = wd.check()
            if wd_event == "idle" and session["state"] != IDLE:
                logger.info("Session idle → returning to IDLE")
                await _cleanup_session(session, tts_task, llm_cancel)
                session["state"] = IDLE
            elif wd_event == "strike_out":
                await ws.send_json({
                    "type": "state", "state": "degraded",
                    "message": "服务繁忙，稍等哦",
                })

            # === 接收消息（带断连保护） ===
            try:
                data = await ws.receive()
            except (WebSocketDisconnect, RuntimeError) as e:
                # RuntimeError: "Cannot call receive once a disconnect..."
                logger.info(f"Client disconnected during receive: {e}")
                break

            if "bytes" in data:
                raw = data["bytes"]
                if len(raw) != CHUNK_SAMPLES * 2:
                    continue

                audio_chunk = np.frombuffer(raw, dtype=np.int16).copy()
                s = session["state"]
                wd.reset()  # 收到音频 = 有活动

                # === SPEAK: 打断检测 ===
                if s == SPEAK:
                    try:
                        energy = vad.process_frame(audio_chunk)
                        if interrupt_detector.process(audio_chunk, energy):
                            logger.info("INTERRUPT — destroy and recreate")
                            await _handle_interrupt(
                                ws, session, tts_task, llm_cancel
                            )
                            continue
                    except Exception as e:
                        logger.error(f"Interrupt detection error: {e}")
                        health.report_error("interrupt_detector", str(e))

                # === LISTEN / IDLE: VAD + 断句 ===
                if s in (LISTEN, IDLE):
                    try:
                        await _process_audio_frame(
                            ws, audio_chunk, session, tts_task, llm_cancel
                        )
                    except Exception as e:
                        logger.error(f"Audio processing error: {e}")
                        health.report_error("audio_processing", str(e))
                        wd.report_error("audio_processing")
                        session["state"] = LISTEN  # 恢复监听

                # === 断句器超时检查 ===
                if session["state"] == LISTEN and sentence_judge.has_pending:
                    pending = sentence_judge.check_timeout()
                    if pending:
                        logger.info(f"SentenceJudge: timeout release → '{pending}'")
                        await ws.send_json({
                            "type": "asr_result", "text": pending, "time_s": 0,
                        })
                        memory.save_turn("user", pending)
                        await _start_llm_tts(
                            ws, pending, llm_cancel, tts_task, session, memory
                        )

            elif "text" in data:
                msg = json.loads(data["text"])
                msg_type = msg.get("type", "")

                if msg_type == "ping":
                    wd.heartbeat()
                    await ws.send_json({"type": "pong"})

                elif msg_type == "reset":
                    logger.info("Client requested reset")
                    await _cleanup_session(session, tts_task, llm_cancel)
                    await memory.end_session(llm)
                    session_id = memory.start_session()
                    llm.reset_history()
                    history = memory.get_history_for_llm()
                    if history:
                        llm.load_history_from_memory(history)
                    memory_ctx = memory.get_memory_context()
                    if memory_ctx:
                        llm.set_memory_context(memory_ctx)
                    session["state"] = IDLE
                    await ws.send_json({"type": "state", "state": IDLE})

    except (WebSocketDisconnect, RuntimeError):
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Unexpected error: {e}\n{traceback.format_exc()}")
        health.report_error("server", str(e))
    finally:
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass

        await _cleanup_session(session, tts_task, llm_cancel)
        await memory.end_session(llm)
        logger.info(f"Session ended, final state={session['state']}")


# === 音频帧处理 ===
async def _process_audio_frame(ws, audio_chunk, session, tts_task, llm_cancel):
    energy = vad.process_frame(audio_chunk)
    event, speech_audio = vad.update(energy, audio_chunk)

    if event == "speech_start" and session["state"] == IDLE:
        session["state"] = LISTEN
        logger.info("State: IDLE → LISTEN")

    if event == "speech_end" and session["state"] == LISTEN:
        if speech_audio is None or len(speech_audio) < SAMPLE_RATE * 0.3:
            return

        silence_ms = vad.last_silence_ms
        vad.reset()
        session["state"] = THINK
        audio_len = len(speech_audio) / SAMPLE_RATE
        logger.info(
            f"State: LISTEN → THINK (audio={audio_len:.1f}s, silence={silence_ms:.0f}ms)"
        )

        # === ASR（带超时保护） ===
        t_asr = time.time()
        try:
            text = await asyncio.wait_for(
                asyncio.to_thread(
                    asr.transcribe, speech_audio.astype(np.float32)
                ),
                timeout=ASR_TIMEOUT_MS / 1000,
            )
            health.report_ok("asr")
        except asyncio.TimeoutError:
            logger.warning(f"ASR timeout ({ASR_TIMEOUT_MS}ms)")
            health.report_timeout("asr")
            session["state"] = LISTEN
            return
        except Exception as e:
            logger.error(f"ASR error: {e}")
            health.report_error("asr", str(e))
            session["state"] = LISTEN
            return

        t_asr = time.time() - t_asr
        logger.info(f"ASR [{t_asr:.2f}s]: {text}")

        if not text.strip():
            session["state"] = LISTEN
            return

        await ws.send_json({
            "type": "asr_result", "text": text, "time_s": round(t_asr, 2),
        })

        # Sentence Judge
        is_complete, judged_text = sentence_judge.judge(text, silence_ms)
        if not is_complete:
            logger.info(f"SentenceJudge: incomplete, buffering '{judged_text[-20:]}'")
            session["state"] = LISTEN
            return

        memory.save_turn("user", judged_text)
        memory_ctx = memory.get_memory_context(judged_text)
        if memory_ctx:
            llm.set_memory_context(memory_ctx)

        await _start_llm_tts(ws, judged_text, llm_cancel, tts_task, session, memory)


# === 打断处理 ===
async def _handle_interrupt(ws, session, tts_task, llm_cancel):
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


# === 会话清理 ===
async def _cleanup_session(session, tts_task, llm_cancel):
    if llm_cancel:
        llm_cancel[0].set()
    if tts_task and not tts_task[0].done():
        tts_task[0].cancel()
    echo_judge.flush()
    vad.reset()
    interrupt_detector.reset()
    sentence_judge.reset()


# === LLM/TTS 流式处理（带超时保护） ===
async def _start_llm_tts(ws, text, llm_cancel, tts_task, session, memory):
    t_llm_start = time.time()
    first_token = True
    llm_text = ""
    t_first_token = 0.0
    llm_timeout = False

    cancel_evt = asyncio.Event()
    llm_cancel.clear()
    llm_cancel.append(cancel_evt)

    async def stream_llm_to_tts():
        nonlocal first_token, llm_text, t_first_token, llm_timeout
        try:
            async for token in llm.chat_stream(text):
                if cancel_evt.is_set():
                    break

                if first_token:
                    t_first_token = time.time() - t_llm_start
                    ttft_ms = t_first_token * 1000
                    logger.info(f"LLM first token: {ttft_ms:.0f}ms")

                    if ttft_ms > LLM_FIRST_TOKEN_MS:
                        logger.warning(f"LLM first token timeout ({ttft_ms:.0f}ms)")
                        health.report_timeout("llm")
                        # 不中断，继续等待
                    first_token = False

                llm_text += token

                # 总超时检查
                elapsed = (time.time() - t_llm_start) * 1000
                if elapsed > LLM_TOTAL_MS:
                    logger.warning(f"LLM total timeout ({elapsed:.0f}ms)")
                    llm_timeout = True
                    health.report_timeout("llm")
                    break

            if llm_timeout and not llm_text.strip():
                llm_text = "我正在想，请稍等..."

            if llm_text.strip() and not cancel_evt.is_set():
                health.report_ok("llm")
                session["state"] = SPEAK
                logger.info(f"State: THINK → SPEAK | LLM: {llm_text[:50]}...")
                await ws.send_json({
                    "type": "llm_result",
                    "text": llm_text,
                    "ttft_ms": round(t_first_token * 1000),
                })

                # === TTS（带超时保护） ===
                chunk_count = 0
                t_tts_start = time.time()
                try:
                    async for audio_bytes in tts.synthesize_stream(llm_text):
                        if cancel_evt.is_set():
                            break

                        # 单 chunk 超时检查
                        chunk_elapsed = (time.time() - t_tts_start) * 1000
                        if chunk_elapsed > TTS_CHUNK_MS and chunk_count == 0:
                            logger.warning(f"TTS first chunk timeout")
                            health.report_timeout("tts")
                            break

                        ref_chunk = np.frombuffer(audio_bytes, dtype=np.int16)
                        echo_judge.feed_reference(ref_chunk)
                        await ws.send_bytes(audio_bytes)
                        chunk_count += 1
                        await asyncio.sleep(0)

                    health.report_ok("tts")
                except Exception as e:
                    logger.error(f"TTS error: {e}")
                    health.report_error("tts", str(e))

                await ws.send_json({"type": "tts_end"})
                logger.info(f"TTS done: {chunk_count} chunks")

                memory.save_turn("assistant", llm_text)
                session["state"] = LISTEN
                logger.info("State: SPEAK → LISTEN")
        except asyncio.CancelledError:
            if llm_text.strip():
                memory.save_turn("assistant", llm_text, interrupted=1)
            logger.info("LLM/TTS cancelled")

    task = asyncio.create_task(stream_llm_to_tts())
    tts_task.clear()
    tts_task.append(task)


# === 心跳发送 ===
async def _send_heartbeat(ws: WebSocket, wd: SessionWatchdog):
    """每 3s 发送一次心跳"""
    try:
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL_MS / 1000)
            try:
                await ws.send_json({"type": "ping"})
            except Exception:
                break  # 连接已断开
    except asyncio.CancelledError:
        pass


# === 测试页面 ===
_TEST_PAGE = """<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Smart RangRang - Phase 5</title></head>
<body>
<h2>Phase 5 稳定性测试</h2>
<button id="start">开始录音</button>
<button id="stop" disabled>停止</button>
<span id="status">就绪</span>
<span id="heartbeat" style="color:green">●</span>
<hr>
<div><b>ASR:</b> <span id="asr"></span></div>
<div><b>LLM:</b> <span id="llm"></span></div>
<div><b>状态:</b> <span id="state"></span></div>
<div><b>心跳:</b> <span id="hb"></span></div>
<hr>
<div id="log"></div>

<script>
let ws, mediaStream, audioCtx, processor, source;
let isRecording = false;
let lastHb = Date.now();

document.getElementById('start').onclick = async () => {
    ws = new WebSocket('ws://' + location.host + '/ws');
    ws.binaryType = 'arraybuffer';
    ws.onopen = () => log('WebSocket 已连接');

    ws.onmessage = (e) => {
        if (e.data instanceof ArrayBuffer) return;
        const msg = JSON.parse(e.data);
        if (msg.type === 'asr_result') {
            document.getElementById('asr').textContent = msg.text;
        } else if (msg.type === 'llm_result') {
            document.getElementById('llm').textContent = msg.text;
            document.getElementById('state').textContent = 'SPEAK';
        } else if (msg.type === 'tts_end') {
            document.getElementById('state').textContent = 'LISTEN';
        } else if (msg.type === 'stop_playback') {
            document.getElementById('state').textContent = 'LISTEN (打断)';
        } else if (msg.type === 'ping') {
            lastHb = Date.now();
            ws.send(JSON.stringify({type: 'ping'}));
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

    // 心跳监控
    setInterval(() => {
        const elapsed = (Date.now() - lastHb) / 1000;
        document.getElementById('hb').textContent = elapsed.toFixed(1) + 's';
        document.getElementById('heartbeat').style.color = elapsed > 8 ? 'red' : 'green';
    }, 1000);
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
    logger.info("=== Smart RangRang Phase 5 - Stability ===")
    logger.info(f"Config: ASR to={ASR_TIMEOUT_MS}ms, "
                f"LLM first={LLM_FIRST_TOKEN_MS}ms total={LLM_TOTAL_MS}ms, "
                f"TTS={TTS_CHUNK_MS}ms, idle={SESSION_IDLE_MS}ms, "
                f"heartbeat={HEARTBEAT_INTERVAL_MS}ms, max_strikes={MAX_CONSECUTIVE_ERRORS}")

    logger.info("Loading models...")
    asr.load()
    vad.load()
    logger.info("All models loaded. Starting server on :8765")

    # 注册组件健康
    health.get("asr")
    health.get("llm")
    health.get("tts")
    health.get("interrupt_detector")
    health.get("audio_processing")
    health.get("server")

    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="info")
