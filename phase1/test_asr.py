"""
Phase 1 - ASR 验证: SenseVoice-small 离线识别

用法:
    python test_asr.py                     # 使用内置测试音频
    python test_asr.py <wav_file>          # 识别指定 WAV 文件
"""

import sys
import time
from pathlib import Path

# --- 配置 ---
MODEL_NAME = "iic/SenseVoiceSmall"
DEVICE = "cuda:0"  # GPU 推理
SAMPLE_RATE = 16000


def load_model():
    """加载 SenseVoice-small 模型"""
    from funasr import AutoModel

    print(f"[ASR] 加载模型: {MODEL_NAME} (device={DEVICE})")
    t0 = time.time()
    model = AutoModel(
        model=MODEL_NAME,
        device=DEVICE,
        disable_update=True,  # 禁止自动更新
    )
    # SenseVoice 返回格式: {key, value} 其中 value 是带时间戳的识别结果
    elapsed = time.time() - t0
    print(f"[ASR] 模型加载完成, 耗时 {elapsed:.1f}s")
    return model


def recognize(model, wav_path: str) -> str:
    """识别 WAV 文件，返回完整文本"""
    print(f"[ASR] 识别文件: {wav_path}")
    t0 = time.time()
    result = model.generate(
        input=wav_path,
        language="zh",
        use_itn=True,       # 逆文本正则化 (数字/日期等)
    )
    elapsed = time.time() - t0
    realtime_factor = elapsed / (1.0 * 60)  # 假设60秒音频，粗略

    # SenseVoice 返回结构:
    # [{'key': 'file_path', 'text': '识别结果文本'}]
    # 或更详细的: [{'key': '...', 'text': '...', 'sentences': [...]}]
    if result and len(result) > 0:
        text = result[0].get("text", "")
        # 去除 SenseVoice 特有的情感/语种标签
        # 格式可能类似: "<|zh|><|NEUTRAL|>..."
        text = _clean_sensevoice_tags(text)
        print(f"[ASR] 识别完成, 耗时 {elapsed:.2f}s, RTF ~{elapsed/max(len(text)/5, 0.001):.3f}")
        print(f"[ASR] 结果: {text}")
        return text
    else:
        print("[ASR] 未识别到内容")
        return ""


def _clean_sensevoice_tags(text: str) -> str:
    """清理 SenseVoice 输出的特殊标签"""
    # 去除 <|...|> 格式的标签
    import re
    text = re.sub(r'<\|[^|]*\|>', '', text)
    return text.strip()


def create_test_audio(output_path: str, duration_sec: float = 4.0):
    """用 PC 自带的麦克风录制一段测试音频"""
    try:
        import pyaudio
        import wave
        import numpy as np
    except ImportError:
        print("[ASR] PyAudio 未安装，跳过录音")
        return False

    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000

    p = pyaudio.PyAudio()
    print(f"[ASR] 可用音频设备:")
    for i in range(p.get_device_count()):
        info = p.get_device_info(i)
        if info["maxInputChannels"] > 0:
            print(f"  [{i}] {info['name']}")

    device_index = None
    for i in range(p.get_device_count()):
        info = p.get_device_info(i)
        if info["maxInputChannels"] > 0:
            device_index = i
            print(f"[ASR] 使用设备 [{device_index}]: {info['name']}")
            break

    if device_index is None:
        print("[ASR] 未找到输入设备")
        p.terminate()
        return False

    print(f"[ASR] 开始录制 {duration_sec}s 测试音频，请说话...")
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=CHUNK,
    )

    frames = []
    for _ in range(int(RATE / CHUNK * duration_sec)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("[ASR] 录制完成")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(output_path, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b"".join(frames))
    wf.close()

    print(f"[ASR] 已保存: {output_path}")
    return True


def main():
    wav_path = None
    if len(sys.argv) > 1:
        wav_path = sys.argv[1]
    else:
        # 尝试录制
        test_path = Path(__file__).parent / "test_data" / "test_input.wav"
        if not test_path.exists():
            ok = create_test_audio(str(test_path))
            if not ok:
                print("[ASR] 请提供 WAV 文件: python test_asr.py <wav_file>")
                return
        wav_path = str(test_path)

    model = load_model()
    text = recognize(model, wav_path)

    if text:
        # 保存识别结果
        out_path = Path(__file__).parent / "output" / "asr_result.txt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"[ASR] 结果已保存: {out_path}")


if __name__ == "__main__":
    main()
