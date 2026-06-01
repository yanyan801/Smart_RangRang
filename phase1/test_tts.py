"""
Phase 1 - TTS 验证: Edge-TTS 语音合成

用法:
    python test_tts.py                          # 内置测试文本
    python test_tts.py "你好，今天天气不错"       # 指定文本合成
"""

import asyncio
import sys
import time
from pathlib import Path

# --- 配置 ---
VOICE = "zh-CN-XiaoxiaoNeural"  # 中文女声，自然度最高
RATE = "+15%"                    # 语速（稍快适合语音助手）
PITCH = "+0Hz"                  # 音调不变
OUTPUT_DIR = Path(__file__).parent / "output"


async def synthesize(text: str, output_path: str) -> float:
    """合成语音并保存到文件，返回耗时(秒)"""
    import edge_tts

    t0 = time.time()
    communicate = edge_tts.Communicate(
        text=text,
        voice=VOICE,
        rate=RATE,
        pitch=PITCH,
    )
    await communicate.save(output_path)
    elapsed = time.time() - t0
    return elapsed


async def list_voices():
    """列出可用的中文语音"""
    import edge_tts

    print("[TTS] 可用中文语音:")
    voices = await edge_tts.VoicesManager.create()
    for v in voices.voices:
        if v["Locale"].startswith("zh-"):
            print(f"  {v['ShortName']:30s} {v['Locale']:10s} {v.get('VoiceTag', {}).get('VoiceRole', '')}")


async def main():
    print(f"[TTS] Edge-TTS 语音: {VOICE} | 语速: {RATE}")

    if len(sys.argv) > 1 and sys.argv[1] == "--voices":
        await list_voices()
        return

    # 测试文本
    if len(sys.argv) > 1:
        text = sys.argv[1]
    else:
        text = "你好！我是小智，一个运行在本地电脑上的智能语音助手。有什么我可以帮你的吗？"

    print(f"[TTS] 合成文本: {text}")
    print(f"[TTS] 合成中...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = str(OUTPUT_DIR / "tts_output.mp3")

    elapsed = await synthesize(text, output_path)

    # 估算音频时长（中文约 3-4 字/秒）
    est_duration = len(text) / 3.5
    rtf = elapsed / est_duration if est_duration > 0 else 0

    print(f"[TTS] 合成完成!")
    print(f"[TTS] 耗时: {elapsed:.2f}s | 文本长度: {len(text)}字 | RTF: {rtf:.2f}")
    print(f"[TTS] 输出: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
