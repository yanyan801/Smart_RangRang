"""
Phase 1 - 端到端链路验证: 音频 → ASR → LLM → TTS → 音频

用法:
    python pipeline.py                             # 录制测试
    python pipeline.py <wav_file>                  # 从 WAV 文件
    python pipeline.py <wav_file> --skip-llm       # 仅测试 ASR+TTS
    python pipeline.py <wav_file> --text "你好"    # 跳过ASR，直接输入文本
"""

import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

# === 导入各模块 ===
sys.path.insert(0, str(Path(__file__).parent))
from test_asr import load_model as load_asr, recognize, create_test_audio
from test_llm import chat as llm_chat, check_ollama
from test_tts import synthesize as tts_synthesize

# --- 配置 ---
OUTPUT_DIR = Path(__file__).parent / "output"
TEST_DATA_DIR = Path(__file__).parent / "test_data"


class PipelineResult:
    """链路各阶段耗时统计"""

    def __init__(self):
        self.asr_text = ""
        self.asr_time = 0.0
        self.llm_response = ""
        self.llm_ttft = 0.0
        self.llm_tokens = 0
        self.llm_total_time = 0.0
        self.tts_time = 0.0
        self.tts_output = ""
        self.total_time = 0.0

    def report(self):
        print("\n" + "=" * 60)
        print("  端到端链路验证报告")
        print("=" * 60)
        print(f"  ASR 识别: {self.asr_text[:60]}{'...' if len(self.asr_text) > 60 else ''}")
        print(f"  ASR 耗时: {self.asr_time:.2f}s")
        print(f"  LLM 回复: {self.llm_response[:60]}{'...' if len(self.llm_response) > 60 else ''}")
        print(f"  LLM 首token: {self.llm_ttft*1000:.0f}ms | 总token: {self.llm_tokens} | 总耗时: {self.llm_total_time:.2f}s")
        print(f"  TTS 耗时: {self.tts_time:.2f}s")
        print(f"  TTS 输出: {self.tts_output}")
        print(f"  ─────────────────────────────")
        print(f"  总耗时: {self.total_time:.2f}s")
        print(f"  端到端首音延迟估算: {(self.asr_time + self.llm_ttft + (self.tts_time * 0.3))*1000:.0f}ms")
        print("=" * 60)

        # 保存报告
        report_path = OUTPUT_DIR / "pipeline_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(
                {
                    "asr_text": self.asr_text,
                    "asr_time_s": self.asr_time,
                    "llm_response": self.llm_response,
                    "llm_ttft_ms": self.llm_ttft * 1000,
                    "llm_tokens": self.llm_tokens,
                    "llm_total_time_s": self.llm_total_time,
                    "tts_time_s": self.tts_time,
                    "tts_output": self.tts_output,
                    "total_time_s": self.total_time,
                    "e2e_first_audio_latency_ms": (
                        self.asr_time + self.llm_ttft + self.tts_time * 0.3
                    )
                    * 1000,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\n  报告已保存: {report_path}")


async def run_pipeline(wav_path: str = None, skip_asr: bool = False,
                       fixed_text: str = None, skip_llm: bool = False):
    """运行完整链路"""
    result = PipelineResult()
    t_total_start = time.time()

    # ===== 1. ASR =====
    if skip_asr or fixed_text:
        if fixed_text:
            result.asr_text = fixed_text
            print(f"[PIPE] 跳过 ASR，使用固定文本: {fixed_text}")
        else:
            result.asr_text = ""
    else:
        print("\n" + "-" * 40)
        print("[PIPE] Stage 1/3: ASR 语音识别")
        print("-" * 40)

        if wav_path is None:
            wav_path = str(TEST_DATA_DIR / "test_input.wav")
            if not Path(wav_path).exists():
                print("[PIPE] 无测试音频，开始录制...")
                ok = create_test_audio(wav_path)
                if not ok:
                    print("[PIPE] 录制失败")
                    return

        asr_model = load_asr()
        result.asr_text = recognize(asr_model, wav_path)
        result.asr_time = time.time() - t_total_start
        print(f"[PIPE] ASR 阶段耗时: {result.asr_time:.2f}s")

    # 保存 ASR 结果
    asr_out = OUTPUT_DIR / "asr_result.txt"
    asr_out.parent.mkdir(parents=True, exist_ok=True)
    asr_out.write_text(result.asr_text, encoding="utf-8")

    if not result.asr_text.strip():
        print("[PIPE] ASR 无结果，链路终止")
        return

    # ===== 2. LLM =====
    if skip_llm:
        result.llm_response = result.asr_text
        print("[PIPE] 跳过 LLM")
    else:
        print("\n" + "-" * 40)
        print("[PIPE] Stage 2/3: LLM 对话生成")
        print("-" * 40)

        if not check_ollama():
            print("[PIPE] Ollama 不可用，链路终止")
            return

        t_llm_start = time.time()
        print(f"[PIPE] 用户输入: {result.asr_text}")
        print(f"[PIPE] 助手回复: ", end="")
        result.llm_response, result.llm_ttft, result.llm_tokens = llm_chat(
            result.asr_text
        )
        print()  # 换行
        result.llm_total_time = time.time() - t_llm_start

    # 保存 LLM 结果
    llm_out = OUTPUT_DIR / "llm_response.txt"
    llm_out.write_text(result.llm_response, encoding="utf-8")

    # ===== 3. TTS =====
    print("\n" + "-" * 40)
    print("[PIPE] Stage 3/3: TTS 语音合成")
    print("-" * 40)

    tts_out = str(OUTPUT_DIR / "pipeline_output.mp3")
    result.tts_time = await tts_synthesize(result.llm_response, tts_out)
    result.tts_output = tts_out
    print(f"[PIPE] TTS 阶段耗时: {result.tts_time:.2f}s")

    result.total_time = time.time() - t_total_start
    result.report()


def main():
    wav_path = None
    skip_asr = False
    fixed_text = None
    skip_llm = False

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--skip-llm":
            skip_llm = True
        elif args[i] == "--text" and i + 1 < len(args):
            fixed_text = args[i + 1]
            skip_asr = True
            i += 1
        elif args[i] == "--skip-asr":
            skip_asr = True
        elif not args[i].startswith("--"):
            wav_path = args[i]
        i += 1

    print("=" * 60)
    print("  Smart RangRang - Phase 1 端到端链路验证")
    print("  音频 → ASR(SenseVoice) → LLM(Qwen2.5:14b) → TTS(Edge-TTS)")
    print("=" * 60)

    asyncio.run(run_pipeline(
        wav_path=wav_path,
        skip_asr=skip_asr,
        fixed_text=fixed_text,
        skip_llm=skip_llm,
    ))


if __name__ == "__main__":
    main()
