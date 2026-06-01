"""
Phase 1 - LLM 验证: Ollama Qwen2.5:14b 对话测试

用法:
    python test_llm.py                        # 交互式对话
    python test_llm.py "今天天气怎么样？"       # 单次问答
"""

import json
import sys
import time
import httpx

# --- 配置 ---
OLLAMA_BASE = "http://127.0.0.1:11434"
MODEL = "qwen2.5:14b"
SYSTEM_PROMPT = """你是一个友好的语音助手，名叫小智。
你的回答应该：
1. 简洁明了，适合语音播报（每次回复不超过100字）
2. 自然口语化，像朋友聊天一样
3. 如果用户的问题不完整或模糊，可以追问澄清"""


def check_ollama() -> bool:
    """检查 Ollama 服务是否可用"""
    try:
        resp = httpx.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            names = [m["name"] for m in models]
            print(f"[LLM] Ollama 可用, 已安装模型: {names}")
            return MODEL in names or any(MODEL.split(":")[0] in n for n in names)
        return False
    except Exception as e:
        print(f"[LLM] Ollama 连接失败: {e}")
        return False


def chat(prompt: str, stream: bool = True) -> tuple[str, float, int]:
    """
    发送对话请求，返回 (完整回复文本, 首token延迟, 总token数)
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": stream,
        "options": {
            "temperature": 0.7,
            "num_predict": 512,  # 限制最大生成 token
        },
    }

    t_start = time.time()
    first_token_time = None
    full_text = ""
    token_count = 0

    with httpx.stream(
        "POST",
        f"{OLLAMA_BASE}/api/chat",
        json=payload,
        timeout=60,
    ) as response:
        for line in response.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue

            if "message" in chunk and chunk["message"].get("content"):
                if first_token_time is None:
                    first_token_time = time.time()
                full_text += chunk["message"]["content"]
                token_count += 1

                if stream:
                    # 流式打印（原地刷新）
                    print(chunk["message"]["content"], end="", flush=True)

            if chunk.get("done"):
                break

    t_end = time.time()
    ttft = (first_token_time - t_start) if first_token_time else (t_end - t_start)
    return full_text, ttft, token_count


def main():
    print(f"[LLM] 模型: {MODEL}")

    if not check_ollama():
        print("[LLM] 请确保 Ollama 已启动且模型已安装")
        print("  安装命令: ollama pull qwen2.5:14b")
        return

    if len(sys.argv) > 1:
        # 单次问答模式
        prompt = sys.argv[1]
        print(f"[LLM] 用户: {prompt}")
        print(f"[LLM] 助手: ", end="")
        response, ttft, tokens = chat(prompt)
        print()  # 换行
        total_time = ttft
        print(f"\n[LLM] 首token延迟: {ttft*1000:.0f}ms | 总token: {tokens}")
    else:
        # 交互模式
        print("[LLM] 进入交互模式 (输入 'quit' 退出)")
        print("-" * 50)
        while True:
            try:
                user_input = input("\n你: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if user_input.lower() in ("quit", "exit", "q"):
                break
            if not user_input:
                continue

            print("小智: ", end="")
            response, ttft, tokens = chat(user_input)
            print()
            print(f"  [首token: {ttft*1000:.0f}ms | tokens: {tokens}]")


if __name__ == "__main__":
    main()
