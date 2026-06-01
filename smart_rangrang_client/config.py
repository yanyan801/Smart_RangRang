"""
Pi 客户端配置
"""
import os

# === 服务器连接 ===
SERVER_HOST = os.getenv("SRR_HOST", "192.168.1.100")
SERVER_PORT = int(os.getenv("SRR_PORT", "8765"))
WS_URL = f"ws://{SERVER_HOST}:{SERVER_PORT}/ws"

# === 音频参数（与服务器保持一致） ===
SAMPLE_RATE = 16000
CHANNELS = 1            # XVF3800 硬件 AEC 后取 ASR 通道（mono）
CHUNK_MS = 40
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_MS / 1000)   # 640
CHUNK_BYTES = CHUNK_SAMPLES * 2                       # 1280
FORMAT = "int16"

# === XVF3800 设备识别 ===
# PyAudio 设备名匹配关键词（按优先级）
MIC_DEVICE_KEYWORDS = [
    "reSpeaker", "seeed", "XMOS", "XVF3800",
    "3800", "mic array", "USB Audio",
]

# === 重连策略 ===
RECONNECT_BASE_DELAY = 1.0    # 初始重连延迟 (秒)
RECONNECT_MAX_DELAY = 30.0    # 最大重连延迟 (秒)
RECONNECT_BACKOFF = 2.0       # 退避倍数

# === 心跳 ===
HEARTBEAT_INTERVAL = 3.0      # 心跳检测间隔 (秒)
HEARTBEAT_TIMEOUT = 10.0      # 心跳超时 (秒)

# === 播放 ===
PLAYER_BUFFER_CHUNKS = 4      # 播放缓冲 chunk 数（减少 underflow）
