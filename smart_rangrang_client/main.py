"""
Smart RangRang - 树莓派客户端入口

用法:
    python main.py                                    # 使用 config.py 默认配置
    python main.py --host 192.168.1.100 --port 8765   # 指定服务器地址
    python main.py --list-devices                     # 列出可用音频设备
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pi.main")

# 确保 smart_rangrang_client 包的父目录在 sys.path，支持直接 `python main.py` 运行
_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

# 兼容模块运行和直接运行
try:
    from . import config
    from .audio_capture import AudioCapture
    from .audio_player import AudioPlayer
    from .ws_client import PiClient
except ImportError:
    import config  # type: ignore
    from audio_capture import AudioCapture  # type: ignore
    from audio_player import AudioPlayer  # type: ignore
    from ws_client import PiClient  # type: ignore


def main():
    parser = argparse.ArgumentParser(description="Smart RangRang - 树莓派客户端")
    parser.add_argument("--host", help="服务器 IP 地址")
    parser.add_argument("--port", type=int, help="服务器端口")
    parser.add_argument("--device", help="麦克风设备名称（部分匹配）")
    parser.add_argument("--list-devices", action="store_true", help="列出可用音频设备")
    args = parser.parse_args()

    if args.list_devices:
        print("可用的输入设备:")
        AudioCapture.list_devices()
        return

    # 构建 URL
    host = args.host or config.SERVER_HOST
    port = args.port or config.SERVER_PORT
    ws_url = f"ws://{host}:{port}/ws"

    # 初始化采集和播放
    capture = AudioCapture(
        sample_rate=config.SAMPLE_RATE,
        chunk_samples=config.CHUNK_SAMPLES,
        device_name=args.device,
    )
    player = AudioPlayer(
        sample_rate=config.SAMPLE_RATE,
        chunk_samples=config.CHUNK_SAMPLES,
        buffer_chunks=config.PLAYER_BUFFER_CHUNKS,
    )

    # 指定设备
    if args.device:
        capture.find_device([args.device])

    # 启动客户端
    client = PiClient(
        ws_url=ws_url,
        capture=capture,
        player=player,
        heartbeat_interval=config.HEARTBEAT_INTERVAL,
        heartbeat_timeout=config.HEARTBEAT_TIMEOUT,
        reconnect_base=config.RECONNECT_BASE_DELAY,
        reconnect_max=config.RECONNECT_MAX_DELAY,
        reconnect_backoff=config.RECONNECT_BACKOFF,
    )

    logger.info(f"Starting Pi client, server={ws_url}")
    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        logger.info("Interrupted")


if __name__ == "__main__":
    main()
