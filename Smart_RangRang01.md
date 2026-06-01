# Smart RangRang — 智能语音交互系统技术方案

## 一、概述

本方案基于**树莓派5（16G）+ XVF3800 4麦环形阵列 + Windows PC（32G/16G显存）+ Ollama**的硬件组合，实现工业级本地化语音交互系统，核心指标对标豆包语音助手的交互体验：**听得清、不卡死、随时打断、聪明断句**。

### 核心设计原则

| 原则 | 说明 |
|------|------|
| **全链路流式** | ASR → LLM → TTS 均为流式处理，首字/首音延迟 < 1.5s |
| **双向双工** | 麦克风输入与扬声器输出可同时工作，支持 barge-in 打断 |
| **边侧分离** | 树莓派负责音频 I/O 与轻量推理，PC 负责重算力推理 |
| **本地闭环** | 所有数据不出局域网，隐私安全，零 API 费用 |

---

## 二、硬件拓扑与角色划分

```
┌─────────────────────────────────────────────────────────────────┐
│                    Windows PC (算力中心)                          │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Docker Compose 容器集群                        │   │
│  │                                                           │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │   │
│  │  │ ASR 容器  │  │ LLM 容器  │  │ TTS 容器  │  │Memory容器 │ │   │
│  │  │ FunASR   │  │ Ollama   │  │ CosyVoice│  │ChromaDB  │ │   │
│  │  │Paraformer│  │ Qwen3    │  │    2     │  │+ SQLite  │ │   │
│  │  │Port:8100 │  │Port:11434│  │Port:8101 │  │Port:8102 │ │   │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘ │   │
│  │       │              │              │              │      │   │
│  │       └──────────────┼──────────────┼──────────────┘      │   │
│  │                      │ localhost HTTP                      │   │
│  │  ┌───────────────────┴────────────────────────────────┐   │   │
│  │  │        Orchestrator 容器 (SessionManager)           │   │   │
│  │  │        WebSocket Server (FastAPI) :8765             │   │   │
│  │  └───────────────────┬────────────────────────────────┘   │   │
│  └──────────────────────┼───────────────────────────────────┘   │
└─────────────────────────┼───────────────────────────────────────┘
                          │  LAN WebSocket
┌─────────────────────────┼───────────────────────────────────────┐
│           树莓派5 (音频前端 + 交互终端)                            │
│  ┌──────────────────────┴────────────────────────────────┐      │
│  │          WebSocket Client                               │      │
│  ├──────────────────┬─────────────────────────────────────┤      │
│  │   Audio Capture  │   Audio Playback (可中断)            │      │
│  │   (PyAudio)      │   (PyAudio)                          │      │
│  ├──────────────────┴─────────────────────────────────────┤      │
│  │  Silero VAD  │  openWakeWord  │  Echo Judge (回声判决器) │      │
│  ├────────────────────────────────────────────────────────┤      │
│  │   XVF3800 USB 音频设备 (即插即用)                        │      │
│  │   硬件 AEC / 波束成形 / NS / VNR / AGC / DOA            │      │
│  ├────────────────────────────────────────────────────────┤      │
│  │     XVF3800 4麦环形阵列    │   扬声器                    │      │
└───────────────────────────────────────────────────────────┘
```

### 角色划分

| 设备 | 角色 | 核心任务 |
|------|------|----------|
| **XVF3800** | 硬件 DSP | 即插即用 USB 音频设备，硬件层完成波束成形、AEC、降噪(NS)、语音对噪声比(VNR)、AGC、到达方向(DOA) — 物理层"听得清"已有工业级保障 |
| **树莓派5** | 音频前端 | 音频采集/播放、唤醒词检测、VAD、回声判决(Echo Judge)、打断检测、与 PC 的 WebSocket 通信 |
| **Windows PC** | 算力中心 | Docker Compose 容器集群：ASR 推理、LLM 推理、TTS 合成、会话管理、记忆管理 |

---

## 三、模型选型

### 3.1 语音识别（ASR）

| 候选模型 | 优势 | 劣势 | 结论 |
|----------|------|------|------|
| **FunASR Paraformer-large** | 中文 SOTA、流式支持、标点/ITN 内置、工业验证 | 需 GPU 推理 | **主选** |
| SenseVoice-small | 多语种、情感识别、速度快 | 中文精度略逊 Paraformer | 备选/辅助 |
| Faster-Whisper large-v3 | 通用性好、生态成熟 | 中文不及专有模型、延迟偏高 | 不推荐 |

**最终选择：FunASR Paraformer-large（流式版 paraformer-online）**

- 推理设备：Windows PC GPU（占用约 2-3GB 显存）
- 实时率(RTF)：< 0.01（GPU），满足实时流式要求
- 内置标点恢复与逆文本正则化(ITN)，直接输出带标点文本
- VAD + 标点双重信号驱动"聪明断句"

### 3.2 大语言模型（LLM）

| 候选模型 | 显存占用(Q4) | 中文能力 | 推理速度 | 结论 |
|----------|-------------|----------|----------|------|
| **Qwen3:14B** | ~9GB | 优秀 | 快 | **品质优先** |
| Qwen3:8B | ~5.5GB | 良好 | 很快 | 速度优先 |
| Qwen2.5:14B | ~9GB | 优秀 | 快 | 稳定选择 |
| Yi-1.5:9B | ~6GB | 良好 | 快 | 备选 |

**最终选择：Qwen3:14B-Q4_K_M（Ollama部署）**

- 16GB 显存下尚有 ~7GB 余量给 ASR 和 TTS 模型
- Qwen3 系列在指令遵循、多轮对话、中文理解上表现优异
- 通过 Ollama 管理，支持 context window 扩展至 32K tokens

### 3.3 语音合成（TTS）

| 候选模型 | 中文自然度 | 流式支持 | 推理速度 | 结论 |
|----------|-----------|----------|----------|------|
| **CosyVoice 2** | 极高 | 支持 | 快(RTF<0.1) | **主选** |
| ChatTTS | 高 | 部分支持 | 快 | 备选 |
| Edge-TTS | 较高 | 原生流式 | 即时 | 降级方案 |
| GPT-SoVITS | 可定制 | 不支持流式 | 中等 | 不推荐 |

**最终选择：CosyVoice 2 + Edge-TTS 降级**

- CosyVoice 2 提供 SOTA 级别中文语音合成，支持情感、语速控制
- Edge-TTS 作为降级兜底方案（无需 GPU，永远可用）
- 推理设备：Windows PC GPU（占用约 2-3GB 显存）

### 3.4 唤醒词检测（Wake Word）

**选择：openWakeWord**

- 运行设备：树莓派5（CPU推理，占用 < 5%）
- 自定义唤醒词："小智小智"、"你好小智"等
- 误唤醒率：< 1次/24小时（合理阈值下）
- 支持多个唤醒词同时检测

### 3.5 语音活动检测（VAD）

**选择：Silero VAD**

- 运行设备：树莓派5（CPU推理，< 3ms/帧）
- 极高准确率，支持多语种
- 输出：语音概率(0-1)逐帧，用于打断检测和端点判断
- 配合说话人定向（利用 XVF3800 波束成形后的单通道信号）

### 3.6 模型部署总表

| 模型 | 部署位置 | 显存/内存 | 推理框架 |
|------|----------|----------|----------|
| openWakeWord | 树莓派5 CPU | ~50MB RAM | ONNX Runtime |
| Silero VAD | 树莓派5 CPU | ~1MB RAM | ONNX Runtime |
| FunASR Paraformer | PC GPU | ~2.5GB VRAM | FunASR / ONNX |
| Qwen3:14B-Q4 | PC GPU | ~9GB VRAM | Ollama (llama.cpp) |
| CosyVoice 2 | PC GPU | ~2.5GB VRAM | CosyVoice SDK |

**PC 16GB 显存使用：9GB(LLM) + 2.5GB(ASR) + 2.5GB(TTS) ≈ 14GB，余量充足。**

---

## 四、软件架构

### 4.1 总体架构图

```
                      Docker Compose (PC 端)
                      ─────────────────────────────────────────────
                      │                                             │
  ┌───────┐           │  ┌─────────────────────────────────────┐   │
  │  Pi 5 │  WebSocket│  │        Orchestrator 容器              │   │
  │       │◄────────►│  │  ┌─────────────────────────────────┐ │   │
  │       │  :8765    │  │  │      SessionManager             │ │   │
  │       │           │  │  │  (唯一状态持有者，所有状态变更    │ │   │
  │       │           │  │  │   均由此发出指令，各节点执行)     │ │   │
  │       │           │  │  ├─────────────────────────────────┤ │   │
  │       │           │  │  │  对话状态机 (IDLE→LISTEN→       │ │   │
  │       │           │  │  │              THINK→SPEAK→...)    │ │   │
  │       │           │  │  ├─────────────────────────────────┤ │   │
  │       │           │  │  │  打断仲裁器 (双模判决)            │ │   │
  │       │           │  │  ├─────────────────────────────────┤ │   │
  │       │           │  │  │  断句决策器 (语义+声学+标点)     │ │   │
  │       │           │  │  ├─────────────────────────────────┤ │   │
  │       │           │  │  │  超时看门狗 (防卡死)             │ │   │
  │       │           │  │  └─────────────────────────────────┘ │   │
  │       │           │  └──────┬──────┬──────┬────────────────┘   │
  │       │           │         │      │      │                    │
  │       │           │    localhost HTTP (各容器独立部署，无依赖冲突) │
  │       │           │         │      │      │                    │
  │       │           │  ┌──────┴──┐ ┌─┴──────┴─┐ ┌────────────┐  │
  │       │           │  │ ASR 容器 │ │ LLM 容器  │ │ Memory容器  │  │
  │       │           │  │ FunASR  │ │ Ollama+  │ │ ChromaDB   │  │
  │       │           │  │ :8100   │ │ Qwen3    │ │ :8102      │  │
  │       │           │  └─────────┘ │ :11434   │ └────────────┘  │
  │       │           │              └──────────┘                 │
  │       │           │  ┌──────────┐                             │
  │       │           │  │ TTS 容器  │                             │
  │       │           │  │ CosyVoice│                             │
  │       │           │  │ 2 :8101  │                             │
  │       │           │  └──────────┘                             │
  │       │           └───────────────────────────────────────────┘
  │       │
  │  ┌────┴──────────────────────────────────────┐
  │  │  Input Pipeline    │   Output Pipeline     │
  │  │  麦克风→VAD→       │   TTS数据→            │
  │  │  回声判决→         │   可中断播放           │
  │  │  音频帧发送        │                       │
  │  ├────────────────────┴──────────────────────┤
  │  │  Echo Judge (回声判决器)                   │
  │  │  输入: mic_signal + speaker_ref           │
  │  │  输出: is_echo (bool)                     │
  │  │  逻辑: 互相关峰值 > 阈值 → 判定为回声      │
  │  └──────────────────────────────────────────┘
  └──────────────────────────────────────────────┘
```

**容器间通信原则**：
- Orchestrator 是唯一的大脑，持有全部会话状态
- ASR / LLM / TTS / Memory 是无状态工作节点，仅响应 Orchestrator 指令
- 容器间通过 localhost HTTP 通信（Docker 内部网络），延迟可忽略
- 任一节点崩溃不影响其他容器，Orchestrator 检测超时后自动重启节点容器

### 4.2 核心模块说明

#### Orchestrator（核心调度器）

```
对话状态机（SessionManager 唯一持有状态，各节点无状态）：

                    wake_word
  ┌─────────┐                    ┌─────────┐
  │  IDLE   │ ─────────────────> │ LISTEN  │
  │ (空闲)   │                    │ (倾听)   │
  └─────────┘                    └────┬────┘
       ^                              │ vad_end / sentence_end
       │                              v
       │                         ┌─────────┐
       │                         │ THINK   │
       │                         │ (思考)   │
       │                         └────┬────┘
       │                              │ first_token
       │                              v
       │ ┌─────────┐             ┌─────────┐
       │ │INTERRUPT│ <────────── │ SPEAK   │
       │ │(被打断) │  user_speak │ (播报)   │
       │ └────┬────┘             └─────────┘
       │      │                      │
       │      │ 打断处理策略:          │ tts_done
       │      │ 销毁而非清理           │ user_speak (打断)
       │      │ ────────────          │
       │      │ 1. ASR: 销毁当前       │
       │      │    session,重建新session
       │      │ 2. LLM: cancel 当前    │
       │      │    /api/generate 请求   │
       │      │ 3. TTS: flush 音频     │
       │      │    buffer,停播         │
       │      │ 4. Pi: 发送 stop_      │
       │      │    playback 指令       │
       │      │ 5. 所有节点回报 ACK    │
       │      │    后 → state=LISTEN  │
       │      v                       v
       │  ┌────────────────────────────┐
       │  │  所有节点 ACK 后            │
       │  │  → state = LISTEN          │
       │  │  (新 session, 干净状态)     │
       │  └────────────────────────────┘
       │
       └──── timeout(5min) ────┘
```

### 4.3 项目目录结构

```
Smart_RangRang/
├── pi/                          # 树莓派端代码
│   ├── main.py                  # 入口
│   ├── audio_capture.py         # 音频采集（PyAudio + XVF3800）
│   ├── audio_player.py          # 音频播放（流式、可中断）
│   ├── vad_engine.py            # Silero VAD 封装
│   ├── echo_judge.py            # 回声判决器（互相关 + 能量对比）
│   ├── wake_word.py             # openWakeWord 封装
│   ├── ws_client.py             # WebSocket 客户端
│   └── config_pi.yaml           # Pi 端配置
│
├── server/                      # PC 服务端代码
│   ├── docker-compose.yml       # 容器编排
│   ├── orchestrator/            # Orchestrator 容器
│   │   ├── Dockerfile
│   │   ├── main.py              # 入口（FastAPI + WebSocket）
│   │   ├── orchestrator.py      # 核心调度器（状态机、打断仲裁、断句）
│   │   ├── session.py           # SessionManager（唯一状态持有者）
│   │   ├── ws_hub.py            # WebSocket 多通道管理
│   │   ├── config.yaml          # Orchestrator 配置
│   │   └── prompts/
│   │       └── system_prompt.txt
│   ├── asr_node/                # ASR 容器
│   │   ├── Dockerfile
│   │   └── server.py            # FunASR 流式推理服务
│   ├── tts_node/                # TTS 容器
│   │   ├── Dockerfile
│   │   └── server.py            # CosyVoice 2 + Edge-TTS 降级服务
│   └── memory_node/             # Memory 容器
│       ├── Dockerfile
│       └── server.py            # ChromaDB + SQLite 记忆服务
│
├── shared/                      # 共享协议定义
│   ├── protocol.py              # 消息协议（数据类）
│   └── audio_utils.py           # 音频格式转换工具
│
├── data/                        # 持久化数据（挂载到 Memory 容器）
│   ├── chat_history.db          # 对话历史（SQLite）
│   └── chroma/                  # 向量记忆（ChromaDB）
│
├── requirements_pi.txt
└── README.md
```

---

## 五、关键功能技术实现

### 5.1 "听得清" — 高品质拾音方案

XVF3800 成品模块已内置完整 DSP 算法链（AEC / NS / VNR / AGC / DOA），USB 即插即用，硬件层信号质量有工业级保障。软件层只需做轻量补充。

```
层次化降噪管道：

  Layer 1: XVF3800 硬件 DSP（即插即用，无需额外配置）
    ├── 4麦波束成形 → 定向拾音，锁定最强声源方向
    ├── 声学回声消除 (AEC) → 消除扬声器线性回采
    ├── 自适应降噪 (ANS) → 稳态噪声抑制 > 20dB
    ├── 语音对噪声比 (VNR) → 实时评估信号质量
    ├── 到达方向 (DOA) → 声源定位，忽略非目标方向
    └── 自动增益控制 (AGC) → 输出电平稳定
    → USB 音频设备，输出 16kHz/16bit 单通道波束成形后信号

  Layer 2: 树莓派软件层（轻量补充）
    ├── 高通滤波 (80Hz) → 去除低频环境噪声（空调、风扇）
    └── 回声判决器 (Echo Judge) → 解决 AEC 残留回声导致的误打断
        （见 5.3 节）

  Layer 3: ASR 输入层
    └── FunASR 前端 VAD + 特征增强 → 进一步提升语音可懂度
```

> **为什么还需要 Echo Judge？** XVF3800 的 AEC 能消除线性回声（直达声 + 早期反射），但大音量下扬声器振膜和功放饱和产生的**非线性失真**无法被 AEC 消除。这些残留信号可能被 Silero VAD 误判为"有人在说话"，导致错误打断。Echo Judge 在软件层做第二级回声判别，是打断准确率的关键保障。

### 5.2 "不卡死" — 全链路防阻塞机制

```
防卡死三层保障：

  1. 组件级超时
     ├── ASR: 10s 无结果 → 丢弃当前音频帧，重置 ASR 状态
     ├── LLM: 15s 无首 token → 超时降级回复（"我思考太久了..."）
     ├── TTS: 5s 未合成 → 切换 Edge-TTS 降级
     └── WebSocket: 3s 心跳包，5s 无响应 → 重连

  2. 看门狗机制
     ├── 全局 30s 对话无进展 → 主动返回 IDLE 状态
     └── 连续 3 次超时 → 通知用户检查服务状态

  3. 资源隔离
     ├── ASR / LLM / TTS 各自独立进程/线程，单点故障不扩散
     └── asyncio 异步架构 + 生产者-消费者队列解耦各节点
```

**流式管道设计**（asyncio Queue 解耦）：

```python
# 每个节点独立的消息队列，避免阻塞传递
audio_frame_queue: asyncio.Queue    # Pi → ASR: 音频帧
asr_text_queue: asyncio.Queue       # ASR → Orchestrator: 识别文本
llm_prompt_queue: asyncio.Queue     # Orchestrator → LLM: 组装后的 prompt
llm_token_queue: asyncio.Queue      # LLM → TTS: token 流
tts_audio_queue: asyncio.Queue      # TTS → Pi: 合成音频帧
interrupt_queue: asyncio.Queue      # Pi → Orchestrator: 打断信号(高优先级)
```

### 5.3 "随时打断" — 双模判决 Barge-In 机制

借鉴小米音箱和豆包的做法：**不靠单一 VAD 做打断决策，而是 VAD + 回声判决器双信号 AND 逻辑**，确保"听到的是人声而非回声"。

```
打断检测双模判决流程（运行在 Pi 端，独立于播报线程）：

  ┌─────────────────────────────────────────────────────────────┐
  │                     双模判决管道                              │
  │                                                             │
  │  扬声器播放中 ──→ speaker_ref_buffer (播放音频环形缓冲区)      │
  │                                                             │
  │  麦克风采集 ──→ mic_buffer (麦克风音频环形缓冲区)              │
  │       │                                                     │
  │       ├──→ 路径A: Silero VAD (声学 VAD)                      │
  │       │    逐帧输出 speech_prob (0-1)                        │
  │       │    连续 N 帧 speech_prob > 0.7 → vad_trigger = True  │
  │       │                                                     │
  │       └──→ 路径B: Echo Judge (回声判决器)                     │
  │            输入: mic_buffer + speaker_ref_buffer              │
  │            计算: 互相关峰值 (cross-correlation peak)          │
  │                   + 能量包络对比                              │
  │            互相关 > 阈值 AND 能量包络相似 → is_echo = True   │
  │            互相关 < 阈值 → is_echo = False (真人说话)        │
  │                                                             │
  │  打断 = vad_trigger AND (NOT is_echo)                       │
  │                                                             │
  │  即: VAD 认为有人在说话 AND Echo Judge 认为不是回声           │
  │       → 确认为真实打断, 发送 Interrupt Signal 到 PC          │
  └─────────────────────────────────────────────────────────────┘
```

**Echo Judge 实现要点**：

```python
# 核心逻辑（伪代码）
class EchoJudge:
    def __init__(self, threshold=0.6, consecutive_frames=10):
        self.threshold = threshold        # 互相关阈值
        self.consecutive_frames = consecutive_frames  # 连续帧确认

    def judge(self, mic_chunk, speaker_ref_chunk):
        """
        mic_chunk: 当前麦克风音频帧 (40ms)
        speaker_ref_chunk: 对应时间段的扬声器播放音频帧 (40ms)
        返回: is_echo (bool)
        """
        # 1. 归一化互相关
        correlation = np.correlate(mic_chunk, speaker_ref_chunk, mode='full')
        peak = np.max(np.abs(correlation)) / (
            np.sqrt(np.sum(mic_chunk**2)) * np.sqrt(np.sum(speaker_ref_chunk**2))
        )

        # 2. 能量包络对比
        mic_env = np.mean(np.abs(mic_chunk))
        ref_env = np.mean(np.abs(speaker_ref_chunk))
        energy_ratio = min(mic_env, ref_env) / max(mic_env, ref_env)

        # 3. 综合判决
        return (peak > 0.5) or (peak > 0.3 and energy_ratio > 0.6)
```

**打断处理流程（PC 端 — 销毁重建策略）**：

```
  Pi 端 Interrupt Signal 到达 Orchestrator

  ┌─────────────────────────────────────────────────────────┐
  │  SessionManager.state → INTERRUPT                        │
  │                                                         │
  │  不增量清理 → 直接销毁 + 重建:                             │
  │                                                         │
  │  1. ASR 容器: 发送 reset 指令 → 销毁当前识别 session      │
  │               → 立即创建新 session (干净状态，无脏数据)     │
  │                                                         │
  │  2. LLM 容器: 发送 cancel 指令 → Ollama /api/generate     │
  │               cancel → 丢弃当前生成的 token 流             │
  │                                                         │
  │  3. TTS 容器: 发送 flush 指令 → 清空合成缓冲              │
  │                                                         │
  │  4. Pi 端:    发送 stop_playback → 立即停止扬声器播放     │
  │                                                         │
  │  5. 等待全部节点回报 ACK                                  │
  │                                                         │
  │  6. state → LISTEN (新 session_id，干净状态)              │
  │     被打断的文本记入 history (标记 interrupted=true)       │
  └─────────────────────────────────────────────────────────┘
```

> **为什么销毁重建而非增量清理**：FunASR 流式模型的 encoder/decoder 内部状态复杂，增量清理容易出现残留文本（下一句识别带上一句的尾巴）。直接销毁 session、创建新 session 是最可靠的方案，延迟代价 < 20ms，远小于 ASR 推理延迟。

**打断延迟分析**：

| 环节 | 延迟 | 说明 |
|------|------|------|
| VAD 连续检测到语音 | < 100ms | 10帧 × 10ms（连续确认，防止瞬时噪声误触发） |
| Echo Judge 判决 | < 5ms | 互相关计算（40ms 窗口 × 16kHz = 640点） |
| WebSocket 传输 Interrupt Signal | < 5ms | LAN |
| Orchestrator 发出 cancel 指令 | < 5ms | 容器间 localhost HTTP |
| Pi 端停止播放 | < 10ms | PyAudio 停止流 |
| 各节点 ACK + 状态切换 | < 20ms | |
| **总计（用户感知打断延迟）** | **< 150ms** | 用户说话后 150ms 内停止播放，体验流畅 |

### 5.4 "聪明断句" — 声学+标点+语义三模断句

单纯的 VAD 静默检测无法区分"句中停顿"和"句末停顿"。本方案结合**声学信号(VAD)**、**标点预测(ASR内置)**、**语义完整性(规则+模型)** 三重信号做断句决策。

```
断句决策器（三模融合）:

  输入：实时 VAD 概率流 + ASR 增量文本流
  输出：出句时机决策

  ┌──────────────────────────────────────────────────────────────┐
  │                                                              │
  │  路径A: 标点触发出句（最快）                                    │
  │    ASR 增量文本包含句子结束标点（。！？）                       │
  │    AND Silero VAD silence > 200ms                             │
  │    → 立即出句                                                 │
  │                                                              │
  │  路径B: 静默超时触发出句（兜底）                                │
  │    Silero VAD silence > 800ms                                 │
  │    → 强制出句（即使没有句末标点）                               │
  │                                                              │
  │  路径C: 长文本保护性截断                                       │
  │    ASR 缓冲区文本 > 100 字                                    │
  │    AND Silero VAD silence > 300ms                             │
  │    → 出句（防止单次输入过长拖垮 LLM）                           │
  │                                                              │
  │  不发句的情况（语义完整性判断 — 延长等待）:                      │
  │    ASR 文本尾部为不完整短语:                                    │
  │    "那个"、"就是"、"然后"、"还有"、"因为"、"比如说"、           │
  │    "嗯"、"呃"等填充词结尾                                      │
  │    AND Silero VAD silence < 1500ms                            │
  │    → 延长等待窗口，暂不出句                                    │
  │                                                              │
  └──────────────────────────────────────────────────────────────┘

  参数调优：
  ├── silence_punct:  200ms   （句末标点后短静默，快速出句）
  ├── silence_normal: 500ms   （无标点中等静默，等待更多内容）
  ├── silence_force:  800ms   （长静默强制出句，兜底）
  ├── silence_extend: 1500ms  （不完整短语时的最大容忍静默）
  ├── text_max_len:   100chars（长文本保护性截断）
  └── incomplete_tails: ["那个","就是","然后","还有","因为","比如说","嗯","呃"]
```

**关键点**：
- FunASR Paraformer 流式模型的标点预测是断句的核心锚点——只要看到 `。！？` 且用户确实停下来了（VAD silence），就说明一句话说完了
- 语义完整性检查解决"用户还没想好说什么"的场景：如果 ASR 最后两个字是"那个..."，说明用户还在组织语言，不能急着出句
- 三条路径是 OR 关系，任意一条满足即出句；不完整短语判断是一个"暂缓出句"的反向规则

### 5.5 "接续对话" — 一次唤醒、持续交互

```
接续对话流程：

  [唤醒词] → [问题1] → [回答1] → [问题2] → [回答2] → ... → [静默5分钟] → [回到IDLE]

  实现要点：
  ├── 唤醒后进入 LISTEN 状态，每次回答结束后自动进入 LISTEN（不需再次唤醒）
  ├── 每次 LISTEN 有 5s 的语音等待窗口（可配置）
  ├── 每次 LISTEN 可触发多轮追问，LLM 上下文持续累积
  ├── 静默超过 5min → 自动回到 IDLE 状态，释放 LLM context
  └── 会话中可随时说"再见"、"退下"等关键词主动结束会话
```

### 5.6 "聊天记忆" — 双层记忆系统

```
记忆架构：

  Layer 1: 短期记忆（会话上下文）
  ├── 存储：当前 LLM context window（最多 32K tokens）
  ├── 内容：本轮会话的全部对话轮次
  └── 管理：超过 context window 时，旧轮次自动滑出

  Layer 2: 长期记忆（持久化存储）
  ├── 结构化对话记录 → SQLite
  │   └── 表结构: (id, session_id, role, content, timestamp, interrupted)
  │
  ├── 语义摘要 → ChromaDB 向量库
  │   ├── 每次会话结束，LLM 生成 2-3 句摘要
  │   ├── 摘要文本 → embedding → ChromaDB 存储
  │   └── 新会话启动时，检索相似历史摘要作为 context 前缀
  │
  └── 用户画像 → JSON 文件
      └── 提取用户偏好、常用话题、称呼偏好等
```

**启动时记忆加载流程**：

```
程序启动
  ├── 加载最近 N 轮对话（SQLite）→ 注入 LLM system prompt
  ├── 语义检索相关历史（ChromaDB）→ 作为参考上下文注入
  └── 加载用户画像 → 个性化 system prompt
```

---

## 六、通信协议设计

### 6.1 WebSocket 通道定义

```
WebSocket URL: ws://<pc_ip>:8765

消息格式（JSON + Binary 混合）:
{
  "channel": "audio_in | audio_out | control | event",
  "type": "audio_frame | vad_event | interrupt | wake_word | heartbeat | ...",
  "seq": 12345,           // 单调递增序列号
  "timestamp": 1717200000.123,  // Unix 时间戳（秒），精确到毫秒
  "payload": { ... }      // 根据 type 变化
}
```

### 6.2 关键消息类型

| Channel | Type | 方向 | 说明 |
|---------|------|------|------|
| `audio_in` | `audio_frame` | Pi → PC | 16kHz/16bit PCM 音频帧（40ms/帧） |
| `audio_out` | `audio_frame` | PC → Pi | TTS 合成音频帧，可流式播放 |
| `control` | `vad_start` | Pi → PC | VAD 检测到语音开始 |
| `control` | `vad_end` | Pi → PC | VAD 检测到语音结束 |
| `control` | `interrupt` | Pi → PC | 打断信号（播报中用户说话） |
| `control` | `wake_word` | Pi → PC | 唤醒词检测到 |
| `control` | `stop_tts` | PC → Pi | 服务端要求停止播放 |
| `control` | `session_end` | PC → Pi | 会话结束（超时或主动） |
| `event` | `heartbeat` | 双向 | 心跳包（每3s） |
| `event` | `state_change` | PC → Pi | 状态机状态变更通知 |
| `event` | `error` | 双向 | 错误报告 |

---

## 七、性能指标与优化

### 7.1 延迟预算（端到端）

| 环节 | 目标延迟 | 说明 |
|------|----------|------|
| VAD 检测说话开始 | < 30ms | 3帧 × 10ms |
| 语音结束→断句决策 | 200-1500ms | 标点+静默+语义完整性复合判断 |
| ASR 推理（流式） | < 200ms | 每40ms输入一帧，输出延迟约5帧 |
| LLM 首 token | < 800ms | Qwen3:14B-Q4，GPU warm |
| TTS 首音频帧 | < 200ms | CosyVoice 2 流式合成 |
| 网络传输（LAN） | < 10ms | Wi-Fi 5GHz 或以太网 |
| **端到端首音延迟** | **< 1.5s** | 从用户说完到听到首音 |
| **打断延迟** | **< 150ms** | 用户说话→Echo Judge+VAD确认→Pi停止播放 |

### 7.2 优化策略

```
  1. LLM 加速
     ├── Ollama 开启 flash_attention（OLLAMA_FLASH_ATTENTION=1）
     ├── 设置 num_predict 限制单轮最大 token
     ├── 使用 Q4_K_M 量化（精度/速度最优平衡点）
     └── KV cache 复用（同会话内）

  2. ASR 加速
     ├── FunASR 热词增强（注入上下文词汇）
     └── GPU FP16 推理

  3. TTS 加速
     ├── CosyVoice 2 流式模式（stream=True）
     └── 首句 warm-up 缓存

  4. 网络
     ├── 树莓派优先使用有线以太网
     └── WebSocket 使用 binary 模式传输音频帧
```

---

## 八、运行流程

### 8.1 启动流程

```
PC 端（Docker Compose 一键启动）:
  docker-compose up -d
    ├── 1. Ollama 容器启动（确保 Qwen3:14B 模型已拉取）
    ├── 2. ASR 容器启动 → 加载 FunASR Paraformer 模型（GPU）
    ├── 3. TTS 容器启动 → 加载 CosyVoice 2 模型 + 预热
    ├── 4. Memory 容器启动 → ChromaDB + SQLite 初始化
    └── 5. Orchestrator 容器启动 → WebSocket Server 监听 8765

树莓派端:
  1. XVF3800 上电，确认 USB 音频设备识别（16kHz/1ch）
  2. 加载 openWakeWord、Silero VAD、Echo Judge
  3. 连接 WebSocket → PC (ws://<pc_ip>:8765)
  4. 进入 IDLE 状态，等待唤醒词
```

### 8.2 一次完整交互时序

```
用户:  "小智小智"                    [唤醒]
Pi:    wake_word → PC
PC:    IDLE → LISTEN，开始接收音频

用户:  "今天天气怎么样？"
Pi:    audio_frame × N → PC
PC:    ASR 流式识别 → "今天天气怎么样？"
       VAD end + 标点? → 触发出句

PC:    LISTEN → THINK
       LLM prompt = system_prompt + memory_context + "今天天气怎么样？"
       LLM stream → "今天天气不错..."

PC:    THINK → SPEAK
       TTS 逐句合成 → audio_out → Pi 流式播放

用户:  "明天呢？"    [打断/追问]
Pi:    VAD 检测到语音 → interrupt → PC
PC:    flush TTS + cancel LLM → LISTEN
       接收新语音 → ASR → LLM（context 保持）→ TTS → 播报

...持续交互...

[5分钟无交互]
PC:   自动回到 IDLE，本次会话 memory 持久化
```

**Docker Compose 启动文件（server/docker-compose.yml）**：

```yaml
version: "3.8"
services:
  ollama:
    image: ollama/ollama:latest
    ports: ["11434:11434"]
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - OLLAMA_KEEP_ALIVE=-1
      - OLLAMA_FLASH_ATTENTION=1

  asr_node:
    build: ./asr_node
    ports: ["8100:8100"]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - FUNASR_MODEL=paraformer-online

  tts_node:
    build: ./tts_node
    ports: ["8101:8101"]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  memory_node:
    build: ./memory_node
    ports: ["8102:8102"]
    volumes:
      - ../data:/app/data

  orchestrator:
    build: ./orchestrator
    ports: ["8765:8765"]
    depends_on: [asr_node, ollama, tts_node, memory_node]
    environment:
      - ASR_URL=http://asr_node:8100
      - LLM_URL=http://ollama:11434
      - TTS_URL=http://tts_node:8101
      - MEMORY_URL=http://memory_node:8102

volumes:
  ollama_data:
```

---

## 九、依赖清单

### 树莓派端（requirements_pi.txt）

```
# 音频
pyaudio==0.2.14
numpy==1.26.4
scipy==1.13.0          # Echo Judge 互相关计算

# 唤醒词
openwakeword==0.6.0

# VAD
silero-vad==5.1

# WebSocket
websockets==12.0

# 配置
pyyaml==6.0.2
```

### PC 端 — 各容器独立依赖（Dockerfile 各自管理，无冲突）

**ASR 容器（asr_node/requirements.txt）**：
```
funasr==1.1.0
modelscope==1.17.0
fastapi==0.115.0
uvicorn[standard]==0.30.0
```

**TTS 容器（tts_node/requirements.txt）**：
```
# CosyVoice 2 — 从 https://github.com/FunAudioLLM/CosyVoice 安装
fastapi==0.115.0
uvicorn[standard]==0.30.0
edge-tts==6.1.0         # 降级方案
```

**Memory 容器（memory_node/requirements.txt）**：
```
chromadb==0.5.0
sqlalchemy==2.0.35
fastapi==0.115.0
uvicorn[standard]==0.30.0
```

**Orchestrator 容器（orchestrator/requirements.txt）**：
```
fastapi==0.115.0
uvicorn[standard]==0.30.0
websockets==12.0
httpx==0.27.0            # 异步 HTTP 客户端（调用各节点）
pyyaml==6.0.2
```

---

## 十、风险与缓解

| 风险 | 概率 | 缓解措施 |
|------|------|----------|
| 回声误触发打断（AEC残留→VAD误判） | **高** | 双模判决：Echo Judge（互相关）+ Silero VAD AND 逻辑；连续多帧确认防瞬时噪声 |
| CosyVoice 2 与 FunASR 依赖冲突 | **高** | Docker 容器隔离部署，各自独立 Python 环境，通过 localhost HTTP 通信 |
| 打断时 ASR 状态残留（下一句带上一句的文本） | 中 | 销毁重建策略：打断时不增量清理，直接销毁 ASR session + 新建 |
| Ollama 模型加载慢（冷启动） | 中 | 设置 OLLAMA_KEEP_ALIVE=-1，模型常驻显存 |
| ASR 在极噪杂环境精度下降 | 低 | XVF3800 硬件 DSP 已做 NS 降噪 + 软件层 ASR 前端 VAD 增强 |
| LLM 生成过长导致 TTS 延迟大 | 低 | system prompt 限制回复长度 < 100 字；流式 TTS 逐句合成 |
| Pi 端 WebSocket 断连 | 低 | 自动重连 + 指数退避；断连时 LED 指示灯提醒 |
| XVF3800 USB 带宽不足 | 低 | 仅传输 1ch/16kHz 波束成形后数据，带宽需求 < 256kbps |
| 多说话人场景无法区分 | 中 | XVF3800 DOA + 波束成形锁定最强声源方向；VAD 忽略低置信度帧 |
| Docker 容器管理复杂度 | 低 | docker-compose up 一键启动；健康检查自动重启异常容器 |

---

## 十一、国产化替代方案（可选）

如未来需要完全国产化或更低成本部署：

| 组件 | 原选型 | 替代方案 |
|------|--------|----------|
| 开发板 | 树莓派5 | 香橙派5 / 瑞芯微RK3588 开发板 |
| 麦克风阵列 | XVF3800 | 科大讯飞 6麦环阵 / 自研 Respeaker 4-Mic |
| ASR | FunASR | 讯飞语音听写（云端） / Whisper.cpp |
| LLM | Qwen3:14B | DeepSeek-V3-Lite / ChatGLM4-9B |
| TTS | CosyVoice 2 | 讯飞语音合成 / ChatTTS |
| GPU | NVIDIA | 摩尔线程 MTT S80 / 昇腾 Atlas 300V |

---

## 十二、后续演进方向

1. **多模态交互**：接入摄像头，配合视觉模型实现"指哪说哪"
2. **情感感知**：SenseVoice 情感识别 + CosyVoice 情感合成，实现有感情的对话
3. **设备控制**：MCP 协议接入智能家居、家电控制（"打开客厅灯"）
4. **声纹识别**：多人场景区分说话人，个性化服务
5. **知识库 RAG**：接入本地文档/知识库，实现基于私有知识的问答

---

*方案版本: v1.1 | 日期: 2026-06-01 | 修订: +Docker Compose容器隔离 +双模打断判决 +销毁重建策略 +三模断句(含语义完整性)*
