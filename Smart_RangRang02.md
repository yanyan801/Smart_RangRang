# Smart RangRang — 全本地智能语音交互系统技术方案

> 版本 v2.0 | 2026-06-01 | 全本地部署，零云端依赖，数据不出局域网

---

## 一、概述

### 1.1 硬件条件

| 设备 | 规格 | 角色 |
|------|------|------|
| 树莓派 5 | 16GB RAM，ARM Cortex-A76 × 4 | 音频前端：采集、播放、VAD、唤醒词、回声判决 |
| XVF3800 4 麦环形阵列 | USB 即插即用，硬件 DSP 内置 AEC / NS / VNR / AGC / DOA | 物理层拾音保障 |
| 扬声器 | 3.5mm 或 USB 音频输出 | TTS 语音播报 |
| Windows PC | 32GB RAM，16GB VRAM（NVIDIA），Ollama Server | 算力中心：ASR / LLM / TTS / 会话管理 / 记忆 |

### 1.2 设计目标

| 目标 | 说明 | 对标体验 |
|------|------|----------|
| **听得清** | 硬件 DSP + 软件补充，家居环境拾音准确率 > 95% | 智能音箱级 |
| **不卡死** | 全链路超时 + 看门狗 + 进程隔离，单点故障不扩散 | 服务 7×24 自愈 |
| **随时打断** | 双模判决（VAD + Echo Judge），播放中用户说话 < 150ms 响应 | 豆包级 |
| **聪明断句** | 声学 + 标点 + 语义三模决策，区分句中停顿与句末停顿 | 豆包级 |
| **接续对话** | 一次唤醒，持续交互，无需反复唤醒 | 自然对话感 |
| **聊天记忆** | SQLite + ChromaDB，重启后自动加载上下文 | 记忆连续性 |

### 1.3 核心设计原则

| 原则 | 说明 |
|------|------|
| **全链路流式** | ASR → LLM → TTS 全部流式处理，首音延迟 < 1.5s |
| **双向双工** | 麦克风采集与扬声器播放独立线程，支持 barge-in 打断 |
| **进程隔离、非容器** | Windows 原生 conda 环境 + 多进程，避免 Docker GPU Passthrough 的虚拟化开销 |
| **销毁重建** | 打断时不增量清理模型状态，直接销毁 session 重建，确保状态干净 |
| **边侧分离** | Pi 负责轻量音频处理，PC 负责 GPU 推理 |
| **本地闭环** | 零云端流量，隐私数据不出局域网 |

---

## 二、硬件拓扑

```
┌──────────────────────────────────────────────────────────────────┐
│                      Windows PC (算力中心)                         │
│                                                                    │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                   进程管理层 (start_all.bat)                 │  │
│  │                                                            │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐   │  │
│  │  │ asr_env     │  │ tts_env     │  │ memory_env      │   │  │
│  │  │ SenseVoice  │  │ Edge-TTS    │  │ ChromaDB+SQLite │   │  │
│  │  │ +CT-Trans   │  │ (主选)       │  │ Port: 8102      │   │  │
│  │  │ Port: 8100  │  │ Port: 8101  │  │                 │   │  │
│  │  └──────┬──────┘  └──────┬──────┘  └────────┬────────┘   │  │
│  │         │                │                   │            │  │
│  │         └────────────────┼───────────────────┘            │  │
│  │                          │ localhost HTTP                  │  │
│  │  ┌───────────────────────┴────────────────────────────┐  │  │
│  │  │           Orchestrator (main_env)                    │  │  │
│  │  │           WebSocket Server (FastAPI) :8765           │  │  │
│  │  │           + SessionManager + 状态机                  │  │  │
│  │  └───────────────────────┬────────────────────────────┘  │  │
│  │                          │                                │  │
│  │  ┌───────────────────────┴────────────────────────────┐  │  │
│  │  │           Ollama (原生 Windows 安装)                 │  │  │
│  │  │           Qwen3:14B-Q4_K_M  Port: 11434            │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────┬─────────────────────────────────────┘
                             │  LAN WebSocket (ws://pc_ip:8765)
┌────────────────────────────┼─────────────────────────────────────┐
│                    树莓派 5 (音频前端)                              │
│  ┌────────────────────────┴─────────────────────────────────┐   │
│  │              WebSocket Client (asyncio)                    │   │
│  ├───────────────────┬───────────────────────────────────────┤   │
│  │  Input Pipeline   │  Output Pipeline                      │   │
│  │  麦克风 → SpeexDSP│  TTS 音频帧 → PyAudio 播放             │   │
│  │  AEC → VAD →      │  (收到 stop_playback 立即停止)        │   │
│  │  Echo Judge →     │                                       │   │
│  │  音频帧发送       │                                       │   │
│  ├───────────────────┴───────────────────────────────────────┤   │
│  │  openWakeWord (唤醒词)  Silero VAD  Echo Judge            │   │
│  ├───────────────────────────────────────────────────────────┤   │
│  │  XVF3800 USB 音频设备 (16kHz / 1ch / 16bit)               │   │
│  │  硬件: AEC + 波束成形 + NS + VNR + AGC + DOA              │   │
│  └───────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 三、模型选型

### 3.1 语音识别（ASR）

| 候选模型 | 流式支持 | 中文精度 | 标点预测 | 显存 | 结论 |
|----------|----------|----------|----------|------|------|
| **SenseVoice-small** | **原生流式** | 高 | 需独立模型 | 1.5GB | **主选** |
| FunASR paraformer-online | 流式 | 中（流式版精度折损） | 内置但弱 | 2.5GB | 备选 |
| Faster-Whisper large-v3 | 非流式 | 中 | 有 | 2.5GB | 不推荐（非流式） |

**最终选择：SenseVoice-small + CT-Transformer 独立标点模型**

- SenseVoice 流式 API 简洁，状态管理清晰，比 FunASR paraformer-online 更稳定
- CT-Transformer（`punc_ct`）是专门的标点恢复模型，不依赖 ASR 的局部上下文，标点精度显著高于流式 ASR 内置标点
- 标点是三模断句的核心锚点，必须优先保证精度
- 标点模型显存 ~0.5GB，延迟 ~10ms，代价可忽略

### 3.2 大语言模型（LLM）

| 候选模型 | 显存(Q4) | 中文能力 | 推理速度 | 结论 |
|----------|----------|----------|----------|------|
| **Qwen3:14B-Q4_K_M** | ~9GB | 优秀 | 25 tok/s | **主选** |
| Qwen3:8B-Q4_K_M | ~5.5GB | 良好 | 40 tok/s | 速度优先降级 |
| DeepSeek-R1-14B-Q4 | ~9GB | 优秀 | 20 tok/s | 备选 |

**最终选择：Qwen3:14B-Q4_K_M**

- Qwen3 系列对话能力、指令遵循、中文理解均为开源第一梯队
- 14B 模型 + 32K context 在 16GB 显存下运行稳定
- Ollama 原生 Windows 安装，无需 Docker，管理简便

### 3.3 语音合成（TTS）

| 候选模型 | 稳定性 | 流式 | 中文自然度 | GPU 需求 | 结论 |
|----------|--------|------|-----------|----------|------|
| **Edge-TTS** | **极高** | 原生流式 | 高 | 零 | **主选** |
| CosyVoice 2 | 中（流式模式有卡死风险） | 支持 | 极高 | 2.5GB | 可选增强 |
| ChatTTS | 中 | 部分支持 | 高 | 2GB | 不推荐 |

**最终选择：Edge-TTS 主选 + CosyVoice 2 可选增强**

- Edge-TTS 是微软免费 TTS 服务的前端封装，通过 `edge-tts` Python 库调用微软语音合成引擎，**本地运行、零 GPU、零费用、100% 稳定**
- 不使用云端 API endpoint，"全本地"指模型和推理都在本地设备完成；Edge-TTS 的合成引擎是本地运行
- 释放 CosyVoice 2 所需的 2.5GB 显存，可用于更长的 LLM context 或给标点模型
- 如未来需要更情感化的音色，可启用 CosyVoice 2 增强模式

### 3.4 唤醒词检测

**选择：openWakeWord v0.6.0**

- Pi 5 CPU 推理，ONNX Runtime，占用 < 5% CPU
- 自定义唤醒词（录制 3-5 个样本即可）
- 同时支持多唤醒词并行检测
- 默认唤醒词："小智小智"

### 3.5 语音活动检测（VAD）

**选择：Silero VAD v5.1**

- Pi 5 CPU 推理，< 3ms/帧
- 输出逐帧语音概率（0-1），支持多语种
- 与 Echo Judge 配合做双模打断判决

### 3.6 模型部署总表

| 模型 | 部署位置 | 运行环境 | 显存 | 内存 |
|------|----------|----------|------|------|
| openWakeWord | Pi 5 CPU | ONNX Runtime | — | ~50MB |
| Silero VAD | Pi 5 CPU | ONNX Runtime | — | ~1MB |
| SpeexDSP AEC | Pi 5 CPU | C 库 | — | ~3MB |
| SenseVoice-small | PC GPU | conda asr_env | ~1.5GB | — |
| CT-Transformer | PC GPU | conda asr_env | ~0.5GB | — |
| Qwen3:14B-Q4_K_M | PC GPU | Ollama (原生Win) | ~10GB* | — |
| Edge-TTS | PC CPU | conda tts_env | — | ~200MB |
| CosyVoice 2 | PC GPU (可选) | conda tts_env | ~2.5GB | — |

> *含 32K context KV cache。9GB 模型权重 + ~1GB KV cache = ~10GB。

**PC 16GB 显存占用：10GB(LLM) + 2.0GB(ASR+标点) + 系统预留 1GB = ~13GB，剩余约 3GB 安全余量。**

---

## 四、软件架构

### 4.1 进程架构（Windows 原生）

```
┌─────────────────────────────────────────────────────────────────┐
│                     Windows PC 进程拓扑                           │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Ollama.exe (原生 Windows 服务)                           │    │
│  │  ollama serve → localhost:11434                          │    │
│  │  模型常驻显存 (OLLAMA_KEEP_ALIVE=-1)                      │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ┌──────────────────┐  ┌──────────────────┐                     │
│  │ ASR Service      │  │ TTS Service      │                     │
│  │ (conda asr_env)  │  │ (conda tts_env)  │                     │
│  │ uvicorn :8100    │  │ uvicorn :8101    │                     │
│  │                  │  │                  │                     │
│  │ POST /recognize  │  │ POST /synthesize │                     │
│  │  (WebSocket 流式) │  │  (WebSocket 流式) │                     │
│  │ POST /reset      │  │ POST /flush      │                     │
│  └──────────────────┘  └──────────────────┘                     │
│                                                                   │
│  ┌──────────────────┐                                            │
│  │ Memory Service   │                                            │
│  │ (conda memory_env)                                            │
│  │ uvicorn :8102    │                                            │
│  │                  │                                            │
│  │ POST /chat/save  │                                            │
│  │ POST /chat/load  │                                            │
│  │ POST /chat/search│                                            │
│  └──────────────────┘                                            │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Orchestrator (conda main_env)                             │    │
│  │ uvicorn :8765                                             │    │
│  │                                                           │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────┐  │    │
│  │  │ 状态机    │ │打断仲裁器│ │断句决策器│ │ 超时看门狗 │  │    │
│  │  │SessionMgr│ │双模判决  │ │三模融合  │ │ 防卡死    │  │    │
│  │  └──────────┘ └──────────┘ └──────────┘ └───────────┘  │    │
│  │                                                           │    │
│  │  对外: WebSocket Server :8765 (Pi 连接)                    │    │
│  │  对内: httpx AsyncClient 调用各 Service                    │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 进程间通信

```
Pi ────WebSocket────► Orchestrator (:8765)
                          │
                          ├── HTTP/WebSocket ──► ASR Service (:8100)
                          ├── HTTP ───────────► Ollama (:11434)
                          ├── HTTP/WebSocket ──► TTS Service (:8101)
                          └── HTTP ───────────► Memory Service (:8102)
```

- **Pi ↔ Orchestrator**：WebSocket，多通道（audio_in / audio_out / control / event）
- **Orchestrator ↔ 各 Service**：localhost HTTP + WebSocket 流式传递
- 每个 Service 独立进程 + 独立 conda 环境，依赖完全隔离
- 单 Service 崩溃不影响其他进程，Orchestrator 检测到超时后自动重启子进程

### 4.3 对话状态机

```
                    wake_word
  ┌─────────┐                    ┌─────────┐
  │  IDLE   │ ─────────────────> │ LISTEN  │
  │ (空闲)   │                    │ (倾听)   │
  └─────────┘                    └────┬────┘
       ^                              │ sentence_ready
       │                              v
       │                         ┌─────────┐
       │                         │ THINK   │
       │                         │ (思考)   │
       │                         └────┬────┘
       │                              │ first_token
       │                              v
       │ ┌─────────┐             ┌─────────┐
       │ │INTERRUPT│ <────────── │ SPEAK   │
       │ │(被打断) │  interrupt  │ (播报)   │
       │ └────┬────┘             └─────────┘
       │      │                      │
       │      │ 销毁重建流程:          │ tts_done
       │      │ ────────────          │ (无新语音)
       │      │ 1. → Pi: stop_        │
       │      │    playback           │
       │      │ 2. → ASR: reset       │
       │      │    (销毁session+重建)  │
       │      │ 3. → Ollama: cancel   │
       │      │    (断开HTTP连接)      │
       │      │ 4. → TTS: flush       │
       │      │    等待所有 ACK       │
       │      │ 5. 新 session_id      │
       │      │ 6. state → LISTEN    │
       │      v                       v
       │  ┌────────────────────────────┐
       │  │ 所有节点 ACK 后             │
       │  │ → state = LISTEN           │
       │  │ (新 session, 干净状态)      │
       │  └────────────────────────────┘
       │
       └──── timeout(5min 无交互) ────┘
```

**SessionManager 是唯一状态持有者**，各 Service（ASR / TTS / Memory）完全无状态。状态变更全部由 SessionManager 发起指令，各节点执行后回报 ACK。

---

## 五、项目目录结构

```
Smart_RangRang/
│
├── pi/                              # 树莓派端代码
│   ├── main.py                      # 入口
│   ├── audio_capture.py             # 音频采集（PyAudio, 16kHz/16bit/1ch）
│   ├── audio_player.py              # 音频播放（流式、可中断）
│   ├── vad_engine.py                # Silero VAD 封装
│   ├── echo_judge.py                # 回声判决器
│   ├── speex_aec.py                 # SpeexDSP AEC 软件补充
│   ├── wake_word.py                 # openWakeWord 封装
│   ├── ws_client.py                 # WebSocket 客户端（多通道）
│   ├── config_pi.yaml               # Pi 端配置
│   └── requirements_pi.txt          # Pi 端依赖
│
├── server/                          # PC 服务端代码
│   │
│   ├── start_all.bat                # Windows 一键启动脚本
│   ├── stop_all.bat                 # Windows 一键停止脚本
│   │
│   ├── orchestrator/                # Orchestrator（main_env）
│   │   ├── main.py                  # 入口（FastAPI + WebSocket）
│   │   ├── orchestrator.py          # 核心调度器
│   │   ├── session.py               # SessionManager（唯一状态持有者）
│   │   ├── sentence_judge.py        # 三模断句决策器
│   │   ├── ws_hub.py                # WebSocket 多通道管理
│   │   ├── config.yaml              # 配置
│   │   ├── prompts/
│   │   │   └── system_prompt.txt    # LLM 系统提示词
│   │   └── requirements.txt
│   │
│   ├── asr_service/                 # ASR Service（asr_env）
│   │   ├── server.py                # SenseVoice + CT-Transformer 服务
│   │   ├── stt_engine.py            # SenseVoice 流式推理引擎
│   │   ├── punctuation.py           # CT-Transformer 标点后处理
│   │   ├── config.yaml
│   │   └── requirements.txt
│   │
│   ├── tts_service/                 # TTS Service（tts_env）
│   │   ├── server.py                # Edge-TTS + CosyVoice 可选
│   │   ├── tts_edge.py              # Edge-TTS 引擎
│   │   ├── tts_cosyvoice.py         # CosyVoice 2 引擎（可选）
│   │   ├── config.yaml
│   │   └── requirements.txt
│   │
│   └── memory_service/              # Memory Service（memory_env）
│       ├── server.py                # ChromaDB + SQLite 服务
│       ├── chat_store.py            # SQLite 对话存储
│       ├── vector_store.py          # ChromaDB 语义检索
│       ├── config.yaml
│       └── requirements.txt
│
├── shared/                          # 共享协议
│   ├── protocol.py                  # 消息协议定义
│   └── audio_utils.py               # 音频格式转换
│
├── data/                            # 持久化数据
│   ├── chat_history.db              # SQLite 对话历史
│   └── chroma/                      # ChromaDB 向量库
│
├── conda_envs/                      # conda 环境创建脚本
│   ├── create_asr_env.bat
│   ├── create_tts_env.bat
│   ├── create_memory_env.bat
│   └── create_main_env.bat
│
└── README.md
```

---

## 六、通信协议设计

### 6.1 Pi ↔ PC WebSocket 协议

```
WebSocket URL: ws://<pc_ip>:8765

通用消息格式（JSON）:
{
  "channel": "audio_in | audio_out | control",
  "type": "...",
  "seq": 12345,
  "timestamp": 1717200000.123,
  "payload": { ... }
}
```

### 6.2 关键消息类型

**audio_in 通道（Pi → PC）**：

| Type | 格式 | 说明 |
|------|------|------|
| `audio_frame` | Binary | 16kHz/16bit/1ch PCM，40ms/帧（640 samples） |
| `audio_frame` | JSON meta | `{"seq": N, "vad_prob": 0.92, "is_echo": false}` 附每帧元信息 |

**audio_out 通道（PC → Pi）**：

| Type | 格式 | 说明 |
|------|------|------|
| `audio_frame` | Binary | TTS 合成音频帧（16kHz/16bit/1ch PCM） |
| `stream_end` | JSON | 当前 TTS 句子播放完毕 |

**control 通道（双向）**：

| Type | 方向 | 说明 |
|------|------|------|
| `wake_word` | Pi → PC | 唤醒词检测到 `{"word": "小智小智", "confidence": 0.93}` |
| `interrupt` | Pi → PC | 打断信号 `{"vad_frames": 10, "echo_score": 0.12}` |
| `vad_start` | Pi → PC | 用户开始说话 |
| `vad_end` | Pi → PC | 用户停止说话 `{"silence_duration_ms": 800}` |
| `stop_playback` | PC → Pi | 立即停止扬声器播放 |
| `state_change` | PC → Pi | `{"from": "SPEAK", "to": "LISTEN"}` |
| `session_end` | PC → Pi | 会话结束（超时或主动） |
| `heartbeat` | 双向 | 每 3s 发送一次 |

### 6.3 PC 内部 Service API

**ASR Service（:8100）**：

```
WebSocket /ws/recognize
  → 接收 audio_frame (binary PCM)
  ← 返回 {"type": "partial", "text": "今天天", "is_final": false}
     {"type": "final",   "text": "今天天气怎么样？", "is_final": true}

POST /reset
  → 销毁当前识别 session，重建新 session
  ← {"status": "ok", "new_session_id": "uuid"}
```

**TTS Service（:8101）**：

```
WebSocket /ws/synthesize
  → 发送 {"text": "今天天气不错..."}
  ← 返回 audio_frame (binary PCM) 流式传输
     {"type": "stream_end"}

POST /flush
  → 清空合成缓冲区，停止当前合成
  ← {"status": "ok"}
```

**Memory Service（:8102）**：

```
POST /chat/save    → {"session_id": "...", "messages": [...]}
POST /chat/load    → {"session_id": "..."}  ← {"messages": [...]}
POST /chat/search  → {"query": "..."}       ← {"results": [...]}
POST /chat/export  → {"format": "json"}     ← 全部对话记录
```

---

## 七、关键功能技术实现

### 7.1 "听得清" — 高品质拾音

```
层次化降噪管道：

Layer 1: XVF3800 硬件 DSP（USB 即插即用）
  ├── 4 麦波束成形 → 空间滤波，锁定最强声源
  ├── AEC（声学回声消除）→ 消除扬声器直达声 + 早期反射线性回声
  ├── ANS（自适应降噪）→ 稳态噪声抑制 > 20dB
  ├── VNR（语音对噪声比）→ 实时评估信号质量
  ├── DOA（到达方向）→ 声源角度估计
  └── AGC（自动增益控制）→ 输出电平归一化

Layer 2: Pi 端软件补充
  ├── SpeexDSP AEC → 消除 XVF3800 AEC 无法处理的非线性失真残留
  │   输入: mic_signal, speaker_ref_signal
  │   输出: 进一步消除回声后的干净信号
  │   CPU 开销: < 3%（Pi 5, 16kHz 单通道）
  │
  └── 高通滤波 (80Hz, 6dB/oct) → 去除空调/风扇低频噪声

Layer 3: ASR 前端（PC 端）
  └── SenseVoice 内置特征增强 → 频域归一化，提升可懂度
```

> XVF3800 的 AEC 能消除线性回声（扬声器→麦克风的直达声和早期反射），但大音量下的谐波失真无法被线性 AEC 处理。Layer 2 的 SpeexDSP AEC + Echo Judge 是**打断准确率**的关键保障。

### 7.2 "不卡死" — 全链路防阻塞

```
防卡死三层体制：

1. 组件级超时
   ├── ASR: 10s 无 partial result → timeout, 重置 ASR session
   ├── LLM: 15s 无首 token → 超时降级回复 ("我正在想，请稍等...")
   ├── TTS: 单句 5s 未合成完 → 跳过当前句，合成下一句
   └── WebSocket: 3s 心跳, PC 5s 无响应 → Pi 自动重连 (指数退避)

2. 全局看门狗（Orchestrator 内置）
   ├── 整个对话 30s 无任何进展 → 主动回 IDLE
   ├── 连续 3 次组件超时 → 通知 Pi 端播放提示音 ("服务繁忙，稍等哦")
   └── 进程健康检查: 每 10s 检查各 Service 是否存活

3. 进程隔离
   ├── ASR / TTS / Memory 各自独立进程 + 独立 conda 环境
   ├── 单进程崩溃不影响其他 Service
   └── Orchestrator 检测到子进程异常 → 自动重启

异步流式管道 (asyncio.Queue 解耦):
  audio_frame_queue: asyncio.Queue   # Pi → ASR: 音频帧
  asr_text_queue: asyncio.Queue      # ASR → Orchestrator: 识别文本
  llm_token_queue: asyncio.Queue     # LLM → Orchestrator: token 流
  tts_text_queue: asyncio.Queue      # Orchestrator → TTS: 待合成文本句
  tts_audio_queue: asyncio.Queue     # TTS → Pi: 合成音频帧
  interrupt_queue: asyncio.Queue     # Pi → Orchestrator: 打断信号 (高优先级)
```

### 7.3 "随时打断" — 双模 Barge-In

```
打断检测 = 声学VAD AND (NOT 回声判决)

┌──────────────────────────────────────────────────────────────┐
│                    双模判决管道 (Pi 端)                        │
│                                                              │
│  speaker_ref_buffer ← 即将播放的 TTS 音频 (环形缓冲, 1s)       │
│  mic_buffer         ← 麦克风实时采集 (环形缓冲, 1s)            │
│                                                              │
│  路径 A: Silero VAD (声学语音检测)                             │
│    ├── 逐帧 (10ms/帧) 输出 speech_prob                        │
│    ├── 连续 10 帧 speech_prob > 0.7 → vad_trigger = True      │
│    └── 防止瞬时噪声误触发                                      │
│                                                              │
│  路径 B: Echo Judge (回声判决器)                               │
│    ├── 输入: mic_chunk (40ms) + speaker_ref_chunk (40ms)      │
│    ├── 步骤 1: 归一化互相关计算                                │
│    │   peak = max(correlate(mic, ref)) / (||mic|| × ||ref||) │
│    ├── 步骤 2: 能量包络对比                                    │
│    │   env_ratio = min(mean_abs(mic), mean_abs(ref))          │
│    │             / max(mean_abs(mic), mean_abs(ref))          │
│    └── 判决:                                                  │
│        peak > 0.5                     → is_echo = True        │
│        peak > 0.3 AND env_ratio > 0.6 → is_echo = True        │
│        其他                           → is_echo = False       │
│                                                              │
│  打断判决 = vad_trigger AND (NOT is_echo)                     │
│            → 确认为真实人声 → 发送 Interrupt Signal             │
└──────────────────────────────────────────────────────────────┘
```

**Echo Judge 核心代码**：

```python
import numpy as np
from scipy.signal import correlate

class EchoJudge:
    """回声判决器：区分真实人声与扬声器回声"""

    def __init__(self, corr_threshold: float = 0.5,
                 env_threshold: float = 0.6,
                 corr_weak_threshold: float = 0.3):
        self.corr_threshold = corr_threshold
        self.env_threshold = env_threshold
        self.corr_weak_threshold = corr_weak_threshold

    def judge(self, mic_chunk: np.ndarray,
              speaker_ref_chunk: np.ndarray) -> bool:
        """
        返回 True = 是回声（不触发打断）
        返回 False = 不是回声（真人说话）
        """
        norm_factor = np.linalg.norm(mic_chunk) * np.linalg.norm(speaker_ref_chunk)
        if norm_factor < 1e-10:
            return False  # 静音，不判为回声

        # 归一化互相关峰值
        corr = correlate(mic_chunk, speaker_ref_chunk, mode='full')
        peak = np.max(np.abs(corr)) / norm_factor

        # 能量包络相似度
        mic_env = np.mean(np.abs(mic_chunk))
        ref_env = np.mean(np.abs(speaker_ref_chunk))
        energy_ratio = min(mic_env, ref_env) / max(mic_env, ref_env + 1e-10)

        # 强相关 → 回声; 中等相关 + 能量接近 → 回声
        if peak > self.corr_threshold:
            return True
        if peak > self.corr_weak_threshold and energy_ratio > self.env_threshold:
            return True
        return False
```

**打断处理流程（PC 端 — 销毁重建）**：

```
Pi 端 Interrupt Signal 到达 → Orchestrator:

  SessionManager.state → INTERRUPT

  销毁重建（不增量清理）:

  1. → Pi:     stop_playback（立即停止扬声器播放）
  2. → ASR:    POST /reset（销毁当前 session + 立即创建新 session）
  3. → Ollama: cancel 当前 /api/generate（断开 HTTP 连接）
  4. → TTS:    POST /flush（清空合成缓冲区）

  5. 等待所有 ACK

  6. 新 session_id = uuid4()
  7. 被打断的文本写入 history（interrupted=true）
  8. state → LISTEN
```

**延迟预算**：

| 环节 | 延迟 |
|------|------|
| VAD 连续帧确认 (10帧 × 10ms) | 100ms |
| Echo Judge 互相关计算 | < 5ms |
| WebSocket 传输 (Pi→PC) | < 5ms |
| Orchestrator → Pi stop_playback | < 5ms |
| Pi 停止 PyAudio 播放流 | < 10ms |
| **总计（用户感知打断延迟）** | **< 130ms** |

### 7.4 "聪明断句" — 三模融合断句

```
断句决策器: 声学 + 标点 + 语义  三重信号

输入: 实时 VAD 概率流 + ASR 增量文本流（带标点）
输出: 出句时机决策

┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  路径A: 标点触发（最快、最可靠）                               │
│    ASR 增量文本包含句末标点（。！？）                          │
│    AND VAD silence > 200ms                                   │
│    → 立即出句                                                 │
│                                                              │
│  路径B: 静默超时触发（兜底）                                   │
│    VAD silence > 800ms（用户长时间不说话）                    │
│    → 强制出句（即使没有句末标点）                              │
│                                                              │
│  路径C: 长文本保护性截断                                       │
│    累计文本 > 100 字                                          │
│    AND VAD silence > 300ms                                   │
│    → 出句（防止单次输入过长）                                  │
│                                                              │
│  暂缓出句（语义完整性判断）：                                    │
│    ASR 文本尾部为填充/不完整词汇:                               │
│    ["那个", "就是", "然后", "还有", "因为",                   │
│     "比如说", "嗯", "呃", "这个", "所以"]                     │
│    AND VAD silence < 1500ms                                  │
│    → 延长等待，暂不出句                                       │
│                                                              │
│  标点状态映射:                                                 │
│    ，、：；   → 句中停顿（不触发出句，除非超时）                │
│    。！？\n  → 句末标点（与短静默组合→触发出句）               │
│    无标点     → 等待静默超时（或语义不完整则继续等）            │
│                                                              │
└──────────────────────────────────────────────────────────────┘

参数表:
  silence_punct:   200ms   （句末标点后短静默）
  silence_force:   800ms   （静默强制出句）
  silence_max:    1500ms   （语义不完整时的最大容忍）
  text_max_len:    100字   （长文本保护性截断）
  incomplete_tails: ["那个","就是","然后","还有","因为",
                      "比如说","嗯","呃","这个","所以"]
```

**关键依赖**：CT-Transformer（独立标点模型）的标点精度是三模断句的核心锚点。流式 ASR 内置标点受限于局部上下文会漏标点、错标点，独立标点模型可以看到完整句子做全局最优解码，精度提升明显。

### 7.5 "接续对话" — 一次唤醒持续交互

```
接续对话流程:

  [唤醒词 "小智小智"]
       │
       ▼
  ┌─────────────────────────────────────────────┐
  │  LISTEN 状态                                 │
  │  ├── 等待语音（最多 5s 静默窗口）             │
  │  ├── 语音输入 → ASR → 出句 → LLM → TTS 播报   │
  │  ├── 播报结束后 → 自动回 LISTEN 状态          │
  │  │   （不需要再次唤醒）                       │
  │  ├── 可多轮追问，LLM context 持续累积          │
  │  ├── 静默 5min → 自动回 IDLE                  │
  │  └── 用户说 "再见/退下/休息吧" → 立即回 IDLE   │
  └─────────────────────────────────────────────┘
```

**LLM Context 窗口管理**：

- 默认 32K tokens context window
- 对话持续累积，旧轮次在超出 context 前通过滑窗逐步丢弃
- 每轮对话后记录到 Memory Service
- 超过 32K 时，保留 system prompt + 最近 N 轮对话 + 历史摘要（由 Memory 检索注入）

### 7.6 "聊天记忆" — 双层记忆

```
记忆架构:

Layer 1: 短期记忆（LLM Context Window）
  ├── 存储: Ollama context window（最多 32K tokens）
  ├── 内容: 当前会话全部对话轮次
  └── 管理: 超 context 时旧轮次滑出

Layer 2: 长期记忆（持久化存储）
  ├── SQLite 对话记录
  │   └── 表: chat_records
  │       (id, session_id, role, content, timestamp,
  │        interrupted, turn_index)
  │
  ├── ChromaDB 语义记忆
  │   ├── 每次会话结束: LLM 生成 2-3 句摘要
  │   ├── 摘要 → embedding (text2vec-base-chinese) → ChromaDB
  │   └── 新会话启动: 语义检索 top-5 相关历史摘要
  │       → 注入 LLM context 前缀
  │
  └── 用户画像 (JSON)
      └── 提取: 称呼偏好、常用话题、问答风格
          存储: data/user_profile.json
```

**启动时记忆加载**：

```
程序启动
  ├── 1. SQLite: 加载最近 20 轮对话 → 注入 system prompt
  ├── 2. ChromaDB: 语义检索 top-5 相关历史 → 注入 context
  └── 3. 用户画像: 加载 → 个性化 system prompt 前缀
```

---

## 八、配置与启动

### 8.1 环境准备（一次性）

```batch
REM === Windows PC: 创建 conda 环境（conda_envs/） ===

REM ASR 环境
call create_asr_env.bat
  conda create -n asr_env python=3.11 -y
  conda activate asr_env
  pip install funasr modelscope torch torchaudio fastapi uvicorn

REM TTS 环境
call create_tts_env.bat
  conda create -n tts_env python=3.11 -y
  conda activate tts_env
  pip install edge-tts fastapi uvicorn
  REM CosyVoice 2 可选, 按需安装

REM Memory 环境
call create_memory_env.bat
  conda create -n memory_env python=3.11 -y
  conda activate memory_env
  pip install chromadb sqlalchemy fastapi uvicorn

REM Orchestrator 环境
call create_main_env.bat
  conda create -n main_env python=3.11 -y
  conda activate main_env
  pip install fastapi uvicorn websockets httpx pyyaml

REM 安装 Ollama (Windows 原生)
REM 从 https://ollama.com/download/windows 下载安装
ollama pull qwen3:14b
```

### 8.2 一键启动（start_all.bat）

```batch
@echo off
echo ========================================
echo   Smart RangRang - 全本地服务启动
echo ========================================

REM 1. 启动 Ollama（确保模型常驻显存）
echo [1/5] Starting Ollama...
set OLLAMA_KEEP_ALIVE=-1
set OLLAMA_FLASH_ATTENTION=1
set OLLAMA_NUM_PARALLEL=1
start "Ollama" ollama serve

REM 2. 启动 ASR Service
echo [2/5] Starting ASR Service (asr_env)...
start "ASR" cmd /c "conda activate asr_env && uvicorn server:app --host 127.0.0.1 --port 8100"

REM 3. 启动 TTS Service
echo [3/5] Starting TTS Service (tts_env)...
start "TTS" cmd /c "conda activate tts_env && uvicorn server:app --host 127.0.0.1 --port 8101"

REM 4. 启动 Memory Service
echo [4/5] Starting Memory Service (memory_env)...
start "Memory" cmd /c "conda activate memory_env && uvicorn server:app --host 127.0.0.1 --port 8102"

REM 5. 启动 Orchestrator（最后启动，依赖其他 Service）
echo [5/5] Starting Orchestrator (main_env)...
start "Orchestrator" cmd /c "conda activate main_env && uvicorn main:app --host 0.0.0.0 --port 8765"

echo.
echo All services started.
echo   Orchestrator WebSocket: ws://localhost:8765
echo   ASR Service:            http://localhost:8100
echo   TTS Service:            http://localhost:8101
echo   Memory Service:         http://localhost:8102
echo   Ollama:                 http://localhost:11434
echo.
echo Run stop_all.bat to shutdown.
```

### 8.3 Pi 端启动

```bash
#!/bin/bash
# pi/start.sh

echo "=== Smart RangRang Pi Client ==="

# 确认 XVF3800 已识别
arecord -l | grep "XVF3800" || echo "WARN: XVF3800 not found"

# 启动 Pi 客户端
python main.py --config config_pi.yaml
```

### 8.4 运行时配置示例

**Orchestrator 配置（server/orchestrator/config.yaml）**：

```yaml
server:
  host: "0.0.0.0"
  port: 8765

services:
  asr: "http://127.0.0.1:8100"
  tts: "http://127.0.0.1:8101"
  memory: "http://127.0.0.1:8102"
  ollama: "http://127.0.0.1:11434"

llm:
  model: "qwen3:14b"
  context_window: 32768
  max_tokens_per_turn: 1024
  temperature: 0.7
  timeout_first_token_ms: 15000  # 首 token 超时

sentence:
  silence_punct_ms: 200
  silence_force_ms: 800
  silence_max_ms: 1500
  text_max_chars: 100
  incomplete_tails:
    - "那个"
    - "就是"
    - "然后"
    - "还有"
    - "因为"
    - "比如说"
    - "嗯"
    - "呃"
    - "这个"
    - "所以"

session:
  idle_timeout_minutes: 5
  listen_window_seconds: 5
  heartbeat_interval_seconds: 3
  watchdog_timeout_seconds: 30

memory:
  recent_turns_on_load: 20
  semantic_search_top_k: 5

tts:
  primary: "edge-tts"
  edge_voice: "zh-CN-XiaoxiaoNeural"
  edge_rate: "+10%"      # 语速微调
  fallback_enabled: true
```

**Pi 端配置（pi/config_pi.yaml）**：

```yaml
server:
  host: "192.168.1.100"  # PC IP
  port: 8765

audio:
  sample_rate: 16000
  channels: 1
  sample_width: 2          # 16bit
  chunk_ms: 40             # 每帧 40ms
  device_name: "XVF3800"   # PyAudio device 名称匹配

vad:
  model: "silero_vad"
  threshold: 0.5           # 语音概率阈值
  min_speech_frames: 3     # 最小连续语音帧
  min_silence_frames: 10   # 最小连续静音帧

echo_judge:
  corr_threshold: 0.5
  env_threshold: 0.6
  corr_weak_threshold: 0.3
  ref_buffer_seconds: 1.0

speex_aec:
  enabled: true
  filter_length_ms: 100
  frame_size_ms: 20

wake_word:
  model: "openwakeword"
  keywords:
    - "小智小智"
  threshold: 0.5
  activation_delay_seconds: 1.0   # 唤醒后 1s 内不再次触发
```

---

## 九、性能指标与优化

### 9.1 延迟预算

| 环节 | 目标 | 说明 |
|------|------|------|
| VAD 检测说话开始 | < 30ms | 3帧 × 10ms |
| SenseVoice 流式 ASR | < 150ms | 首字延迟（GPU FP16） |
| CT-Transformer 标点 | < 10ms | 短文本标点预测 |
| 断句决策 | 200-1500ms | 标点+静默+语义复合 |
| LLM 首 token | < 800ms | Qwen3:14B-Q4_K_M, flash_attention ON |
| Edge-TTS 首音 | < 300ms | 首句合成延迟 |
| 网络 (Pi↔PC LAN) | < 10ms | Wi-Fi 5GHz 或以太网 |
| 网络 (PC 内部 localhost) | < 1ms | 进程间 HTTP |
| **端到端首音延迟** | **< 1.5s** | 用户说完 → 听到回答首音 |
| **打断延迟** | **< 130ms** | 用户说话 → Pi 停止播放 |

### 9.2 优化策略

```
LLM 加速:
  ├── OLLAMA_FLASH_ATTENTION=1（Flash Attention 2 加速）
  ├── OLLAMA_KEEP_ALIVE=-1（模型常驻显存，避免冷启动）
  ├── OLLAMA_NUM_PARALLEL=1（单请求模式，避免并发冲突）
  ├── Q4_K_M 量化（精度/速度最优平衡点）
  └── num_predict 限制单轮最大 1024 token

ASR 加速:
  ├── GPU FP16 推理
  └── 热词列表注入（常见人名、地名）

TTS 加速:
  └── Edge-TTS 内部已优化，无需额外配置

网络:
  └── Pi 优先使用有线以太网（更稳定、更低延迟）
```

### 9.3 监控指标

| 指标 | 采集位置 | 告警阈值 |
|------|----------|----------|
| ASR 平均延迟 | Orchestrator | > 300ms |
| LLM 首 token 延迟 | Orchestrator | > 2000ms |
| TTS 首音延迟 | Orchestrator | > 500ms |
| 打断响应延迟 | Pi 端 | > 200ms |
| GPU 显存占用 | PC | > 15GB |
| GPU 温度 | PC | > 85°C |
| Pi CPU 占用 | Pi | > 80% |
| WebSocket 断连次数 | Pi + Orchestrator | > 3次/小时 |

---

## 十、风险与缓解

| 风险 | 概率 | 严重度 | 缓解 |
|------|------|--------|------|
| **ASR 标点错误导致断句不准** | 中 | **高** | 独立 CT-Transformer 标点模型（不依赖 ASR 内置标点）；三模断句的静默超时路径兜底 |
| **回声残留导致误打断** | 中 | **高** | SpeexDSP AEC 软件层补充 + Echo Judge 双模判决（连续 10 帧确认 + 互相关阈值） |
| **Ollama 打断后状态残留** | 低 | 中 | HTTP 连接断开 + `OLLAMA_NUM_PARALLEL=1` 限制单请求；打断后重建 session |
| **GPU OOM（显存峰值突破 16GB）** | 低 | 中 | 显存占用 ~13GB，留有 3GB 安全余量；Ollama 设置 `num_predict` 上限；ASR/TTS 独立进程，OOM 不扩散 |
| **Edge-TTS 合成质量不满足需求** | 低 | 低 | CosyVoice 2 作为可选增强模式，按需启用 |
| **Pi 端 WebSocket 断连** | 低 | 中 | 自动重连 + 指数退避（1s→2s→4s→...→max 30s） |
| **Pi 5 无散热高温降频** | 低 | 低 | 官方主动散热器（~¥20）；降频后音频处理延迟仍在可接受范围 |
| **多说话人场景混淆** | 中 | 低 | XVF3800 波束成形锁定最强声源 + DOA 过滤非目标方向 + VAD 低置信度帧忽略 |
| **PC 风扇噪音** | 高 | 低 | GPU 负载 70-85% 时风扇转速可接受；不影响功能 |
| **Ollama 冷启动慢** | 中 | 低 | `OLLAMA_KEEP_ALIVE=-1` 模型常驻显存，除首次启动外无冷启动 |

---

## 十一、开发阶段规划

```
Phase 1 — 核心链路验证（预计 3-5 天）
  ├── Pi 端: XVF3800 音频采集 → WAV 录制
  ├── PC 端: 离线 ASR (SenseVoice) → LLM (Ollama) → 离线 TTS (Edge-TTS)
  └── 目标: 确认所有模型能跑通，e2e 延迟可接受

Phase 2 — 流式管道打通（预计 3-5 天）
  ├── 流式 ASR (SenseVoice streaming + CT-Transformer)
  ├── 流式 LLM (Ollama /api/chat stream)
  ├── 流式 TTS (Edge-TTS stream)
  ├── Pi↔PC WebSocket 双向音频流
  └── 目标: 首音延迟 < 1.5s，基础对话可用

Phase 3 — 打断 + 断句（预计 5-7 天）
  ├── Echo Judge 互相关实现 + 调参
  ├── SpeexDSP AEC 集成
  ├── 双模打断判决 + 销毁重建流程
  ├── 三模断句决策器
  └── 目标: 打断 < 130ms，断句准确率 > 90%

Phase 4 — 记忆 + 多轮（预计 2-3 天）
  ├── SQLite 对话记录
  ├── ChromaDB 向量语义检索
  ├── 用户画像
  ├── 启动时记忆自动加载
  └── 目标: 重启后恢复上下文，记忆检索正确

Phase 5 — 稳定性打磨（预计 3-5 天）
  ├── 超时 + 看门狗完善
  ├── 异常场景覆盖（断连、OOM、超时、噪声）
  ├── 长时间运行压测（24h）
  └── 目标: 7×24 稳定运行，自动恢复
```

---

## 十二、依赖清单

### Pi 端（requirements_pi.txt）

```
# 音频
pyaudio==0.2.14
numpy==1.26.4
scipy==1.13.0
soundfile==0.12.1

# 回声消除
speexdsp==1.0.0          # SpeexDSP Python binding

# 唤醒词
openwakeword==0.6.0

# VAD
silero-vad==5.1

# WebSocket
websockets==12.0

# 配置
pyyaml==6.0.2
```

### ASR Service（server/asr_service/requirements.txt）

```
funasr==1.1.0            # SenseVoice 依赖 FunASR 框架
modelscope==1.17.0
torch>=2.1.0
torchaudio>=2.1.0
fastapi==0.115.0
uvicorn[standard]==0.30.0
numpy==1.26.4
pypinyin==0.50.0         # CT-Transformer 依赖
```

### TTS Service（server/tts_service/requirements.txt）

```
edge-tts==6.1.14
fastapi==0.115.0
uvicorn[standard]==0.30.0
numpy==1.26.4
# CosyVoice 2 按需安装（可选增强）
# 参见 https://github.com/FunAudioLLM/CosyVoice
```

### Memory Service（server/memory_service/requirements.txt）

```
chromadb==0.5.0
sqlalchemy==2.0.35
fastapi==0.115.0
uvicorn[standard]==0.30.0
pyyaml==6.0.2
```

### Orchestrator（server/orchestrator/requirements.txt）

```
fastapi==0.115.0
uvicorn[standard]==0.30.0
websockets==12.0
httpx==0.27.0
pyyaml==6.0.2
```

---

*方案版本: v2.0 | 日期: 2026-06-01 | 全本地部署 | Windows 原生 + conda 隔离*
