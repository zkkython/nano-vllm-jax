# nano-vllm-jax

高性能 JAX 版本的 LLM 推理引擎，支持 Tensor Parallel 和高效的 KV Cache 管理。

## 目录

- [特性](#特性)
- [项目架构](#项目架构)
- [核心组件](#核心组件)
- [安装](#安装)
- [快速开始](#快速开始)
- [高级用法](#高级用法)
- [日志配置](#日志配置)
- [开发和测试](#开发和测试)

## 特性

- ✅ **Tensor Parallel 支持**: 支持多 GPU 并行推理，提升吞吐量
- ✅ **高效 KV Cache**: 基于 Block 的 KV Cache 管理，支持 Paged Attention
- ✅ **Varlen Attention**: 支持变长序列的高效批处理
- ✅ **GQA 支持**: 完整支持 Grouped Query Attention
- ✅ **RoPE 位置编码**: 支持 RoPE (Rotary Position Embedding)
- ✅ **灵活的日志系统**: 支持全局和模块级别的日志控制
- ✅ **模型支持**: 目前支持 Qwen3 系列模型

## 项目架构

### 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                          LLM (User API)                          │
│  - generate(): 生成文本                                           │
│  - add_request(): 添加请求                                        │
└───────────────────────────────┬─────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐      ┌────────────────┐     ┌──────────────┐
│   Scheduler   │      │  ModelRunner   │     │   Sampler    │
│  - schedule() │      │  - run_model() │     │  - sample()  │
│  - postprocess│      │  - prepare_*() │     └──────────────┘
└───────────────┘      └────────┬───────┘
                                │
                ┌───────────────┼───────────────┐
                │               │               │
                ▼               ▼               ▼
        ┌──────────────┐ ┌──────────┐  ┌──────────────┐
        │ Qwen3Model   │ │ KV Cache │  │   Context    │
        │ - layers     │ │ - blocks │  │ - cu_seqlens │
        │ - attention  │ └──────────┘  │ - slot_map   │
        │ - mlp        │               └──────────────┘
        └──────┬───────┘
               │
       ┌───────┼────────┐
       │       │        │
       ▼       ▼        ▼
  ┌────────┐ ┌────┐ ┌─────────┐
  │Attention│ │MLP │ │LayerNorm│
  │- QKV   │ │    │ │         │
  │- RoPE  │ │    │ │         │
  │- Varlen│ │    │ │         │
  └────────┘ └────┘ └─────────┘
```

### 目录结构

```
nano-vllm-jax/
├── nanovllm_jax/              # 核心库代码
│   ├── configs/               # 配置相关
│   │   └── model_config.py    # 模型配置（TP、GQA 等）
│   ├── engine/                # 推理引擎
│   │   ├── block_manager.py   # Block 管理器
│   │   ├── model_runner.py    # 模型运行器（核心）
│   │   ├── scheduler.py       # 请求调度器
│   │   └── sequence.py        # 序列管理
│   ├── layers/                # 模型层实现
│   │   ├── attention.py       # Attention 层（Varlen/Paged）
│   │   ├── embed_head.py      # Embedding 和 LM Head
│   │   ├── layernorm.py       # RMSNorm
│   │   ├── linear.py          # Linear 层（支持 TP）
│   │   └── sampler.py         # Token 采样器
│   ├── models/                # 模型实现
│   │   └── qwen3.py           # Qwen3 模型
│   ├── utils/                 # 工具函数
│   │   ├── common_utils.py    # 通用工具
│   │   ├── context.py         # 全局上下文管理
│   │   ├── jax_utils.py       # JAX 工具（TP 相关）
│   │   ├── logging_utils.py   # 日志工具
│   │   └── weight_utils.py    # 权重加载工具
│   ├── config.py              # 全局配置
│   ├── llm.py                 # LLM 用户 API
│   └── sampling_params.py     # 采样参数
├── test_llm.py                # 端到端测试
├── test_tp_*.py               # Tensor Parallel 单元测试
├── test_logging.py            # 日志功能测试
└── LOGGING_USAGE.md           # 日志使用文档
```

## 核心组件

### 1. LLM API (`llm.py`)

用户入口，提供简单的生成接口：

```python
from nanovllm_jax.llm import LLM, SamplingParams

# 初始化模型
llm = LLM(
    model_path="/path/to/model",
    max_model_len=2048,
    tensor_parallel_size=2  # 使用 2 个 GPU
)

# 生成文本
outputs = llm.generate(
    ["你好，介绍一下你自己。"],
    SamplingParams(temperature=0.7, max_tokens=100)
)
```

### 2. ModelRunner (`engine/model_runner.py`)

核心推理引擎，负责：
- **Tensor Parallel 管理**: 创建 mesh，分配权重
- **KV Cache 分配**: 按 TP 切分 KV cache
- **上下文准备**: 
  - `prepare_prefill()`: 准备 prefill 阶段的输入
  - `prepare_decode()`: 准备 decode 阶段的输入
- **模型执行**: `run_model()` 在 mesh 上下文中执行

### 3. Attention 层 (`layers/attention.py`)

支持两种 attention 模式：
- **Varlen Attention**: 用于 prefill 阶段，支持变长序列批处理
- **Paged Attention**: 用于 decode 阶段，高效利用 KV cache

关键函数：
- `varlen_attention()`: 处理变长序列
- `paged_attention()`: 从 block 中读取 KV
- `store_kvcache()`: 存储 KV 到 cache

### 4. Qwen3 模型 (`models/qwen3.py`)

完整的 Qwen3 模型实现：
- **SelfAttentionVarlen**: 支持 TP 的 Attention 层
  - Q/K/V 投影使用 Column Parallel
  - O 投影使用 Row Parallel
  - 支持 GQA（Grouped Query Attention）
  - RoPE 位置编码
- **MLP**: SwiGLU 激活的 MLP 层
- **权重加载**: 自动处理 TP 切分和 KV head 复制

### 5. Scheduler (`engine/scheduler.py`)

请求调度器，负责：
- 管理请求队列（waiting, running, finished）
- 决定 prefill 还是 decode
- 分配和回收 KV cache blocks
- 处理序列完成状态

## 安装

### 依赖

```bash
# JAX (CUDA 版本)
pip install jax[cuda12]

# Flax NNX
pip install flax

# 其他依赖
pip install transformers safetensors tqdm
```

### 从源码安装

```bash
git clone https://github.com/your-repo/nano-vllm-jax.git
cd nano-vllm-jax
pip install -e .
```

## 快速开始

### 基本使用

```python
from nanovllm_jax.llm import LLM, SamplingParams

# 1. 初始化 LLM
llm = LLM(
    model_path="/path/to/Qwen3-8B",
    max_model_len=2048,
    tensor_parallel_size=1  # 单 GPU
)

# 2. 生成文本
prompts = ["你好，介绍一下你自己。"]
outputs = llm.generate(
    prompts,
    SamplingParams(
        temperature=0.7,
        max_tokens=100,
        top_p=0.9
    )
)

# 3. 查看结果
for output in outputs:
    print(output["text"])
```

### 使用 Tensor Parallel

```python
# 使用 2 个 GPU
llm = LLM(
    model_path="/path/to/Qwen3-8B",
    max_model_len=2048,
    tensor_parallel_size=2  # 2-way TP
)

# 其他使用方式相同
outputs = llm.generate(["Hello"], SamplingParams(max_tokens=50))
```

### 批量生成

```python
prompts = [
    "中国的首都在哪里？",
    "解释一下量子力学。",
    "列出100以内的质数。"
]

# 支持不同的采样参数
sampling_params = [
    SamplingParams(temperature=0.0, max_tokens=50),  # 确定性
    SamplingParams(temperature=0.9, max_tokens=200), # 更随机
    SamplingParams(temperature=0.5, max_tokens=100)
]

outputs = llm.generate(prompts, sampling_params)
```

## 高级用法

### 自定义配置

```python
from nanovllm_jax.config import Config

config = Config(
    model="/path/to/model",
    max_num_batched_tokens=16384,  # 最大批处理 token 数
    max_num_seqs=512,              # 最大序列数
    max_model_len=4096,            # 最大序列长度
    gpu_memory_utilization=0.9,    # GPU 内存利用率
    tensor_parallel_size=2,        # TP size
    kvcache_block_size=256,        # KV cache block 大小
)
```

### 调整采样参数

```python
from nanovllm_jax.sampling_params import SamplingParams

params = SamplingParams(
    temperature=0.7,      # 温度，控制随机性
    top_p=0.9,           # nucleus sampling
    top_k=50,            # top-k sampling
    max_tokens=100,      # 最大生成 token 数
    repetition_penalty=1.1,  # 重复惩罚
)
```

### 检查 TP 一致性

```bash
# 比较 TP=1 和 TP=2 的输出是否一致
python test_tp_consistency.py --model-path /path/to/model
```

## 日志配置

### 基本用法

```bash
# 默认 INFO 级别
python test_llm.py --tp-size 2

# 设置为 WARNING（减少输出）
python test_llm.py --tp-size 2 --log-level WARNING

```

### 模块级别日志

只对特定模块启用 DEBUG，其他保持 WARNING：

```bash
# 只看 attention 层的详细日志
python test_llm.py --tp-size 2 \
    --log-level WARNING \
    --debug-modules nanovllm_jax.layers.attention

# 多个模块
python test_llm.py --tp-size 2 \
    --log-level WARNING \
    --debug-modules nanovllm_jax.layers.attention,nanovllm_jax.engine.model_runner
```

### 使用环境变量

```bash
# 全局设置
export NANO_VLLM_LOG_LEVEL=WARNING
export NANO_VLLM_DEBUG_MODULES=nanovllm_jax.layers.attention

# 所有脚本都会使用这个配置
python test_llm.py --tp-size 2
```

详细文档请查看 [LOGGING_USAGE.md](LOGGING_USAGE.md)


### 调试技巧

#### 1. 查看 shape 和 sharding

```bash
# 启用 attention 层的 DEBUG 日志
python test_llm.py --tp-size 2 \
    --log-level WARNING \
    --debug-modules nanovllm_jax.layers.attention
```

#### 2. 检查 KV cache 配置

```bash
# 启用 model_runner 的 DEBUG 日志
python test_llm.py --tp-size 2 \
    --log-level WARNING \
    --debug-modules nanovllm_jax.engine.model_runner
```

#### 3. 验证数值一致性

```bash
# 比较 TP=1 和 TP=2 的输出
python test_tp_consistency.py --model-path /path/to/model
```

## 核心设计思想

### 1. Tensor Parallel 策略

- **Column Parallel**: Q/K/V 投影、MLP gate/up 投影
  - 权重 sharding: `(None, "tensor")`
  - 输出在最后一维分片
  
- **Row Parallel**: O 投影、MLP down 投影
  - 权重 sharding: `("tensor", None)`
  - 输出需要 all-reduce

- **KV Cache Sharding**: 在 num_kv_heads 维度分片
  - Shape: `[num_blocks, block_size, num_kv_heads, head_dim]`
  - Sharding: `P(None, None, "tensor", None)`

### 2. Varlen Attention

支持变长序列的批处理：
- 使用 `cu_seqlens_q/k` 标记每个序列的边界
- 支持 prefix caching（prefill 时复用已有 KV）
- 自动处理 GQA 的 KV head repeat

### 3. Block-based KV Cache

高效的内存管理：
- 固定大小的 block（如 256 tokens）
- Block 可以在序列间共享（prefix caching）
- Paged attention 从 block table 读取 KV

### 4. GQA 支持

- 自动检测 GQA 模型（`num_kv_heads < num_heads`）
- TP 时需要 KV head replication（当 tp_size > num_kv_heads）
- Attention 中自动 repeat KV heads

## 性能优化

### 当前优化

- ✅ Tensor Parallel for 多 GPU
- ✅ Block-based KV cache
- ✅ Varlen attention for 批处理
- ✅ GQA for 减少 KV cache
- ✅ BFloat16 计算

### 待优化（TODO）

- ⏳ JIT 编译优化
- ⏳ Continuous batching
- ⏳ FlashAttention 集成
- ⏳ 更多模型支持（LLaMA, Mistral 等）
- ⏳ INT8/INT4 量化

## 常见问题

### Q1: 内存不足怎么办？

调整配置参数：
```python
llm = LLM(
    model_path="/path/to/model",
    gpu_memory_utilization=0.8,     # 降低内存占用
    max_num_batched_tokens=8192,    # 减少批大小
    kvcache_block_size=128,         # 减小 block size
)
```

## 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启 Pull Request

## 许可证

MIT License

## 致谢

本项目受到以下项目的启发：
- [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)
- [vLLM](https://github.com/vllm-project/vllm)
- [JAX](https://github.com/google/jax)
- [Flax](https://github.com/google/flax)
