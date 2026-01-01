# 日志配置使用指南

本项目使用统一的日志配置系统，支持灵活的日志级别控制。

## 快速开始

### 1. 使用默认配置

```bash
# 默认 INFO 级别
python test_llm.py --tp-size 2
```

### 2. 通过命令行参数控制

```bash
# 设置为 WARNING 级别（只显示警告和错误）
python test_llm.py --tp-size 2 --log-level WARNING

# 设置为 DEBUG 级别（显示所有日志）
python test_llm.py --tp-size 2 --log-level DEBUG

# 完全关闭日志（只显示 CRITICAL）
python test_llm.py --tp-size 2 --log-level CRITICAL
```

### 3. 通过环境变量控制（全局）

```bash
# 设置全局日志级别为 WARNING
NANO_VLLM_LOG_LEVEL=WARNING python test_llm.py --tp-size 2

# 组合使用：全局 WARNING，但命令行可覆盖
NANO_VLLM_LOG_LEVEL=WARNING python test_llm.py --tp-size 2 --log-level INFO
```

## 高级用法

### 针对特定模块启用 DEBUG

当你只想看某个模块的详细日志时：

```bash
# 只对 attention 模块启用 DEBUG，其他模块保持 WARNING
python test_llm.py --tp-size 2 \
    --log-level WARNING \
    --debug-modules nanovllm_jax.layers.attention

# 对多个模块启用 DEBUG（逗号分隔）
python test_llm.py --tp-size 2 \
    --log-level WARNING \
    --debug-modules nanovllm_jax.layers.attention,nanovllm_jax.engine.model_runner
```

### 使用环境变量指定 DEBUG 模块

```bash
# 全局 WARNING，但 attention 和 model_runner 使用 DEBUG
NANO_VLLM_LOG_LEVEL=WARNING \
NANO_VLLM_DEBUG_MODULES=nanovllm_jax.layers.attention,nanovllm_jax.engine.model_runner \
python test_llm.py --tp-size 2
```

### 组合使用（最灵活）

```bash
# 环境变量设置基础配置
export NANO_VLLM_LOG_LEVEL=WARNING
export NANO_VLLM_DEBUG_MODULES=nanovllm_jax.layers.attention

# 命令行可以覆盖或补充
python test_llm.py --tp-size 2 --log-level INFO
```

## 常见使用场景

### 场景 1：正常运行，只看重要信息

```bash
python test_llm.py --tp-size 2 --log-level WARNING
```

### 场景 2：调试 TP 问题，只看 attention 层

```bash
python test_llm.py --tp-size 2 \
    --log-level WARNING \
    --debug-modules nanovllm_jax.layers.attention
```

### 场景 3：调试 KV cache 问题

```bash
python test_llm.py --tp-size 2 \
    --log-level WARNING \
    --debug-modules nanovllm_jax.engine.model_runner,nanovllm_jax.layers.attention
```

### 场景 4：运行测试时减少输出

```bash
# 一致性测试，只看结果
python test_tp_consistency.py --log-level ERROR

# 完全安静模式
python test_tp_consistency.py --log-level CRITICAL
```

### 场景 5：开发调试，查看所有细节

```bash
python test_llm.py --tp-size 2 --log-level DEBUG
```

## 在代码中使用

### 基本用法

```python
from nanovllm_jax.utils.logging_utils import get_logger

logger = get_logger(__name__)

logger.debug("详细的调试信息")
logger.info("一般信息")
logger.warning("警告信息")
logger.error("错误信息")
logger.critical("严重错误")
```

### 在入口脚本初始化

```python
from nanovllm_jax.utils.logging_utils import setup_logging

# 程序启动时调用一次
setup_logging(level='INFO')

# 或者启用特定模块的 DEBUG
setup_logging(
    level='WARNING',
    enable_debug_modules=['nanovllm_jax.layers.attention']
)
```

### 动态修改日志级别

```python
from nanovllm_jax.utils.logging_utils import set_log_level, set_module_log_level

# 动态修改全局日志级别
set_log_level('DEBUG')

# 动态修改特定模块的日志级别
set_module_log_level('nanovllm_jax.layers.attention', 'DEBUG')
```

## 日志级别说明

- **DEBUG**: 最详细，显示所有日志（包括中间变量、shape、sharding 信息等）
- **INFO**: 一般信息（默认级别，显示关键步骤）
- **WARNING**: 警告信息（潜在问题，但不影响运行）
- **ERROR**: 错误信息（出现问题，但程序可能继续）
- **CRITICAL**: 严重错误（程序可能崩溃）

## 环境变量参考

- `NANO_VLLM_LOG_LEVEL`: 全局日志级别 (DEBUG/INFO/WARNING/ERROR/CRITICAL)
- `NANO_VLLM_DEBUG_MODULES`: 需要启用 DEBUG 的模块列表（逗号分隔）

## 注意事项

1. **优先级**: 命令行参数 > 环境变量 > 默认值(INFO)
2. **模块名称**: 必须使用完整的模块名（如 `nanovllm_jax.layers.attention`）
3. **性能**: DEBUG 级别会输出大量信息，可能影响性能，生产环境建议使用 INFO 或 WARNING
4. **覆盖**: 使用 `setup_logging(force=True)` 可以覆盖已有的日志配置
