#!/usr/bin/env python3
"""测试日志配置功能"""

from nanovllm_jax.utils.logging_utils import setup_logging, get_logger

# 创建不同模块的 logger
logger_main = get_logger(__name__)
logger_attention = get_logger('nanovllm_jax.layers.attention')
logger_model = get_logger('nanovllm_jax.engine.model_runner')


def test_basic_logging():
    """测试基本日志功能"""
    print("\n" + "=" * 80)
    print("测试 1: 基本日志级别（默认 INFO）")
    print("=" * 80)
    
    setup_logging(level='INFO')
    
    logger_main.debug("这条 DEBUG 信息不应该显示")
    logger_main.info("这条 INFO 信息应该显示")
    logger_main.warning("这条 WARNING 信息应该显示")
    logger_main.error("这条 ERROR 信息应该显示")


def test_warning_level():
    """测试 WARNING 级别"""
    print("\n" + "=" * 80)
    print("测试 2: WARNING 级别（应该只显示 WARNING 及以上）")
    print("=" * 80)
    
    setup_logging(level='WARNING')
    
    logger_main.debug("这条 DEBUG 信息不应该显示")
    logger_main.info("这条 INFO 信息不应该显示")
    logger_main.warning("这条 WARNING 信息应该显示")
    logger_main.error("这条 ERROR 信息应该显示")


def test_module_specific_debug():
    """测试模块特定 DEBUG"""
    print("\n" + "=" * 80)
    print("测试 3: 全局 WARNING，但 attention 模块 DEBUG")
    print("=" * 80)
    
    setup_logging(
        level='WARNING',
        enable_debug_modules=['nanovllm_jax.layers.attention']
    )
    
    logger_main.debug("main 的 DEBUG 不应该显示")
    logger_main.info("main 的 INFO 不应该显示")
    logger_main.warning("main 的 WARNING 应该显示")
    
    logger_attention.debug("attention 的 DEBUG 应该显示 ✓")
    logger_attention.info("attention 的 INFO 应该显示 ✓")
    logger_attention.warning("attention 的 WARNING 应该显示 ✓")
    
    logger_model.debug("model 的 DEBUG 不应该显示")
    logger_model.info("model 的 INFO 不应该显示")
    logger_model.warning("model 的 WARNING 应该显示")


def test_multiple_debug_modules():
    """测试多模块 DEBUG"""
    print("\n" + "=" * 80)
    print("测试 4: 全局 WARNING，attention 和 model_runner 都 DEBUG")
    print("=" * 80)
    
    setup_logging(
        level='WARNING',
        enable_debug_modules=[
            'nanovllm_jax.layers.attention',
            'nanovllm_jax.engine.model_runner'
        ]
    )
    
    logger_main.debug("main 的 DEBUG 不应该显示")
    logger_main.warning("main 的 WARNING 应该显示")
    
    logger_attention.debug("attention 的 DEBUG 应该显示 ✓")
    logger_model.debug("model_runner 的 DEBUG 应该显示 ✓")


def test_env_var_simulation():
    """模拟环境变量使用"""
    print("\n" + "=" * 80)
    print("测试 5: 使用环境变量（模拟）")
    print("=" * 80)
    print("提示: 实际使用时可以这样:")
    print("  NANO_VLLM_LOG_LEVEL=WARNING python test_llm.py")
    print("  NANO_VLLM_DEBUG_MODULES=nanovllm_jax.layers.attention python test_llm.py")
    print("=" * 80)
    
    import os
    os.environ['NANO_VLLM_LOG_LEVEL'] = 'WARNING'
    os.environ['NANO_VLLM_DEBUG_MODULES'] = 'nanovllm_jax.layers.attention'
    
    # 重新设置以读取环境变量
    setup_logging()
    
    logger_main.info("main 的 INFO 不应该显示（因为环境变量设置为 WARNING）")
    logger_main.warning("main 的 WARNING 应该显示")
    logger_attention.debug("attention 的 DEBUG 应该显示（环境变量指定）✓")
    
    # 清理环境变量
    del os.environ['NANO_VLLM_LOG_LEVEL']
    del os.environ['NANO_VLLM_DEBUG_MODULES']


if __name__ == '__main__':
    print("=" * 80)
    print("日志配置功能测试")
    print("=" * 80)
    
    test_basic_logging()
    test_warning_level()
    test_module_specific_debug()
    test_multiple_debug_modules()
    test_env_var_simulation()
    
    print("\n" + "=" * 80)
    print("所有测试完成！")
    print("=" * 80)
    print("\n使用说明:")
    print("1. 命令行参数: python test_llm.py --log-level WARNING")
    print("2. 环境变量: NANO_VLLM_LOG_LEVEL=WARNING python test_llm.py")
    print("3. 模块调试: python test_llm.py --log-level WARNING --debug-modules nanovllm_jax.layers.attention")
    print("\n详细文档请查看: LOGGING_USAGE.md")
