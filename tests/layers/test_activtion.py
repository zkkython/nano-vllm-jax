import torch
from jax import numpy as jnp
import numpy as np
import unittest

from vllm_jax.layers.activation import SiluAndMul, TorchSiluAndMul

"""
python -m unittest tests.layers.test_activtion.TestSiluAndMulConsistency
if we want to test one function only, we can use like:
python -m unittest tests.layers.test_activtion.TestSiluAndMulConsistency.test_basic_functionality
"""


class TestSiluAndMulConsistency(unittest.TestCase):

    def setUp(self):
        """设置测试环境和随机种子"""
        # 设置随机种子以确保可重复性
        torch.manual_seed(42)
        np.random.seed(42)

        # 初始化模型
        self.torch_model = TorchSiluAndMul()
        self.jax_model = SiluAndMul()

    def test_basic_functionality(self):
        """测试基本功能一致性"""
        print("测试基本功能一致性...")

        # 创建测试输入
        test_input = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)

        # Torch 计算
        torch_input = torch.from_numpy(test_input)
        torch_output = self.torch_model(torch_input).detach().numpy()

        # JAX 计算
        jax_input = jnp.array(test_input)
        jax_output = self.jax_model(jax_input)
        jax_output_np = np.array(jax_output)

        # 验证结果
        self.assertEqual(torch_output.shape, jax_output_np.shape)
        np.testing.assert_allclose(torch_output, jax_output_np, rtol=1e-6, atol=1e-6)
        print("✓ 基本功能测试通过")

    def test_various_input_shapes(self):
        """测试各种输入形状"""
        print("测试各种输入形状...")

        test_shapes = [
            (1, 4),  # 最小形状
            (2, 6),  # 批量大小 2
            (5, 8),  # 批量大小 5
            (10, 12),  # 批量大小 10
            (3, 16),  # 较大的特征维度
            (1, 2),  # 最小有效形状
        ]

        for shape in test_shapes:
            with self.subTest(shape=shape):
                # 生成随机数据
                np_input = np.random.randn(*shape).astype(np.float32)

                # Torch 计算
                torch_input = torch.from_numpy(np_input)
                torch_output = self.torch_model(torch_input).detach().numpy()

                # JAX 计算
                jax_input = jnp.array(np_input)
                jax_output = self.jax_model(jax_input)
                jax_output_np = np.array(jax_output)

                # 验证形状和数值一致性
                self.assertEqual(torch_output.shape, jax_output_np.shape)
                np.testing.assert_allclose(
                    torch_output, jax_output_np, rtol=1e-5, atol=1e-5
                )

        print("✓ 各种输入形状测试通过")

    def test_edge_cases(self):
        """测试边界情况"""
        print("测试边界情况...")

        test_cases = [
            ("zeros", np.zeros((2, 4))),
            ("ones", np.ones((1, 6))),
            ("negative", np.array([[-1.0, -2.0, -3.0, -4.0]])),
            ("mixed_signs", np.array([[1.0, -1.0, 2.0, -2.0]])),
            ("large_values", np.array([[100.0, -100.0, 50.0, -50.0]])),
            ("small_values", np.array([[1e-6, -1e-6, 1e-8, -1e-8]])),
        ]

        for case_name, test_input in test_cases:
            with self.subTest(case=case_name):
                # Torch 计算
                torch_input = torch.from_numpy(test_input.astype(np.float32))
                torch_output = self.torch_model(torch_input).detach().numpy()

                # JAX 计算
                jax_input = jnp.array(test_input.astype(np.float32))
                jax_output = self.jax_model(jax_input)
                jax_output_np = np.array(jax_output)

                # 验证
                np.testing.assert_allclose(
                    torch_output, jax_output_np, rtol=1e-6, atol=1e-6
                )

        print("✓ 边界情况测试通过")

    def test_manual_verification(self):
        """手动验证数学正确性"""
        print("手动验证数学正确性...")

        # 测试用例 1: [0, 1, 2, 3]
        # silu([0, 1]) = [0, 0.73105858]
        # 乘以 [2, 3] = [0, 2.19317574]
        test_input1 = np.array([[0.0, 1.0, 2.0, 3.0]], dtype=np.float32)
        expected1 = np.array([[0.0, 2.19317574]])

        # 测试用例 2: [1, 0, 1, 1]
        # silu([1, 0]) = [0.73105858, 0]
        # 乘以 [1, 1] = [0.73105858, 0]
        test_input2 = np.array([[1.0, 0.0, 1.0, 1.0]], dtype=np.float32)
        expected2 = np.array([[0.73105858, 0.0]])

        test_cases = [
            (test_input1, expected1),
            (test_input2, expected2),
        ]

        for i, (test_input, expected) in enumerate(test_cases):
            with self.subTest(case=f"manual_{i+1}"):
                # Torch 计算
                torch_input = torch.from_numpy(test_input)
                torch_output = self.torch_model(torch_input).detach().numpy()

                # JAX 计算
                jax_input = jnp.array(test_input)
                jax_output = self.jax_model(jax_input)
                jax_output_np = np.array(jax_output)

                # 验证与期望值的一致性
                np.testing.assert_allclose(torch_output, expected, rtol=1e-6)
                np.testing.assert_allclose(jax_output_np, expected, rtol=1e-6)
                np.testing.assert_allclose(torch_output, jax_output_np, rtol=1e-6)

        print("✓ 手动验证通过")

    def test_numerical_precision(self):
        """测试数值精度"""
        print("测试数值精度...")

        # 生成大量随机数据测试数值稳定性
        np_input = np.random.randn(1000, 8).astype(np.float32)

        # Torch 计算
        torch_input = torch.from_numpy(np_input)
        torch_output = self.torch_model(torch_input).detach().numpy()

        # JAX 计算
        jax_input = jnp.array(np_input)
        jax_output = self.jax_model(jax_input)
        jax_output_np = np.array(jax_output)

        # 计算统计信息
        diff = np.abs(torch_output - jax_output_np)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)

        print(f"最大差异: {max_diff:.2e}")
        print(f"平均差异: {mean_diff:.2e}")
        print(f"标准差: {std_diff:.2e}")

        # 验证高精度一致性
        np.testing.assert_allclose(torch_output, jax_output_np, rtol=1e-6, atol=1e-6)

        print("✓ 数值精度测试通过")

    def test_multiple_executions(self):
        """测试多次执行的一致性"""
        print("测试多次执行一致性...")

        np_input = np.random.randn(5, 6).astype(np.float32)

        # Torch 多次执行
        torch_input = torch.from_numpy(np_input)
        torch_results = [
            self.torch_model(torch_input).detach().numpy() for _ in range(5)
        ]

        # JAX 多次执行
        jax_input = jnp.array(np_input)
        jax_results = [np.array(self.jax_model(jax_input)) for _ in range(5)]

        # 验证每次执行结果一致
        for i in range(1, 5):
            np.testing.assert_allclose(torch_results[0], torch_results[i], rtol=1e-10)
            np.testing.assert_allclose(jax_results[0], jax_results[i], rtol=1e-10)

        # 验证 Torch 和 JAX 之间的一致性
        np.testing.assert_allclose(torch_results[0], jax_results[0], rtol=1e-6)

        print("✓ 多次执行测试通过")

    def test_jit_compilation(self):
        """测试 JIT 编译效果"""
        print("测试 JIT 编译效果...")

        # 第一次调用（编译）
        test_input = np.random.randn(3, 4).astype(np.float32)
        jax_input = jnp.array(test_input)

        # 编译时间测试
        import time

        start_time = time.time()
        result1 = self.jax_model(jax_input)
        compile_time = time.time() - start_time

        # 第二次调用（应该使用编译后的版本）
        start_time = time.time()
        result2 = self.jax_model(jax_input)
        execution_time = time.time() - start_time

        print(f"编译时间: {compile_time:.4f}s")
        print(f"执行时间: {execution_time:.4f}s")
        print(f"加速比: {compile_time/execution_time:.1f}x")

        # 验证结果一致性
        np.testing.assert_allclose(np.array(result1), np.array(result2), rtol=1e-10)

        print("✓ JIT 编译测试通过")


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("Torch 与 JAX SiluAndMul 一致性测试")
    print("=" * 60)

    # 创建测试套件
    suite = unittest.TestSuite()
    test_methods = [
        "test_basic_functionality",
        "test_various_input_shapes",
        "test_edge_cases",
        "test_manual_verification",
        "test_numerical_precision",
        "test_multiple_executions",
        "test_jit_compilation",
    ]

    for method in test_methods:
        suite.addTest(TestSiluAndMulConsistency(method))

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 输出总结
    print("\n" + "=" * 60)
    print("测试总结:")
    print(f"运行测试: {result.testsRun}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")

    if result.wasSuccessful():
        print("🎉 所有测试通过！Torch 和 JAX 版本完全一致")
        return True
    else:
        print("❌ 存在测试失败")
        for failure in result.failures:
            print(f"\n失败测试: {failure[0]}")
            print(f"错误信息: {failure[1]}")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
