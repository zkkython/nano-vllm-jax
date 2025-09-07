# filename: simple_pmap.py
from functools import partial
import jax
import jax.numpy as jnp
from jax._src import config
from jax import pmap, random

config.update("jax_platform_name", "cpu")  # 在 CPU 上模拟

# 1. 模拟有 4 个设备
jax.distributed.initialize(
    coordinator_address="localhost:1234", num_processes=1, process_id=0
)
# 在实际多机多卡环境中，需要在不同进程/机器上运行此脚本，并配置正确的 coordinator_address

# 2. 检查设备
devices = jax.devices()
print(f"Number of devices: {len(devices)}")


def create_data():
    """创建示例数据"""
    key = random.PRNGKey(0)
    # 模拟输入数据：批量大小16，特征维度10
    x = random.normal(key, (16, 10))
    # 模拟目标数据
    y = random.randint(key, (16,), 0, 2)
    return x, y


def simple_model(params, x):
    """简单线性模型"""
    w, b = params
    return jnp.dot(x, w) + b


def loss_fn(params, x, y):
    """均方误差损失函数"""
    predictions = simple_model(params, x)
    return jnp.mean((predictions - y) ** 2)


# 正确使用 pmap 作为装饰器:cite[2]:cite[9]
@partial(pmap, axis_name="devices")
def pmapped_train_step(params, x_batch, y_batch):
    """并行化的训练步骤"""
    # 计算损失和梯度
    loss, grads = jax.value_and_grad(loss_fn)(params, x_batch, y_batch)

    # 跨设备同步梯度（数据并行的关键）:cite[3]
    grads = jax.lax.pmean(grads, axis_name="devices")
    loss = jax.lax.pmean(loss, axis_name="devices")

    # 简单参数更新（实际中应使用优化器）
    new_params = jax.tree.map(lambda p, g: p - 0.01 * g, params, grads)
    return new_params, loss


def main():
    print("可用设备:", jax.devices())

    # 准备数据
    x, y = create_data()

    # 初始化模型参数
    key = random.PRNGKey(42)
    w = random.normal(key, (10, 1))
    b = random.normal(key, (1,))
    params = (w, b)

    # 复制参数到所有设备（pmap 会自动处理）:cite[3]
    replicated_params = jax.tree.map(
        lambda x: jnp.array([x] * len(jax.devices())), params
    )

    # 将数据分片到设备上
    num_devices = len(jax.devices())
    assert x.shape[0] % num_devices == 0, "批量大小必须能被设备数整除"

    x_sharded = x.reshape(num_devices, -1, x.shape[1])
    y_sharded = y.reshape(num_devices, -1)

    # 执行并行训练
    print("开始并行训练...")
    for step in range(5):
        replicated_params, loss = pmapped_train_step(
            replicated_params, x_sharded, y_sharded
        )
        print(f"步骤 {step}, 损失: {loss[0]:.4f}")


if __name__ == "__main__":
    main()
