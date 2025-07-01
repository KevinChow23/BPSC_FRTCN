import torch
import matplotlib.pyplot as plt
from TCN_train import TCN  # 确保从您的模型定义文件中导入TCN

def plot_parameter_histogram(model, layer_name, parameter_type='weight'):
    """
    绘制模型特定层的参数直方图。

    :param model: PyTorch模型。
    :param layer_name: 要绘制参数直方图的层的名称。
    :param parameter_type: 'weight' 或 'bias'，指定是绘制权重还是偏置的直方图。
    """
    # 模型参数的字典
    state_dict = model.state_dict()

    # 检索特定层的参数
    param_key = f'{layer_name}.{parameter_type}'
    if param_key in state_dict:
        param_data = state_dict[param_key].cpu().numpy().flatten()

        # 绘制直方图
        plt.hist(param_data, bins=50)
        plt.title(f'Histogram of {parameter_type} in layer {layer_name}')
        plt.xlabel('Parameter Value')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()
    else:
        print(f"No parameter named {param_key} found in this model.")

# 实例化模型
# 实例化模型
num_inputs = 5
num_channels = [32, 16, 4, 1]
kernel_size = 3
dropout = 0.3
model = TCN(num_inputs, num_channels, kernel_size, dropout)

# 打印模型的状态字典中的所有键（key）
for param_tensor in model.state_dict():
    print(param_tensor)

# 假设你想要绘制名为'temporal_block_1.conv1'的层的权重直方图
layer_name = 'network.3.net.4.weight_g'

# 绘制权重直方图
plot_parameter_histogram(model, layer_name, parameter_type='weight')

