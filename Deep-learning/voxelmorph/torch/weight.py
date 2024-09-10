import numpy as np
import matplotlib.pyplot as plt

def create_complex_weight_map(volume_shape, edge_weight=1.5, center_weight=1.0, inner_cube1_ratio=0.8, inner_cube2_ratio=0.6):
    D, H, W = volume_shape
    weight_map = np.ones(volume_shape) * center_weight
    
    # 计算第一个小正方体的尺寸和位置
    inner1_D = int(inner_cube1_ratio * D)
    inner1_H = int(inner_cube1_ratio * H)
    inner1_W = int(inner_cube1_ratio * W)
    
    start1_D = 0
    end1_D = start1_D + inner1_D
    start1_H = (H - inner1_H) // 2
    end1_H = start1_H + inner1_H
    start1_W = (W - inner1_W) // 2
    end1_W = start1_W + inner1_W
    
    # 设置第一个小正方体部分的权重
    weight_map[start1_D:end1_D, start1_H:end1_H, start1_W:end1_W] = edge_weight
    
    # 计算第二个小正方体的尺寸和位置
    inner2_D = int(inner_cube2_ratio * D)
    inner2_H = int(inner_cube2_ratio * H)
    inner2_W = int(inner_cube2_ratio * W)
    
    start2_D = 0
    end2_D = start2_D + inner2_D
    start2_H = (H - inner2_H) // 2
    end2_H = start2_H + inner2_H
    start2_W = (W - inner2_W) // 2
    end2_W = start2_W + inner2_W
    
    # 设置第二个小正方体部分的权重
    weight_map[start2_D:end2_D, start2_H:end2_H, start2_W:end2_W] = center_weight

    return weight_map

# 可视化函数
def visualize_weight_map(weight_map):
    mid_slice = weight_map.shape[0] // 2
    plt.figure(figsize=(10, 10))
    plt.imshow(weight_map[mid_slice], cmap='Reds', vmin=1.0, vmax=1.5)
    plt.title('Weight Map Cross-Section')
    plt.colorbar()
    plt.axis('off')
    plt.show()

# 示例使用
volume_shape = (256, 256, 256)
weight_map = create_complex_weight_map(volume_shape, edge_weight=1.5, center_weight=1.0, inner_cube1_ratio=0.8, inner_cube2_ratio=0.6)

# 可视化
visualize_weight_map(weight_map)