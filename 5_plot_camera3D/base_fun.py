import numpy as np


def get_cam_pose_vert(center:np.ndarray, rotate:np.ndarray, scale:float=20):
    rotation_matrix = rotate.T

    # 提取边缘的坐标
    edge_x = []
    edge_y = []
    edge_z = []

    vertices = [
        [-1, -1, 0],   # 底部中心点
        [1, -1, 0],   # 顶部顶点1
        [1, 1, 0],  # 底部顶点2
        [-1, 1, 0], # 底部顶点3
        [-1, -1, 3],   # 底部中心点
        [1, -1, 3],   # 顶部顶点1
        [1, 1, 3],  # 底部顶点2
        [-1, 1, 3], # 底部顶点3
        [-5, -3, 5],   # 底部中心点
        [5, -3, 5],   # 顶部顶点1
        [5, 3, 5],  # 底部顶点2
        [-5, 3, 5], # 底部顶点3
    ]
    vertices = np.array(vertices) * scale
    vertices[:,-1]
    edges = [
        (0, 1), (1, 2), (2, 3), (0, 3),  # 底部边缘
        (0+4, 1+4), (1+4, 2+4), (2+4, 3+4), (0+4, 3+4),   # 顶部的边缘
        (0, 4), (1, 1+4), (2, 2+4), (3, 3+4),   # 底到顶
        (0+8, 1+8), (1+8, 2+8), (2+8, 3+8), (0+8, 3+8),   # 顶部的边缘
        (4, 8), (5, 1+8), (6, 2+8), (7, 3+8),   # 顶部的边缘
    ]
    x, y, z = zip(*vertices)
    x, y, z = rotation_matrix @ np.array([x, y, z]) + center[:,None]
    # 提取边缘的坐标
    edge_x = []
    edge_y = []
    edge_z = []
    for s, e in edges:
        edge_x += [x[s], x[e], None]  # 在边缘两个顶点之间添加None，以绘制单独的线段
        edge_y += [y[s], y[e], None]
        edge_z += [z[s], z[e], None]
    
    return edge_x, edge_y, edge_z
