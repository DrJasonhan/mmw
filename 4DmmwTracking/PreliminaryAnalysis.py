"""
根据以前的点云数据，逐帧展示
"""
import pandas as pd
import open3d as o3d
import numpy as np
import os, time


def label_to_rgb(label):
    colors = {
        0: (255, 0, 0),  # Red
        1: (255, 165, 0),  # Orange
        2: (255, 255, 0),  # Yellow
        3: (0, 128, 0),  # Green
        4: (0, 255, 255),  # Cyan
        5: (0, 0, 255),  # Blue
        6: (128, 0, 128),  # Purple
    }
    return colors.get(label, (255, 255, 255))  # Default to white if label is not recognized


def read_pcd(file):
    """读取pcd文件, 并返回o3d.geometry.PointCloud对象"""
    data = pd.read_csv(file, skiprows=11, sep=' ', header=None)

    points = data.iloc[:, 0:3].values
    labels = data.iloc[:, 4].values
    rgb = np.array([label_to_rgb(label) for label in labels])
    rgb = rgb / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    return pcd


folder_path = 'Data/tennis'
file_list = sorted([os.path.join(folder_path, file)
                    for file in os.listdir(folder_path)
                    if file.endswith('.pcd')])

vis = o3d.visualization.Visualizer()
vis.create_window()

view_control = vis.get_view_control()
view_control.set_lookat([0, 0, 0])  # 设置视点位置
view_control.set_up([0, -1, 0])  # 设置上方向
view_control.set_front([0, 0, -1])  # 设置前方向

for file in file_list:
    pcd = read_pcd(file)

    vis.clear_geometries()
    vis.add_geometry(pcd)
    vis.update_geometry(pcd)

    vis.poll_events()
    vis.update_renderer()
    time.sleep(0.1)  # 暂停0.1秒

vis.destroy_window()
