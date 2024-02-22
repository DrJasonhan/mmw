"""
根据以前的点云数据，逐帧展示
"""
import pandas as pd
import open3d as o3d
import numpy as np
import os, time
import matplotlib.pyplot as plt

is_paused = False


def on_key(vis, action, mods):
    """ 暂停/继续展示
    action：检查按键是被按下还是被释放，1按下，0释放；否则，会触发两次    """
    global is_paused
    if action == 1:
        is_paused = not is_paused
    return True


def show_pcd(pcd):
    vis.create_window()
    vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    vis.run()
    vis.destroy_window()


def label_to_rgb(label):
    colors = {
        0: [255, 0, 0],  # Red
        1: [255, 165, 0],  # Orange
        2: [255, 255, 0],  # Yellow
        3: [0, 128, 0],  # Green
        4: [0, 255, 255],  # Cyan
        5: [0, 0, 255],  # Blue
        6: [128, 0, 128],  # Purple
    }
    return np.array(colors.get(label, (255, 255, 255))) / 255


def read_pcd(file):
    """读取pcd文件, 并返回o3d.geometry.PointCloud对象"""
    data = pd.read_csv(file, skiprows=11, sep=' ', header=None)

    points = data.iloc[:, 0:3].values
    labels = data.iloc[:, 4].values


    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # # 根据人工标注的标签，设置颜色
    # rgb = np.array([label_to_rgb(label) for label in labels])
    # pcd.colors = o3d.utility.Vector3dVector(rgb)

    return pcd


## 读取数据
folder_path = 'Data/tennis'
file_list = sorted([os.path.join(folder_path, file)
                    for file in os.listdir(folder_path)
                    if file.endswith('.pcd')])
##

# 创建窗口，并设置视角
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window()
vis.register_key_action_callback(32, on_key)  # 32 空格的 ASCII码
point_cloud = read_pcd(file_list[0])
vis.add_geometry(point_cloud)
view_control = vis.get_view_control()
# ctr = vis.get_view_control()
# ctr.change_field_of_view(step=70)

pcd_accumulated = o3d.geometry.PointCloud()
for file in file_list:
    if not is_paused:
        pcd = read_pcd(file)
        pcd_accumulated += pcd

        # 聚类，并设置颜色
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(pcd.cluster_dbscan(eps=0.3, min_points=25, print_progress=True))
        max_label = max(labels)  # 最大的类别值
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 1  # 类别为0的，颜色设置为黑色
        # 更新要显示的点云对象
        point_cloud.points = pcd.points
        point_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])  # ndarray to vector3d

        vis.update_geometry(point_cloud)
        vis.poll_events()
        vis.update_renderer()

        time.sleep(0.05)  # 暂停0.05秒

    while is_paused:
        vis.poll_events()
        vis.update_renderer()

vis.run()
vis.destroy_window()

# show_pcd(pcd_accumulated)
