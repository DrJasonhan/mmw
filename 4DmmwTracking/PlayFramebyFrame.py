"""
根据以前的点云数据，逐帧展示
"""
import pandas as pd
import open3d as o3d
import numpy as np
import os, time
import matplotlib.pyplot as plt

is_paused = False


#

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


# 创建一个网格平面的函数，作为地面
def create_ground_plane(size=10, divisions=10, color=[0.5, 0.5, 0.5], height=-1):
    # 生成网格线的点
    x_offset = size / 2
    y_offset = size / 2
    lines = []
    for i in range(divisions + 1):
        lines.append([i - x_offset, 0 - y_offset, height])
        lines.append([i - x_offset, size - y_offset, height])
        lines.append([0 - x_offset, i - y_offset, height])
        lines.append([size - x_offset, i - y_offset, height])

    # 创建LineSet对象，用于绘制线条
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(lines),
        lines=o3d.utility.Vector2iVector([[i, i + 1] for i in range(0, len(lines), 2)])
    )

    # 设置线条颜色
    colors = [color for i in range(len(line_set.lines))]
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


def create_axis(size=1.0):
    # 创建三个箭头分别代表X、Y、Z轴，分别对应 红绿蓝
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    return mesh_frame


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
# 创建地平面网格
ground_plane = create_ground_plane(size=30, divisions=30)
vis.add_geometry(ground_plane)

# 创建坐标轴箭头
axis = create_axis(size=0.5)
vis.add_geometry(axis)

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

        # time.sleep(0.05)  # 暂停0.05秒

    while is_paused:
        vis.poll_events()
        vis.update_renderer()

vis.run()
vis.destroy_window()

# show_pcd(pcd_accumulated)
