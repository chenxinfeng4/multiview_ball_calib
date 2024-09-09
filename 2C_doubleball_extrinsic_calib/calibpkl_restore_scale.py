# python -m lilab.multiview_scripts_dev.p2_calibpkl_refine_byglobal calibpkl globalrefpkl
# %%
import numpy as np
import pickle
import os.path as osp
from multiview_calib.calibpkl_predict import CalibPredict
import cv2
import argparse
import matplotlib.pyplot as plt

calibpkl = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/LZTxWT_230505/2023-05-04_16-27-48ball.calibpkl'
globalrefpkl = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/LZTxWT_230505/ball/2023-05-04_13-11-04Scheckboard.globalrefpkl'

axis_length = 220

def plot_axis(calibPredict:CalibPredict, background_imgs:list):
    axis_points = np.float32([[0, 0, 0], [axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]])
    axis_recentor = np.float32([[0,0,0]])
    axis_points_recentor = axis_points + axis_recentor
    axis_points_recentor_xy = calibPredict.p3d_to_p2d(axis_points_recentor).astype(int)
    labeled_imgs = []
    for ix, (img, image_points) in enumerate(zip(background_imgs, axis_points_recentor_xy)):
        img = img.copy()
        origin_point = tuple(image_points[0].ravel())  # 原点坐标
        x_axis_end_point = tuple(image_points[1].ravel())  # X轴坐标
        cv2.line(img, origin_point, x_axis_end_point, (0, 0, 255), 3)  # 绘制X轴 color red
        cv2.putText(img, 'X', x_axis_end_point, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        y_axis_end_point = tuple(image_points[2].ravel())  # Y轴坐标
        cv2.line(img, origin_point, y_axis_end_point, (0, 255, 0), 3)  # 绘制Y轴 color green
        cv2.putText(img, 'Y', y_axis_end_point, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        z_axis_end_point = tuple(image_points[3].ravel())  # Z轴坐标
        cv2.line(img, origin_point, z_axis_end_point, (255, 0, 0), 3)  # 绘制Z轴 color blue
        cv2.putText(img, 'Z', z_axis_end_point, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        labeled_imgs.append(img)

    return labeled_imgs


def main(calibpkl, matpkl, armlen_world):
    data_calib = pickle.load(open(calibpkl, 'rb'))
    data_mat = pickle.load(open(matpkl, 'rb'))
    assert 'ba_poses' in data_calib, "ba_poses is expected in calibpkl"
    assert {'keypoints'} <= set(data_mat), "keypoint is expected in matpkl"
    if 'keypoint_xyz' in data_mat:
        keypoint_xy = data_mat['keypoint_xy']
    elif 'keypoints' in data_mat:
        keypoint_xy = data_mat['keypoints'][..., :2]
        keypoint_p = data_mat['keypoints'][..., 2]
        keypoint_xy[keypoint_p < 0.5] = np.nan
    
    calibPredict = CalibPredict(data_calib)
    keypoint_xyz = calibPredict.p2d_to_p3d(keypoint_xy)
    ball1 = keypoint_xyz[..., 0, :]
    ball3 = keypoint_xyz[..., -1, :]
    armlen = np.linalg.norm(ball1 - ball3, axis=-1).flatten()
    armlen_mean = np.nanmean(armlen)
    scale = armlen_world / armlen_mean

    ba_poses = data_calib['ba_poses']
    for CamParam in ba_poses.values():
        CamParam['t'] = [t*scale for t in CamParam['t']]

    outjpg = osp.splitext(calibpkl)[0]+'_carlbal.jpg'
    plt.figure()
    plt.hist((armlen-armlen_mean)*scale, bins=100, density=True)
    plt.ylabel('Density')
    plt.yticks([])
    plt.xlabel('Arm length register error (mm)')
    plt.savefig(outjpg, dpi=300)
    plt.close()
    
    outpkl = osp.splitext(calibpkl)[0]+'.recalibpkl'
    with open(outpkl, 'wb') as f:
        pickle.dump(data_calib, f)
        print(f'recalibpkl saved as {outpkl}')

    return outpkl


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('calibpkl', type=str)
    parser.add_argument('matpkl', type=str)
    parser.add_argument('--arm-length-mm', type=float, default=141)
    args = parser.parse_args()
    assert osp.exists(args.calibpkl)
    assert osp.exists(args.matpkl)
    main(args.calibpkl, args.matpkl, args.arm_length_mm)
