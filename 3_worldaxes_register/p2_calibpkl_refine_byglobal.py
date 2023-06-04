# python -m lilab.multiview_scripts_dev.p2_calibpkl_refine_byglobal calibpkl globalrefpkl
# %%
import numpy as np
import pickle
import os.path as osp
from lilab.multiview_scripts_dev.s6_calibpkl_predict import CalibPredict
from multiview_calib.extrinsics_short import global_registration_np
import cv2
import argparse

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


def main(calibpkl, globalrefpkl):
    data_calib = pickle.load(open(calibpkl, 'rb'))
    data_globalref = pickle.load(open(globalrefpkl, 'rb'))
    assert 'ba_poses' in data_calib, "ba_poses is expected in calibpkl"
    assert {'keypoint_xy', 'keypoint_xyz'} <= set(data_globalref), "keypoint_xy and keypoint_xyz are expected in globalrefpkl"
    
    calibPredict = CalibPredict(data_calib)
    reg_xyz_mm = data_globalref['keypoint_xyz']
    keypoint_xy = data_globalref['keypoint_xy']
    keypoint_old_xyz = calibPredict.p2d_to_p3d(keypoint_xy)
    ba_poses = data_calib['ba_poses']
    global_poses, _ = global_registration_np(ba_poses, keypoint_old_xyz, reg_xyz_mm)

    new_calibPredict = CalibPredict({'ba_poses': global_poses})
    keypoint_new_xyz = new_calibPredict.p2d_to_p3d(keypoint_xy)

    error_dist = np.linalg.norm((keypoint_new_xyz - reg_xyz_mm), axis=-1)
    error_dist_mean = np.mean(error_dist)
    error_dist_std = np.std(error_dist)
    error_dist_max = np.max(error_dist)
    print(f'error_dist: {error_dist_mean :.2f} ± {error_dist_std:.2f} mm, max: {error_dist_max:.2f} mm')
    if error_dist_mean > 2 or error_dist_max > 4:
        print('!!!!! too large error distance, warning.')
    else:
        print('Global registration success.')

    data_calib['ba_poses'] = global_poses

    if 'background_img' in data_calib:
        background_imgs = data_calib['background_img']
        label_imgs = plot_axis(new_calibPredict, background_imgs)
        for ix, label_img in enumerate(label_imgs):
            fout = osp.splitext(calibpkl)[0] + f'-label-{ix}.jpg'
            cv2.imwrite(fout, label_img)
    
    outpkl = osp.splitext(calibpkl)[0]+'.recalibpkl'
    with open(outpkl, 'wb') as f:
        pickle.dump(data_calib, f)
        print(f'recalibpkl saved as {outpkl}')

    return outpkl


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('calibpkl', type=str)
    parser.add_argument('globalrefpkl', type=str)
    args = parser.parse_args()
    assert osp.exists(args.calibpkl)
    assert osp.exists(args.globalrefpkl)
    main(args.calibpkl, args.globalrefpkl)
