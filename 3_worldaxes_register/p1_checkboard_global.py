# python -m lilab.multiview_scripts_dev.p1_checkboard_global A/B/C.mp4 --board_size 6 9 --square_size 23
"""
python -m lilab.multiview_scripts_dev.p1_checkboard_global \
    /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/LZTxWT_230505/ball/2023-05-04_13-11-04Scheckboard.mp4 \
    --board_size 11 8 --square_size_mm 20
"""
# %%
import numpy as np
import cv2
import os.path as osp
import matplotlib.pyplot as plt
import pickle
import argparse
import json

vfile='/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/LZTxWT_230505/ball/2023-05-04_13-11-04Scheckboard.mp4'
board_size = (11, 8)
square_size = 20.0

def get_view_xywh_1280x800x9():
    w, h = 1280, 800
    crop_xywh = [[w*0,h*0,w,h],
                [w*1,h*0,w,h],
                [w*2,h*0,w,h],
                [w*0,h*1,w,h],
                [w*1,h*1,w,h],
                [w*2,h*1,w,h],
                [w*0,h*2,w,h],
                [w*1,h*2,w,h],
                [w*2,h*2,w,h]]
    return crop_xywh

# %%
def __convert(img_crop_l, intrinces_dict, board_size, square_size, outdir):
    # all the image are the checkboard image, so we next detect the corner of the checkboard using cv2
    object_corners = np.zeros((board_size[0]*board_size[1], 3), np.float32)
    object_corners[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    object_corners[:, 0] = object_corners[:, 0].max() - object_corners[:, 0]
    object_corners *= square_size
    axis_length = square_size * (board_size[0] // 2)

    # set axix origin point in the centor
    object_corners[:, 0] -= (board_size[0]//2)*square_size
    object_corners[:, 1] -= (board_size[1]//2)*square_size

    keypoint_xy = np.ones((len(img_crop_l), board_size[0]*board_size[1], 2), float) * np.nan

    for i, image in enumerate(img_crop_l):
        dist = np.float64(intrinces_dict[str(i)]['dist']).flatten()
        mtx = np.float64(intrinces_dict[str(i)]['K'])
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, board_size, None)

        if ret:
            keypoint_xy[i,:] = corners[:,0,:]
            # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            # corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            # 在输入图像上绘制角点
            cv2.drawChessboardCorners(image, board_size, corners, ret)

            # 计算相机内部参数, 默认不做畸变考虑
            case_name = 2
            if case_name==0:
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([object_corners], [corners], gray.shape[::-1], None, None)
            elif case_name==1:
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([object_corners], [corners], gray.shape[::-1], None, None, 
                    flags=cv2.CALIB_FIX_K1 + cv2.CALIB_FIX_K2 + cv2.CALIB_FIX_K3 + cv2.CALIB_ZERO_TANGENT_DIST)
            elif case_name==2:
                # use cv2.solvePnP
                ret, rvec, tvec = cv2.solvePnP(object_corners, corners, mtx, dist)
                rvecs, tvecs = [rvec], [tvec]
            else:
                raise Exception('Unknown case')
            
           # 校正图像
            img_corrected = cv2.undistort(image, mtx, dist)

            # 绘制 x、y、z 坐标轴

            # 计算坐标轴在相机坐标系中的位置
            # 定义坐标系的空间点坐标
            axis_points = np.float32([[0, 0, 0], [axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]])

            # 将坐标系的空间点坐标投影到图像平面
            image_points, _ = cv2.projectPoints(axis_points, rvecs[0], tvecs[0], mtx, dist)
            image_points = np.int32(image_points)

            # 在图像上绘制坐标轴
            origin_point = tuple(image_points[0].ravel())  # 原点坐标
            x_axis_end_point = tuple(image_points[1].ravel())  # X轴坐标
            cv2.line(img_corrected, origin_point, x_axis_end_point, (0, 0, 255), 3)  # 绘制X轴 color red
            cv2.putText(img_corrected, 'X', x_axis_end_point, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            y_axis_end_point = tuple(image_points[2].ravel())  # Y轴坐标
            cv2.line(img_corrected, origin_point, y_axis_end_point, (0, 255, 0), 3)  # 绘制Y轴 color green
            cv2.putText(img_corrected, 'Y', y_axis_end_point, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            z_axis_end_point = tuple(image_points[3].ravel())  # Z轴坐标
            cv2.line(img_corrected, origin_point, z_axis_end_point, (255, 0, 0), 3)  # 绘制Z轴 color blue
            cv2.putText(img_corrected, 'Z', z_axis_end_point, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            cv2.imwrite(osp.join(outdir, f'global_ref_axis{i}.jpg'), img_corrected)

    keypoint_xyz = object_corners

    outdict = {'board_size': board_size,
            'square_size': square_size,
            'keypoint_xyz': keypoint_xyz,
            'keypoint_xy': keypoint_xy}

    detected_nviews = np.sum(~np.isnan(keypoint_xy[:,0,0]))
    assert detected_nviews>=2, 'Not enough checkboard detected, maybe wrong board_size.'
    outpkl = outdir + osp.splitext(vfile)[0]+'.globalrefpkl'
    outpkl = osp.join(outdir, f'calibration.globalrefpkl')
    with open(outpkl, 'wb') as f:
        pickle.dump(outdict, f)


def convert_from_img(img_folder, intrinsic_json, board_size, square_size):
    import glob
    intrinces_dict = json.load(open(intrinsic_json, 'r'))
    nview = len(intrinces_dict)
    img_files = []
    for iview in range(nview):
        img_file = glob.glob(osp.join(img_folder, f'{iview}.jpg')) + \
                   glob.glob(osp.join(img_folder, f'{iview}.jpeg')) + \
                    glob.glob(osp.join(img_folder, f'{iview}.bmp')) + \
                   glob.glob(osp.join(img_folder, f'{iview}.png'))
        assert len(img_file) == 1
        img_files.append(img_file[0])

    img_crop_l = [cv2.imread(img_file) for img_file in img_files]
    __convert(img_crop_l, intrinces_dict, board_size, square_size, img_folder)


def convert(vfile, intrinsic_json, board_size, square_size):
    vid = cv2.VideoCapture(vfile)
    ret, img = vid.read()
    assert ret, "Failed to read frame {}".format(0)
    crop_xywh_l = get_view_xywh_1280x800x9()
    intrinces_dict = json.load(open(intrinsic_json, 'r'))

    # get the crop image by crop_xywh
    img_crop_l = [img[xywh[1]:xywh[1]+xywh[3], xywh[0]:xywh[0]+xywh[2], :] for xywh in crop_xywh_l]
    __convert(img_crop_l, intrinces_dict, board_size, square_size, osp.dirname(vfile))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', type=str, help='path to video or folder')
    parser.add_argument('--intrinsic_json', type=str, default='bob')
    parser.add_argument('--board_size', type=int, nargs='+', default=board_size)
    parser.add_argument('--square_size_mm', type=float, default=square_size)
    args = parser.parse_args()
    assert len(args.board_size) == 2, "board_size should be 2 elements"
    assert sum(args.board_size) % 2 == 1, "board size should be (even, odd) or (odd, even)"

    assert osp.isfile(args.video_path)
    convert(args.video_path, args.intrinsic_json, args.board_size, args.square_size_mm)
