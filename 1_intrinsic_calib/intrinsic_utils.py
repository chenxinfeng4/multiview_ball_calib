import cv2
import numpy as np
import glob
import os
import os.path as osp
import datetime
import json
import re
import argparse
from multiprocessing import Pool

# 设置棋盘格大小
board_size = (11,8)
datetime_now = datetime.datetime.now()


def calibration(imageFolder, board_size):
    outdir = osp.join(imageFolder, 'calib')
    os.makedirs(outdir, exist_ok=True)

    # 获取图片路径
    images = glob.glob(osp.join(imageFolder, '*.jpg')) + glob.glob(osp.join(imageFolder, '*.png'))
    # %%设置物体点坐标, square_size 不会影响相机内参
    object_corners = np.zeros((board_size[0]*board_size[1], 3), np.float32)
    object_corners[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)

    # 存储棋盘识别的物体点和图像点
    objpoints = [] # 转换为物体坐标系的点
    imgpoints = [] # 转换为图像坐标系的点

    # %%遍历所有图片
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        image_shape = gray.shape

        # 查找棋盘格及其角点
        ret, corners = cv2.findChessboardCorners(gray, board_size, None)

        # 如果找到棋盘格
        if ret:
            # 存储物体点和图像点
            objpoints.append(object_corners)
            imgpoints.append(corners)

            # 可视化识别结果
            cv2.drawChessboardCorners(img, board_size, corners, ret)
            cv2.imwrite(osp.join(outdir, osp.basename(fname)), img)

    # 计算相机内参
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], 
                                                    None, None, flags=cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST)
    print("相机内参矩阵：\n", mtx.astype(int))
    with np.printoptions(precision=3):
        print("相机失真参数：\n", dist.flatten())

    # %%计算重新投影误差并筛选优质图片
    nimg = len(objpoints)
    img_error = np.zeros(nimg)
    for i in range(nimg):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        img_error[i] = error

    good_images = np.where(img_error < 0.5)[0]

    print("重新投影误差: ", np.mean(img_error))

    if len(good_images) == 0:
        print("没有优质图片！")
    elif len(good_images) == nimg:
        pass
    else:
        print("优质图片数量: ", len(good_images))

        # 选取优质图片，并重新计算相机内参
        objpoints2 = [objpoints[i] for i in good_images]
        imgpoints2 = [imgpoints[i] for i in good_images]
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints2, imgpoints2, gray.shape[::-1], None, None)

        print("相机内参矩阵：\n", mtx)

        with np.printoptions(precision=3):
            print("相机失真参数：\n", dist.flatten())

    # %%保存相机内参
    dist = dist.flatten()
    return mtx, dist, image_shape


def create_json_data(mtx, dist, image_shape):
    data = dict()
    # save the date as "2022-10-10 21:55:50"
    data['date'] = datetime_now.strftime("%Y-%m-%d %H:%M:%S")
    data['description'] = 'by opencv'
    data['K'] = mtx.astype(int).tolist()
    data['dist'] = [round(f, 3) for f in dist.flatten()]
    data['image_shape'] = image_shape
    return data


def calibrate_nview(prj_imgfolder, board_size):
    imageFolders = glob.glob(osp.join(prj_imgfolder, '[0-9]'))
    outjson = osp.join(prj_imgfolder, 'intrinsics_calib.json')
    labels = [osp.basename(imageFolder) for imageFolder in imageFolders]
    result = [calibration(imageFolder, board_size) for imageFolder in imageFolders]
    
    mtxs, dists, image_shapes = zip(*result)
    outdict = {label: create_json_data(mtx, dist, image_shape) 
               for label, mtx, dist, image_shape in zip(labels, mtxs, dists, image_shapes)}

    output = json.dumps(outdict, indent=4)
    output1 = re.sub(r'(\d+,?)\n\s+', r'\1 ', output)
    output2 = re.sub(r'\[\n\s*', r'[', output1)
    output3 = re.sub(r']\n\s*]', r']]', output2)
    output4 = re.sub(r'\s*]', r']', output3)
    with open(outjson, 'w') as f:
        f.write(output4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('prj_imgfolder', type=str)
    parser.add_argument('--board_size', type=int, nargs='+', default=board_size)
    args = parser.parse_args()
    assert len(args.board_size) == 2, "board_size should be 2 elements"
    assert sum(args.board_size) % 2 == 1, "board size should be (even, odd) or (odd, even)"

    assert osp.isdir(args.prj_imgfolder), "prj_imgfolder should be a folder"
    assert osp.isdir(osp.join(args.prj_imgfolder, '1')), "prj_imgfolder should contain subfolders"
    board_size = tuple(args.board_size)
    calibrate_nview(args.prj_imgfolder)
