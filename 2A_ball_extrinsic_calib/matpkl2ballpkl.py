# python -m lilab.multiview_scripts_dev.s2_matpkl2ballpkl ../data/matpkl/ball.matpkl --time 1 12 24 35 48
import pickle
import numpy as np
import argparse
import os.path as osp
import cv2
import json

second_based = True
pthr = 0.7
matfile = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/wtxwt_social/ball/2022-04-29_17-58-45_ball.matpkl'


def load_mat(matfile):
    data = pickle.load(open(matfile, 'rb'))
    keypoint = data['keypoints'].copy()
    fps = data['info']['fps']
    vfile = osp.splitext(matfile)[0] + '.mp4'
    views_xywh = data['views_xywh'].astype(int)

    assert keypoint.ndim == 4 and keypoint.shape[2] == 1, "Only one class and one instance is supported"
    assert keypoint.shape[-1] == 3, "xyp is expected"
    keypoint = keypoint[:,:,0,:]
    return keypoint, fps, vfile, views_xywh


def auto_find_thr(pvals):
    pvals_ravel = pvals.ravel()
    pvals_ravel = pvals_ravel[~np.isnan(pvals_ravel)]
    pthr = np.percentile(pvals_ravel, 50) * 0.9
    return pthr

def split_keypoint(keypoint, fps, global_time):
    keypoint_p = keypoint[...,2]
    keypoint_xy = keypoint[...,:2]  # VIEWxTIMExXY
    pthr = auto_find_thr(keypoint_p)
    keypoint_xy[keypoint_p < pthr] = np.nan
    move_time = global_time[-1] + 3
    if second_based:
        global_index = np.array(np.array(global_time)*fps, dtype=int)
        move_index = int(move_time*fps)
    else:
        global_index = np.array(global_time, dtype=int)
        move_index = int(move_time)
    keypoint_xy_global = keypoint_xy[:, global_index, :]      # VIEWxTIMExXY
    if keypoint_xy.shape[1] > 1000:
        keypoint_xy_move = keypoint_xy[:, move_index:(-5*30), :]  # VIEWxTIMExXY
    else:
        keypoint_xy_move = keypoint_xy
    return keypoint_xy_global, keypoint_xy_move, global_index


def get_background_img(global_iframe, vfile, views_xywh):
    background_img = None
    img_stack = []
    if osp.exists(vfile):
        vin = cv2.VideoCapture(vfile)
        for i in global_iframe:
            vin.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, img = vin.read()
            assert ret, "Failed to read frame {}".format(i)
            img_stack.append(img)
        vin.release()
        img_stack = np.stack(img_stack, axis=0)
        background_canvas = np.median(img_stack, axis=0).astype(np.uint8) #HxWx3
        background_img = []
        for crop_xywh in views_xywh:
            x, y, w, h = crop_xywh
            background_img.append(background_canvas[y:y+h, x:x+w, :])
        return background_img
    else:
        print("Video file {} does not exist".format(vfile))
        background_img = []
        for crop_xywh in views_xywh:
            x, y, w, h = crop_xywh
            background_img = np.zeros((h, w, 3), dtype=np.uint8)
        return background_img


def downsampe_keypoint(keypoint_xy_move):
    ind_notnan = ~np.isnan(keypoint_xy_move[:,:,0]) #(nview, nframe)
    ind_3notnan = np.sum(ind_notnan, axis=0) >=4
    keypoint_xy_move = keypoint_xy_move[:, ind_3notnan]
    ind_notnan = np.isnan(keypoint_xy_move[:,:,0]) #(nview, nframe)
    nframe = keypoint_xy_move.shape[1]
    mycase=1
    nchoose=1000

    if mycase==0 or nframe<nchoose:
        keypoint_xy_move_downsample = keypoint_xy_move
    elif mycase==1:
        keypoint_xy_move_downsample = keypoint_xy_move[:,::3,:]
    elif mycase==2:
        ind_notnan_iframe = np.all(ind_notnan, axis=0) #(nframe)
        keypoint_xy_move_downsample = keypoint_xy_move[:, ind_notnan_iframe, :]
    elif mycase==3:
        count_notnan_iframe = np.mean(ind_notnan, axis=0) #(nframe)
        p_iframe = np.clip(count_notnan_iframe, 0.3, 1)
        p_iframe = (p_iframe-np.min(p_iframe))/(np.max(p_iframe)-np.min(p_iframe))
        ind_rand = np.random.random(count_notnan_iframe.shape) < p_iframe
        keypoint_xy_move_downsample = keypoint_xy_move[:, ind_rand, :]
    nview, nget=keypoint_xy_move_downsample.shape[:2]

    if nget>nchoose:
        ind_down = np.random.choice(nget, nchoose, replace=False)
        keypoint_xy_move_downsample = keypoint_xy_move_downsample[:, ind_down, :]
    return keypoint_xy_move_downsample


def convert(matfile, intrinsics_json, setup_json, global_time=None):
    if global_time is None:
        global_time = [0, 0, 0, 0, 0] # obsoleted feature
    intrinsics_json = json.load(open(intrinsics_json, 'r'))
    setup_json_data = json.load(open(setup_json, 'r'))

    keypoint, fps, vfile, views_xywh = load_mat(matfile)
    keypoint_xy_global, keypoint_xy_move, global_index = split_keypoint(keypoint, fps, global_time)
    keypoint_xy_move_downsample = downsampe_keypoint(keypoint_xy_move)
    background_img = get_background_img(global_index, vfile, views_xywh)
    fitball_xyz_global =  np.zeros((len(global_index), 3))+np.nan # obsoleted feature
    
    assert len(intrinsics_json) == len(views_xywh)
    
    outdict = {'landmarks_global_xy': keypoint_xy_global,         # VIEWxTIMExXY
               'landmarks_move_xy': keypoint_xy_move_downsample,  # VIEWxTIMExXY
               'global_iframe': global_index,                    # TIME, obsoleted feature
               'landmarks_global_cm':  fitball_xyz_global,        # TIMExXYZ
               'background_img': background_img,                 # VIEWxHxWx3
               'setup': setup_json_data,                         # setup.json content
               'intrinsics': intrinsics_json,               # intrinsics.json content
               }
    outfile = osp.splitext(matfile)[0] + '.ballpkl'
    pickle.dump(outdict, open(outfile, 'wb'))
    print('python -m lilab.multiview_scripts_dev.s3_ballpkl2calibpkl',
            "{}".format(outfile))
    return outfile

# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('matfile', type=str)
    parser.add_argument('intrinsic_json', type=str)
    parser.add_argument('setup_json', type=str)
    parser.add_argument('--time', type=float, nargs='+', help='obsoleted feature')
    args = parser.parse_args()
    if args.time is None or len(args.time) == 0:
        args.time = [0, 0, 0, 0, 0]
    assert len(args.time) == 5, "global_time should be 5 elements"
    convert(args.matfile, args.intrinsic_json, args.setup_json, args.time)
