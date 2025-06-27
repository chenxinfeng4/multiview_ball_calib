import pickle
import numpy as np
import os.path as osp
import json


def convert_dannce(calibpkl) -> str:
    import scipy.io as sio
    ba_poses = pickle.load(open(calibpkl, 'rb'))['ba_poses']
    ncam = len(ba_poses)
    camParams = []
    for i in range(ncam):
        pose = ba_poses[i]
        K0 = np.array(pose['K'])
        dist0 = np.array(pose['dist'])
        R0 = np.array(pose['R'])
        t0 = np.array(pose['t'])
        K = K0.T + [[0,0,0], [0,0,0], [1,1,0]]
        t = t0.reshape(1,-1)
        r = R0.T
        RDistort = dist0[[0,1,4]].reshape(1,-1)
        TDistort = dist0[[2,3]].reshape(1,-1)
        camParams.append({
            'K': K,
            'RDistort': RDistort,
            'TDistort': TDistort,
            'r': r,
            't': t
        })

    # 将字典数据写入 matlab 文件
    dannce_mat = osp.join(osp.splitext(calibpkl)[0] + '_dannce.mat')
    sio.savemat(dannce_mat, 
                {'camParams': np.array(camParams).reshape(-1,1)})
    print("DANNCE :", dannce_mat)
    return dannce_mat


def convert_json(calibpkl) -> str:
    ba_poses = pickle.load(open(calibpkl, 'rb'))['ba_poses']
    ncam = len(ba_poses)
    for i in range(ncam):
        pose = ba_poses[i]
        pose['K'] = np.array(pose['K']).tolist()
        pose['dist'] = np.array(pose['dist']).tolist()
        pose['R'] = np.array(pose['R']).tolist()
        pose['t'] = np.array(pose['t']).tolist()
    
    # 将字典数据写入JSON文件
    json_file = osp.join(osp.splitext(calibpkl)[0] + '_json.json')
    with open(json_file, 'w') as f:
        json.dump(ba_poses, f, indent=4)
    print("json :", json_file)
    return json_file


def convert_anipose(calibpkl, cam_wh=[1280, 800]):
    import cv2
    import toml
    ba_poses = pickle.load(open(calibpkl, 'rb'))['ba_poses']
    ncam = len(ba_poses)
    toml_data = {}
    for i in range(ncam):
        pose0 = ba_poses[i]
        pose = {}
        pose['name'] = f'cam_{i}'
        pose['size'] = cam_wh
        pose['matrix'] = np.array(pose0['K']).tolist()
        pose['distortions'] = np.array(pose0['dist']).tolist()[:4]
        pose['translation'] = np.array(pose0['t']).tolist()
        pose['rotation'] = cv2.Rodrigues(np.array(pose0['R']))[0].ravel().tolist()
        pose['fisheye'] = True
        toml_data[f"cam_{i}"] = pose

    toml_data["metadata"] = {'adjusted': False,
                             'error': 1.0}

    # 将TOML数据写入文件
    toml_file = osp.join(osp.splitext(calibpkl)[0] + '_anipose.toml')
    with open(toml_file, "w") as f:
        toml.dump(toml_data, f)
    print("DEEPLABCUT / ANIPOSE :", toml_file)
    return toml_file
