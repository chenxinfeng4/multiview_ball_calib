# python -m lilab.multiview_scripts_dev.s3_ballpkl2calibpkl a-b-l.ballpkl
# %%
import argparse
import matplotlib
import logging
import time
import os
import os.path as osp
import warnings
import pickle
import numpy as np
import copy
import itertools
import argparse

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from multiview_calib import utils
from multiview_calib.extrinsics_short import (compute_relative_poses_robust, visualise_epilines, 
                                        verify_view_tree, global_registration, visualise_global_registration,
                                        concatenate_relative_poses, visualise_cameras_and_triangulated_points,
                                        verify_landmarks, global_registration_np)
from multiview_calib.bundle_adjustment_scipy_short import (build_input, bundle_adjustment, evaluate, 
                                                     visualisation, unpack_camera_params, error_measure)
from multiview_calib.singleview_geometry import reprojection_error


logger = logging.getLogger(__name__)

ballcalibfile = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/wtxwt_social/ball/2022-04-25ball.calibpkl'

def convert_landmarks_global_mm(landmarks_global_cm_mat):
    return  {'ids': np.arange(len(landmarks_global_cm_mat)),
             'landmarks_global': landmarks_global_cm_mat.astype(np.float32)*10}

def convert_landmarks_xy(landmarks_xy_mat):
    landmarks = dict()
    for i, landmarks_iview in enumerate(landmarks_xy_mat):
        ids = np.where(~np.isnan(landmarks_iview[:, 0]))[0]
        landmarks_now = landmarks_iview[ids]
        landmarks[i] = {'ids': ids, 'landmarks': landmarks_now}
    return landmarks

def convert_filenames(background_img):
    return {i:background_img[i] for i in range(len(background_img))}

def get_sub_ba_poses(subviews, setup, intrinsics, landmarks_move_xy, background_img):
    setup = copy.deepcopy(setup)
    setup['views'] = subviews
    setup['minimal_tree'] = list(zip(subviews[1:], itertools.repeat(subviews[0])))
    relative_poses = a1_relative_poses(setup, intrinsics, landmarks_move_xy, background_img)
    extrinsics = a2_concatenate_relative_poses(setup, relative_poses)
    poses = intri_extrin_to_ba_poses(intrinsics, extrinsics)
    config = {'th_outliers_early':100, "th_outliers":20}
    ba_poses, ba_points = a3_bundle_ajustment(setup, poses, landmarks_move_xy, background_img,
                                              config = config, iter1=100, iter2=300)
    return setup, ba_poses, ba_points

def ba_poses_to_intrin_extrin(ba_poses):
    intrinsics = {view:{'K':data['K'], 'dist':data['dist']} for view,data in ba_poses.items()}
    extrinsics = {view:{'R':data['R'], 't':data['t']} for view,data in ba_poses.items()} 
    return intrinsics, extrinsics

def intri_extrin_to_ba_poses(intrin, extrin):
    views = set(intrin.keys()) & set(extrin.keys())
    ba_poses = {view:{'K':intrin[view]['K'], 
                      'dist':intrin[view]['dist'], 
                      'R':extrin[view]['R'], 
                      't':extrin[view]['t'],
                      'image_shape': intrin[view]['image_shape']}
                 for view in views}
    return ba_poses

def b1_merge_ballfile(ballfile='ballfile_mat.ballpkl',**kargs):
    balldata = pickle.load(open(ballfile, 'rb'))
    balldata.update(kargs)
    outpkl = osp.splitext(ballfile)[0] + '.calibpkl'
    pickle.dump(balldata, open(outpkl, 'wb'))
    return outpkl


def a0_read_ballfile(ballfile='ballfile_mat.pkl'):
    balldata = pickle.load(open(ballfile, 'rb'))
    setup = balldata['setup']
    intrinsics = balldata['intrinsics']
    intrinsics = {int(key):value for key, value in intrinsics.items()}
    landmarks_move_xy_mat   = balldata['landmarks_move_xy']
    landmarks_global_xy_mat = balldata['landmarks_global_xy']
    landmarks_global_cm_mat = balldata['landmarks_global_cm']
    background_img = balldata['background_img'].copy()

    # convert
    landmarks_move_xy   = convert_landmarks_xy(landmarks_move_xy_mat)
    landmarks_global_xy = convert_landmarks_xy(landmarks_global_xy_mat)
    landmarks_global_mm = convert_landmarks_global_mm(landmarks_global_cm_mat)
    background_img = convert_filenames(background_img)

    return setup, intrinsics, landmarks_move_xy, landmarks_global_xy, landmarks_global_mm, background_img


def a1_relative_poses(setup,
         intrinsics,
         landmarks,
         background_img,
         method='8point',
         th=20,
         max_paths=5,
         output_path="output/relative_poses/"):
    utils.mkdir(output_path)    
    utils.config_logger(os.path.join(output_path, "relative_poses_robust.log"))
    
    if not verify_view_tree(setup['minimal_tree']):
        raise ValueError("minimal_tree is not a valid tree!")     
    
    relative_poses = compute_relative_poses_robust(setup['views'], setup['minimal_tree'], intrinsics, 
                                                   landmarks, method=method, th=th, max_paths=max_paths,
                                                   verbose=1)
    visualise_epilines(setup['minimal_tree'], relative_poses, intrinsics, landmarks, 
                        background_img, output_path=output_path)
    
    return relative_poses


def a2_concatenate_relative_poses(setup,
         relative_poses,
         method='cross-ratios',
         output_path="output/relative_poses/"):
    
    utils.mkdir(output_path)
    utils.config_logger(os.path.join(output_path, "concat_relative_poses.log"))
    
    if not verify_view_tree(setup['minimal_tree']):
        raise ValueError("minimal_tree is not a valid tree!")         
    
    poses, triang_points = concatenate_relative_poses(setup['minimal_tree'], relative_poses, method)
    visualise_cameras_and_triangulated_points(setup['views'], setup['minimal_tree'], poses, triang_points, 
                                              max_points=100, path=output_path)

    return poses


def a3_bundle_ajustment(
         setup,
         pose,
         landmarks,
         filenames_images,
         config=None,
         iter1=200,
         iter2=200):

    __config__ = {
            "each_training": 1,
            "each_visualisation": 1,
            "th_outliers_early": 500,
            "th_outliers": 10,
            "optimize_points": True,
            "optimize_camera_params": True,
            "bounds": True,  
            "bounds_cp": [ 
                0.3, 0.3, 0.3,
                100, 100, 100,
                300, 300, 300, 300,
                0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0, 0, 0, 0, 0, 0, 0, 0],
            "bounds_pt": [1000, 1000, 1000],
            "max_nfev": 100,
            "max_nfev2": 100,
            "ftol": 1e-08,
            "xtol": 1e-08,  
            "loss": "linear",
            "f_scale": 1,
            "output_path": "output/bundle_adjustment/"
        }
    if config is not None:
        __config__.update(config)
    if iter1 is not None:
        __config__["max_nfev"] = iter1
    if iter2 is not None:
        __config__["max_nfev2"] = iter2   
        
    utils.mkdir(__config__["output_path"])
    utils.config_logger(os.path.join(__config__["output_path"], "bundle_adjustment.log"))

    intrinsics, extrinsics = ba_poses_to_intrin_extrin(pose)
    n_dist_coeffs = len(list(intrinsics.values())[0]['dist'])
    
    if not verify_view_tree(setup['minimal_tree']):
        raise ValueError("minimal_tree is not a valid tree!")  
        
    res, msg = verify_landmarks(landmarks)
    if not res:
        raise ValueError(msg)    

    views = setup['views']
    logging.info("-"*20)
    logging.info("Views: {}".format(views))
    
    if __config__["each_training"]<2 or __config__["each_training"] is None:
        logging.info("Use all the landmarks.")
    else:
        logging.info("Subsampling the landmarks to 1 every {}.".format(__config__["each_training"]))  

    logging.info("Preparing the input data...(this can take a while depending on the number of points to triangulate)")
    start = time.time()
    camera_params, points_3d, points_2d,\
    camera_indices, point_indices, \
    n_cameras, n_points, ids, views_and_ids = build_input(views, intrinsics, extrinsics, 
                                                          landmarks, each=__config__["each_training"], 
                                                          view_limit_triang=4)
    logging.info("The preparation of the input data took: {:0.2f}s".format(time.time()-start))
    logging.info("Sizes:")
    logging.info("\t camera_params: {}".format(camera_params.shape))
    logging.info("\t points_3d: {}".format(points_3d.shape))
    logging.info("\t points_2d: {}".format(points_2d.shape))
    
    f0 = evaluate(camera_params, points_3d, points_2d,
                  camera_indices, point_indices,
                  n_cameras, n_points) 

    class PlotWrapper(object):
        def __init__(self, title, path):
            self.title = title
            self.path = path
            
        def __enter__(self):
            plt.figure()

        def __exit__(self, *args):
            plt.title(self.title)
            plt.ylabel("X and Y coordinates")
            plt.xlabel("X and Y coordinates")
            plt.legend()
            plt.grid()
            plt.show()
            plt.savefig(self.path, bbox_inches='tight')

    with PlotWrapper("Residuals at initialization", 
                     os.path.join(__config__["output_path"], "1_initial_residuals.jpg")):
        camera_indices_rav = np.vstack([camera_indices]*2).T.ravel()
        for view_idx in range(n_cameras):
            m = np.where(camera_indices_rav==view_idx)[0]
            plt.plot(f0[m], label='{}'.format(views[view_idx]))
            
    if __config__["th_outliers_early"]==0 or __config__["th_outliers_early"] is None:
        logging.info("No early outlier rejection.")
    else:
        logging.info("Early Outlier rejection:")
        logging.info("\t threshold outliers: {}".format(__config__["th_outliers_early"])) 
        
        f0_ = np.abs(f0.reshape(-1,2))
        mask_outliers = np.logical_or(f0_[:,0]>__config__["th_outliers_early"],f0_[:,1]>__config__["th_outliers_early"])
                
        point_indices = point_indices[~mask_outliers]
        camera_indices = camera_indices[~mask_outliers]
        points_2d = points_2d[~mask_outliers]
        views_and_ids = [views_and_ids[i] for i,m in enumerate(~mask_outliers) if m]
        optimized_points = np.int32(list(set(point_indices)))
        logging.info("\t Number of points considered outliers: {}".format(sum(mask_outliers)))

        if sum(mask_outliers)/len(mask_outliers)>0.5:
            logging.info("!"*20)
            logging.info("More than half of the data points have been considered outliers! Something may have gone wrong.")
            logging.info("!"*20) 
        
    f01 = evaluate(camera_params, points_3d, points_2d,
                    camera_indices, point_indices,
                    n_cameras, n_points)         
    with PlotWrapper("2_Residuals after early outlier rejection",
                        os.path.join(__config__["output_path"], "2_early_outlier_rejection_residuals.jpg")):
        camera_indices_rav = np.vstack([camera_indices]*2).T.ravel()
        for view_idx in range(n_cameras):
            m = np.where(camera_indices_rav==view_idx)[0]
            plt.plot(f01[m], label='{}'.format(views[view_idx]))      
    
    n_camer_coeffs = camera_params.shape[1]
    assert n_camer_coeffs<=len(__config__["bounds_cp"]), n_dist_coeffs>=6
    __config__["bounds_cp"] = __config__["bounds_cp"][:n_camer_coeffs]

    new_camera_params, new_points_3d = bundle_adjustment(camera_params, points_3d, points_2d, camera_indices, 
                                                         point_indices, n_cameras, n_points, ids, 
                                                         optimize_camera_params=__config__["optimize_camera_params"], 
                                                         optimize_points=__config__["optimize_points"], 
                                                         ftol=__config__["ftol"], xtol=__config__["xtol"],
                                                         loss=__config__['loss'], f_scale=__config__['f_scale'],
                                                         max_nfev=__config__["max_nfev"], 
                                                         bounds=__config__["bounds"], 
                                                         bounds_cp = __config__["bounds_cp"],
                                                         bounds_pt = __config__["bounds_pt"],
                                                         verbose=True, eps=1e-12,
                                                         n_dist_coeffs=n_dist_coeffs)

    
    f1 = evaluate(new_camera_params, new_points_3d, points_2d, 
                  camera_indices, point_indices, 
                  n_cameras, n_points)

    avg_abs_res = np.abs(f1[:]).mean()
    logging.info("Average absolute residual: {:0.2f} over {} points.".format(avg_abs_res, len(f1)/2))
    if avg_abs_res>15:
        logging.info("!"*20)
        logging.info("The average absolute residual error ({}) is high! Something may have gone wrong.".format(avg_abs_res))
        logging.info("!"*20)
            
    with PlotWrapper("Residuals after bundle adjustment",
                    os.path.join(__config__["output_path"], "3_optimized_residuals.jpg")):
        camera_indices_rav = np.vstack([camera_indices]*2).T.ravel()
        for view_idx in range(n_cameras):
            m = np.where(camera_indices_rav==view_idx)[0]
            plt.plot(f1[m], label='{}'.format(views[view_idx]))            

    # Find ouliers points and remove them form the optimization.
    # These might be the result of inprecision in the annotations.
    # in this case we remove the resduals higher than 20 pixels.
    if not (__config__["th_outliers"]==0 or __config__["th_outliers"] is None):
        logging.info("Outlier rejection:")
        logging.info("\t threshold outliers: {}".format(__config__["th_outliers"])) 

        f1_ = np.abs(f1.reshape(-1,2))
        mask_outliers = np.logical_or(f1_[:,0]>__config__["th_outliers"],f1_[:,1]>__config__["th_outliers"])
                
        point_indices = point_indices[~mask_outliers]
        camera_indices = camera_indices[~mask_outliers]
        points_2d = points_2d[~mask_outliers]
        views_and_ids = [views_and_ids[i] for i,m in enumerate(~mask_outliers) if m]
        optimized_points = np.int32(list(set(point_indices)))
        logging.info("\t Number of points considered outliers: {}".format(sum(mask_outliers)))        
        
        if sum(mask_outliers)==0:
            logging.info("\t Exit.")
        else:
        
            if sum(mask_outliers)/len(mask_outliers)>0.5:
                logging.info("!"*20)
                logging.info("More than half of the data points have been considered outliers! Something may have gone wrong.")
                logging.info("!"*20)            

            logging.info("\t New sizes:")
            logging.info("\t\t camera_params: {}".format(camera_params.shape))
            logging.info("\t\t points_3d: {}".format(points_3d.shape))
            logging.info("\t\t points_2d: {}".format(points_2d.shape))
            
            if len(points_2d)==0:
                logging.info("No points left! Exit.")
                return

            new_camera_params, new_points_3d = bundle_adjustment(camera_params, points_3d, points_2d, camera_indices, 
                                                                 point_indices, n_cameras, n_points, ids,
                                                                 optimize_camera_params=__config__["optimize_camera_params"], 
                                                                 optimize_points=__config__["optimize_points"], 
                                                                 ftol=__config__["ftol"], xtol=__config__["xtol"],
                                                                 loss=__config__['loss'], f_scale=__config__['f_scale'],
                                                                 max_nfev=__config__["max_nfev2"], 
                                                                 bounds=__config__["bounds"], 
                                                                 bounds_cp = __config__["bounds_cp"],
                                                                 bounds_pt = __config__["bounds_pt"],
                                                                 verbose=True, eps=1e-12,
                                                                 n_dist_coeffs=n_dist_coeffs)

            f2 = evaluate(new_camera_params, new_points_3d, points_2d, 
                          camera_indices, point_indices, 
                          n_cameras, n_points)

            avg_abs_res = np.abs(f2[:]).mean()
            logging.info("Average absolute residual: {:0.2f} over {} points.".format(avg_abs_res, len(f2)/2))
            if avg_abs_res>15:
                logging.info("!"*20)
                logging.info("The average absolute residual error (after outlier removal) is high! Something may have gone wrong.".format(avg_abs_res))
                logging.info("!"*20)

            with PlotWrapper("4_Residuals after outlier removal",
                            os.path.join(__config__["output_path"], "4_optimized_residuals_outliers_removal.jpg")):
                camera_indices_rav = np.vstack([camera_indices]*2).T.ravel()
                for view_idx in range(n_cameras):
                    m = np.where(camera_indices_rav==view_idx)[0]
                    plt.plot(f2[m], label='{}'.format(views[view_idx]))

    logging.info("Reprojection errors (mean+-std pixels):")
    ba_poses = {}
    for i,(view, cp) in enumerate(zip(views, new_camera_params)):
        print('pose keys', pose[view].keys())
        K, R, t, dist = unpack_camera_params(cp)
        ba_poses[view] = {  "R":R.tolist(),
                            "t":t.tolist(), 
                            "K":K.tolist(),
                            "dist":dist.tolist(),
                            "image_shape": pose[view]["image_shape"]}
        
        points3d = new_points_3d[point_indices[camera_indices==i]]
        points2d = points_2d[camera_indices==i]
        
        if len(points3d)==0:
            raise RuntimeError("All 3D points have been discarded/considered outliers.")
        
        mean_error, std_error = reprojection_error(R, t, K, dist, points3d, points2d)
        logging.info("\t {} n_points={}: {:0.3f}+-{:0.3f}".format(view, len(points3d), mean_error, std_error))      

    ba_points = {"points_3d": new_points_3d[optimized_points], 
                 "ids":np.array(ids)[optimized_points]}  
        
    path = __config__['output_path']
    visualisation(setup, landmarks, filenames_images, 
                  new_camera_params, new_points_3d, 
                  points_2d, camera_indices, each=__config__["each_visualisation"], path=path)    
    return ba_poses, ba_points


def b0_landmarksto3d(ba_poses, landmarks, dump_images=False, setup=None, filenames=None): 
    views = list(set(landmarks.keys()) & set(ba_poses.keys()))
    intrinsics = {view:{'K':data['K'], 'dist':data['dist']} for view,data in ba_poses.items()}
    extrinsics = {view:{'R':data['R'], 't':data['t']} for view,data in ba_poses.items()}  
    camera_params, points_3d, points_2d,\
    camera_indices, point_indices, \
    n_cameras, n_points, ids, views_and_ids = build_input(views, intrinsics, extrinsics, 
                                                            landmarks, each=1,   #xxx
                                                            view_limit_triang=4)
    ba_points = {'ids': np.array(ids), 'points_3d': points_3d}

    each_forplot = max(1, len(ids)//200)
    if dump_images:
        visualisation(setup, landmarks, filenames, 
                  camera_params, points_3d, 
                  points_2d, camera_indices, each=each_forplot, path='output/custom')  

    return ba_points


def a4_global_bundle_adjustment(setup,
         ba_poses,
         ba_points,
         landmarks,
         landmarks_global,
         background_img=None,
         dump_images=True,
         output_path="output/global_registration"):
    
    utils.mkdir(output_path)
    utils.config_logger(os.path.join(output_path, "global_registration.log"))
    
    global_poses, global_triang_points = global_registration(ba_poses, ba_points, landmarks_global)  
    
    if dump_images:
        filenames = background_img
        visualise_global_registration(global_poses, landmarks_global, ba_poses, ba_points, 
                                      filenames, output_path=output_path)
    avg_dist, std_dist, median_dist = error_measure(setup, landmarks, global_poses, global_triang_points, 
                                                    scale=1, view_limit_triang=5)
    logging.info("Per pair of view average error:")
    logging.info("\t mean+-std: {:0.3f}+-{:0.3f} [unit of destination (dst) point set]".format(avg_dist, std_dist))
    logging.info("\t median:    {:0.3f}        [unit of destination (dst) point set]".format(median_dist))

    return global_poses, global_triang_points


def a5_global_registrate_short(zscale):
    fun_old2new = lambda p3d_old: p3d_old * [[1.0, 1.0, zscale]]
    fun_new2old = lambda p3d_new: p3d_new * [[1.0, 1.0, 1/zscale]]
    kargs = {'zscale': zscale}
    strfun_ba2global = 'lambda p3d_ba, kargs: p3d_ba * [[1.0, 1.0, kargs["zscale"]]] '
    strfun_global2ba = 'lambda p3d_global, kargs: p3d_global * [[1.0, 1.0, 1/kargs["zscale"]]]'
    return kargs, fun_old2new, fun_new2old, strfun_ba2global, strfun_global2ba

# %%
def a6_global_regist(ba_poses, landmarks_global_xy, setup, background_img, landmarks_global_mm):
    ba_points_global_xyz = b0_landmarksto3d(ba_poses, landmarks_global_xy, False, setup, background_img)
    ba_posessrc_refine, ba_points_global_xyz_refine = a4_global_bundle_adjustment(setup, ba_poses, 
            ba_points_global_xyz, landmarks_global_xy, landmarks_global_mm, background_img)
    if ba_points_global_xyz_refine['points_3d'][-1][-1]<0:
        L = landmarks_global_mm['landmarks_global']
        tmp = copy.copy(L[3]); L[3] = L[1]; L[1] = tmp
        ba_posessrc_refine, ba_points_global_xyz_refine = a4_global_bundle_adjustment(setup, ba_poses, 
            ba_points_global_xyz, landmarks_global_xy, landmarks_global_mm, background_img)
    
    ba_poses = ba_posessrc_refine
    ba_points_global_xyz = b0_landmarksto3d(ba_poses, landmarks_global_xy)
    zscale =  landmarks_global_mm['landmarks_global'][-1,-1] / ba_points_global_xyz_refine['points_3d'][-1][-1]
    assert 0.5 < zscale < 2
    return ba_poses, ba_points_global_xyz

def landmark_to_numpy(landmarks_global_xy:dict):
    views = sorted(list(landmarks_global_xy.keys()))
    nframe = max([max(lm['ids']) for lm in landmarks_global_xy.values()])+1
    ndim = landmarks_global_xy[views[0]]['landmarks'][0].size
    landmarks_np = np.zeros((len(views), nframe, ndim)) + np.nan
    for iview in views:
        id_lm = landmarks_global_xy[iview]
        ids, lm = id_lm['ids'], np.array(id_lm['landmarks'])
        landmarks_np[iview, ids, :] = lm

    return landmarks_np


def a6_global_regist_short(ba_poses, landmarks_global_xy, landmarks_global_mm):
    from multiview_calib.calibpkl_predict import CalibPredict
    calibPredict = CalibPredict({'ba_poses':ba_poses})
    landmarks_np = landmark_to_numpy(landmarks_global_xy)
    landmarks_3d = calibPredict.p2d_to_p3d(landmarks_np)
    landmarks_3d_mm = np.array(landmarks_global_mm['landmarks_global'])
    is_sorted = lambda a: np.all(a[:-1] <= a[1:])
    assert np.all(~np.isnan(landmarks_3d))
    assert is_sorted(landmarks_global_mm['ids'])
    assert landmarks_3d.shape == landmarks_3d_mm.shape
    global_poses, _ = global_registration_np(ba_poses, landmarks_3d[:4], landmarks_3d_mm[:4])
    global_poses, _ = global_registration_np(ba_poses, landmarks_3d, landmarks_3d_mm)

    calibPredict2 = CalibPredict({'ba_poses':global_poses})
    ba_points_global_xyz = calibPredict2.p2d_to_p3d(landmarks_np)
    print('Error distance:', np.linalg.norm(ba_points_global_xyz - landmarks_3d_mm, axis=1).round(1))
    return global_poses, ba_points_global_xyz


def main_calibrate(ballfile, skip_global:bool, skip_camera_intrinsic:bool):
    setup, intrinsics, landmarks_move_xy, landmarks_global_xy, landmarks_global_mm, background_img = a0_read_ballfile(ballfile)

    # %% do bundle adjustment
    relative_poses = a1_relative_poses(setup, intrinsics, landmarks_move_xy, background_img)
    extrinsics = a2_concatenate_relative_poses(setup, relative_poses)
    poses = intri_extrin_to_ba_poses(intrinsics, extrinsics)
    config = {'th_outliers_early':100, "th_outliers":20}
    if skip_camera_intrinsic:
        print("Fixing camera instrincis!")
        intrinsics_bounds = np.zeros(18) + 0.0001  #4+6+n
        config["bounds_cp"] = [ 0.3, 0.3, 0.3, 100, 100, 100, *intrinsics_bounds]
    ba_poses, ba_points = a3_bundle_ajustment(setup, poses, landmarks_move_xy, background_img,
                                            config = config, iter1=100, iter2=200)

    if skip_global:
        print('Skipping global BA')
        ba_poses_refine = ba_poses
    else:
        # %% do global registration
        # ba_poses_refine, ba_points_global_xyz_refine = a6_global_regist(ba_poses, landmarks_global_xy, setup, background_img, landmarks_global_mm)
        ba_poses_refine, ba_points_global_xyz_refine = a6_global_regist_short(ba_poses, landmarks_global_xy, landmarks_global_mm)
        b0_landmarksto3d(ba_poses_refine, landmarks_move_xy, True, setup, background_img)
        
        # %% do global registration - post hoc
        error_global = np.array(ba_points_global_xyz_refine) - landmarks_global_mm['landmarks_global']
        error_global_dist = np.linalg.norm(error_global, axis=1).round(1).tolist()
        print('Global Markers Error {}(mm)'.format(error_global_dist))

    # %% save data
    calibpklfile = b1_merge_ballfile(ballfile, ba_poses=ba_poses_refine)
    print('python -m lilab.multiview_scripts_dev.s4_matpkl2matcalibpkl ', 
            ballfile.replace('.ballpkl', '.matpkl'),
            ballfile.replace('.ballpkl', '.calibpkl'))
    return calibpklfile


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('ballfile', type=str)
    argparser.add_argument('--skip-global', action="store_true")
    argparser.add_argument('--skip-camera-intrinsic', action="store_true")
    arg = argparser.parse_args()
    main_calibrate(arg.ballfile, arg.skip_global, arg.skip_camera_intrinsic)
