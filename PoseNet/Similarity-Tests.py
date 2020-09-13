import torch
import cv2
import time
import argparse
import numpy as np
import os
import posenet
from scipy.spatial.transform import Rotation as R
import error_code_numero_dos

def ralign(X,Y):
    m, n = X.shape

    mx = X.mean(1)
    my = Y.mean(1)
    Xc =  X - np.tile(mx, (n, 1)).T
    Yc =  Y - np.tile(my, (n, 1)).T

    sx = np.mean(np.sum(Xc*Xc, 0))
    sy = np.mean(np.sum(Yc*Yc, 0))

    Sxy = np.dot(Yc, Xc.T) / n

    U,D,V = np.linalg.svd(Sxy,full_matrices=True,compute_uv=True)
    V=V.T.copy()
    #print U,"\n\n",D,"\n\n",V
    r = np.linalg.matrix_rank(Sxy)
    d = np.linalg.det(Sxy)
    S = np.eye(m)
    if r > (m - 1):
        if ( np.det(Sxy) < 0 ):
            S[m, m] = -1;
        elif (r == m - 1):
            if (np.det(U) * np.det(V) < 0):
                S[m, m] = -1  
        else:
            R = np.eye(2)
            c = 1
            t = np.zeros(2)
            return R,c,t

    R = np.dot( np.dot(U, S ), V.T)

    c = np.trace(np.dot(np.diag(D), S)) / sx
    t = my - c * np.dot(R, mx)
    print(X)
    print(Y)
    return rmsd((R*c).dot(X)+np.expand_dims(t,axis=1), Y)

def rmsd(V, W):
    """
    Calculate Root-mean-square deviation from two sets of vectors V and W.
    Parameters
    ----------
    V : array
        (N,D) matrix, where N is points and D is dimension.
    W : array
        (N,D) matrix, where N is points and D is dimension.
    Returns
    -------
    rmsd : float
        Root-mean-square deviation between the two vectors
    """
    diff = np.array(V) - np.array(W)
    N = len(V)
    return np.sqrt((diff * diff).sum() / N)

#matrix1, matrix2
def find_error(coords1, coords2, score1, score2):
    idx = (score1 > 0.1) & (score2 > 0.2)
    mat1 = coords1[idx]
    mat2 = coords2[idx]
    n = len(mat1)
    if n == 0:
        return -1
    # return error_code_numero_dos.calc(mat1, mat2)
    return ralign(mat1, mat2)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default= 800)
parser.add_argument('--cam_height', type=int, default= 450)
parser.add_argument('--scale_factor', type=float, default=0.7125) # 
args = parser.parse_args()

def main():        
    model = posenet.load_model(args.model)
    # model = model.cuda()
    output_stride = model.output_stride
    num_parts = 17
    
    cap = cv2.VideoCapture(args.cam_id)
    cap.set(3, args.cam_width)
    cap.set(4, args.cam_height)
    
    target_pose_scores = np.load('pose_scores.npy')
    target_keypoint_scores = np.load('keypoint_scores.npy')
    target_keypoint_coords = np.load('keypoint_coords.npy')
    
    start = time.time()
    frame_count = 0
    error_sum = 0
    while True:
        input_image, display_image, output_scale = posenet.read_cap(
            cap, scale_factor=args.scale_factor, output_stride=output_stride)

        with torch.no_grad():
            input_image = torch.Tensor(input_image)
            # input_image = torch. Tensor(input_image).cuda()

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
                heatmaps_result.squeeze(0),
                offsets_result.squeeze(0),
                displacement_fwd_result.squeeze(0),
                displacement_bwd_result.squeeze(0),
                output_stride=output_stride,
                max_pose_detections=1,
                min_pose_score=0.15)

        keypoint_coords *= output_scale

        """
        overlay_image = posenet.draw_skel_and_kp(
            display_image, pose_scores, keypoint_scores, keypoint_coords,
            min_pose_score=0.15, min_part_score=0.1)
        cv2.imshow('posenet', overlay_image)
        
        # save pose
        if frame_count == 50:
            np.save('bad_pose_scores.npy', pose_scores)
            np.save('bad_keypoint_scores.npy', keypoint_scores)
            np.save('bad_keypoint_coords.npy', keypoint_coords)
            break
        """
    
        overlay_image_target = posenet.draw_skel_and_kp(
            display_image, target_pose_scores, target_keypoint_scores, target_keypoint_coords,
            min_pose_score=0.15, min_part_score=0.1)
        cv2.imshow('posenet', overlay_image_target)

        # check if pose is similar
        if frame_count % 5 == 0:
            print('{:.1f}'.format(error_sum / 5))
            error_sum = 0
        else:
            error_sum += find_error(target_keypoint_coords[0], keypoint_coords[0], target_keypoint_scores[0], keypoint_scores[0])

        
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print('Average FPS: ', frame_count / (time.time() - start))


if __name__ == "__main__":
    main()
