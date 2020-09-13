import torch
import cv2
import time
import argparse
import numpy as np
import os
import error_code_numero_dos

import posenet

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default= 800)
parser.add_argument('--cam_height', type=int, default= 450)
parser.add_argument('--scale_factor', type=float, default=0.7125) # 
args = parser.parse_args()
    
#matrix1, matrix2
def find_error(coords1, coords2, score1, score2):
    idx = (score1 > 0.1) & (score2 > 0.2)
    mat1 = coords1[idx]
    mat2 = coords2[idx]
    n = len(mat1)
    if n == 0:
        return -1

    return error_code_numero_dos.calc(mat1, mat2)


def main():
    pose_scores_arr = np.load('pose_scores_arr.npy')
    keypoint_scores_arr = np.load('keypoint_scores_arr.npy')
    keypoint_coords_arr = np.load('keypoint_coords_arr.npy')
    keypoint_coords_arr *= 1.6
    frame_count = pose_scores_arr.shape[0]

    
    model = posenet.load_model(args.model)
    # model = model.cuda()
    output_stride = model.output_stride
    num_parts = 17
    
    # access webcam
    cap = cv2.VideoCapture(args.cam_id)
    cap.set(3, args.cam_width)
    cap.set(4, args.cam_height)
        
    start = time.time()
    paused = False
    frame_num = 0
    frames = 0
    error_sum = 0
    while True:
        if paused == False:
            input_image, display_image, output_scale = posenet.read_cap(
                cap, scale_factor=args.scale_factor, output_stride=output_stride)

            # display frames
            target_overlay_image = posenet.draw_skel_and_kp(
                    display_image, pose_scores_arr[frame_num % frame_count], keypoint_scores_arr[frame_num % frame_count], keypoint_coords_arr[frame_num % frame_count],
                    min_pose_score=0.15, min_part_score=0.1)
            cv2.imshow('posenet', target_overlay_image)
            
            frame_num += 1
            frames += 1

            # press 'p' to pause video, 'q' to exit video
            if cv2.waitKey(1) & 0xFF == ord('p'):
                paused = True
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if frame_num == 1:
                for i in range(3):
                    print(3-i)
                    time.sleep(1)
        else:
            input_image, display_image, output_scale = posenet.read_cap(
                cap, scale_factor=args.scale_factor, output_stride=output_stride)

            # determine pose
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
                    max_pose_detections=10,
                    min_pose_score=0.15)

            keypoint_coords *= output_scale

            if frames % 5 == 0:
                print('{:.1f}%'.format(min(error_sum / 2.5, 100)))
                error_sum = 0
            else:
                error_sum += find_error(keypoint_coords_arr[frame_num % frame_count][0], keypoint_coords[0], keypoint_scores_arr[frame_num % frame_count][0], keypoint_scores[0])
            frames += 1
            
            # display frames
            target_overlay_image = posenet.draw_skel_and_kp(
                    display_image, pose_scores_arr[frame_num % frame_count], keypoint_scores_arr[frame_num % frame_count], keypoint_coords_arr[frame_num % frame_count],
                    min_pose_score=0.15, min_part_score=0.1)
            cv2.imshow('posenet', target_overlay_image)

            # press 'p' to pause video, 'q' to exit video
            if cv2.waitKey(1) & 0xFF == ord('p'):
                paused = False 
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    print('Average FPS: ', frame_count / (time.time() - start))


if __name__ == "__main__":
    main()
