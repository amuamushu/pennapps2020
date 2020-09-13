import torch
import cv2
import time
import argparse
import numpy as np
import os

import posenet

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default= 800)
parser.add_argument('--cam_height', type=int, default= 450)
parser.add_argument('--scale_factor', type=float, default=0.7125) # 
args = parser.parse_args()


def main():
    pose_scores_arr = np.load('pose_scores_arr.npy')
    keypoint_scores_arr = np.load('keypoint_scores_arr.npy')
    keypoint_coords_arr = np.load('keypoint_coords_arr.npy')
    frame_count = pose_scores_arr.shape[0]
    
    model = posenet.load_model(args.model)
    # model = model.cuda()
    output_stride = model.output_stride
    num_parts = 17

    # access webcam
    cap = cv2.VideoCapture(args.cam_id)
    cap.set(3, args.cam_width)
    cap.set(4, args.cam_height)

    # load data
    target_pose_scores = np.load('pose_scores.npy')
    target_keypoint_scores = np.load('keypoint_scores.npy')
    target_keypoint_coords = np.load('keypoint_coords.npy')

    start = time.time()
    paused = False
    frame_num = 0
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

            # press 'p' to pause video, 'q' to exit video
            if cv2.waitKey(1) & 0xFF == ord('p'):
                paused = True
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
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