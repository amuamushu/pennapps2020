import torch
import cv2
import time
import argparse
import numpy as np
import os

import posenet

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--scale_factor', type=float, default=0.7125)
args = parser.parse_args()

def count_frames(filename):
    # there exists a faster method but its unreliable, looping through video doesn't take much time
    frames = 0
    cap = cv2.VideoCapture(filename)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frames += 1
        else:
            break
    cap.release()
    return frames

def preprocess(filename):
    """saves the positional data into 3 numpy files.  pose_scores_arr stores pose confidence, keypoint_scores_arr stores keypoint condidence, keypoint_coords_arr stores keypoint coordinates"""
    start = time.time()
    frame_count = count_frames(filename)
    print('Frames:', frame_count)
          
    # preprocess video
    pose_scores_arr = np.zeros((frame_count, 10))
    keypoint_scores_arr = np.zeros((frame_count, 10, 17))
    keypoint_coords_arr = np.zeros((frame_count, 10, 17, 2))
    
    model = posenet.load_model(args.model)
    # model = model.cuda() # uncomment if cuda is available
    output_stride = model.output_stride

    cap = cv2.VideoCapture(filename)

    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    frame_num = 0
    while(cap.isOpened()):
        ret, img = cap.read()
        if ret == True:
            input_image, display_image, output_scale = posenet.read_vid(img, args.scale_factor, output_stride)
            with torch.no_grad():
                input_image = torch.Tensor(input_image)
                # input_image = torch. Tensor(input_image).cuda()

                heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)

                target_pose_scores, target_keypoint_scores, target_keypoint_coords = posenet.decode_multiple_poses(
                    heatmaps_result.squeeze(0),
                    offsets_result.squeeze(0),
                    displacement_fwd_result.squeeze(0),
                    displacement_bwd_result.squeeze(0),
                    output_stride=output_stride,
                    max_pose_detections=10,
                    min_pose_score=0.15)

            target_keypoint_coords *= output_scale

            pose_scores_arr[frame_num] = target_pose_scores
            keypoint_scores_arr[frame_num] = target_keypoint_scores
            keypoint_coords_arr[frame_num] = target_keypoint_coords

            if frame_num % 100 == 0:
                print('Frame {} completed in {:.2f}'.format(frame_num, time.time() - start))
            frame_num += 1
        else:
            break

    np.save('pose_scores_arr.npy', pose_scores_arr)
    np.save('keypoint_scores_arr.npy',  keypoint_scores_arr)
    np.save('keypoint_coords_arr.npy',  keypoint_coords_arr)
    print('completed in {:.2f}'.format(time.time() - start))


def main():
    filename = 'TestVideo2.mp4' # change video here
    """IMPORTANT: PREPROCESS VIDEO ONLY IF VID IS NEW. OTHERWISE, COMMENT OUT NEXT LINE!!!!!"""
    # preprocess(filename)
    
    pose_scores_arr = np.load('pose_scores_arr.npy')
    keypoint_scores_arr = np.load('keypoint_scores_arr.npy')
    keypoint_coords_arr = np.load('keypoint_coords_arr.npy')

    model = posenet.load_model(args.model)
    # model = model.cuda() # uncomment if cuda is available
    output_stride = model.output_stride
    
    cap = cv2.VideoCapture(filename)
    frame_num = 0
    start = time.time()
    while(cap.isOpened()):
        ret, img = cap.read()
        if ret == True:
            input_image, display_image, output_scale = posenet.read_vid(
                img, scale_factor=args.scale_factor, output_stride=output_stride)
                
            target_overlay_image = posenet.draw_skel_and_kp(
                display_image, pose_scores_arr[frame_num], keypoint_scores_arr[frame_num], keypoint_coords_arr[frame_num],
                min_pose_score=0.15, min_part_score=0.1)
            cv2.imshow('posenet', target_overlay_image)           
            frame_num += 1
            
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    print(time.time() - start)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
