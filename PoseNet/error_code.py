import numpy as np

target_keypoint_scores = np.load('keypoint_scores.npy')
target_keypoint_coords = np.load('keypoint_coords.npy')
good_keypoint_scores = np.load('bad_keypoint_scores.npy')
good_keypoint_coords = np.load('bad_keypoint_coords.npy')

def foo(coords1, coords2, score1, score2):
    idx = (score1 > 0.1) & (score2 > 0.2)
    mat1 = coords1[idx]
    mat2 = coords2[idx]
    return mat1, mat2

target, good_coords = foo(target_keypoint_coords[0], good_keypoint_coords[0], target_keypoint_scores[0], good_keypoint_scores[0])
print(good_coords)
