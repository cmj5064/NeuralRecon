import sys
sys.path.insert(0, "/data/ephemeral/home/NeuralRecon")
import cv2
import os
import numpy as np
import argparse
from tqdm import tqdm
from tools.kp_reproject import *

def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', help='datapath')
    args = parser.parse_args()
    return args


def set_intrinsics(K, width, height):
    fx = K[0, 0]
    fy = K[1, 1]
    paspect = fy / fx
    dim_aspect = width / height
    img_aspect = dim_aspect * paspect
    if img_aspect < 1.0:
        flen = fy / height
    else:
        flen = fx / width
    ppx = K[0, 2] / width
    ppy = K[1, 2] / height
    return [flen, 0, 0, paspect, ppx, ppy]


if __name__ == '__main__':
    args = args_parse()
    intrinsic_path = os.path.join(args.datapath, 'intrinsics')
    pose_path = os.path.join(args.datapath, 'poses')

    intrinsic_list = sorted(os.listdir(intrinsic_path))
    for i in tqdm(intrinsic_list, desc='Processing camera file...'):
        id = i[:-4]
        K = np.loadtxt(
            os.path.join(intrinsic_path, i),
            delimiter=' '
        )
        P = np.loadtxt(
            os.path.join(pose_path, i),
            delimiter=' '
        )

        # Moving down the X-Y plane in the ARKit coordinate to meet the training settings in ScanNet.
        # The coordinate system of ScanNet is Z Up.
        P[2, 3] += 1.5
        # P is the transformation matrix (camera to world ), and we need the transformation matrix (world to camera).
        # So we need to inverse P matrix, and then P_inv is the transformation matrix (world to camera).
        P_inv = np.linalg.inv(P)
        img = cv2.imread(os.path.join(args.datapath, 'images', f'{id}.jpg'))
        cam_path = os.path.join(args.datapath, 'images', f'{id}.cam')
        intrinsics = set_intrinsics(K, img.shape[1], img.shape[0])
        with open(cam_path, 'w') as f:
            s1, s2 = '', ''
            for i in range(3):
                for j in range(3):
                    s1 += str(P_inv[i][j]) + ' '
                s2 += str(P_inv[i][3]) + ' '
            f.write(s2 + s1[:-1] + '\n')
            f.write(str(intrinsics[0]) + ' 0 0 ' + str(intrinsics[3]) + ' ' + str(intrinsics[4]) + ' ' + str(intrinsics[5]) + '\n')