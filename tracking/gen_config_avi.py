import os
import json
import numpy as np

import cv2


def gen_config(args):

    if args.seq != '':
        # generate config from a sequence name

        seq_home = 'datasets/OTB'
        result_home = 'results'

        seq_name = args.seq
        img_dir = os.path.join(seq_home, seq_name, 'img')
        gt_path = os.path.join(seq_home, seq_name, 'groundtruth_rect.txt')

        img_list = os.listdir(img_dir)
        img_list.sort()
        img_list = [os.path.join(img_dir, x) for x in img_list]
        
        with open(gt_path) as f:
            gt = np.loadtxt((x.replace('\t',',') for x in f), delimiter=',')
        init_bbox = gt[0]

        result_dir = os.path.join(result_home, seq_name)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        savefig_dir = os.path.join(result_dir, 'figs')
        result_path = os.path.join(result_dir, 'result.json')
        frame = None

    elif args.json != '':
        # load config from a json file

        param = json.load(open(args.json, 'r'))
        seq_name = param['seq_name']
        img_list = param['img_list']
        init_bbox = param['init_bbox']
        savefig_dir = param['savefig_dir']
        result_path = param['result_path']
        gt = None
        frame = None

    elif args.avi != '':
        cap = cv2.VideoCapture('/home/quinn/Robotics_Final/PyMDNet/Robotic_Perception_Final/datasets/avi/test_video.avi')
        ret, frame = cap.read()
        frame_height, frame_width = frame.shape[:2]
        if not ret:
            print('cannot read the video')
        
        init_bbox = cv2.selectROI(frame, False)
        img_list = cap
        gt = None
        savefig_dir = cv2.VideoWriter(f'/datasets/avi/results/PyMDNet.avi', 
                         cv2.VideoWriter_fourcc(*'XVID'), 60.0, 
                         (frame_width, frame_height), True)
        result_path = '/home/quinn/Robotics_Final/PyMDNet/Robotic_Perception_Final/results/test_avi'


    '''
    if args.savefig:
        if not os.path.exists(savefig_dir):
            os.makedirs(savefig_dir)
    else:
        savefig_dir = ''
    '''

    return img_list, frame, init_bbox, gt, savefig_dir, args.display, result_path
