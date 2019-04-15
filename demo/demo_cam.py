import os
import cv2
import pdb
import torch
import numpy as np
import argparse
from datetime import datetime
from imutils.video import FPS

import _init_paths
from model.module import networks
from SSH.model.SSH import SSH
from SSH.model.utils.config import cfg
from SSH.model.network import load_check_point
from SSH.model.nms.nms_wrapper import nms
from SSH.model.utils.test_utils import _get_image_blob, _compute_scaling_factor

import time

def detect(img, netF, thresh=0.5):
    with torch.no_grad():
        img_scale = _compute_scaling_factor(img.shape, cfg.TEST.SCALES[0], cfg.TEST.MAX_SIZE)
        img_blob = _get_image_blob(img, [img_scale])[0]

        img_info = np.array([[img_blob['data'].shape[2], img_blob['data'].shape[3], img_scale]])
        img_data = img_blob['data']

        img_info = torch.from_numpy(img_info).cuda()
        img_data = torch.from_numpy(img_data).cuda()

        ssh_rois = netF(img_data, img_info)
        inds = (ssh_rois[:, :, 4] > thresh)
        ssh_roi_keep = ssh_rois[:, inds[0], :]

        # unscale back
        ssh_roi_keep[:, :, 0:4] /= img_scale
        ssh_roi_single = ssh_roi_keep[0].cpu().numpy()
        nms_keep = nms(ssh_roi_single, cfg.TEST.RPN_NMS_THRESH)
        cls_dets_single = ssh_roi_single[nms_keep, :]

        return cls_dets_single


def coner_check(x1, x2, w, scale=1.05):
    x1 = max(x1 - (scale/2-0.5)*w, 0)
    x1 = min(x1, w-2)

    x2 = min(x2 + (scale/2-0.5)*w, w-1)
    x2 = max(x2, x1+1)
    return int(x1), int(x2)


def demo(netF, netG):
    cap = cv2.VideoCapture(0)

    cap.set(3, 1280)
    cap.set(4, 720)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while(cap.isOpened()):
        ret, img = cap.read() 
        
        if ret == False:
            break

        h, w, _ = img.shape
        im_G = np.copy(img)

        box_info = detect(img, netF)
        boxes, prob = box_info[:,:4], box_info[:,-1]
        boxes = boxes[prob > 0.41]

        G_input = torch.zeros(boxes.shape[0], 3, 256, 256)
        for i, (box) in enumerate(boxes):
            x1, x2 = coner_check(int(box[0]), int(box[2]), w)
            y1, y2 = coner_check(int(box[1]), int(box[3]), h)

            crop = cv2.resize(img[y1:y2, x1:x2, :], (256,256))
            img_i = np.copy(crop[:,:,::-1]) / 127.5 - 1
            G_input[i] = torch.from_numpy(img_i).permute(2,0,1)

        if len(G_input):
            with torch.no_grad():
                out_G = netG(G_input.float().cuda())

            for i, (box) in enumerate(boxes):
                x1, x2 = coner_check(int(box[0]), int(box[2]), w)
                y1, y2 = coner_check(int(box[1]), int(box[3]), h)
                
                crop_G = (out_G[i].detach().cpu().float().permute(1,2,0).numpy() + 1) * 127.5
                im_crop_G = cv2.resize(crop_G[:,:,::-1].astype(np.uint8), (x2-x1, y2-y1))
                im_G[y1:y2, x1:x2, :] = im_crop_G
        else:
            im_G = np.copy(img)

        # concatenate images
        new_im = np.concatenate((img, im_G), axis=1)
        new_im = cv2.resize(new_im, dsize=(width*2, height), interpolation=cv2.INTER_AREA)

        # print text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(new_im, 'Before', (10, 80), font, 3, (0,255,0), 2, cv2.LINE_AA)
        cv2.putText(new_im, 'After', (width+10, 80), font, 3, (0,255,0), 2, cv2.LINE_AA)

        window_name = 'Before and After Modification'
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        cv2.imshow(window_name, new_im)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()



if __name__ == '__main__':
    # Face detector F
    netF = SSH()

    saved_model_path = '../data/pretrained_model/check_point.zip'
    check_point = load_check_point(saved_model_path)
    netF.load_state_dict(check_point['model_state_dict'])
    netF.eval().cuda()
    print('==> Face detector loaded')

    # Initialize modifier G
    G_name = 'resnet_9blocks'
    netG = networks.define_G(input_nc=3, output_nc=3, ngf=64, which_model_netG=G_name, norm='instance')
    netG = networks.load_networks(netG, '../data/pretrained_model/model_1.pth')
    netG.cuda()
    print('==> netG loaded')

    # demo cam start
    demo(netF, netG)
