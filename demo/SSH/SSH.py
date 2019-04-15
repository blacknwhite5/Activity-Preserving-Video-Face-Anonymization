import os
import numpy as np
from SSH.model.utils.config import cfg
import torch
from SSH.model.SSH import SSH
from SSH.model.network import load_check_point
import cv2

from SSH.model.nms.nms_wrapper import nms
from SSH.model.utils.test_utils import _get_image_blob, _compute_scaling_factor


def ssh(img, thresh=0.5):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    saved_model_path = 'SSH/check_point.zip'
    assert os.path.isfile(saved_model_path), 'Pretrained model not found'

    net = SSH(vgg16_image_net=False)

    if (os.path.isfile(saved_model_path)):
        check_point = load_check_point(saved_model_path)
        net.load_state_dict(check_point['model_state_dict'])

    net.to(device)
    net.eval()

    with torch.no_grad():
        img_scale = _compute_scaling_factor(img.shape, cfg.TEST.SCALES[0], cfg.TEST.MAX_SIZE)
        img_blob = _get_image_blob(img, [img_scale])[0]

        img_info = np.array([[img_blob['data'].shape[2], img_blob['data'].shape[3], img_scale]])
        img_data = img_blob['data']

        img_info = torch.from_numpy(img_info).to(device)
        img_data = torch.from_numpy(img_data).to(device)

        # batch_size = img_data.size()[0]
        ssh_rois = net(img_data, img_info)

        inds = (ssh_rois[:, :, 4] > thresh)
        ssh_roi_keep = ssh_rois[:, inds[0], :]

        # unscale back
        ssh_roi_keep[:, :, 0:4] /= img_scale


        ssh_roi_single = ssh_roi_keep[0].cpu().numpy()
        nms_keep = nms(ssh_roi_single, cfg.TEST.RPN_NMS_THRESH)
        cls_dets_single = ssh_roi_single[nms_keep, :]

        return cls_dets_single 


if __name__ == '__main__':
    img = cv2.imread('00080.png')
    ssh(img)
