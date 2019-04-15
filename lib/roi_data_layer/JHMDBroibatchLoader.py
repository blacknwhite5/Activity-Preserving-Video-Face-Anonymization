"""The data layer used during training to train a Fast R-CNN network.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image
import torch
import torch.utils.data as data

from model.utils.config import cfg
from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
from roi_data_layer.minibatch import get_minibatch, get_minibatch
from roi_data_layer.matlab_cp2tform import get_similarity_transform_for_cv2

import os
import numpy as np
import random
import time
import pdb

class roibatchLoader(data.Dataset):
  def __init__(self, roidb, ratio_list, ratio_index, batch_size, num_classes, training=True, normalize=None):
    self._roidb = roidb
    self._num_classes = num_classes
    # we make the height of image consistent to trim_height, trim_width
    self.trim_height = cfg.TRAIN.TRIM_HEIGHT
    self.trim_width = cfg.TRAIN.TRIM_WIDTH
    self.max_num_box = cfg.MAX_NUM_GT_BOXES
    self.training = training
    self.normalize = normalize
    self.ratio_list = ratio_list
    self.ratio_index = ratio_index
    self.batch_size = batch_size
    self.data_size = len(self.ratio_list)

    # given the ratio_list, we want to make the ratio same for each batch.
    self.ratio_list_batch = torch.Tensor(self.data_size).zero_()
    num_batch = int(np.ceil(len(ratio_index) / batch_size))
    for i in range(num_batch):
        left_idx = i*batch_size
        right_idx = min((i+1)*batch_size-1, self.data_size-1)

        if ratio_list[right_idx] < 1:
            # for ratio < 1, we preserve the leftmost in each batch.
            target_ratio = ratio_list[left_idx]
        elif ratio_list[left_idx] > 1:
            # for ratio > 1, we preserve the rightmost in each batch.
            target_ratio = ratio_list[right_idx]
        else:
            # for ratio cross 1, we make it to be 1.
            target_ratio = 1

        self.ratio_list_batch[left_idx:(right_idx+1)] = target_ratio


  def __getitem__(self, index):
    if self.training:
        index_ratio = int(self.ratio_index[index])
    else:
        index_ratio = index

    # get the anchor index for current sample index
    # here we set the anchor index to the last one
    # sample in this group
    minibatch_db = [self._roidb[index_ratio]]
    blobs = get_minibatch(minibatch_db, self._num_classes)
    data = torch.from_numpy(blobs['data'])
    im_info = torch.from_numpy(blobs['im_info'])
    # we need to random shuffle the bounding box.
    data_height, data_width = data.size(1), data.size(2)

    if self.training:
        np.random.shuffle(blobs['gt_boxes'])
        gt_boxes = torch.from_numpy(blobs['gt_boxes'])

        ########################################################
        # padding the input image to fixed size for each group #
        ########################################################

        # NOTE1: need to cope with the case where a group cover both conditions. (done)
        # NOTE2: need to consider the situation for the tail samples. (no worry)
        # NOTE3: need to implement a parallel data loader. (no worry)
        # get the index range

        # if the image need to crop, crop to the target size.
        ratio = self.ratio_list_batch[index]

        if self._roidb[index_ratio]['need_crop']:
            if ratio < 1:
                # this means that data_width << data_height, we need to crop the
                # data_height
                min_y = int(torch.min(gt_boxes[:,1]))
                max_y = int(torch.max(gt_boxes[:,3]))
                trim_size = int(np.floor(data_width / ratio))
                box_region = max_y - min_y + 1
                if min_y == 0:
                    y_s = 0
                else:
                    if (box_region-trim_size) < 0:
                        y_s_min = max(max_y-trim_size, 0)
                        y_s_max = min(min_y, data_height-trim_size)
                        if y_s_min == y_s_max:
                            y_s = y_s_min
                        else:
                            y_s = np.random.choice(range(y_s_min, y_s_max))
                    else:
                        y_s_add = int((box_region-trim_size)/2)
                        if y_s_add == 0:
                            y_s = min_y
                        else:
                            y_s = np.random.choice(range(min_y, min_y+y_s_add))
                # crop the image
                data = data[:, y_s:(y_s + trim_size), :, :]

                # shift y coordiante of gt_boxes
                gt_boxes[:, 1] = gt_boxes[:, 1] - float(y_s)
                gt_boxes[:, 3] = gt_boxes[:, 3] - float(y_s)

                # update gt bounding box according the trip
                gt_boxes[:, 1].clamp_(0, trim_size - 1)
                gt_boxes[:, 3].clamp_(0, trim_size - 1)

            else:
                # this means that data_width >> data_height, we need to crop the
                # data_width
                min_x = int(torch.min(gt_boxes[:,0]))
                max_x = int(torch.max(gt_boxes[:,2]))
                trim_size = int(np.ceil(data_height * ratio))
                box_region = max_x - min_x + 1
                if min_x == 0:
                    x_s = 0
                else:
                    if (box_region-trim_size) < 0:
                        x_s_min = max(max_x-trim_size, 0)
                        x_s_max = min(min_x, data_width-trim_size)
                        if x_s_min == x_s_max:
                            x_s = x_s_min
                        else:
                            x_s = np.random.choice(range(x_s_min, x_s_max))
                    else:
                        x_s_add = int((box_region-trim_size)/2)
                        if x_s_add == 0:
                            x_s = min_x
                        else:
                            x_s = np.random.choice(range(min_x, min_x+x_s_add))
                # crop the image
                data = data[:, :, x_s:(x_s + trim_size), :]

                # shift x coordiante of gt_boxes
                gt_boxes[:, 0] = gt_boxes[:, 0] - float(x_s)
                gt_boxes[:, 2] = gt_boxes[:, 2] - float(x_s)
                # update gt bounding box according the trip
                gt_boxes[:, 0].clamp_(0, trim_size - 1)
                gt_boxes[:, 2].clamp_(0, trim_size - 1)

        # based on the ratio, padding the image.
        if ratio < 1:
            # this means that data_width < data_height
            trim_size = int(np.floor(data_width / ratio))
            padding_data = torch.FloatTensor(int(np.ceil(data_width / ratio)), \
                                             data_width, 3).zero_()

            padding_data[:data_height, :, :] = data[0]
            # update im_info
            im_info[0, 0] = padding_data.size(0)
            # print("height %d %d \n" %(index, anchor_idx))
        elif ratio > 1:
            # this means that data_width > data_height
            # if the image need to crop.
            padding_data = torch.FloatTensor(data_height, \
                                             int(np.ceil(data_height * ratio)), 3).zero_()
            padding_data[:, :data_width, :] = data[0]
            im_info[0, 1] = padding_data.size(1)
        else:
            trim_size = min(data_height, data_width)
            padding_data = torch.FloatTensor(trim_size, trim_size, 3).zero_()
            padding_data = data[0][:trim_size, :trim_size, :]
            gt_boxes.clamp_(0, trim_size)
            im_info[0, 0] = trim_size
            im_info[0, 1] = trim_size

        # check the bounding box:
        not_keep = (gt_boxes[:,0] == gt_boxes[:,2]) | (gt_boxes[:,1] == gt_boxes[:,3])
        keep = torch.nonzero(not_keep == 0).view(-1)

        gt_boxes_padding = torch.FloatTensor(self.max_num_box, gt_boxes.size(1)).zero_()
        if keep.numel() != 0:
            gt_boxes = gt_boxes[keep]
            num_boxes = min(gt_boxes.size(0), self.max_num_box)
            gt_boxes_padding[:num_boxes,:] = gt_boxes[:num_boxes]
        else:
            num_boxes = 0

            # permute trim_data to adapt to downstream processing
        padding_data = padding_data.permute(2, 0, 1).contiguous()
        im_info = im_info.view(3)

        # NEW
        img_path = self._roidb[index_ratio]['image']
        _temp_img = Image.open(img_path)
        w, h = _temp_img.size
        act, vid, img_name = img_path.strip('.png').split('/')[-3:]
        _name = vid.split('_')[-1] + '_' + img_name
        keys_folder = '/home/SSD3/jason-data/JHMDB'
        keys_path = os.path.join(keys_folder, 'face_bbox/all', act, vid, _name+'.npz')
        keys = np.load(keys_path)
        face_thres = 0.6
        temp_keys = keys['gt']
        good_face = temp_keys[temp_keys[:,4]>face_thres]
        #  pdb.set_trace()
        if len(good_face) > 0:
            rand_idx = np.random.randint(len(good_face))
            face_bbox = torch.from_numpy(good_face[rand_idx, :]).double()
            #  print(face_bbox)
            #  print(w,h)
            #  print(data_width, data_height)
            #  adjust the bbox according to new ratio
            w_ratio = float(w) / data_width
            h_ratio = float(h) / data_height
            #  print(ratio, w_ratio, h_ratio)
            face_bbox[0] /= w_ratio
            face_bbox[1] /= h_ratio
            face_bbox[2] /= w_ratio
            face_bbox[3] /= h_ratio
            #  print(face_bbox[0:4])

            # enlarge the bbox 2.2x
            _x1 = face_bbox[0]
            _y1 = face_bbox[1]
            _x2 = face_bbox[2]
            _y2 = face_bbox[3]
            face_bbox[0] = max(0, 1.6*_x1 - 0.6*_x2)
            face_bbox[1] = max(0, 1.6*_y1 - 0.6*_y2)
            face_bbox[2] = min(data_width-1,  1.6*_x2 - 0.6*_x1)
            face_bbox[3] = min(data_height-1, 1.6*_y2 - 0.6*_y1)
            #  print(face_bbox[0:4])
            # flipping
            if self._roidb[index_ratio]['flipped']:
                oldx1 = face_bbox[0]
                oldx2 = face_bbox[2]
                face_bbox[0] = data_width - oldx2 - 1
                face_bbox[2] = data_width - oldx1 - 1
                assert face_bbox[0] < face_bbox[2]
            face_flag = 1
        else:
            face_bbox = torch.DoubleTensor(5).zero_()
            face_flag = 0

        tfm_tensor = torch.DoubleTensor(2, 3).zero_()
        return padding_data, im_info, gt_boxes_padding, num_boxes, face_bbox, tfm_tensor, face_flag

    else:
        data = data.permute(0, 3, 1, 2).contiguous().view(3, data_height, data_width)
        im_info = im_info.view(3)
        gt_boxes = torch.FloatTensor([1,1,1,1,1])
        num_boxes = 0
        return data, im_info, gt_boxes, num_boxes

  def __len__(self):
    return len(self._roidb)
