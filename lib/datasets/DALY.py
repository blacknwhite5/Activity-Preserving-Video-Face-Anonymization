from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Jason, pengxj, Ross Girshick
# --------------------------------------------------------
import os, sys
import datasets
from datasets.imdb import imdb
import numpy as np
import scipy.sparse
import scipy.io as sio
import pdb
from PIL import Image
import glob
import pickle

from datasets.voc_eval import voc_ap

class DALY(imdb):
    def __init__(self, image_set):
        imdb.__init__(self, image_set)
        # self.DALY_path = '/home/SSD3/jason-data/DALY'
        self.DALY_path = '/workspace/ppad/dataset/DALY'
        self.config = {'cleanup' : False} # DALY specific config options
        OPERATION, PHASE = image_set.split('_')[-2:]
        #  pdb.set_trace()
        self._image_set = self.DALY_path + '/debug_img.txt' if PHASE == 'train' else self.DALY_path + '/test_img.txt'

        gt_name = os.path.join(self.DALY_path, 'daly1.1.0.pkl')
        with open(gt_name, 'rb') as f_gt:
            self.daly_annot = pickle.load(f_gt, encoding='latin1')
        if 'RGB' in image_set and 'FLOW' in image_set: # for 2stream fusion
            raise NotImplementedError
        else:
            self._MOD = image_set.split('_')[1]
            self._LEN = image_set.split('_')[2]
            self._data_path = None
            if self._MOD=='RGB':
                if PHASE == 'train':
                    self._data_path = self.DALY_path + '/images_w_act_' + OPERATION
                elif PHASE == 'test':
                    self._data_path = self.DALY_path + '/images_test_1_11_14023/'
            if self._MOD=='FLOW': raise NotImplementedError #self._data_path = './data/DALY/flows_color'

        self._classes = ('__background__',
                        'ApplyingMakeUpOnLips', 'BrushingTeeth', 'CleaningFloor',
                        'CleaningWindows', 'Drinking', 'FoldingTextile', 'Ironing',
                        'Phoning', 'PlayingHarmonica', 'TakingPhotosOrVideos') # 11

        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._image_index = self._load_image_set_index()
        self.train_videos = self.get_train_videos()
        self.test_videos = self.get_test_videos()
        self._roidb_handler = self.gt_roidb

    def _load_gt(self):
        with open(self.DALY_path + '/daly1.1.0.pkl', 'rb') as f:
            gt = pickle.load(f)
        return gt
    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return os.path.join(self._data_path, self._image_index[i])

    def image_id_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return i

    def prepare_traintest(self):
        """
        Generating train/test file list according to self.videos
        """
        pass

    def get_train_videos(self, split=1):
        """
        train / test splits
        """
        ts_videos = self.get_test_videos()
        all_videos = glob.glob(os.path.join(self.DALY_path, 'videos/*.mp4'))
        tr_videos = []
        for v in all_videos:
            v_name = v.split('/')[-1]
            if v_name not in ts_videos:
                tr_videos.append(v_name)
        return tr_videos

    def get_test_videos(self, split=1):
        ts_videos = self.daly_annot['splits'][0]
        return ts_videos

    def get_annot_image_boxes(self, videoname, n):
        #  pdb.set_trace()
        mask = self.get_annot_image_mask(videoname, n)
        m = self.mask_to_bbox(mask)
        if m is None:
            pdb.set_trace()
            m = np.zeros((0,4), dtype=np.float32)
        if m.shape[0]>1:
            pdb.set_trace()

        self.daly_annot['annot'][vid]['annot'][act_id][0]['keyframes'][0]['boundingBox']

        return m

    def _get_video_resolution(self, videoname):
        """
        Get original resolution of video frames
        Different video has diff. resolution
        """
        vids = glob.glob(os.path.join(self._data_path, videoname + '/*.jpg'))
        if len(vids)< 1:
            print(self._data_path, videoname)
            raise ValueError('should exist at least 1 frm')
        else:
            img = Image.open(vids[0])
        return img.size

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        if not os.path.exists(self._image_set):
            print('Path does not exist: {}'.format(self._image_set))
            raise NotADirectoryError
            #  print('Preparing {}'.format(self._image_set))
            #  self.prepare_traintest()
        with open(self._image_set) as f:
            image_index = [x.strip() for x in f.readlines()]
        #  pdb.set_trace()
        return image_index

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb
        roidb = [self._load_DALY_annotation(index) for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            pickle.dump(roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))
        return roidb

    def _load_DALY_annotation(self, index):
        """
        Load image and bounding boxes info
        """
        #  index = index.split(',')[-1] # to support 2 stream filelist input
        videoname, imagename = index.split('/')
        timestamp, clip, act, fid = imagename.strip('.jpg').split('_')[1:]
        print(videoname, imagename, timestamp, clip, act, fid)
        width, height = self._get_video_resolution(videoname)

        bbox_data = self.daly_annot['annot'][videoname]['annot'][act][int(clip)]['keyframes'][int(fid)]['boundingBox']
        num_objs = len(bbox_data)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        for i in range(num_objs):
            x1 = np.max((0, int(bbox_data[i][0]*width)))
            y1 = np.max((0, int(bbox_data[i][1]*height)))
            x2 = np.min((int(bbox_data[i][2]*width), width))
            y2 = np.min((int(bbox_data[i][3]*height), height))
            if x2 >= x1 and y2 >= y1:
                boxes[i:] = [x1, y1, x2, y2]
            cls = self._class_to_ind[act]
            gt_classes[i] = cls
            overlaps[i, cls] = 1.0
        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False}

    def get_test_video_annotations(self):
        assert self._phase == 'TEST'
        res = {}
        for v in self.test_videos:
            assert not v in res
            res[v] = {}
            tubes = []
            # only one object in DALY
            mask = sio.loadmat(self._get_puppet_mask_file(v))["part_mask"]
            tube = np.empty((mask.shape[2], 5), dtype=np.int32)
            for i in range(mask.shape[2]):
                box = self.mask_to_bbox(mask[:,:, i])
                tube[i, 0] = i + 1
                tube[i, 1:] = box
            tubes.append(tube)
            res[v] = {'tubes': tubes, 'gt_classes': self._class_to_ind[v.split('/')[0]]}
        return res

    def _get_DALY_results_file_template(self, output_dir):
        tem = self._image_set.split('/')[-1].split('.')[0]
        filename = 'detections_' + tem + '_{:s}.txt'
        path = os.path.join(output_dir, filename)
        return path

    def _write_voc_results_file(self, all_boxes, output_dir):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('Writing "{}" DALY results file'.format(cls))
            filename = self._get_DALY_results_file_template(output_dir).format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(str(index), dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def DALY_eval(self, detpath, gt_roidb, image_index, classindex, ovthresh=0.5, use_07_metric=False):
        """
        Top level function that does the JHMDB evaluation.
        detpath: Path to detections
        gt_roidb: List of ground truth structs.
        image_index: List of image ids.
        classindex: Category index
        [ovthresh]: Overlap threshold (default = 0.5)
        [use_07_metric]: Whether to use VOC07's 11 point AP computation (default False)
        """
        # extract gt objects for this class
        class_recs = {}
        npos = 0
        for item,imagename in zip(gt_roidb,image_index):
            bbox = item['boxes'][np.where(item['gt_classes'] == classindex)[0], :]
            difficult = np.zeros((bbox.shape[0],)).astype(np.bool)
            det = [False] * bbox.shape[0]
            npos = npos + sum(~difficult)
            class_recs[str(imagename)] = {'bbox': bbox, 'difficult': difficult, 'det': det}
        if npos == 0: # No ground truth examples
            return 0,0,0

        # read dets
        with open(detpath, 'r') as f:
            lines = f.readlines()
        if len(lines) == 0: # No detection examples
            return 0,0,0

        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = -np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)

            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                # union
                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                       (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                       (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)

        return rec, prec, ap

    def _do_python_eval(self, output_dir):
        # We re-use parts of the pascal voc python code for DALY
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = False
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        # Load ground truth
        gt_roidb = self.gt_roidb()
        classes = self._classes
        for i, cls in enumerate(classes):
            if cls == '__background__' or cls == '__no_attribute__':
                continue
            filename = self._get_DALY_results_file_template(output_dir).format(cls)
            rec, prec, ap = self.DALY_eval(filename, gt_roidb, self.image_index, i,
                                        ovthresh=0.5, use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('~~~~~~~~')
        print('')


    def evaluate_detections(self, all_boxes, output_dir):
        self._write_voc_results_file(all_boxes, output_dir)
        self._do_python_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_DALY_results_file_template(output_dir).format(cls)
                os.remove(filename)

if __name__ == '__main__':
    d = DALY('DALY_RGB_1_split_1_raw_train')
    pdb.set_trace()
    res = d.roidb
    import cv2
    from model.utils.net_utils import vis_detections
    for i in range(50):
        ri = np.random.randint(d.num_images)
        im = cv2.imread(d.image_path_at(ri))
        gt_cls = d.classes[res[ri]['gt_classes'][0]]
        gt_bbox = res[ri]['boxes']
        im2show = vis_detections(im, gt_cls, gt_bbox, 0.5)
        cv2.imwrite(str(i)+'.jpg', im2show)
    #  from IPython import embed; embed()
