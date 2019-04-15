from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
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
import pickle
from PIL import Image
from .voc_eval import voc_ap

class JHMDB(imdb):
    def __init__(self, image_set):
        imdb.__init__(self, image_set)
        self.jhmdb_path = '/home/SSD3/jason-data/JHMDB'
        # JHMDB specific config options
        self.config = {'cleanup' : False}
        OPERATION, PHASE = image_set.split('_')[-2:]
        #  pdb.set_trace()
        #  temp_name = "_".join(image_set.split('_')[:-2])
        if PHASE == 'train':
            self._image_set = self.jhmdb_path + '/listfiles/' + image_set.strip('_train') + '.trainlist'
            # you need a full path for image list and data path
        else:
            self._image_set = self.jhmdb_path + '/listfiles/' + image_set.strip('_test') + '.testlist'

        self._annot_path = self.jhmdb_path + "/puppet_mask" # you only have annotations in RGB data folder
        self._SPLIT = int(image_set.split('_')[4])

        if 'RGB' in image_set and 'FLOW' in image_set: # for 2stream fusion
            raise NotImplementedError
            #  self._data_path = '/home/lear/xpeng/data/JHMDB/flows_color'
        else:
            self._MOD = image_set.split('_')[1]
            self._LEN = image_set.split('_')[2]
            self._data_path = None
            if self._MOD=='RGB':
                self._data_path = self.jhmdb_path + '/images_' + OPERATION
            if self._MOD=='FLOW': raise NotImplementedError #self._data_path = './data/JHMDB/flows_color'

        self._classes = ('__background__',
                         'brush_hair', 'catch', 'clap', 'climb_stairs', 'golf',
                         'jump', 'kick_ball', 'pick', 'pour', 'pullup', 'push',
                         'run', 'shoot_ball', 'shoot_bow', 'shoot_gun', 'sit',
                         'stand', 'swing_baseball', 'throw', 'walk', 'wave') # 22

        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_index = self._load_image_set_index()
        self.test_videos = self.get_test_videos(self._SPLIT)
        # Default to roidb handler
        self._roidb_handler = self.gt_roidb

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

    def prepare_traintest(self): # generating train/test file list according to self.videos
        pass

    def get_train_videos(self, split):
        """
        train / test splits
        """
        assert split<3
        tr_videos = [os.path.join(label,l.split()[0][:-4]) for label in self._classes[1:]
                                   for l in file(self.jhmdb_path + "/listfiles/JHMDB_splits/%s_test_split%d.txt"%(label,split+1))
                                   if l.split()[1][0]=="1"]
        return tr_videos

    def get_test_videos(self, split):
        assert split<3
        ts_videos = [os.path.join(label,l.split()[0][:-4]) for label in self._classes[1:]
                                 for l in file(self.jhmdb_path + "/listfiles/JHMDB_splits/%s_test_split%d.txt"%(label,split+1))
                                 if l.split()[1][0]=="2"]
        return ts_videos

    def _get_puppet_mask_file(self, videoname):
        """
        Annotation: warning few images do not have annotations
        """
        return os.path.join(self._annot_path, videoname, "puppet_mask.mat")

    def get_annot_image_mask(self, videoname, n):
        assert os.path.exists(self._get_puppet_mask_file(videoname))
        m = sio.loadmat(self._get_puppet_mask_file(videoname))["part_mask"]
        if n-1 < m.shape[2]:
            return m[:,:,n-1]>0
        else:
            return m[:,:,-1]>0

    def get_annot_image_boxes(self, videoname, n):
        #  pdb.set_trace()
        mask = self.get_annot_image_mask(videoname, n)
        m = self.mask_to_bbox(mask)
        if m is None:
            pdb.set_trace()
            m = np.zeros((0,4), dtype=np.float32)
        if m.shape[0]>1:
            pdb.set_trace()
        return m

    def mask_to_bbox(self,mask):
         # you are aware that only 1 box for each frame
        return np.array(Image.fromarray(mask.astype(np.uint8)).getbbox(), dtype=np.float32).reshape(1,4)-np.array([0,0,1,1])

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        temp_name = "_".join(self._image_set.split('_')[:-1]) + '.' +  self._image_set.split('.')[1]
        if not os.path.exists(temp_name):
            print('Path does not exist: {}'.format(temp_name))
            raise NotADirectoryError
            #  print('Preparing {}'.format(temp_name))
            #  self.prepare_traintest()
        with open(temp_name) as f:
            image_index = [x.strip() for x in f.readlines()]
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

        roidb = [self._load_JHMDB_annotation(index) for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            pickle.dump(roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))
        return roidb

    def _load_JHMDB_annotation(self, index):
        """
        Load image and bounding boxes info
        """
        index = index.split(',')[-1] # to support 2 stream filelist input
        videoname = os.path.dirname(index)
        print(videoname)
        #  pdb.set_trace()
        frm = int(index.split('/')[-1].split('.')[0])

        num_objs = 1 # num_objs is num humans, for JHMDB only one instance in a frame

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        boxes[0,:] = self.get_annot_image_boxes(videoname, frm)
        cls = self._class_to_ind[videoname.split('/')[0]]
        gt_classes[0] = cls
        overlaps[0, cls] = 1.0
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
            # only one object in JHMDB
            mask = sio.loadmat(self._get_puppet_mask_file(v))["part_mask"]
            tube = np.empty((mask.shape[2], 5), dtype=np.int32)
            for i in range(mask.shape[2]):
                box = self.mask_to_bbox(mask[:,:, i])
                tube[i, 0] = i + 1
                tube[i, 1:] = box
            tubes.append(tube)
            res[v] = {'tubes': tubes, 'gt_classes': self._class_to_ind[v.split('/')[0]]}
        return res

    def _get_JHMDB_results_file_template(self, output_dir):
        tem = self._image_set.split('/')[-1].split('.')[0]
        filename = 'detections_' + tem + '_{:s}.txt'
        path = os.path.join(output_dir, filename)
        return path

    def _write_voc_results_file(self, all_boxes, output_dir):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('Writing "{}" JHMDB results file'.format(cls))
            filename = self._get_JHMDB_results_file_template(output_dir).format(cls)
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

    def JHMDB_eval(self, detpath, gt_roidb, image_index, classindex,
                 ovthresh=0.5, use_07_metric=False):
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
        if npos == 0:
            # No ground truth examples
            return 0,0,0

        # read dets
        with open(detpath, 'r') as f:
            lines = f.readlines()
        if len(lines) == 0:
            # No detection examples
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
        # We re-use parts of the pascal voc python code for JHMDB
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
            filename = self._get_JHMDB_results_file_template(output_dir).format(cls)
            rec, prec, ap = self.JHMDB_eval(filename, gt_roidb, self.image_index, i, ovthresh=0.5,
                                            use_07_metric=use_07_metric)
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
                filename = self._get_JHMDB_results_file_template(output_dir).format(cls)
                os.remove(filename)

if __name__ == '__main__':
    d = JHMDB('JHMDB_RGB_1_split_1', 'TRAIN')
    res = d.roidb
    import cv2
    from model.utils.net_utils import vis_detections
    for i in range(5):
        ri = np.random.randint(d.num_images)
        im = cv2.imread(d.image_path_at(ri))
        gt_cls = d.classes[res[ri]['gt_classes'][0]]
        gt_bbox = res[ri]['boxes']
        im2show = vis_detections(im, gt_cls, gt_bbox, 0.5)
        cv2.imwrite(str(i)+'.jpg', im2show)
    #  from IPython import embed; embed()
