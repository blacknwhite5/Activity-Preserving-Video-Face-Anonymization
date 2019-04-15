import numpy as np
import torch
import torch.utils.data as data
from PIL import Image, ImageFile
import os, random, pdb, glob, cv2

from roi_data_layer.matlab_cp2tform import get_similarity_transform_for_cv2

ImageFile.LOAD_TRUNCATED_IMAGES = True

class CelebA_CLS(data.Dataset):
    def __init__(self, data_dir='data', text_file='annotation/train.txt', transform=None):
        super(CelebA_CLS, self).__init__()

        # load split file
        text_path = os.path.join(data_dir, text_file)
        self.indexlist = [line.rstrip('\n') for line in open(text_path,'r')][1:]

        # load bbox
        bbox_path = os.path.join(data_dir, 'annotation/list_bbox_celeba.txt')
        self.bboxes = [line.rstrip('\n') for line in open(bbox_path,'r')][1:]

        # load key-points
        keys_path = os.path.join(data_dir, 'annotation/list_landmarks_celeba.txt')
        self.keys = [line.rstrip('\n') for line in open(keys_path,'r')][1:]

        # load images
        self.data_dir = data_dir
        self.img_dir = 'img-wild/train'
        print('using images from %s' % self.img_dir)

        self.transform = transform

    def alignment(self, src_pts):
        ref_pts = [ [30.2946, 51.6963],[65.5318, 51.5014],
                    [48.0252, 71.7366],[33.5493, 92.3655],[62.7299, 92.2041] ]
        src_pts = np.array(src_pts).reshape(5,2)
        s = np.array(src_pts).astype(np.float32)
        r = np.array(ref_pts).astype(np.float32)
        # center normalize for spatial transformer
        r[:, 0] = r[:, 0] / 48. - 1
        r[:, 1] = r[:, 1] / 56. - 1
        tfm = get_similarity_transform_for_cv2(r, s)
        return tfm

    def sample_negative(self, a_cls):
        while True:
            rand = random.randint(0, len(self.indexlist)-1)
            name, cls = self.indexlist[rand].split()
            if cls != a_cls:
                break
        return name, cls

    def sample_positive(self, name, cls):
        cls_path = os.path.join(self.data_dir, self.img_dir, cls)
        pics = glob.glob(cls_path + '/*.jpg')
        if len(pics) == 1:
            return name
        else:
            while True:
                rand = random.randint(0, len(pics)-1)
                p_name = pics[rand].split('/')[-1]
                if p_name != name:
                    return p_name

    def load_img(self, name, cls):
        # load key-points
        key_points = self.keys[int(name.strip('.jpg'))].split()
        assert key_points[0] == name
        landmark = [float(k) for k in key_points[1:]]

        # load bbox
        box = self.bboxes[int(name.strip('.jpg'))].split()
        assert box[0] == name
        bbox = [float(k) for k in box[1:]]

        # load and align images
        img_path = os.path.join(self.data_dir, self.img_dir, cls, name)
        img = cv2.imread(img_path)
        x1 = max(bbox[1]-0.3*bbox[3], 0)
        y1 = max(bbox[0]-0.3*bbox[2], 0)
        x2 = min(bbox[1]+1.3*bbox[3], img.shape[0])
        y2 = min(bbox[0]+1.3*bbox[2], img.shape[1])
        # center normalize for STN
        center_x = (x1 + x2) / 2.
        center_y = (y1 + y2) / 2.
        for p in range(5):
            landmark[2*p] = 2.*(landmark[2*p] - center_y) / (y2-y1)
            landmark[2*p+1] = 2.*(landmark[2*p+1] - center_x) / (x2-x1)

        # crop the image
        img_new = img[int(x1):int(x2), int(y1):int(y2), :]
        # cv2: BGR --> PIL.Image: RGB
        img_a = Image.fromarray(np.uint8(cv2.cvtColor(img_new,cv2.COLOR_BGR2RGB)), 'RGB')

        return img_a, landmark

    def __getitem__(self, index):
        # Get the index of each image in the triplet
        a_name, a_cls = self.indexlist[index].split()
        n_name, n_cls = self.sample_negative(a_cls)

        _a, a_keys = self.load_img(a_name, a_cls)
        _n, n_keys = self.load_img(n_name, n_cls)

        # transform images if required
        if self.transform:
            img_a = self.transform(_a)
            img_n = self.transform(_n)

        tfm_a = self.alignment(a_keys)
        tfm_n = self.alignment(n_keys)

        return img_a, img_n, tfm_a, tfm_n, int(a_cls)-1, int(n_cls)-1

    def __len__(self):
        return len(self.indexlist)


class TripletCelebA(data.Dataset):
    def __init__(self, data_dir='data', text_file='annotation/train.txt', transform=None):
        super(TripletCelebA, self).__init__()

        # load split file
        text_path = os.path.join(data_dir, text_file)
        self.indexlist = [line.rstrip('\n') for line in open(text_path,'r')][1:]

        # load bbox
        bbox_path = os.path.join(data_dir, 'annotation/list_bbox_celeba.txt')
        self.bboxes = [line.rstrip('\n') for line in open(bbox_path,'r')][1:]

        # load key-points
        keys_path = os.path.join(data_dir, 'annotation/list_landmarks_celeba.txt')
        self.keys = [line.rstrip('\n') for line in open(keys_path,'r')][1:]

        # load images
        self.data_dir = data_dir
        self.img_dir = 'img-wild/train'
        print('using images from %s' % self.img_dir)

        self.transform = transform

    def sample_negative(self, a_cls):
        while True:
            rand = random.randint(0, len(self.indexlist)-1)
            name, cls = self.indexlist[rand].split()
            if cls != a_cls:
                break
        return name, cls

    def sample_positive(self, name, cls):
        cls_path = os.path.join(self.data_dir, self.img_dir, cls)
        pics = glob.glob(cls_path + '/*.jpg')
        if len(pics) == 1:
            return name
        else:
            while True:
                rand = random.randint(0, len(pics)-1)
                p_name = pics[rand].split('/')[-1]
                if p_name != name:
                    return p_name

    def load_img(self, name, cls):
        # load key-points
        key_points = self.keys[int(name.strip('.jpg'))].split()
        assert key_points[0] == name
        landmark = [float(k) for k in key_points[1:]]

        # load bbox
        box = self.bboxes[int(name.strip('.jpg'))].split()
        assert box[0] == name
        bbox = [float(k) for k in box[1:]]

        # load and align images
        img_path = os.path.join(self.data_dir, self.img_dir, cls, name)
        img = cv2.imread(img_path)
        x1 = max(bbox[1]-0.3*bbox[3], 0)
        y1 = max(bbox[0]-0.3*bbox[2], 0)
        x2 = min(bbox[1]+1.3*bbox[3], img.shape[0])
        y2 = min(bbox[0]+1.3*bbox[2], img.shape[1])
        # center normalize for STN
        center_x = (x1 + x2) / 2.
        center_y = (y1 + y2) / 2.
        for p in range(5):
            landmark[2*p] = 2.*(landmark[2*p] - center_y) / (y2-y1)
            landmark[2*p+1] = 2.*(landmark[2*p+1] - center_x) / (x2-x1)
        img_new = img[int(x1):int(x2), int(y1):int(y2), :]
        # cv2: BGR --> PIL.Image: RGB
        img_a = Image.fromarray(np.uint8(cv2.cvtColor(img_new,cv2.COLOR_BGR2RGB)), 'RGB')

        return img_a, landmark

    def __getitem__(self, index):
        # Get the index of each image in the triplet
        a_name, cls = self.indexlist[index].split()
        p_name = self.sample_positive(a_name, cls)
        n_name, n_cls = self.sample_negative(cls)

        _a, a_keys = self.load_img(a_name, cls)
        _p, p_keys = self.load_img(p_name, cls)
        _n, n_keys = self.load_img(n_name, n_cls)

        # transform images if required
        if self.transform:
            img_a = self.transform(_a)
            img_p = self.transform(_p)
            img_n = self.transform(_n)

        keys_a = torch.FloatTensor(a_keys)
        keys_p = torch.FloatTensor(p_keys)
        keys_n = torch.FloatTensor(n_keys)

        return img_a, img_p, img_n, keys_a, keys_p, keys_n

    def __len__(self):
        return len(self.indexlist)


def main():
    import torch
    from torchvision import transforms, utils
    from torch.utils.data import DataLoader
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    g_data = CelebA_CLS(data_dir='/home/SSD2/jason-data/celebA',
            text_file='annotation/train.txt',
            transform=transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(), normalize]))
    dataloader = DataLoader(g_data, batch_size=8, shuffle=False, num_workers=1)
    for i, data in enumerate(dataloader):
        a, n = data[0:2]
        ak, nk = data[2:4]
        a_id, n_id = data[4:6]
        print(i, len(g_data), a.size(), n.size(), ak.size(), nk.size(), a_id.size(), n_id.size())
        #  print(a_id, n_id)
        utils.save_image(torch.cat((a, n), dim=0), str(i) + '.jpg', normalize=True)

    #  for i, data in enumerate(dataloader):
        #  a, p, n = data[0:3]
        #  ak, pk, nk = data[3:6]
        #  print(i, len(g_data), a.size(), p.size(), n.size())
        #  print(ak)
    #      utils.save_image(torch.cat((a, p, n), dim=0), str(i) + '.jpg', normalize=True)

if __name__ == "__main__":
    main()
