import numpy as np
import torch
import torch.utils.data as data
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os, random, glob, cv2

class DALY2DEMO(data.Dataset):
    def __init__(self, data_dir='data', transform=None):
        super(DALY2DEMO, self).__init__()
        # load split file
        self.data_dir = data_dir
        img_path = os.path.join(data_dir, 'demo_video/test_frames')
        self.indexlist = glob.glob(img_path + '/*/*.jpeg')
        self.transform = transform

    def load_img(self, index):
        info = self.indexlist[index]
        vid, im_name = info.split('/')[-2:]

        img_path = info
        img = Image.open(img_path).convert('RGB')
        w, h = img.size

        annot_path = os.path.join(self.data_dir, 'demo_video/face_bbox', vid, im_name.strip('.jpeg')+'.npz')
        annot = np.load(annot_path)

        face_bbox = torch.DoubleTensor(50,5).zero_()
        if len(annot['gt']) > 0:
            #  print('len:', len(annot['gt']))
            for i in range(len(annot['gt'])):
                #  print('i:' , i)
                _bbox = annot['gt'][i]
                # enlarge the bbox 2.2x
                x1 = _bbox[0]
                y1 = _bbox[1]
                x2 = _bbox[2]
                y2 = _bbox[3]
                _bbox[0] = max(0, 1.3*x1 - 0.3*x2)
                _bbox[1] = max(0, 1.3*y1 - 0.3*y2)
                _bbox[2] = min(w-1, 1.3*x2 - 0.3*x1)
                _bbox[3] = min(h-1, 1.3*y2 - 0.3*y1)
                face_bbox[i] = torch.DoubleTensor(_bbox)
            face_count = len(annot['gt'])
        else:
            face_count = 0

        return img, face_bbox, face_count

    def __getitem__(self, index):
        _img, face_bbox, face_count = self.load_img(index)

        if self.transform:
            img = self.transform(_img)
        #  print(img.size(), face_bbox.size(), face_count)
        return img, face_bbox, face_count, index

    def __len__(self):
        return len(self.indexlist)

def main():
    import torch
    from torchvision import transforms, utils
    from torch.utils.data import DataLoader
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    g_data = DALY2DEMO(data_dir='/home/SSD5/jason-data/Privacy/dataset/DALY',
            transform=transforms.Compose([
                        transforms.Resize((600, 800)),
                        transforms.ToTensor(), normalize]))
    print(len(g_data))
    dataloader = DataLoader(g_data, batch_size=4, shuffle=False, num_workers=1)
    for i, data in enumerate(dataloader):
        im, bbox, flag,idx = data
        print(im.size(), bbox.size(), flag.size())
        utils.save_image(im, str(i) + '.jpg', normalize=True)

if __name__ == "__main__":
    main()
