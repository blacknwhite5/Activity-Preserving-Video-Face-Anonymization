import numpy as np
import torch
import torch.utils.data as data
from PIL import Image, ImageFile
import os, random, pdb, glob, cv2

ImageFile.LOAD_TRUNCATED_IMAGES = True

class CASIA(data.Dataset):
    def __init__(self, data_dir='data', transform=None):
        super(CASIA, self).__init__()

        # make labels
        img_folder_name = os.listdir(data_dir)
        img_folder_name.sort()

        self.labels = {}
        for index, label in enumerate(img_folder_name):
            self.labels[label] = index

        # make images path
        images = glob.glob(data_dir+'/**/*.jpg', recursive=True)
        
        # make indexlist
        self.indexlist = []
        for image in images:
            self.indexlist.append(image + ' ' + str(self.labels[image.split('/')[-2]]))

        random.shuffle(self.indexlist)

        # transform
        self.transform = transform

    def sample_negative(self, a_cls):
        while True:
            rand = random.randint(0, len(self.indexlist)-1)
            name, cls = self.indexlist[rand].split()[0:2]
            if cls != a_cls:
                break
        return rand

    def load_img(self, index):
        info = self.indexlist[index].split()
        cls = int(info[1])

        img = Image.open(info[0]).convert('RGB')
        return img, cls

    def __getitem__(self, index):
        # Get the index of each image in the triplet
        a_name, a_cls = self.indexlist[index].split()
        n_index = self.sample_negative(a_cls)

        _a, a_cls = self.load_img(index)
        _n, n_cls = self.load_img(n_index)

        # transform images if required
        if self.transform:
            img_a = self.transform(_a)
            img_n = self.transform(_n)

        return img_a, img_n, a_cls, n_cls

    def __len__(self):
        return len(self.indexlist)


def main():
    import torch
    from torchvision import transforms, utils
    from torch.utils.data import DataLoader
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    g_data = CASIA('../../data/CASIA/CASIA-maxpy-clean/',
            transform=transforms.Compose([
                        transforms.Resize((250, 250)),
                        transforms.ToTensor(), normalize]))
    dataloader = DataLoader(g_data, batch_size=8, shuffle=False, num_workers=1)
    for i, data in enumerate(dataloader):
        a, n = data[0:2]
        ak, nk = data[2:4]
        # print(i, len(g_data), a.size(), n.size(), ak.size(), nk.size())
        print(i, ak, nk)
        # utils.save_image(torch.cat((a, n), dim=0), str(i) + '.jpg', normalize=True)

if __name__ == "__main__":
    main()
