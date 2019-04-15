import numpy as np
import torch
import torch.utils.data as data
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os, random, glob, cv2

class LFW2G(data.Dataset):
    def __init__(self, data_dir='data', transform=None):
        super(LFW2G, self).__init__()
        self.indexlist = glob.glob(os.path.join(data_dir, 'images/*/*.jpg'))
        # load images
        self.data_dir = data_dir
        self.img_dir = 'images'
        self.transform = transform

    def load_img(self, index):
        info = self.indexlist[index]
        img_path = info
        img = Image.open(img_path).convert('RGB')
        return img

    def __getitem__(self, index):
        _img = self.load_img(index)
        if self.transform:
            img = self.transform(_img)
        return img, index

    def __len__(self):
        return len(self.indexlist)

def main():
    import torch
    from torchvision import transforms, utils
    from torch.utils.data import DataLoader
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    g_data = LFW2G(data_dir='/home/SSD5/jason-data/Privacy/dataset/LFW',
            transform=transforms.Compose([
                        transforms.Resize((256, 256)),
                        transforms.ToTensor(), normalize]))
    print(len(g_data))
    dataloader = DataLoader(g_data, batch_size=4, shuffle=False, num_workers=1)
    for i, data in enumerate(dataloader):
        im,_ = data
        print(im.size())
        utils.save_image(im, str(i) + '.jpg', normalize=True)

if __name__ == "__main__":
    main()
