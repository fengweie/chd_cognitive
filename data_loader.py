from PIL import Image
from torch.utils.data import Dataset

def default_loader(img_path, path):
    try:
        img = Image.open(img_path+path)
        return img.convert('RGB')
    except:
        print("Cannot read image: {}".format(path))

def default_loader_jpeg(img_path, path):
    try:
        img = Image.open(img_path+path+'.jpeg')
        return img.convert('RGB')
    except:
        print("Cannot read image: {}".format(path))

class customData(Dataset):
    def __init__(self, img_name, img_label, img_path, dataset = '', data_transforms=None, loader = 'jpg'):
        self.img_name = img_name
        self.img_label = img_label
        self.data_transforms = data_transforms
        self.dataset = dataset
        if loader == 'jpeg':
            self.loader = default_loader_jpeg
        else:
            self.loader = default_loader
        self.path = img_path

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, item):
        img_name = self.img_name[item]
        label = self.img_label[item]
        img = self.loader(self.path, img_name)

        if self.data_transforms is not None:
            try:
                img = self.data_transforms[self.dataset](img)
            except Exception as e:
                print("Cannot transform image: {}".format(img_name),e)
        return img_name, img, label  