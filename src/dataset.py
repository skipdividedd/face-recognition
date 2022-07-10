from collections import defaultdict
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
import secrets
import os

os.getcwd()
path4 = os.path.abspath('../') # ... / указывает предыдущий каталог каталога, в котором находится текущий файл

def data(md_name):
    def loader(ds):
        transform = transforms.Compose([
            transforms.Resize(160),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        aug = transforms.Compose([
            transforms.CenterCrop((100, 100)),
            transforms.RandomCrop((80, 80)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=(-90, 90)),
        ])

        train_data = ds('train', transform, aug)
        val_data = ds('val', transform)
        test_data = ds('test', transform)

        batch_size = 64
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, test_loader

    def get_paths(dataset_type='train'):
        '''
        a function that returnes list of images paths for a given type of the dataset
        params:
          dataset_type: one of 'train', 'val', 'test'
        '''

        labels_dict = {
            'train': 0,
            'val': 1,
            'test': 2,
        }

        f = open(path4 + '\data\celebA_train\celebA_train_split.txt', 'r')
        lines = f.readlines()
        f.close()

        lines = [x.strip().split() for x in lines]
        lines = [x[0] for x in lines if int(x[1]) == labels_dict[dataset_type]]

        images_paths = []
        for filename in lines:
            images_paths.append(os.path.join(path4 + '\data\celebA_train\celebA_imgs', filename))
        return np.array(images_paths)

    if md_name == 'resnet' or md_name == 'ArcFace':
        class celebADataset(Dataset):
            def __init__(self, dataset_type, transform, aug=None):
                self.images = get_paths(dataset_type=dataset_type)
                f = open(path4+'\data\celebA_train\celebA_anno.txt', 'r')
                labels = f.readlines()
                f.close()
                labels = [x.strip().split() for x in labels]
                labels = {x : y for x, y in labels}
                self.labels = [labels[x.split('\\')[-1]] for x in self.images]
                self.transform = transform
                self.aug = aug

            def __len__(self):
                return len(self.images)

            def __getitem__(self, idx):
                img_name = self.images[idx]
                label = self.labels[idx]
                image = Image.open(img_name)
                label = torch.tensor(int(label))
                if self.aug:
                    sample = self.aug(
                        image)
                image = self.transform(image)
                sample = (image, label)
                return sample

        return loader(celebADataset)

    if md_name == 'Triplet':

        f = open(path4+'\data\celebA_train\celebA_anno.txt', 'r')
        class_lines = f.readlines()
        f.close()
        class_lines = [x.strip().split() for x in class_lines]
        class_img_names = [x[0] for x in class_lines]
        # dictionary with info of which images belong to which class
        # format: {class: [image_1, image_2, ...]}
        class_dict = defaultdict(list)
        for img_name, img_class in class_lines:
            class_dict[img_class].append(img_name)

        class celebADatasetTriplet(Dataset):
            def __init__(self, dataset_type, transform, aug=None, ):
                self.images = get_paths(dataset_type=dataset_type)
                f = open(path4+'\data\celebA_train\celebA_anno.txt', 'r')
                labels = f.readlines()
                f.close()
                labels = [x.strip().split() for x in labels]
                labels = {x: y for x, y in labels}
                self.labels = [labels[x.split('\\')[-1]] for x in self.images]
                self.transform = transform
                self.aug = aug

            def __len__(self):
                return len(self.images)

            def __getitem__(self, idx):
                img_name = self.images[idx]
                label = self.labels[idx]
                image = Image.open(img_name)
                label = int(label)

                if self.aug:
                    sample = self.aug(
                        image)

                image = self.transform(image)
                positive_img_name = secrets.choice(class_dict[str(label)])
                positive_image = Image.open(path4+'\data\celebA_train\celebA_imgs\\' + positive_img_name)
                positive_image = self.transform(positive_image)

                if int(label) < 495:
                    negative_img_name = secrets.choice(class_dict[str(label + 1)])
                else:
                    negative_img_name = secrets.choice(class_dict[str(label - 1)])
                negative_image = Image.open(path4+'\data\celebA_train\celebA_imgs\\' + negative_img_name)
                negative_image = self.transform(negative_image)
                sample = (image, positive_image, negative_image, float(label))

                return sample

        return loader(celebADatasetTriplet)


if __name__ == "__main__":
    print("Hello, World!")