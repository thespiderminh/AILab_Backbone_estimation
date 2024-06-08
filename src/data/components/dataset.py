from matplotlib import pyplot as plt
import numpy as np
import cv2
import scipy.io
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Dataset, random_split



def data_extractor(path):
    key_points_path = path + "/joints.mat"
    key_points = scipy.io.loadmat(key_points_path)['joints'][:,:-1,:].transpose(2, 0, 1)
    images_path = []

    for i in range(10000):
        images_path.append(path + "/images/im" + "{:05}".format(i + 1) + ".jpg")

    images_path = np.array(images_path)
    key_points = np.array(key_points)
    return images_path, key_points

class Customed_Dataset(Dataset):
    def __init__(self, transform=None):
        path = "data/lsp-master"

        self.images_path, self.key_points = data_extractor(path=path)
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):
        img_path = self.images_path[index]
        keypoints = self.key_points[index]

        image = cv2.imread(img_path)

        if self.transform:
            transformed = self.transform(image=image, keypoints=keypoints)
            image, keypoints = transformed["image"], transformed["keypoints"]

        # plt.imshow(image)
        # for _, (x, y, z) in enumerate(keypoints):
        #     if z == 1:
        #         plt.plot(x, y, "r+")
        # plt.show()

        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        keypoints = torch.Tensor(keypoints).flatten()

        return image, keypoints
    
def main():
    train_dataset = Customed_Dataset(transform=None)
    data_loader = DataLoader(dataset=train_dataset)
    for batch in data_loader:
        image, keypoints = batch
        print(image.shape, keypoints.shape)
        break


if __name__ == "__main__":
    main()
