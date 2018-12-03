import os

import numpy
from PIL import Image
import six

import numpy as np

from io import BytesIO
import os
import pickle
import json
import numpy as np
import glob

import skimage.io as io

from chainer.dataset import dataset_mixin

class NTURGBDDataset(dataset_mixin.DatasetMixin):

    subject_ids = dict()
    subject_ids["train"] = [
        1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
    subject_ids["test"] = [
        3, 6, 7, 10, 11, 12, 20, 21, 22, 23, 24, 26, 29, 30, 32, 33, 36, 37, 39, 40]
    view_ids = dict()
    view_ids["train"] = [1]
    view_ids["test"] = [2, 3]

    def __init__(
        self, subset="train", dataDir='./data/nturgbd', split_criterion="view",
        n_samples=-1
    ):
        print("load dataset start")
        print("\tdataDir: %s"%dataDir)
        print("\tsubset:", subset)
        print("\tsplit criterion:", split_criterion)
        
        rgb_paths = []
        edge_paths = []
        joint_paths = []
        if split_criterion == "subject":
            for i in self.subject_ids[subset]:
                for rgb_p in glob.glob(os.path.join(dataDir, "S{0:03d}C*/rgb_*.jpg".format(i))):
                    edge_p = rgb_p.replace("rgb_", "edge_")
                    if os.path.exists(edge_p):
                        rgb_paths.append(rgb_p)
                        edge_paths.append(edge_p)
        elif split_criterion == "view":
            for i in self.view_ids[subset]:
                for rgb_p in glob.glob(os.path.join(dataDir, "*C{0:03d}P*/rgb_*.jpg".format(i))):
                    edge_p = rgb_p.replace("rgb_", "edge_")
                    if os.path.exists(edge_p):
                        rgb_paths.append(rgb_p)
                        edge_paths.append(edge_p)
        else:
            pass

        print("\t# of image pairs:", len(rgb_paths))
        if n_samples > 0:
            indices = np.random.permutation(len(rgb_paths))[:n_samples]
            self.rgb_paths = [rgb_paths[i] for i in indices]
            self.edge_paths = [edge_paths[i] for i in indices]
        else:
            self.rgb_paths = rgb_paths
            self.edge_paths = edge_paths
    
    def __len__(self):
        return len(self.rgb_paths)

    # return (img1, img2)
    def get_example(self, i, crop_width=256):
        return load_paired_images(self.edge_paths[i], self.rgb_paths[i], crop_width=crop_width)
    


def load_paired_images(path1, path2, crop_width=256):
    im1 = np.asarray(Image.open(path1), dtype="f").transpose(2, 0, 1) / 128.0 - 1.0
    im2 = np.asarray(Image.open(path2), dtype="f").transpose(2, 0, 1) / 128.0 - 1.0
    assert im1.shape == im2.shape, "Mismatch shape: {} vs {}".format(im1.shape, im2.shape)

    _, h, w = im1.shape
    assert h >= crop_width and w >= crop_width, "Image shape is smaller than crop_width"

    x_l = np.random.randint(0,w-crop_width) if w > crop_width else 0
    x_r = x_l+crop_width if w > crop_width else crop_width
    y_l = np.random.randint(0,h-crop_width) if h > crop_width else 0
    y_r = y_l+crop_width if h > crop_width else crop_width
    return im1[:,y_l:y_r,x_l:x_r], im2[:,y_l:y_r,x_l:x_r]

