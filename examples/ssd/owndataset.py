import csv
import numpy as np
import chainer

from chainercv.utils import read_image

from os import listdir
from os.path import isfile, join


class OwnDataset(chainer.dataset.DatasetMixin):

    def __init__(self, csv_path, img_path):
        self._imgs = [join(img_path, f) for f in listdir(img_path) if
                      isfile(join(img_path, f))]
        self._csv_files = [join(csv_path, f) for f in listdir(csv_path) if
                           isfile(join(csv_path, f))]

    def __len__(self):
        return len(self._imgs)

    def get_example(self, i):
        with open(self._csv_files[i], 'r') as bbox_file:
            bbox = []
            csvreader = csv.reader(bbox_file, delimiter=';')
            for row in csvreader:
                bbox.append(row)
        label = np.stack(['0' for _ in bbox]).astype(np.int32)
        bbox = np.stack(bbox).astype(np.float32)
        img = read_image(self._imgs[i], color=True)
        return img, bbox, label
