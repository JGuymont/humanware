import glob
from PIL import Image
import numpy as np
import os
import time
from multiprocessing import Manager
import multiprocessing
import pickle as pkl

import multiprocessing

def append_img(args):
    print(args[0])
    meta = args[2]
    min_left = min(meta['left'])
    max_left = max(meta['left']) + max(meta['width'])
    min_top = min(meta['top'])
    max_top = max(meta['top']) + max(meta['height'])
    image = Image.open(args[1])
    image = image.crop((
        (1 - 0.3) * min_left,
        (1 - 0.3) * min_top,
        (1 + 0.3) * max_left,
        (1 + 0.3) * max_top))
    image = image.resize((64,64))
    return np.array(image)

if __name__ == '__main__':
    metadata = pkl.load(open("./data/SVHN/train_metadata.pkl", "rb"))
    imgs_paths = [path for path in glob.glob("./data/SVHN/train/*.png")]
    indexes = [int(path[path.rfind("\\") + 1:path.rfind(".")]) for path in imgs_paths]
    metas = [metadata[index - 1]["metadata"] for index in indexes]
    args = zip(indexes, imgs_paths, metas)

    p = multiprocessing.Pool()
    imgs = [p.map(append_img, args)]
    imgs = np.array(imgs).squeeze()
    print(imgs.mean(axis=(0,1,2)) / 255.0)
    print(imgs.std(axis=(0,1,2)) / 255.0)