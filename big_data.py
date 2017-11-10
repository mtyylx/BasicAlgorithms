import os
import numpy as np
import cv2

def read_chunk(path, size):
    f = open(path, 'r')
    while True:
        chunk = f.read(size)
        if not chunk:
            break
        else:
            yield chunk
    f.close()


def get_batch(path, batch_size):
    files = os.listdir(path)
    imgs = np.zeros((batch_size, 299, 299, 3))
    while True:
        start = 0
        for i in range(batch_size):
            filename = files[start + i]
            imgs[start + i] = cv2.imread(path + filename)
        yield imgs
        start += batch_size



