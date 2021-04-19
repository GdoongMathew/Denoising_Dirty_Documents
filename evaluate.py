
import tensorflow as tf
import os
from glob import glob
from models import unet
import pandas as pd
import numpy as np
import cv2

gpus = tf.config.experimental.list_physical_devices('GPU')

# Currently, memory growth needs to be the same across GPUs
for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


# path to zipped & working directories
path_zip = r'E:\Data\denoising-dirty-documents'

wd = os.path.join(path_zip, 'working')
test_img = sorted(glob(os.path.join(wd, 'test', '*.png')))

INPUT_SHAPE = (256, 256, 1)

model = unet(INPUT_SHAPE, 1)
model.load_weights('unet/unet.h5')


def crop_img(img, crop_shape):
    img_shape = img.shape
    imgs = []
    points = []
    for y in range(0, img_shape[0], crop_shape[0]):
        for x in range(0, img_shape[1], crop_shape[1]):
            if y + crop_shape[0] >= img_shape[0]:
                y = img_shape[0] - crop_shape[0] - 1
            if x + crop_shape[1] >= img_shape[1]:
                x = img_shape[1] - crop_shape[1] - 1
            _crop_img = img[y: y + crop_shape[1], x: x + crop_shape[0]]
            imgs.append(_crop_img)
            points.append([y, x])
    return imgs, points


def concat_img(imgs, points, ori_shape):
    blk_img = np.zeros(ori_shape, dtype=np.float64)
    for img, pt in zip(imgs, points):
        img_shape = img.shape
        # s = blk_img[pt[0]: img_shape[0] + pt[0], pt[1]: img_shape[1] + pt[1]]
        blk_img[pt[0]: img_shape[0] + pt[0], pt[1]: img_shape[1] + pt[1]] = img

    return blk_img

ids = []
vals = []
for img_path in test_img:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    ori_shape = img.shape
    imgs, points = crop_img(img, INPUT_SHAPE[:2])
    imgs = np.asarray(imgs)
    batch_imgs = imgs[:, :, :, np.newaxis]
    rest = model.predict_on_batch(batch_imgs.astype('float') / 255.)

    rest = np.reshape(rest, rest.shape[:-1])
    cnt_img = concat_img(rest, points, ori_shape)

    img_shape = cnt_img.shape
    _id = os.path.splitext(os.path.basename(img_path))[0]
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            ids.append(f'{_id}_{i + 1}_{j + 1}')
            vals.append(cnt_img[i, j])


#     rest = model.predict_on_batch(test)
#     for p, r, s in zip(path, rest, ori_shape):
#         x = tuple(s.numpy())
#         r = cv2.resize(r, (x[1], x[0]))
#         # r = cv2.resize(r.astype('uint8'), tuple(s.numpy()[:2]))
#         img_shape = r.shape
#         _id = os.path.splitext(os.path.basename(p.numpy().decode('utf-8')))[0]
#         for i in range(img_shape[0]):
#             for j in range(img_shape[1]):
#                 ids.append(f'{_id}_{i + 1}_{j + 1}')
#                 vals.append(r[i, j])
#
pd.DataFrame({'id': ids, 'value': vals}).to_csv('submission.csv', index=False)
# import matplotlib.pyplot as plt
#
# plt.figure(figsize=(15, 25))
# for i in range(0, 8, 2):
#     plt.subplot(4, 2, i + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(test[i][:, :, 0], cmap="gray")
#     plt.title('Noise image')
#
#     plt.subplot(4, 2, i + 2)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(rest[i][:, :, 0], cmap="gray")
#     plt.title('Denoised image')
#
# plt.show()