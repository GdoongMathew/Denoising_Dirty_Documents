import tensorflow as tf
import zipfile
import os
from glob import glob
import cv2
from models import cnn
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint

# path to zipped & working directories
path_zip = r'E:\Data\denoising-dirty-documents'

wd = os.path.join(path_zip, 'working')

# with zipfile.ZipFile(os.path.join(path_zip, 'train.zip'), 'r') as zip_ref:
#     zip_ref.extractall(wd)
#
# with zipfile.ZipFile(os.path.join(path_zip, 'test.zip'), 'r') as zip_ref:
#     zip_ref.extractall(wd)
#
# with zipfile.ZipFile(os.path.join(path_zip, 'train_cleaned.zip'), 'r') as zip_ref:
#     zip_ref.extractall(wd)
#
# with zipfile.ZipFile(os.path.join(path_zip, 'sampleSubmission.csv.zip'), 'r') as zip_ref:
#     zip_ref.extractall(wd)

# For later use, we will store image names into list, so we can draw them simply.

# store image names in list for later use
train_img = sorted(glob(os.path.join(wd, 'train', '*.png')))
train_cleaned_img = sorted(glob(os.path.join(wd, 'train_cleaned', '*.png')))
test_img = sorted(glob(os.path.join(wd, 'test', '*.png')))


def read_imgs(train_path, clean_image_path):
    ori_img = tf.io.read_file(train_path)
    ori_img = tf.image.decode_png(ori_img, channels=1)

    clean_img = tf.io.read_file(clean_image_path)
    clean_img = tf.image.decode_png(clean_img, channels=1)

    ori_img = tf.cast(ori_img, tf.float64) / 255.
    clean_img = tf.cast(clean_img, tf.float64) / 255.

    return ori_img, clean_img

INPUT_SHAPE = (256, 256, 1)

rg = tf.random.Generator.from_non_deterministic_state()

def aug(ori_img, clean_img):
    shape = ori_img.shape

    img = tf.concat([ori_img, clean_img], -1)

    img = tf.cond(
        tf.less(rg.uniform([1], maxval=1), 0.5),
        lambda: tf.image.random_crop(img, [*INPUT_SHAPE[:2], img.shape[-1]]),
        lambda: tf.cast(tf.image.resize(img, INPUT_SHAPE[:2]), tf.float64)
    )

    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)

    ori_img, clean_img = img[:, :, :shape[-1]], img[:, :, shape[-1]:]

    return ori_img, clean_img


AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 4
EPOCHS = 500
LEARNING_RATE = 2e-3
RATE_DECAY = 0.8
model_path = 'cnn/cnn_{epoch:02d}.h5'

train_df = tf.data.Dataset.zip((tf.data.Dataset.list_files(train_img, shuffle=False),
                                tf.data.Dataset.list_files(train_cleaned_img, shuffle=False)))

train_df = (train_df.
            shuffle(len(train_img)).
            repeat(EPOCHS).
            map(read_imgs, num_parallel_calls=AUTOTUNE).
            map(aug, num_parallel_calls=AUTOTUNE).
            batch(BATCH_SIZE, drop_remainder=True).
            prefetch(1))

model = cnn(INPUT_SHAPE, 1)
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='mse'
)

lr_sch = ReduceLROnPlateau(monitor='loss',
                           factor=RATE_DECAY,
                           verbose=1)
model_checkpoints = ModelCheckpoint(model_path,
                                    monitor='loss',
                                    verbose=1)

model.fit(train_df,
          shuffle=False,
          verbose=1,
          steps_per_epoch=len(train_img) // BATCH_SIZE,
          epochs=EPOCHS,
          callbacks=[lr_sch, model_checkpoints])