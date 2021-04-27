import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')

# Currently, memory growth needs to be the same across GPUs
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
import zipfile

from helper import wasserstein_gen_loss_fn, wasserstein_disc_loss_fn
from helper import CustomReduceLROnPlateau
import os
from glob import glob
from models import unet, discriminator, DEGAN
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ReduceLROnPlateau

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

INPUT_SHAPE = (256, 256, 1)


def read_imgs(train_path, clean_image_path):
    ori_img = tf.io.read_file(train_path)
    ori_img = tf.image.decode_png(ori_img, channels=1)

    clean_img = tf.io.read_file(clean_image_path)
    clean_img = tf.image.decode_png(clean_img, channels=1)

    ori_img = (tf.cast(ori_img, tf.float64) / 255.) # * 2. - 1.
    clean_img = (tf.cast(clean_img, tf.float64) / 255.) # * 2. - 1.

    return ori_img, clean_img


rg = tf.random.Generator.from_non_deterministic_state()


def aug(ori_img, clean_img):
    shape = ori_img.shape

    img = tf.concat([ori_img, clean_img], -1)
    img = tf.image.random_crop(img, [*INPUT_SHAPE[:2], img.shape[-1]])
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)

    ori_img, clean_img = img[:, :, :shape[-1]], img[:, :, shape[-1]:]

    ori_img = tf.image.random_brightness(ori_img, 0.08)
    ori_img = tf.clip_by_value(ori_img, 0.0, 1.0)

    return ori_img, clean_img


def train_unet(input_shape, final_channels):
    _model = unet(input_shape, final_channels,
                  use_pooling=False,
                  skip_layers='inception',
                  final_activation='sigmoid')
    _model.load_weights('degan/generator.h5')

    lr = 2e-3
    _model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='mse',
        metrics='mae'
    )

    tensorboard = TensorBoard(profile_batch='10, 20')

    model_path = 'unet/unet.h5'

    model_checkpoints = ModelCheckpoint(model_path,
                                        monitor='mae',
                                        save_best_only=True,
                                        verbose=1)

    lr_sch = ReduceLROnPlateau(monitor='mae',
                               factor=0.5,
                               verbose=1)

    callbacks = [lr_sch, tensorboard, model_checkpoints]
    return _model, callbacks


def train_wgan(input_shape, final_channels):
    generator = unet(input_shape, final_channels,
                     use_pooling=False,
                     skip_layers='inception',
                     final_activation='tanh')
    disc_model = discriminator(INPUT_SHAPE, final_activation='linear')

    # generator.load_weights('degan/generator.h5')
    # discriminator.load_weights('degan/discriminator.h5')

    _model = DEGAN(generator, disc_model)

    gen_decay_rate = 5e-4
    disc_decay_rate = 1e-4
    _model.compile(tf.keras.optimizers.Adam(learning_rate=gen_decay_rate),
                   tf.keras.optimizers.RMSprop(learning_rate=disc_decay_rate),
                   wasserstein_gen_loss_fn,
                   wasserstein_disc_loss_fn)

    gen_lr = CustomReduceLROnPlateau(
        _model.gen_optimizer,
        'gen_lr',
        monitor='generator_loss',
        patience=200,
        factor=RATE_DECAY,
        verbose=1)

    disc_lr = CustomReduceLROnPlateau(
        _model.disc_optimizer,
        'disc_lr',
        monitor='discriminator_loss',
        patience=200,
        factor=RATE_DECAY,
        verbose=1)

    model_path = 'degan/model_name.h5'

    model_checkpoints = ModelCheckpoint(model_path,
                                        monitor='generator_mae',
                                        save_best_only=True,
                                        verbose=1)

    tensorboard = TensorBoard(profile_batch='10, 20')

    callbacks = [gen_lr, disc_lr, tensorboard, model_checkpoints]
    return _model, callbacks


if __name__ == '__main__':
    train_img = sorted(glob(os.path.join(wd, 'train', '*.png')))
    train_cleaned_img = sorted(glob(os.path.join(wd, 'train_cleaned', '*.png')))
    test_img = sorted(glob(os.path.join(wd, 'test', '*.png')))

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    BATCH_SIZE = 8
    EPOCHS = 3000

    RATE_DECAY = 0.8

    train_df = tf.data.Dataset.zip((tf.data.Dataset.list_files(train_img, shuffle=False),
                                    tf.data.Dataset.list_files(train_cleaned_img, shuffle=False)))

    train_df = (train_df.
                shuffle(len(train_img)).
                repeat(EPOCHS).
                map(read_imgs, num_parallel_calls=AUTOTUNE).
                map(aug, num_parallel_calls=AUTOTUNE).
                batch(BATCH_SIZE, drop_remainder=True).
                prefetch(1))

    models, callbacks = train_unet(INPUT_SHAPE, 1)

    models.summary()

    models.fit(train_df,
               shuffle=False,
               verbose=1,
               steps_per_epoch=len(train_img) // BATCH_SIZE,
               epochs=EPOCHS,
               callbacks=callbacks)
