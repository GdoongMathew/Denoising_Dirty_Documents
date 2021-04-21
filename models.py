import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Concatenate
from tensorflow.python.keras.engine import data_adapter


def cnn(input_dim, output_channels, name='CNN'):
    inputs = Input(shape=input_dim)

    def down_conv(_in, channels):
        x = Conv2D(channels, 3, kernel_regularizer='l2', padding='same')(_in)
        x = Activation('elu')(x)
        x = BatchNormalization()(x)

        x1 = Conv2D(channels, 3, strides=2, kernel_regularizer='l2', padding='same')(x)
        x1 = Activation('elu')(x1)
        x1 = BatchNormalization()(x1)

        x2 = Conv2D(channels, 3, strides=1, dilation_rate=2, kernel_regularizer='l2', padding='same')(x)
        x2 = Activation('elu')(x2)
        x2 = BatchNormalization()(x2)

        x2 = MaxPooling2D((2, 2))(x2)
        x = Concatenate()([x1, x2])
        return x

    def up_conv(_in, channels):
        x = Conv2DTranspose(channels, 3, strides=(2, 2), kernel_regularizer='l2', padding='same')(_in)
        x = Activation('elu')(x)

        x = Conv2D(channels, 3, kernel_regularizer='l2', padding='same')(x)
        x = Activation('elu')(x)
        x = BatchNormalization()(x)

        x = Conv2D(channels, 3, kernel_regularizer='l2', padding='same')(x)
        x = Activation('elu')(x)
        x = BatchNormalization()(x)

        return x
    channels = [128, 256]
    x = inputs
    for ch in channels:
        x = down_conv(x, ch)

    for ch in channels[::-1]:
        x = up_conv(x, ch)

    x = Conv2D(output_channels, 3, activation='sigmoid', padding='same')(x)
    _model = Model(inputs=inputs, outputs=x, name=name)
    return _model


def unet(input_dim,
         output_channels,
         name='Unet',
         use_pooling=True,
         final_activation='sigmoid'):
    inputs = Input(shape=input_dim)

    def conv_down(_x, channels):
        _x = Conv2D(channels, 3, padding='same')(_x)
        _x = Activation('elu')(_x)
        _x = Conv2D(channels, 3, padding='same')(_x)
        _x = Activation('elu')(_x)
        _x = BatchNormalization()(_x)
        if use_pooling:
            d = AveragePooling2D((2, 2))(_x)
        else:
            d = Conv2D(channels, 3, strides=(2, 2), padding='same')(_x)
            d = Activation('elu')(d)

        return d, _x

    def conv_up(x1, x2, channels):
        # x1 = UpSampling2D((2, 2))(x1)

        x1 = Conv2DTranspose(channels, 3, strides=(2, 2), padding='same')(x1)
        x1 = Activation('elu')(x1)
        x1 = BatchNormalization()(x1)

        x = Concatenate()([x1, x2])
        x = Conv2D(channels, 3, padding='same')(x)
        x = Activation('elu')(x)
        x = Conv2D(channels, 3, padding='same')(x)
        x = Activation('elu')(x)
        x = BatchNormalization()(x)

        return x

    x = inputs
    channels = [64, 128, 256, 512]

    layers = []
    for ch in channels:
        x, cnv_ly = conv_down(x, ch)
        layers.append(cnv_ly)

    x = Conv2D(1024, 3, padding='same')(x)
    x = Activation('elu')(x)
    x = Conv2D(1024, 3, padding='same')(x)
    x = Activation('elu')(x)
    x = BatchNormalization()(x)

    for ch, ly in zip(channels[::-1], layers[::-1]):
        x = conv_up(x, ly, ch)

    x = Conv2D(output_channels, 1, activation=final_activation, padding='same')(x)
    _model = Model(inputs=inputs, outputs=x, name=name)
    return _model


def discriminator(noise_shape,
                  ori_shape,
                  name='Discriminator',
                  final_activation='sigmoid'):
    in_1 = Input(shape=noise_shape)
    in_2 = Input(shape=ori_shape)
    x1 = Conv2D(32, 3, padding='same')(in_1)
    x1 = Activation('elu')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Conv2D(32, 3, strides=2, padding='same')(x1)
    x1 = Activation('elu')(x1)
    x1 = BatchNormalization()(x1)

    x2 = Conv2D(32, 3, padding='same')(in_2)
    x2 = Activation('elu')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Conv2D(32, 3, strides=2, padding='same')(x2)
    x2 = Activation('elu')(x2)
    x2 = BatchNormalization()(x2)

    x = Concatenate()([x1, x2])
    x = Conv2D(128, 5, padding='same')(x)
    x = Activation('elu')(x)
    x = Conv2D(256, 5, strides=(2, 2), padding='same')(x)
    x = Activation('elu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(256, 5, padding='same')(x)
    x = Activation('elu')(x)
    x = Conv2D(256, 5, strides=(2, 2), padding='same')(x)
    x = Activation('elu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(1, 1, activation=final_activation, padding='same')(x)
    _model = Model(inputs=[in_1, in_2], outputs=x, name=name)
    return _model


class DEGAN(Model):
    def __init__(self, _generator, _discriminator):
        super(DEGAN, self).__init__()
        assert isinstance(_generator, Model)
        assert isinstance(_discriminator, Model)
        self._generator = _generator
        self._discriminator = _discriminator


    def compile(self, gen_optimizer, disc_optimizer, gen_loss_fn, disc_loss_fn):
        super(DEGAN, self).compile()

        assert isinstance(gen_optimizer, tf.keras.optimizers.Optimizer)
        assert isinstance(disc_optimizer, tf.keras.optimizers.Optimizer)
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.gen_loss_fn = gen_loss_fn
        self.disc_loss_fn = disc_loss_fn

    def summary(self, line_length=None, positions=None, print_fn=print):
        print_fn('==================================================================================================')
        print_fn('==========================================Generator===============================================')
        print_fn('==================================================================================================')
        self._generator.summary(line_length=line_length, positions=positions, print_fn=print_fn)
        print_fn('==================================================================================================')
        print_fn('==========================================Discriminator===========================================')
        print_fn('==================================================================================================')
        self._discriminator.summary(line_length=line_length, positions=positions, print_fn=print_fn)

    @ tf.function
    def train_step(self, data):
        data = data_adapter.expand_1d(data)
        noise_img, clean_img, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_img = self._generator(noise_img, training=True)
            real_output = self._discriminator([noise_img, clean_img], training=True)
            gen_output = self._discriminator([noise_img, generated_img], training=True)

            disc_loss = self.disc_loss_fn(real_output, gen_output)
            gen_loss = self.gen_loss_fn(generated_img, clean_img, gen_output)

        gen_gradient = gen_tape.gradient(gen_loss, self._generator.trainable_variables)
        disc_gradient = disc_tape.gradient(disc_loss, self._discriminator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(gen_gradient, self._generator.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(disc_gradient, self._discriminator.trainable_variables))
        mae = tf.reduce_mean(tf.keras.metrics.mean_absolute_error(clean_img, generated_img))

        return {'generator_loss': gen_loss,
                'generator_mae': mae,
                'discriminator_loss': disc_loss}

    @ tf.function
    def test_step(self, data):
        data = data_adapter.expand_1d(data)
        noise_img, clean_img, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        generated_img = self._generator(noise_img, training=True)
        real_output = self._discriminator(noise_img, clean_img, training=True)
        gen_output = self._discriminator(noise_img, generated_img, training=True)

        disc_loss = self.disc_loss_fn(real_output, gen_output)
        gen_loss = self.gen_loss_fn(clean_img, generated_img, gen_output)
        mae = tf.reduce_mean(tf.keras.metrics.mean_absolute_error(clean_img, generated_img))

        return {'generator_loss': gen_loss,
                'generator_mae': mae,
                'discriminator_loss': disc_loss}

    def save_weights(self, filepath, **kwargs):
        gen_filepath = filepath.replace('model_name', 'generator')
        disc_filepath = filepath.replace('model_name', 'discriminator')
        self._generator.save_weights(gen_filepath, **kwargs)
        self._discriminator.save_weights(disc_filepath, **kwargs)

    def save(self, filepath, **kwargs):
        gen_filepath = filepath.replace('model_name', 'generator')
        disc_filepath = filepath.replace('model_name', 'discriminator')
        self._generator.save_weights(gen_filepath, **kwargs)
        self._discriminator.save_weights(disc_filepath, **kwargs)


if __name__ == '__main__':
    input_dim = (256, 256, 3)
    out = 1

    m1 = cnn(input_dim, out)
    m2 = unet(input_dim, out)
    d = discriminator(input_dim, input_dim)

    # m1.summary()
    # print('===============================')
    # m2.summary()
    # print('===============================')
    d.summary()