
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Concatenate

def cnn(input_dim, output_channels):
    inputs = Input(shape=input_dim)

    def down_conv(_in, channels):
        x = Conv2D(channels, 3, activation='elu', kernel_regularizer='l2', padding='same')(_in)
        x = BatchNormalization()(x)

        x1 = Conv2D(channels, 3, activation='elu', strides=2, kernel_regularizer='l2', padding='same')(x)
        x1 = BatchNormalization()(x1)

        x2 = Conv2D(channels, 3, activation='elu', strides=1, dilation_rate=2, kernel_regularizer='l2', padding='same')(x)
        x2 = BatchNormalization()(x2)

        x2 = MaxPooling2D((2, 2))(x2)
        x = Concatenate()([x1, x2])
        return x

    def up_conv(_in, channels):
        x = Conv2DTranspose(channels, 3, strides=(2, 2), activation='relu', kernel_regularizer='l2', padding='same')(_in)

        x = Conv2D(channels, 3, activation='relu', kernel_regularizer='l2', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(channels, 3, activation='relu', kernel_regularizer='l2', padding='same')(x)

        x = BatchNormalization()(x)

        return x
    channels = [128, 256]
    x = inputs
    for ch in channels:
        x = down_conv(x, ch)

    for ch in channels[::-1]:
        x = up_conv(x, ch)

    x = Conv2D(output_channels, 3, activation='sigmoid', padding='same')(x)
    _model = Model(inputs=inputs, outputs=x)
    return _model

def unet(input_dim, output_channels):
    inputs = Input(shape=input_dim)

    def conv_down(x, channels):
        x = Conv2D(channels, 3, activation='relu', padding='same')(x)
        x = Conv2D(channels, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)

        d = MaxPooling2D((2, 2))(x)
        return d, x

    def conv_up(x1, x2, channels):
        # x1 = UpSampling2D((2, 2))(x1)

        x1 = Conv2DTranspose(channels, 3, strides=(2, 2), activation='relu', padding='same')(x1)
        x1 = BatchNormalization()(x1)

        x = Concatenate()([x1, x2])
        x = Conv2D(channels, 3, activation='relu', padding='same')(x)
        x = Conv2D(channels, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)

        return x

    x = inputs
    channels = [64, 128, 256, 512]

    layers = []
    for ch in channels:
        x, cnv_ly = conv_down(x, ch)
        layers.append(cnv_ly)

    x = Conv2D(1024, 3, activation='relu', padding='same')(x)
    x = Conv2D(1024, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    for ch, ly in zip(channels[::-1], layers[::-1]):
        x = conv_up(x, ly, ch)

    x = Conv2D(output_channels, 1, activation='sigmoid', padding='same')(x)
    _model = Model(inputs=inputs, outputs=x)
    return _model

if __name__ == '__main__':
    input_dim = (256, 256, 3)
    out = 1

    m1 = cnn(input_dim, out)
    m2 = unet(input_dim, out)

    m1.summary()
    print('===============================')
    m2.summary()