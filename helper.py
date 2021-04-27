import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.platform import tf_logging as logging
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.constraints import Constraint

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
mse = tf.keras.losses.MeanSquaredError()


def gen_loss_fn(gen_img, real_img, gen_disc, factor=50.0):
    disc_loss = cross_entropy(tf.ones_like(gen_disc), gen_disc)
    mse_loss = mse(real_img, gen_img)
    return mse_loss * factor + disc_loss


def disc_loss_fn(real_output, gen_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(gen_output), gen_output)
    return real_loss + fake_loss


def wasserstein_disc_loss_fn(real_output, gen_output):
    return tf.reduce_mean(gen_output) - tf.reduce_mean(real_output)


def wasserstein_gen_loss_fn(gen_img, real_img, gen_disc, factor=10.0):
    disc_loss = -tf.reduce_mean(gen_disc)
    mse_loss = mse((real_img + 1.) / 2., (gen_img + 1.) / 2.)
    return factor * mse_loss + disc_loss


class CustomReduceLROnPlateau(ReduceLROnPlateau):
    def __init__(self, optimizer, log_name, **kwargs):
        super(CustomReduceLROnPlateau, self).__init__(**kwargs)
        self.optimizer = optimizer
        self.log_name = log_name

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs[self.log_name] = K.get_value(self.optimizer.lr)
        current = logs.get(self.monitor)
        if current is None:
            logging.warning('Reduce LR on plateau conditioned on metric `%s` '
                            'which is not available. Available metrics are: %s',
                            self.monitor, ','.join(list(logs.keys())))

        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                self.wait += 1
                if self.wait >= self.patience:
                    old_lr = float(K.get_value(self.optimizer.lr))
                    if old_lr > self.min_lr:
                        new_lr = old_lr * self.factor
                        new_lr = max(new_lr, self.min_lr)
                        K.set_value(self.optimizer.lr, new_lr)
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s reducing learning '
                                  'rate to %s.' % (epoch + 1, self.log_name, new_lr))
                        self.cooldown_counter = self.cooldown
                        self.wait = 0


class ClipConstraint(Constraint):
    def __init__(self, clip_value=1e-2):
        super(ClipConstraint, self).__init__()
        self.clip_value = clip_value

    def __call__(self, w):
        w = tf.clip_by_value(w, -self.clip_value, self.clip_value)
        return w

    def get_config(self):
        return {'clip_value': self.clip_value}