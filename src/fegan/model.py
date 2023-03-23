"""
    Migrated from tf v1 to tf v2: https://www.tensorflow.org/guide/migrate/upgrade

    The gate convolution is made with reference to Deepfillv1 (https://github.com/JiahuiYu/generative_inpainting)

    https://paperswithcode.com/paper/free-form-image-inpainting-with-gated

    https://github.com/avalonstrel/GatedConvolution_pytorch/blob/master/models/sa_gan.py
    https://github.com/csqiangwen/DeepFillv2_Pytorch/blob/master/network.py

    https://github.com/nipponjo/deepfillv2-pytorch (latest)


"""
from functools import lru_cache

import numpy as np
import tensorflow as tf

from config import CONFIG

# https://stackoverflow.com/questions/57932584/equivalent-of-from-tensorflow-contrib-framework-python-ops-import-arg-scope-in-t
# from tensorflow.contrib.framework.python.ops import add_arg_scope


# https://stackoverflow.com/questions/56561734/runtimeerror-tf-placeholder-is-not-compatible-with-eager-execution
tf.compat.v1.disable_eager_execution()


def gate_conv(
        x_in,
        cnum,
        ksize,
        stride=1,
        rate=1,
        name='conv',
        activation='leaky_relu',
        use_lrn=True
):
    """
    :param x_in: input tensor
    :param cnum: output channel number
    :param ksize: kernel size
    :param stride: stride
    :param rate: dilation rate, https://arxiv.org/abs/1511.07122
    :param name: name
    :param activation: activation function
    :param use_lrn: use local response normalization
    :return: output tensor
    """
    x = tf.compat.v1.layers.conv2d(
        x_in,
        cnum,
        ksize,
        stride,
        dilation_rate=rate,
        activation=None,
        padding='same',
        name=name
    )
    if use_lrn:
        x = tf.nn.lrn(x, bias=0.00005)
    if activation == 'leaky_relu':
        x = tf.nn.leaky_relu(x)
    g = tf.compat.v1.layers.conv2d(
        x_in,
        cnum,
        ksize,
        stride,
        dilation_rate=rate,
        activation=tf.nn.sigmoid,
        padding='same',
        name=name + '_g'
    )
    x = tf.multiply(x, g)
    return x, g


def gate_deconv(
        input_,
        output_shape,
        k_h=5,
        k_w=5,
        d_h=2,
        d_w=2,
        stddev=0.02,
        name="deconv"
):
    with tf.compat.v1.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.compat.v1.get_variable(
            'w',
            [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
            initializer=tf.compat.v1.random_normal_initializer(stddev=stddev)
        )

        deconv = tf.nn.conv2d_transpose(
            input_,
            w,
            output_shape=output_shape,
            strides=[1, d_h, d_w, 1]
        )

        biases = tf.compat.v1.get_variable(
            'biases1',
            [output_shape[-1]],
            initializer=tf.compat.v1.constant_initializer(0.0)
        )
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        deconv = tf.nn.leaky_relu(deconv)

        g = tf.nn.conv2d_transpose(
            input_,
            w,
            output_shape=output_shape,
            strides=[1, d_h, d_w, 1]
        )
        b = tf.compat.v1.get_variable(
            'biases2',
            [output_shape[-1]],
            initializer=tf.compat.v1.constant_initializer(0.0)
        )
        g = tf.reshape(tf.nn.bias_add(g, b), deconv.get_shape())
        g = tf.nn.sigmoid(deconv)
        deconv = tf.multiply(g, deconv)
        return deconv, g


class Model:
    """
    """
    input_size = CONFIG.model.input_size
    batch_size = CONFIG.model.batch_size
    ckpt_path = CONFIG.model.ckpt_path

    sess = None

    demo_output = None

    def __init__(self):
        image_dims = [self.input_size, self.input_size, 3]
        sk_dims = [self.input_size, self.input_size, 1]
        color_dims = [self.input_size, self.input_size, 3]
        masks_dims = [self.input_size, self.input_size, 1]
        noises_dims = [self.input_size, self.input_size, 1]

        self.dtype = tf.float32

        self.images = tf.compat.v1.placeholder(self.dtype, [self.batch_size] + image_dims, name='real_images')
        self.sketches = tf.compat.v1.placeholder(self.dtype, [self.batch_size] + sk_dims, name='sketches')
        self.color = tf.compat.v1.placeholder(self.dtype, [self.batch_size] + color_dims, name='color')
        self.masks = tf.compat.v1.placeholder(self.dtype, [self.batch_size] + masks_dims, name='masks')
        self.noises = tf.compat.v1.placeholder(self.dtype, [self.batch_size] + noises_dims, name='noises')

        self.load_demo_graph()
        self.warmup()

    def build_gen(self, x, mask, name='generator', reuse=False):
        """
        Generator is based on encoder-decoder architecture like the U-Net
        and all convolution layers use gated convolution.

        U-Net: https://towardsdatascience.com/unet-line-by-line-explanation-9b191c76baf5
        Gated Convolution: https://arxiv.org/abs/1806.03589

        Local signal normalization (LRN) is applied after feature map convolution layers excluding other soft gates.
        LRN is applied to all convolution layers except input and output layers.

        The encoder of our generator receives input tensor of size 512×512×9:
        - an incomplete RGB channel image with a removed region to be edited,
        - a binary sketch that describes the structure of removed parts,
        - an RGB color stroke map,
        - a binary mask and a noise

        The encoder downsamples input 7 times using 2 stride kernel convolutions,
        followed by dilated convolutions before upsampling.

        The decoder uses transposed convolutions for upsampling.
        Then, skip connections were added to allow concatenation with previous layer with the same spatial resolution.

        We used the leaky ReLU activation function after each layer
        except for the output layer, which uses a tanh function.

        Overall, our generator consists of 16 convolution layers and the output of the network
        is an RGB image of same size of input (512×512).

        """
        s_h, s_w = self.input_size, self.input_size
        s_h2, s_w2 = int(self.input_size / 2), int(self.input_size / 2)
        s_h4, s_w4 = int(self.input_size / 4), int(self.input_size / 4)
        s_h8, s_w8 = int(self.input_size / 8), int(self.input_size / 8)
        s_h16, s_w16 = int(self.input_size / 16), int(self.input_size / 16)
        s_h32, s_w32 = int(self.input_size / 32), int(self.input_size / 32)
        s_h64, s_w64 = int(self.input_size / 64), int(self.input_size / 64)

        cnum = 64

        with tf.compat.v1.variable_scope(name, reuse=reuse):
            # encoder
            x_now = x
            x1, _ = gate_conv(x, cnum, ksize=7, stride=2, use_lrn=False, name='gconv1_ds')
            x2, _ = gate_conv(x1, 2 * cnum, ksize=5, stride=2, name='gconv2_ds')
            x3, _ = gate_conv(x2, 4 * cnum, ksize=5, stride=2, name='gconv3_ds')
            x4, _ = gate_conv(x3, 8 * cnum, ksize=3, stride=2, name='gconv4_ds')
            x5, _ = gate_conv(x4, 8 * cnum, ksize=3, stride=2, name='gconv5_ds')
            x6, _ = gate_conv(x5, 8 * cnum, ksize=3, stride=2, name='gconv6_ds')
            x7, _ = gate_conv(x6, 8 * cnum, ksize=3, stride=2, name='gconv7_ds')

            # dilated conv
            x7, _ = gate_conv(x7, 8 * cnum, ksize=3, stride=1, rate=2, name='co_conv1_dlt')
            x7, _ = gate_conv(x7, 8 * cnum, ksize=3, stride=1, rate=4, name='co_conv2_dlt')
            x7, _ = gate_conv(x7, 8 * cnum, ksize=3, stride=1, rate=8, name='co_conv3_dlt')
            x7, _ = gate_conv(x7, 8 * cnum, ksize=3, stride=1, rate=16, name='co_conv4_dlt')

            # decoder
            x8, _ = gate_deconv(x7, output_shape=[self.batch_size, s_h64, s_w64, 8 * cnum], name='deconv1')
            x8 = tf.concat([x6, x8], axis=3)
            x8, _ = gate_conv(x8, 8 * cnum, ksize=3, stride=1, name='gconv8')

            x9, _ = gate_deconv(x8, output_shape=[self.batch_size, s_h32, s_w32, 8 * cnum], name='deconv2')
            x9 = tf.concat([x5, x9], axis=3)
            x9, _ = gate_conv(x9, 8 * cnum, ksize=3, stride=1, name='gconv9')

            x10, _ = gate_deconv(x9, output_shape=[self.batch_size, s_h16, s_w16, 8 * cnum], name='deconv3')
            x10 = tf.concat([x4, x10], axis=3)
            x10, _ = gate_conv(x10, 8 * cnum, ksize=3, stride=1, name='gconv10')

            x11, _ = gate_deconv(x10, output_shape=[self.batch_size, s_h8, s_w8, 4 * cnum], name='deconv4')
            x11 = tf.concat([x3, x11], axis=3)
            x11, _ = gate_conv(x11, 4 * cnum, ksize=3, stride=1, name='gconv11')

            x12, _ = gate_deconv(x11, output_shape=[self.batch_size, s_h4, s_w4, 2 * cnum], name='deconv5')
            x12 = tf.concat([x2, x12], axis=3)
            x12, _ = gate_conv(x12, 2 * cnum, ksize=3, stride=1, name='gconv12')

            x13, _ = gate_deconv(x12, output_shape=[self.batch_size, s_h2, s_w2, cnum], name='deconv6')
            x13 = tf.concat([x1, x13], axis=3)
            x13, _ = gate_conv(x13, cnum, ksize=3, stride=1, name='gconv13')

            x14, _ = gate_deconv(x13, output_shape=[self.batch_size, s_h, s_w, 3], name='deconv7')
            x14 = tf.concat([x_now, x14], axis=3)
            x14, mask14 = gate_conv(x14, 3, ksize=3, stride=1, activation=None, use_lrn=False, name='gconv14')

            output = tf.tanh(x14)

            return output, mask14

    def build_demo_graph(self):
        """
        Build the graph for demo

        :return: None

        We replaced the remaining parts of image outside the mask with the input image before applying the loss
        functions to it. This replacement allows the generator to be trained on the edited region exclusively.
        """
        input_images = self.images * (1 - self.masks)
        batch_data = tf.concat([input_images, self.sketches, self.color, self.masks, self.noises], axis=3)

        gen_img, output_mask = self.build_gen(batch_data, self.masks)

        self.demo_output = gen_img * self.masks + input_images

    def load_demo_graph(self):
        sess_config = tf.compat.v1.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=sess_config)
        self.build_demo_graph()
        init_op = tf.compat.v1.global_variables_initializer()
        self.sess.run(init_op)
        vars_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        print(f'Loading model from {self.ckpt_path}...')
        for var in vars_list:
            var_value = tf.train.load_variable(self.ckpt_path, var.name)
            assign_ops.append(tf.compat.v1.assign(var, var_value))
        self.sess.run(assign_ops)
        print(f'Model loaded from {self.ckpt_path}')

    def warmup(self):
        size = self.input_size
        bc = self.batch_size
        _ = self.sess.run(
            self.demo_output,
            feed_dict={
                self.images: np.zeros([bc, size, size, 3]),
                self.sketches: np.zeros([bc, size, size, 1]),
                self.color: np.zeros([bc, size, size, 3]),
                self.masks: np.zeros([bc, size, size, 1]),
                self.noises: np.zeros([bc, size, size, 1])
            }
        )

    def demo(self, batch):
        demo_output = self.sess.run(
            self.demo_output,
            feed_dict={
                self.images: batch[:, :, :, :3],
                self.sketches: batch[:, :, :, 3:4],
                self.color: batch[:, :, :, 4:7],
                self.masks: batch[:, :, :, 7:8],
                self.noises: batch[:, :, :, 8:9]
            }
        )
        return demo_output


@lru_cache()
def get_model():
    return Model()


if __name__ == '__main__':
    CONFIG.model.ckpt_path = 'ckpt/SC-FEGAN.ckpt'
    model = get_model()
