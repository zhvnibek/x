import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope

"""
    The gate convolution is made with reference to Deepfillv1 (https://github.com/JiahuiYu/generative_inpainting)

    https://paperswithcode.com/paper/free-form-image-inpainting-with-gated

    https://github.com/avalonstrel/GatedConvolution_pytorch/blob/master/models/sa_gan.py
    https://github.com/csqiangwen/DeepFillv2_Pytorch/blob/master/network.py

    https://github.com/nipponjo/deepfillv2-pytorch (latest)

"""


@add_arg_scope
def gate_conv(x_in, cnum, ksize, stride=1, rate=1, name='conv',
              padding='SAME', activation='leaky_relu', use_lrn=True):
    assert padding in ['SYMMETRIC', 'SAME', 'REFELECT']
    if padding == 'SYMMETRIC' or padding == 'REFELECT':
        p = int(rate * (ksize - 1) / 2)
        x = tf.pad(x_in, [[0, 0], [p, p], [p, p], [0, 0]], mode=padding)
        padding = 'VALID'
    x = tf.layers.conv2d(
        x_in, cnum, ksize, stride, dilation_rate=rate,
        activation=None, padding=padding, name=name)
    if use_lrn:
        x = tf.nn.lrn(x, bias=0.00005)
    if activation == 'leaky_relu':
        x = tf.nn.leaky_relu(x)

    g = tf.layers.conv2d(
        x_in, cnum, ksize, stride, dilation_rate=rate,
        activation=tf.nn.sigmoid, padding=padding, name=name + '_g')

    x = tf.multiply(x, g)
    return x, g


@add_arg_scope
def gate_deconv(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="deconv"):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                        strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases1', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        deconv = tf.nn.leaky_relu(deconv)

        g = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                   strides=[1, d_h, d_w, 1])
        b = tf.get_variable('biases2', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        g = tf.reshape(tf.nn.bias_add(g, b), deconv.get_shape())
        g = tf.nn.sigmoid(deconv)

        deconv = tf.multiply(g, deconv)

        return deconv, g


class Model:
    """
    """
    input_size = 512
    batch_size = 1
    ckpt_path = 'fegan/ckpt/SC-FEGAN.ckpt'

    sess = None

    demo_output = None

    def __init__(self):
        image_dims = [self.input_size, self.input_size, 3]
        sk_dims = [self.input_size, self.input_size, 1]
        color_dims = [self.input_size, self.input_size, 3]
        masks_dims = [self.input_size, self.input_size, 1]
        noises_dims = [self.input_size, self.input_size, 1]

        self.dtype = tf.float32

        self.images = tf.placeholder(self.dtype, [self.batch_size] + image_dims, name='real_images')
        self.sketches = tf.placeholder(self.dtype, [self.batch_size] + sk_dims, name='sketches')
        self.color = tf.placeholder(self.dtype, [self.batch_size] + color_dims, name='color')
        self.masks = tf.placeholder(self.dtype, [self.batch_size] + masks_dims, name='masks')
        self.noises = tf.placeholder(self.dtype, [self.batch_size] + noises_dims, name='noises')

        self.load_demo_graph()
        self.warmup()

    def build_gen(self, x, mask, name='generator', reuse=False):
        cnum = 64
        s_h, s_w = self.input_size, self.input_size
        s_h2, s_w2 = int(self.input_size / 2), int(self.input_size / 2)
        s_h4, s_w4 = int(self.input_size / 4), int(self.input_size / 4)
        s_h8, s_w8 = int(self.input_size / 8), int(self.input_size / 8)
        s_h16, s_w16 = int(self.input_size / 16), int(self.input_size / 16)
        s_h32, s_w32 = int(self.input_size / 32), int(self.input_size / 32)
        s_h64, s_w64 = int(self.input_size / 64), int(self.input_size / 64)
        with tf.variable_scope(name, reuse=reuse):
            # encoder
            x_now = x
            x1, mask1 = gate_conv(x, cnum, 7, 2, use_lrn=False, name='gconv1_ds')
            x2, mask2 = gate_conv(x1, 2 * cnum, 5, 2, name='gconv2_ds')
            x3, mask3 = gate_conv(x2, 4 * cnum, 5, 2, name='gconv3_ds')
            x4, mask4 = gate_conv(x3, 8 * cnum, 3, 2, name='gconv4_ds')
            x5, mask5 = gate_conv(x4, 8 * cnum, 3, 2, name='gconv5_ds')
            x6, mask6 = gate_conv(x5, 8 * cnum, 3, 2, name='gconv6_ds')
            x7, mask7 = gate_conv(x6, 8 * cnum, 3, 2, name='gconv7_ds')

            # dilated conv
            x7, _ = gate_conv(x7, 8 * cnum, 3, 1, rate=2, name='co_conv1_dlt')
            x7, _ = gate_conv(x7, 8 * cnum, 3, 1, rate=4, name='co_conv2_dlt')
            x7, _ = gate_conv(x7, 8 * cnum, 3, 1, rate=8, name='co_conv3_dlt')
            x7, _ = gate_conv(x7, 8 * cnum, 3, 1, rate=16, name='co_conv4_dlt')

            # decoder
            x8, _ = gate_deconv(x7, [self.batch_size, s_h64, s_w64, 8 * cnum], name='deconv1')
            x8 = tf.concat([x6, x8], axis=3)
            x8, mask8 = gate_conv(x8, 8 * cnum, 3, 1, name='gconv8')

            x9, _ = gate_deconv(x8, [self.batch_size, s_h32, s_w32, 8 * cnum], name='deconv2')
            x9 = tf.concat([x5, x9], axis=3)
            x9, mask9 = gate_conv(x9, 8 * cnum, 3, 1, name='gconv9')

            x10, _ = gate_deconv(x9, [self.batch_size, s_h16, s_w16, 8 * cnum], name='deconv3')
            x10 = tf.concat([x4, x10], axis=3)
            x10, mask10 = gate_conv(x10, 8 * cnum, 3, 1, name='gconv10')

            x11, _ = gate_deconv(x10, [self.batch_size, s_h8, s_w8, 4 * cnum], name='deconv4')
            x11 = tf.concat([x3, x11], axis=3)
            x11, mask11 = gate_conv(x11, 4 * cnum, 3, 1, name='gconv11')

            x12, _ = gate_deconv(x11, [self.batch_size, s_h4, s_w4, 2 * cnum], name='deconv5')
            x12 = tf.concat([x2, x12], axis=3)
            x12, mask12 = gate_conv(x12, 2 * cnum, 3, 1, name='gconv12')

            x13, _ = gate_deconv(x12, [self.batch_size, s_h2, s_w2, cnum], name='deconv6')
            x13 = tf.concat([x1, x13], axis=3)
            x13, mask13 = gate_conv(x13, cnum, 3, 1, name='gconv13')

            x14, _ = gate_deconv(x13, [self.batch_size, s_h, s_w, 3], name='deconv7')
            x14 = tf.concat([x_now, x14], axis=3)
            x14, mask14 = gate_conv(x14, 3, 3, 1, activation=None, use_lrn=False, name='gconv14')

            output = tf.tanh(x14)

            return output, mask14

    def build_demo_graph(self):
        input_images = self.images * (1 - self.masks)
        batch_data = tf.concat([input_images, self.sketches, self.color, self.masks, self.noises], axis=3)
        gen_img, output_mask = self.build_gen(batch_data, self.masks)
        self.demo_output = gen_img * self.masks + input_images

    def load_demo_graph(self):
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        self.build_demo_graph()
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:
            var_value = tf.contrib.framework.load_variable(self.ckpt_path, var.name)
            assign_ops.append(tf.assign(var, var_value))
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
