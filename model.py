import tensorflow as tf
import numpy as np


class Model:

    def residualconv(self, input, filter_size):
        padding = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        pad1 = tf.pad(input, padding, 'REFLECT')
        localConv = self.convblock(pad1, filter_size, padding='VALID')
        pad2 = tf.pad(localConv, padding, 'REFLECT')
        localConv2 = self.convblock(pad2, filter_size, isReLU=False, padding='VALID')
        return tf.add(localConv2, input)

    def convblock(self, input, filter_size, kernel_size = 3, stride_size = 1, isReLU = True, padding='SAME'):
        convLocal = tf.layers.conv2d(input, filters = filter_size, kernel_size=kernel_size,
                                     strides = stride_size, padding=padding)
        bnLocal = tf.contrib.layers.batch_norm(convLocal)

        if isReLU:
            return tf.nn.relu(bnLocal)
        return bnLocal

    def deconvblock(self, input, filter_size, kernel_size = 3, stride_size = 1):
        deconvLocal = tf.layers.conv2d_transpose(input, filters=filter_size , kernel_size=kernel_size,
                                                 strides = stride_size, padding='SAME')
        bnLocal = tf.contrib.layers.batch_norm(deconvLocal)
        ReLULocal = tf.nn.relu(bnLocal)

        return ReLULocal

    def discriminator(self, input, filter_size=64, kernel_size = 3,layer_size = 4):
        conv0 = tf.layers.conv2d(input, filters= filter_size, kernel_size = kernel_size, padding='SAME')
        Leaky = tf.nn.leaky_relu(conv0)
        for i in range(1,layer_size):
            convLoc = tf.layers.conv2d(Leaky, filter_size*(2**i), kernel_size = kernel_size,
                                       strides = 2, padding='SAME')
            bnLoc = tf.contrib.layers.batch_norm(convLoc)
            Leaky = tf.nn.leaky_relu(bnLoc)

        finalConv = tf.layers.conv2d(Leaky, filters=1, kernel_size=kernel_size, padding='VALID')
        reduced = tf.reduce_mean(finalConv, [1,2])
        print(reduced.get_shape())
        sigmoided = tf.sigmoid(reduced)

        D = reduced
        D_sigmoided = sigmoided
        return D, D_sigmoided

    def generator(self, inputImage, filter_size = 64):
        f = 7
        p=(f-1)//2
        padding_constant = tf.constant([[0,0],[p,p],[p,p],[0,0]])
        padres = tf.pad(inputImage, padding_constant, mode = 'REFLECT')

        # Auto Encoder + 행렬 합 중간 중간 넣어주기!
        #7*7 SAME , Generate Layer level
        conv1 = self.convblock(padres, filter_size = filter_size, kernel_size=f, padding='VALID')
        conv2 = self.convblock(conv1, filter_size * 2, stride_size = 2)
        goRes = self.convblock(conv2, filter_size * 4, stride_size = 2)
        print(goRes.get_shape())
        for i in range(0,5):
            goRes = self.residualconv(goRes, filter_size*4)

        # DeConv 시작
        deconv1 = self.deconvblock(goRes, filter_size = filter_size * 2, stride_size = 2)
        deconv2 = self.deconvblock(deconv1, filter_size = filter_size, stride_size = 2)
        pad = tf.pad(deconv2, padding_constant, mode = 'REFLECT')
        finalConv = tf.layers.conv2d(pad, 3, kernel_size = 7)
        G = tf.nn.tanh(finalConv)

        return G

    def __init__(self, in_width = 128, in_height = 128):
        self.inputX = tf.placeholder(tf.float32, [None, in_height, in_width, 3])
        self.inputY = tf.placeholder(tf.float32, [None, in_height, in_width, 3])

        with tf.variable_scope('XtoYGEN'):
            self.XtoYGEN = self.generator(self.inputX)

        with tf.variable_scope('YtoXGEN'):
            self.YtoXGEN = self.generator(self.inputY)

        with tf.variable_scope('XtoYGEN', reuse=True):
            self.YXYGEN = self.generator(self.YtoXGEN)

        with tf.variable_scope('YtoXGEN', reuse=True):
            self.XYXGEN = self.generator(self.XtoYGEN)

        with tf.variable_scope('XDis'):
            self.XDis, self.XDisSigmoid = self.discriminator(self.inputX)

        with tf.variable_scope('YDis'):
            self.YDis, self.YDisSigmoid = self.discriminator(self.inputY)

        with tf.variable_scope('XDis', reuse=True):
            self.YtoXDis, self.YtoXDisSigmoid = self.discriminator(self.YtoXGEN)

        with tf.variable_scope('YDis', reuse=True):
            self.XtoYDis, self.XtoYDisSigmoid = self.discriminator(self.XtoYGEN)

    def getLoss(self, cycleConst=10):
        DisXLoss = tf.square(self.XDisSigmoid-1)+tf.square(self.YtoXDisSigmoid)
        DisYLoss = tf.square(self.YDisSigmoid-1)+tf.square(self.XtoYDisSigmoid)
        GenXtoYLoss = tf.square(self.XtoYDisSigmoid-1)
        GenYtoXLoss = tf.square(self.YtoXDisSigmoid-1)
        XCycleLoss = tf.abs(self.XYXGEN-self.inputX)
        YCycleLoss = tf.abs(self.YXYGEN-self.inputY)
        DLoss = DisXLoss+DisYLoss
        GLoss = cycleConst*(XCycleLoss+YCycleLoss)+GenXtoYLoss+GenYtoXLoss

        return GLoss, DLoss

