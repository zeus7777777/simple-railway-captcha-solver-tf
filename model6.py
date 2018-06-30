import os

import tensorflow as tf
import numpy as np

def show_trainable_variable(scope):
    variables = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)]
    num_vars = 0
    for v in variables:
        print(v)
        num_vars += np.prod([dim.value for dim in v.get_shape()])
    print('Total trainable variables ('+scope+'):', num_vars)

def _model(image, train=True):
    net = tf.layers.conv2d(image, 32, 3, padding='SAME', activation=tf.nn.relu)
    net = tf.layers.conv2d(net, 32, 3, padding='SAME', activation=tf.nn.relu)
    net = tf.layers.batch_normalization(net, training=train)
    net = tf.layers.max_pooling2d(net, 2, 2, padding='SAME')
    if train:
        net = tf.nn.dropout(net, 0.7)
    net = tf.layers.conv2d(net, 64, 3, padding='SAME', activation=tf.nn.relu)
    net = tf.layers.conv2d(net, 64, 3, padding='SAME', activation=tf.nn.relu)
    net = tf.layers.batch_normalization(net, training=train)
    net = tf.layers.max_pooling2d(net, 2, 2, padding='SAME')
    if train:
        net = tf.nn.dropout(net, 0.7)
    net = tf.layers.conv2d(net, 128, 3, padding='SAME', activation=tf.nn.relu)
    net = tf.layers.conv2d(net, 128, 3, padding='SAME', activation=tf.nn.relu)
    net = tf.layers.batch_normalization(net, training=train)
    net = tf.layers.max_pooling2d(net, 2, 2, padding='SAME')
    if train:
        net = tf.nn.dropout(net, 0.7)
    net = tf.layers.conv2d(net, 256, 3, padding='SAME', activation=tf.nn.relu)
    net = tf.layers.batch_normalization(net, training=train)
    net = tf.layers.max_pooling2d(net, 2, 2, padding='SAME')
    net = tf.reshape(net, [-1, 4*13*256])
    if train:
        net = tf.nn.dropout(net, 0.7)
    fc1 = tf.layers.dense(net, 37)
    fc2 = tf.layers.dense(net, 37)
    fc3 = tf.layers.dense(net, 37)
    fc4 = tf.layers.dense(net, 37)
    fc5 = tf.layers.dense(net, 37)
    fc6 = tf.layers.dense(net, 37)

    return fc1, fc2, fc3, fc4, fc5, fc6


class TrainModel():
    def __init__(self, batch_size=0, learning_rate=-1, tfrecord_file=''):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.tfrecord_file = tfrecord_file

        self._generate_input_pipeline(tfrecord_file)
        self._build_model()
    
    def _generate_input_pipeline(self, tfrecord_file):
        def _parse(e):
            features = {
                'jpeg_str': tf.FixedLenFeature([], tf.string),
                'captcha_ints': tf.FixedLenFeature([6], tf.int64),
            }
            features = tf.parse_single_example(e, features)
            image = tf.image.decode_jpeg(features['jpeg_str'], 3)
            image = tf.cast(image, tf.float32) / 255.0
            image = tf.reshape(image, [60, 200, 3])
            return image, features['captcha_ints'][0], features['captcha_ints'][1], features['captcha_ints'][2], features['captcha_ints'][3], features['captcha_ints'][4], features['captcha_ints'][5]
        dataset = tf.data.TFRecordDataset(tfrecord_file)
        dataset = dataset.map(_parse)
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size))
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(self.batch_size//4, count=None))

        iterator = dataset.make_one_shot_iterator()

        self.image, self.captcha_0, self.captcha_1, self.captcha_2, self.captcha_3, self.captcha_4, self.captcha_5 = iterator.get_next()
    
    def _build_model(self):
        fc1, fc2, fc3, fc4, fc5, fc6 = _model(self.image, train=True)

        loss1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.captcha_0, logits=fc1)
        loss2 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.captcha_1, logits=fc2)
        loss3 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.captcha_2, logits=fc3)
        loss4 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.captcha_3, logits=fc4)
        loss5 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.captcha_4, logits=fc5)
        loss6 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.captcha_5, logits=fc6)

        self.loss = tf.reduce_mean(loss1 + loss2 + loss3 + loss4 + loss5 + loss6)

        self.accu1 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(fc1, axis=-1), self.captcha_0), tf.float32))
        self.accu2 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(fc2, axis=-1), self.captcha_1), tf.float32))
        self.accu3 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(fc3, axis=-1), self.captcha_2), tf.float32))
        self.accu4 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(fc4, axis=-1), self.captcha_3), tf.float32))
        self.accu5 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(fc5, axis=-1), self.captcha_4), tf.float32))
        self.accu6 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(fc6, axis=-1), self.captcha_5), tf.float32))

        self.accu = self.accu1 * self.accu2 * self.accu3 * self.accu4 * self.accu5 * self.accu6

        self.global_step = tf.get_variable('global_step', initializer=tf.constant(0), dtype=tf.int32, trainable=False)

        trainer = tf.train.AdamOptimizer(self.learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = trainer.minimize(self.loss, global_step=self.global_step)

        self.init_step = tf.global_variables_initializer()

        self.saver = tf.train.Saver()
        self.summary_writer = None

        show_trainable_variable('')

    def train(self, sess):
        loss, accu, _ = sess.run([self.loss, self.accu, self.train_step])
        return loss, accu
    
    def get_global_step(self, sess):
        return sess.run(self.global_step)
    
    def init(self, sess):
        sess.run(self.init_step)
        print('Model initialized')

    def load(self, sess, path):
        ckpt = tf.train.get_checkpoint_state(path)
        self.saver.restore(sess, ckpt.model_checkpoint_path)
        print('Model restored')

    def save(self, sess, path):
        name = os.path.join(path, 'model.ckpt')
        self.saver.save(sess, name, global_step=self.global_step)
        print('Model saved')

class ValidModel():
    def __init__(self, batch_size=0, tfrecord_file=''):
        self.batch_size = batch_size
        self.tfrecord_file = tfrecord_file

        self._generate_input_pipeline(tfrecord_file)
        self._build_model()

    def _generate_input_pipeline(self, tfrecord_file):
        def _parse(e):
            features = {
                'jpeg_str': tf.FixedLenFeature([], tf.string),
                'captcha_ints': tf.FixedLenFeature([6], tf.int64),
            }
            features = tf.parse_single_example(e, features)
            image = tf.image.decode_jpeg(features['jpeg_str'], 3)
            image = tf.cast(image, tf.float32) / 255.0
            image = tf.reshape(image, [60, 200, 3])
            return image, features['captcha_ints'][0], features['captcha_ints'][1], features['captcha_ints'][2], features['captcha_ints'][3], features['captcha_ints'][4], features['captcha_ints'][5]
        dataset = tf.data.TFRecordDataset(tfrecord_file)
        dataset = dataset.map(_parse)
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size))
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(self.batch_size//4, count=None))

        iterator = dataset.make_one_shot_iterator()

        self.image, self.captcha_0, self.captcha_1, self.captcha_2, self.captcha_3, self.captcha_4, self.captcha_5 = iterator.get_next()
    
    def _build_model(self):
        fc1, fc2, fc3, fc4, fc5, fc6 = _model(self.image, train=False)

        loss1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.captcha_0, logits=fc1)
        loss2 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.captcha_1, logits=fc2)
        loss3 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.captcha_2, logits=fc3)
        loss4 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.captcha_3, logits=fc4)
        loss5 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.captcha_4, logits=fc5)
        loss6 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.captcha_5, logits=fc6)

        self.loss = tf.reduce_mean(loss1 + loss2 + loss3 + loss4 + loss5 + loss6)

        self.accu1 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(fc1, axis=-1), self.captcha_0), tf.float32))
        self.accu2 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(fc2, axis=-1), self.captcha_1), tf.float32))
        self.accu3 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(fc3, axis=-1), self.captcha_2), tf.float32))
        self.accu4 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(fc4, axis=-1), self.captcha_3), tf.float32))
        self.accu5 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(fc5, axis=-1), self.captcha_4), tf.float32))
        self.accu6 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(fc6, axis=-1), self.captcha_5), tf.float32))

        self.accu = self.accu1 * self.accu2 * self.accu3 * self.accu4 * self.accu5 * self.accu6

        self.saver = tf.train.Saver()
    
    def valid(self, sess):
        loss, accu = sess.run([self.loss, self.accu])
        return loss, accu
    
    def load(self, sess, path):
        ckpt = tf.train.get_checkpoint_state(path)
        self.saver.restore(sess, ckpt.model_checkpoint_path)
        print('Model restored')

class TestModel():
    def __init__(self, batch_size=0):
        self.batch_size = batch_size
        self._build_model()

    def _build_model(self):
        self.image = tf.placeholder(tf.float32, [1, 60, 200, 3])
        fc1, fc2, fc3, fc4, fc5, fc6 = _model(self.image, train=False)

        self.predict1 = tf.argmax(fc1, axis=-1)
        self.predict2 = tf.argmax(fc2, axis=-1)
        self.predict3 = tf.argmax(fc3, axis=-1)
        self.predict4 = tf.argmax(fc4, axis=-1)
        self.predict5 = tf.argmax(fc5, axis=-1)
        self.predict6 = tf.argmax(fc6, axis=-1)

        self.saver = tf.train.Saver()

    def load(self, sess, path):
        ckpt = tf.train.get_checkpoint_state(path)
        self.saver.restore(sess, ckpt.model_checkpoint_path)
        print('Model restored')
    
    def predict(self, sess, image):
        a,b,c,d,e,f = sess.run([self.predict1, self.predict2, self.predict3, self.predict4, self.predict5, self.predict6], feed_dict={self.image:[image]})
        return a,b,c,d,e,f