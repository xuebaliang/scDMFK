import tensorflow as tf
import keras.backend as K
from keras.layers import GaussianNoise, Dense, Activation
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import os
import numpy as np


MeanAct = lambda x: tf.clip_by_value(x, 1e-5, 1e6)
DispAct = lambda x: tf.clip_by_value(tf.nn.softplus(x), 1e-4, 1e4)


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def cal_dist(hidden, clusters):
    dist1 = K.sum(K.square(K.expand_dims(hidden, axis=1) - clusters), axis=2)
    temp_dist1 = dist1 - tf.reshape(tf.reduce_min(dist1, axis=1), [-1, 1])
    q = K.exp(-temp_dist1)
    q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
    q = K.pow(q, 2)
    q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
    dist2 = dist1 * q
    return dist1, dist2


def adapative_dist(hidden, clusters, sigma):
    dist1 = K.sum(K.square(K.expand_dims(hidden, axis=1) - clusters), axis=2)
    dist2 = K.sqrt(dist1)
    dist = (1 + sigma) * dist1 / (dist2 + sigma)
    return dist


def fuzzy_kmeans(hidden, clusters, sigma, theta, adapative = True):
    if adapative:
        dist = adapative_dist(hidden, clusters, sigma)
    else:
        dist = K.sum(K.square(K.expand_dims(hidden, axis=1) - clusters), axis=2)
    temp_dist = dist - tf.reshape(tf.reduce_min(dist, axis=1), [-1, 1])
    q = K.exp(-temp_dist / theta)
    q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
    fuzzy_dist = q * dist
    return dist, fuzzy_dist


def multinomial(x, p):
    loss = tf.reduce_mean(-x * tf.log(tf.clip_by_value(p, 1e-12, 1.0)))
    return loss


def weight_mse(x_count, x, recon_x):
    weight_loss = x_count * tf.square(x - recon_x)
    return tf.reduce_mean(weight_loss)


def mask_mse(x_count, x, recon_x):
    loss = tf.sign(x_count) * tf.square(x - recon_x)
    return tf.reduce_mean(loss)


def _nan2zero(x):
    return tf.where(tf.is_nan(x), tf.zeros_like(x), x)

def _nan2inf(x):
    return tf.where(tf.is_nan(x), tf.zeros_like(x)+np.inf, x)

def _nelem(x):
    nelem = tf.reduce_sum(tf.cast(~tf.is_nan(x), tf.float32))
    return tf.cast(tf.where(tf.equal(nelem, 0.), 1., nelem), x.dtype)

def _reduce_mean(x):
    nelem = _nelem(x)
    x = _nan2zero(x)
    return tf.divide(tf.reduce_sum(x), nelem)

def NB(theta, y_true, y_pred, mask = False, debug = False, mean = False):
    eps = 1e-10
    scale_factor = 1.0
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32) * scale_factor
    if mask:
        nelem = _nelem(y_true)
        y_true = _nan2zero(y_true)
    theta = tf.minimum(theta, 1e6)
    t1 = tf.lgamma(theta + eps) + tf.lgamma(y_true + 1.0) - tf.lgamma(y_true + theta + eps)
    t2 = (theta + y_true) * tf.log(1.0 + (y_pred / (theta + eps))) + (y_true * (tf.log(theta + eps) - tf.log(y_pred + eps)))
    if debug:
        assert_ops = [tf.verify_tensor_all_finite(y_pred, 'y_pred has inf/nans'),
                      tf.verify_tensor_all_finite(t1, 't1 has inf/nans'),
                      tf.verify_tensor_all_finite(t2, 't2 has inf/nans')]
        with tf.control_dependencies(assert_ops):
            final = t1 + t2
    else:
        final = t1 + t2
    final = _nan2inf(final)
    if mean:
        if mask:
            final = tf.divide(tf.reduce_sum(final), nelem)
        else:
            final = tf.reduce_mean(final)
    return final

def ZINB(pi, theta, y_true, y_pred, ridge_lambda, mean = True, mask = False, debug = False):
    eps = 1e-10
    scale_factor = 1.0
    nb_case = NB(theta, y_true, y_pred, mean=False, debug=debug) - tf.log(1.0 - pi + eps)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32) * scale_factor
    theta = tf.minimum(theta, 1e6)

    zero_nb = tf.pow(theta / (theta + y_pred + eps), theta)
    zero_case = -tf.log(pi + ((1.0 - pi) * zero_nb) + eps)
    result = tf.where(tf.less(y_true, 1e-8), zero_case, nb_case)
    ridge = ridge_lambda * tf.square(pi)
    result += ridge
    if mean:
        if mask:
            result = _reduce_mean(result)
        else:
            result = tf.reduce_mean(result)

    result = _nan2inf(result)
    return result


class scDMFK(object):
    def __init__(self, dataname, dims, cluster_num, alpha, sigma, theta, learning_rate, noise_sd=1.5, init='glorot_uniform', act='relu', adaptative = True, model = "multinomial", mode = "indirect"):
        self.dataname = dataname
        self.dims = dims
        self.cluster_num = cluster_num
        self.alpha = alpha
        self.sigma = sigma
        self.theta = theta
        self.learning_rate = learning_rate
        self.noise_sd = noise_sd
        self.init = init
        self.act = act
        self.adaptative = adaptative
        self.model = model
        self.mode = mode

        self.n_stacks = len(self.dims) - 1
        # input
        self.x = tf.placeholder(dtype=tf.float32, shape=(None, self.dims[0]))
        self.x_count = tf.placeholder(dtype=tf.float32, shape=(None, self.dims[0]))
        self.sf_layer = tf.placeholder(dtype=tf.float32, shape=(None, 1))
        self.clusters = tf.get_variable(name=self.dataname + "/clusters_rep", shape=[self.cluster_num, self.dims[-1]],
                                        dtype=tf.float32, initializer=tf.glorot_uniform_initializer())

        self.h = self.x
        self.h = GaussianNoise(self.noise_sd, name='input_noise')(self.h)
        for i in range(self.n_stacks - 1):
            self.h = Dense(units=self.dims[i + 1], kernel_initializer=self.init, name='encoder_%d' % i)(self.h)
            self.h = GaussianNoise(self.noise_sd, name='noise_%d' % i)(self.h)  # add Gaussian noise
            self.h = Activation(self.act)(self.h)

        self.latent = Dense(units=self.dims[-1], kernel_initializer=self.init, name='encoder_hidden')(self.h)  # hidden layer, features are extracted from here
        self.latent_dist1, self.latent_dist2 = fuzzy_kmeans(self.latent, self.clusters, self.sigma, self.theta, adapative=self.adaptative)
        self.h = self.latent

        for i in range(self.n_stacks - 1, 0, -1):
            self.h = Dense(units=self.dims[i], activation=self.act, kernel_initializer=self.init,
                           name='decoder_%d' % i)(self.h)

        if self.model == "multinomial":
            if mode == "indirect":
                self.mean = Dense(units=self.dims[0], activation=MeanAct, kernel_initializer=self.init, name='mean')(self.h)
                self.output = self.mean * tf.matmul(self.sf_layer, tf.ones((1, self.mean.get_shape()[1]), dtype=tf.float32))
                self.pi = Dense(units=self.dims[0], activation='sigmoid', kernel_initializer=self.init, name='pi')(self.h)
                self.P = tf.transpose(tf.transpose(self.pi * self.output) / tf.reduce_sum(self.pi * self.output, axis=1))
                self.pre_loss = multinomial(self.x_count, self.P)
            else:
                self.P = Dense(units=self.dims[0], activation=tf.nn.softmax, kernel_initializer=self.init, name='pi')(self.h)
                self.pre_loss = multinomial(self.x_count, self.P)
        elif self.model == "ZINB":
            self.pi = Dense(units=self.dims[0], activation='sigmoid', kernel_initializer=self.init, name='pi')(self.h)
            self.disp = Dense(units=self.dims[0], activation=DispAct, kernel_initializer=self.init, name='dispersion')(self.h)
            self.mean = Dense(units=self.dims[0], activation=MeanAct, kernel_initializer=self.init, name='mean')(self.h)
            self.output = self.mean * tf.matmul(self.sf_layer, tf.ones((1, self.mean.get_shape()[1]), dtype=tf.float32))
            self.pre_loss = ZINB(self.pi, self.disp, self.x_count, self.output, ridge_lambda=1.0)
        elif self.model == "weight mse":
            self.recon_x = Dense(units=self.dims[0], kernel_initializer=self.init, name='reconstruction')(self.h)
            self.weight_mse = weight_mse(self.x_count, self.x, self.recon_x)
        else:
            self.recon_x = Dense(units=self.dims[0], kernel_initializer=self.init, name='reconstruction')(self.h)
            self.mask_mse = mask_mse(self.x_count, self.x, self.recon_x)

        self.kmeans_loss = tf.reduce_mean(tf.reduce_sum(self.latent_dist2, axis=1))
        self.total_loss = self.pre_loss + self.kmeans_loss * self.alpha
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.pretrain_op = self.optimizer.minimize(self.pre_loss)
        self.train_op = self.optimizer.minimize(self.total_loss)

    def pretrain(self, X, count_X, size_factor, batch_size, pretrain_epoch, gpu_option):
        print("begin the pretraining")
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_option
        config_ = tf.ConfigProto()
        config_.gpu_options.allow_growth = True
        config_.allow_soft_placement = True
        self.sess = tf.Session(config=config_)
        self.sess.run(init)
        self.latent_repre = np.zeros((X.shape[0], self.dims[-1]))
        pre_index = 0
        for ite in range(pretrain_epoch):
            while True:
                if (pre_index + 1) * batch_size > X.shape[0]:
                    last_index = np.array(list(range(pre_index * batch_size, X.shape[0])) + list(
                        range((pre_index + 1) * batch_size - X.shape[0])))
                    _, pre_loss, latent = self.sess.run(
                        [self.pretrain_op, self.pre_loss, self.latent],
                        feed_dict={
                            self.sf_layer: size_factor[last_index],
                            self.x: X[last_index],
                            self.x_count: count_X[last_index]})
                    self.latent_repre[last_index] = latent
                    pre_index = 0
                    break
                else:
                    _, pre_loss, latent = self.sess.run(
                        [self.pretrain_op, self.pre_loss, self.latent],
                        feed_dict={
                            self.sf_layer: size_factor[(pre_index * batch_size):(
                                    (pre_index + 1) * batch_size)],
                            self.x: X[(pre_index * batch_size):(
                                    (pre_index + 1) * batch_size)],
                            self.x_count: count_X[(pre_index * batch_size):(
                                    (pre_index + 1) * batch_size)]})
                    self.latent_repre[(pre_index * batch_size):((pre_index + 1) * batch_size)] = latent
                    pre_index += 1

    def funetrain(self, X, count_X, Y, size_factor, batch_size, funetrain_epoch, update_epoch, error):
        kmeans = KMeans(n_clusters=self.cluster_num, init="k-means++", random_state=888)
        self.latent_repre = np.nan_to_num(self.latent_repre)
        self.kmeans_pred = kmeans.fit_predict(self.latent_repre)
        self.last_pred = np.copy(self.kmeans_pred)
        self.sess.run(tf.assign(self.clusters, kmeans.cluster_centers_))
        print("begin the funetraining")

        fune_index = 0
        for i in range(1, funetrain_epoch + 1):
            if i % update_epoch == 0:
                dist, pre_loss, kmeans_loss, latent_repre = self.sess.run(
                    [self.latent_dist1, self.pre_loss, self.kmeans_loss, self.latent],
                    feed_dict={
                        self.sf_layer: size_factor,
                        self.x: X,
                        self.x_count: count_X})
                self.Y_pred = np.argmin(dist, axis=1)
                if np.sum(self.Y_pred != self.last_pred) / len(self.last_pred) < error:
                    break
                else:
                    self.last_pred = self.Y_pred
            else:
                while True:
                    if (fune_index + 1) * batch_size > X.shape[0]:
                        last_index = np.array(list(range(fune_index * batch_size, X.shape[0])) + list(
                            range((fune_index + 1) * batch_size - X.shape[0])))
                        _, pre_loss, Kmeans_loss = self.sess.run(
                            [self.train_op, self.pre_loss, self.kmeans_loss],
                            feed_dict={
                                self.sf_layer: size_factor[last_index],
                                self.x: X[last_index],
                                self.x_count: count_X[last_index]})
                        fune_index = 0
                        break
                    else:
                        _, pre_loss, Kmeans_loss = self.sess.run(
                            [self.train_op, self.pre_loss, self.kmeans_loss],
                            feed_dict={
                                self.sf_layer: size_factor[(fune_index * batch_size):(
                                        (fune_index + 1) * batch_size)],
                                self.x: X[(fune_index * batch_size):(
                                        (fune_index + 1) * batch_size)],
                                self.x_count: count_X[(fune_index * batch_size):(
                                        (fune_index + 1) * batch_size)]})
                        fune_index += 1

        self.sess.close()
        self.accuracy = np.around(cluster_acc(Y, self.Y_pred), 4)
        self.ARI = np.around(adjusted_rand_score(Y, self.Y_pred), 4)
        self.NMI = np.around(normalized_mutual_info_score(Y, self.Y_pred), 4)
        return self.accuracy, self.ARI, self.NMI





