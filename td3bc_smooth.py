import tensorflow as tf
from tensorflow.keras.layers import Dense
from copy import deepcopy

class Qnet(tf.keras.Model):
    def __init__(self):
        super(Qnet, self).__init__()
        initializer = tf.keras.initializers.Orthogonal()
        regularizer = tf.keras.regularizers.L2(0.01)
        self.fc1 = Dense(400, activation='relu', kernel_initializer=initializer, kernel_regularizer=regularizer)
        self.fc2 = Dense(300, activation='relu', kernel_initializer=initializer, kernel_regularizer=regularizer)
        self.fc3 = Dense(1, activation=None, kernel_initializer=initializer, kernel_regularizer=regularizer)

    def forward(self, obs_input, action_input):
        inputs = tf.concat([obs_input, action_input], axis=1)
        layer = self.fc1(inputs)
        layer = self.fc2(layer)
        qval = self.fc3(layer)
        return qval

class PolicyNet(tf.keras.Model):
    def __init__(self, a_dim):
        super(PolicyNet, self).__init__()
        self.a_dim = a_dim
        self.fc1 = Dense(400, activation='relu')
        self.fc2 = Dense(300, activation='relu')
        self.fc3 = Dense(a_dim, activation='tanh')

    def forward(self, o):
        layer = self.fc1(o)
        layer = self.fc2(layer)
        action = self.fc3(layer)
        return action


class Agent:
    def __init__(self, o_dim, a_dim, args):

        self.o_dim, self.a_dim = o_dim, a_dim

        # SAC Hyperparameters
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.tau = args.tau
        self.alpha = args.alpha
        self.lamd_s = args.lamd_s
        self.lr = args.lr
        self.pi_noise = args.pi_noise
        self.noise_clip = args.noise_clip
        self.pi_freq = args.pi_freq
        self.std_k = args.std_k

        # Define networks
        self.q1 = Qnet()
        self.q2 = Qnet()
        self.target_q1 = deepcopy(self.q1)
        self.target_q2 = deepcopy(self.q2)
        self.pi = PolicyNet(a_dim)
        self.target_pi = deepcopy(self.pi)

        # Define optmizers
        self.update_step = 0
        self.q1_optimizer = tf.optimizers.Adam(self.lr)
        self.q2_optimizer = tf.optimizers.Adam(self.lr)
        self.pi_optimizer = tf.optimizers.Adam(self.lr)

    @tf.function
    def _pi_action(self, o):
        action = self.pi.forward(o)
        return action

    def get_action(self, o):
        tf_obs = tf.convert_to_tensor(o.reshape([1, -1]))
        action = self._pi_action(tf_obs)[0].numpy()
        return action

    @tf.function
    def _q_update(self, o_batch, a_batch, r_batch, no_batch, done_batch):
        with tf.GradientTape(persistent=True) as tape:
            na = self.target_pi.forward(no_batch)
            na_noise = self.pi_noise*tf.random.normal(tf.shape(na), 0.0, 1.0)
            na_noise = tf.clip_by_value(na_noise,-self.noise_clip,self.noise_clip)
            na += na_noise
            next_q1 = self.target_q1.forward(no_batch, na)
            next_q2 = self.target_q2.forward(no_batch, na)
            min_q = tf.minimum(next_q1, next_q2)
            target_q = r_batch + self.gamma * (1 - done_batch) * min_q
            q1 = self.q1.forward(o_batch, a_batch)
            q2 = self.q2.forward(o_batch, a_batch)
            q1_loss = 0.5 * tf.reduce_mean((q1 - tf.stop_gradient(target_q)) ** 2)
            q2_loss = 0.5 * tf.reduce_mean((q2 - tf.stop_gradient(target_q)) ** 2)
        q1_grad = tape.gradient(q1_loss, self.q1.trainable_weights)
        q2_grad = tape.gradient(q2_loss, self.q2.trainable_weights)
        self.q1_optimizer.apply_gradients(zip(q1_grad, self.q1.trainable_weights))
        self.q2_optimizer.apply_gradients(zip(q2_grad, self.q2.trainable_weights))

    @tf.function
    def _pi_update(self, o_batch, a_batch, no_batch):
        # Update Policy networks
        with tf.GradientTape() as tape:
            std = self.std_k*tf.abs(no_batch - o_batch)
            o_noise = std*tf.random.normal(tf.shape(o_batch), 0.0, 1.0)
            o_spatial = o_batch + o_noise
            spa_policy = self.pi.forward(o_spatial)
            a_policy = self.pi.forward(o_batch)
            q1_actor = self.q1.forward(o_batch, a_policy)
            q2_actor = self.q2.forward(o_batch, a_policy)
            q_actor = tf.minimum(q1_actor, q2_actor)
            coefficient = tf.stop_gradient(self.alpha/tf.reduce_mean(tf.abs(q_actor)))
            pi_loss = tf.reduce_mean(-coefficient * tf.minimum(q1_actor, q2_actor) + \
                                     tf.linalg.norm(a_policy - a_batch, axis=1, keepdims=True) + \
                                     self.lamd_s*tf.reduce_sum((tf.stop_gradient(a_policy) - spa_policy)**2, axis=1, keepdims=True))
        pi_grad = tape.gradient(pi_loss, self.pi.trainable_weights)
        self.pi_optimizer.apply_gradients(zip(pi_grad, self.pi.trainable_weights))

    @tf.function
    def _target_update(self):
        # Update target-networks
        q_params = self.q1.trainable_weights + self.q2.trainable_weights + self.pi.trainable_weights
        target_q_params = self.target_q1.trainable_weights + self.target_q2.trainable_weights + self.target_pi.trainable_weights
        for target_param, param in zip(target_q_params, q_params):
            target_param.assign((1 - self.tau) * target_param + self.tau * param)

    def train(self, o_batch, a_batch, r_batch, no_batch, done_batch):

        o_batch = tf.convert_to_tensor(o_batch)
        a_batch = tf.convert_to_tensor(a_batch)
        r_batch = tf.convert_to_tensor(r_batch)
        no_batch = tf.convert_to_tensor(no_batch)
        done_batch = tf.convert_to_tensor(done_batch)

        # Update networks
        self._q_update(o_batch, a_batch, r_batch, no_batch, done_batch)
        self.update_step += 1
        if self.update_step % self.pi_freq == 0:
            self._pi_update(o_batch, a_batch, no_batch)
            self._target_update()

    def save_policy(self, path, post_fix=""):
        print("[INFO] Saving policy model...")
        self.pi.save_weights(path + "/pi"+post_fix)

    def load_policy(self, path, post_fix=""):
        print("[INFO] Loding policy model...")
        self.pi.load_weights(path + "/pi"+post_fix)