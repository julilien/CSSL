### Extension note: ###
#
# Code follows the implementation of FixMatch and co. from the origin repository and replaces necessary parts to
# implement the new baseline as described in our paper.
#
### Copyright note from original code: ###
#
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###

import functools
import os

import numpy as np
import tensorflow.compat.v1 as tf
from absl import app
from absl import flags
from tqdm import trange

from cta.cta_remixmatch import CTAReMixMatch
from libml import data, utils, augment, ctaugment, layers

from tensorflow.python.keras.losses import kullback_leibler_divergence

from libml.utils import EasyDict

FLAGS = flags.FLAGS


class AugmentPoolCTACutOut(augment.AugmentPoolCTA):
    @staticmethod
    def numpy_apply_policies(arglist):
        x, cta, probe = arglist
        if x.ndim == 3:
            assert probe
            policy = cta.policy(probe=True)
            return dict(policy=policy,
                        probe=ctaugment.apply(x, policy),
                        image=x)
        assert not probe
        cutout_policy = lambda: cta.policy(probe=False) + [ctaugment.OP('cutout', (1,))]
        return dict(image=np.stack([x[0]] + [ctaugment.apply(y, cutout_policy()) for y in x[1:]]).astype('f'))


class CSSL(CTAReMixMatch):
    """
    Credal Self-Supervised Learning for image classification embedded into the FixMatch framework.
    """

    AUGMENT_POOL_CLASS = AugmentPoolCTACutOut

    @staticmethod
    def cssl_train(obj, train_nimg, report_nimg):
        """
        Basic training procedure shared by most of the algorithms. Static method that allows for re-usage regardless
        the concrete superclass implementation.

        :param train_nimg: Number of training images
        :param report_nimg: Number of images for which results are reported (epoch-wise)
        """

        if FLAGS.eval_ckpt:
            obj.eval_checkpoint(None)
            return

        batch = FLAGS.batch
        train_labeled = obj.dataset.train_labeled.repeat().shuffle(FLAGS.shuffle).parse().augment()
        train_labeled = train_labeled.batch(batch).prefetch(
            tf.data.experimental.AUTOTUNE).make_one_shot_iterator().get_next()
        train_unlabeled = obj.dataset.train_unlabeled.repeat().shuffle(FLAGS.shuffle).parse().augment()
        train_unlabeled = train_unlabeled.batch(batch * obj.params['uratio']).prefetch(tf.data.experimental.AUTOTUNE)
        train_unlabeled = train_unlabeled.make_one_shot_iterator().get_next()
        scaffold = tf.train.Scaffold(saver=tf.train.Saver(max_to_keep=FLAGS.keep_ckpt,
                                                          pad_step_number=10))

        with tf.Session(config=utils.get_config()) as sess:
            obj.session = sess
            obj.cache_eval()

        with tf.train.MonitoredTrainingSession(
                scaffold=scaffold,
                checkpoint_dir=obj.checkpoint_dir,
                config=utils.get_config(),
                save_checkpoint_steps=FLAGS.save_kimg << 10,
                save_summaries_steps=report_nimg - batch) as train_session:
            obj.session = train_session._tf_sess()
            gen_labeled = obj.gen_labeled_fn(train_labeled)
            gen_unlabeled = obj.gen_unlabeled_fn(train_unlabeled)
            obj.tmp.step = obj.session.run(obj.step)
            while obj.tmp.step < train_nimg:
                loop = trange(obj.tmp.step % report_nimg, report_nimg, batch,
                              leave=False, unit='img', unit_scale=batch,
                              desc='Epoch %d/%d' % (1 + (obj.tmp.step // report_nimg), train_nimg // report_nimg))
                for _ in loop:
                    obj.train_step(train_session, gen_labeled, gen_unlabeled)
                    while obj.tmp.print_queue:
                        loop.write(obj.tmp.print_queue.pop(0))
            while obj.tmp.print_queue:
                print(obj.tmp.print_queue.pop(0))

    def train(self, train_nimg, report_nimg):
        CSSL.cssl_train(self, train_nimg, report_nimg)

    @staticmethod
    def guess_label(p_model_y, p_data, p_model, **kwargs):
        """
        Distribution alignment as discussed in the paper's Section 3.4.

        :param p_model_y: Model prediction
        :param p_data: Data prior
        :param p_model: Historical predictions
        :param kwargs: Ignored (for compatibility reasons)
        :return: Returns the aligned prediction and the original prediction as dictionary
        """
        del kwargs
        p_ratio = (1e-6 + p_data) / (1e-6 + p_model)
        p_target = p_model_y * p_ratio
        p_target /= tf.reduce_sum(p_target, axis=1, keep_dims=True)

        return EasyDict(p_target=p_target, p_model=p_model_y)

    @staticmethod
    def determine_imprecisiation(guess, alpha_bound):
        guess_p_target = tf.stop_gradient(guess.p_target)
        max_prob = tf.math.reduce_max(guess_p_target, axis=-1)
        relax_alpha = tf.ones_like(max_prob) - max_prob
        relax_alpha = tf.clip_by_value(relax_alpha, alpha_bound, 1.)
        tf.summary.scalar('monitors/alpha', tf.reduce_mean(relax_alpha))
        return relax_alpha

    @staticmethod
    def determine_label_relaxation_loss(logits_strong, pseudo_labels, relax_alpha):
        pred_softmax = tf.nn.softmax(logits_strong)
        sum_y_hat_prime = tf.reduce_sum((1. - pseudo_labels) * pred_softmax, axis=-1)
        y_pred_hat = tf.expand_dims(relax_alpha, axis=-1) * pred_softmax / (
                tf.expand_dims(sum_y_hat_prime, axis=-1) + 1e-7)
        y_true_credal = tf.where(tf.greater(pseudo_labels, 0.1),
                                 tf.ones_like(pseudo_labels) - tf.expand_dims(relax_alpha, axis=-1), y_pred_hat)
        divergence = kullback_leibler_divergence(y_true_credal, pred_softmax)
        preds = tf.reduce_sum(pred_softmax * pseudo_labels, axis=-1)
        return tf.where(tf.greater_equal(preds, 1. - relax_alpha), tf.zeros_like(divergence),
                        divergence)

    @staticmethod
    def cssl_model(obj, batch, lr, wd, wu, confidence, alpha_bound, uratio, ema=0.999, dbuf=128, **kwargs):
        hwc = [obj.dataset.height, obj.dataset.width, obj.dataset.colors]
        xt_in = tf.placeholder(tf.float32, [batch] + hwc, 'xt')  # Training labeled
        x_in = tf.placeholder(tf.float32, [None] + hwc, 'x')  # Eval images
        y_in = tf.placeholder(tf.float32, [batch * uratio, 2] + hwc, 'y')  # Training unlabeled (weak, strong)
        l_in = tf.placeholder(tf.int32, [batch], 'labels')  # Labels

        lrate = tf.clip_by_value(tf.to_float(obj.step) / (FLAGS.train_kimg << 10), 0, 1)
        lr *= tf.cos(lrate * (7 * np.pi) / (2 * 8))
        tf.summary.scalar('monitors/lr', lr)

        # Compute logits for xt_in and y_in
        classifier = lambda x, **kw: obj.classifier(x, **kw, **kwargs).logits
        skip_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        x = utils.interleave(tf.concat([xt_in, y_in[:, 0], y_in[:, 1]], 0), 2 * uratio + 1)
        logits = utils.para_cat(lambda x: classifier(x, training=True), x)
        logits = utils.de_interleave(logits, 2 * uratio + 1)
        post_ops = [v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if v not in skip_ops]
        logits_x = logits[:batch]
        logits_weak, logits_strong = tf.split(logits[batch:], 2)
        del logits, skip_ops

        # Labeled cross-entropy
        loss_xe = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=l_in, logits=logits_x)
        loss_xe = tf.reduce_mean(loss_xe)
        tf.summary.scalar('losses/xe', loss_xe)

        # Pseudo-labels generation: First, determine reference class, followed by the imprecisiation degree alpha
        orig_pseudo_labels = tf.stop_gradient(tf.nn.softmax(logits_weak))
        pseudo_labels = tf.one_hot(tf.argmax(orig_pseudo_labels, axis=1), depth=tf.shape(orig_pseudo_labels)[1])

        # Maintain alignment moving averages
        p_model = layers.PMovingAverage('p_model', obj.nclass, dbuf)
        p_target = layers.PMovingAverage('p_target', obj.nclass, dbuf)
        p_data = layers.PData(obj.dataset)
        p_data_tf = p_data()
        p_model_tf = p_model()

        guess = CSSL.guess_label(orig_pseudo_labels, p_data_tf, p_model_tf)

        # Determine instance-wise imprecisiation (relaxation) alpha
        relax_alpha = CSSL.determine_imprecisiation(guess, alpha_bound)

        # Calculate label relaxation loss
        loss_xeu = CSSL.determine_label_relaxation_loss(logits_strong, pseudo_labels, relax_alpha)
        # Optionally, filter out instances by confidence
        pseudo_mask = tf.to_float(tf.reduce_max(orig_pseudo_labels, axis=1) >= confidence)
        tf.summary.scalar('monitors/mask', tf.reduce_mean(pseudo_mask))
        loss_xeu = tf.reduce_mean(loss_xeu * pseudo_mask)
        tf.summary.scalar('losses/xeu', loss_xeu)
        obj.distribution_summary(p_data(), p_model(), p_target())

        # L2 regularization
        loss_wd = sum(tf.nn.l2_loss(v) for v in utils.model_vars('classify') if 'kernel' in v.name)
        tf.summary.scalar('losses/wd', loss_wd)

        # Apply EMA
        ema = tf.train.ExponentialMovingAverage(decay=ema)
        ema_op = ema.apply(utils.model_vars())
        ema_getter = functools.partial(utils.getter_ema, ema)
        post_ops.extend([ema_op,
                         p_model.update(guess.p_model),
                         p_target.update(guess.p_target)])

        train_op = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True).minimize(
            loss_xe + wu * loss_xeu + wd * loss_wd, colocate_gradients_with_ops=True)
        with tf.control_dependencies([train_op]):
            train_op = tf.group(*post_ops)

        return utils.EasyDict(
            xt=xt_in, x=x_in, y=y_in, label=l_in, train_op=train_op,
            classify_raw=tf.nn.softmax(classifier(x_in, training=False)),  # No EMA, for debugging.
            classify_op=tf.nn.softmax(classifier(x_in, getter=ema_getter, training=False)))

    def model(self, batch, lr, wd, wu, confidence, alpha_bound, uratio, ema=0.999, dbuf=128, **kwargs):
        return CSSL.cssl_model(self, batch, lr, wd, wu, confidence, alpha_bound, uratio, ema, dbuf, **kwargs)


def main(argv):
    utils.setup_main()
    del argv
    dataset = data.PAIR_DATASETS()[FLAGS.dataset]()
    log_width = utils.ilog2(dataset.width)
    model = CSSL(
        os.path.join(FLAGS.train_dir, dataset.name, CSSL.cta_name()),
        dataset,
        lr=FLAGS.lr,
        wd=FLAGS.wd,
        arch=FLAGS.arch,
        batch=FLAGS.batch,
        nclass=dataset.nclass,
        wu=FLAGS.wu,
        confidence=FLAGS.confidence,
        alpha_bound=FLAGS.alpha_bound,
        uratio=FLAGS.uratio,
        scales=FLAGS.scales or (log_width - 2),
        filters=FLAGS.filters,
        repeat=FLAGS.repeat)

    model.train(FLAGS.train_kimg << 10, FLAGS.report_kimg << 10)


if __name__ == '__main__':
    utils.setup_tf()
    flags.DEFINE_float('confidence', 0.0, 'Confidence threshold.')
    flags.DEFINE_float('alpha_bound', 0.0, 'Lower bound for alpha values.')
    flags.DEFINE_float('wd', 0.0005, 'Weight decay.')
    flags.DEFINE_float('wu', 1, 'Pseudo label loss weight.')
    flags.DEFINE_integer('filters', 32, 'Filter size of convolutions.')
    flags.DEFINE_integer('repeat', 4, 'Number of residual layers per stage.')
    flags.DEFINE_integer('scales', 0, 'Number of 2x2 downscalings in the classifier.')
    flags.DEFINE_integer('uratio', 7, 'Unlabeled batch size ratio.')
    FLAGS.set_default('augment', 'd.d.d')
    FLAGS.set_default('dataset', 'cifar10.3@250-1')
    FLAGS.set_default('batch', 64)
    FLAGS.set_default('lr', 0.03)
    FLAGS.set_default('train_kimg', 1 << 16)
    app.run(main)
