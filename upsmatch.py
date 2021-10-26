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

from cta.cta_remixmatch import CTAReMixMatch

from cssl import CSSL
from libml import data, utils, augment, ctaugment
from libml.models import MultiModel

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


class UPSMatch(CTAReMixMatch):
    AUGMENT_POOL_CLASS = AugmentPoolCTACutOut

    def train(self, train_nimg, report_nimg):
        CSSL.cssl_train(self, train_nimg, report_nimg)

    def model(self, batch, lr, wd, wu, conf_p, uncertainty_p, uratio, ema=0.999, **kwargs):
        hwc = [self.dataset.height, self.dataset.width, self.dataset.colors]
        xt_in = tf.placeholder(tf.float32, [batch] + hwc, 'xt')  # Training labeled
        x_in = tf.placeholder(tf.float32, [None] + hwc, 'x')  # Eval images
        y_in = tf.placeholder(tf.float32, [batch * uratio, 2] + hwc, 'y')  # Training unlabeled (weak, strong)
        l_in = tf.placeholder(tf.int32, [batch], 'labels')  # Labels

        lrate = tf.clip_by_value(tf.to_float(self.step) / (FLAGS.train_kimg << 10), 0, 1)
        lr *= tf.cos(lrate * (7 * np.pi) / (2 * 8))
        tf.summary.scalar('monitors/lr', lr)

        # Compute logits for xt_in and y_in
        classifier = lambda x, **kw: self.classifier(x, **kw, **kwargs).logits
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

        # Pseudo-label cross entropy for unlabeled data
        pseudo_labels = tf.stop_gradient(tf.nn.softmax(logits_weak))

        classifier_do = lambda x, **kw: self.classifier(x, **kw, **kwargs).logits

        # Perform MCDropout uncertainty quantification
        inner_logits = utils.para_cat(lambda x: classifier_do(x, prob=0.7, training=False), x)
        inner_logits = utils.de_interleave(inner_logits, 2 * uratio + 1)
        inner_logits_weak, _ = tf.split(inner_logits[batch:], 2)
        del inner_logits

        raw_predictions = []
        for _ in range(8):
            inner_logits = utils.para_cat(lambda x: classifier_do(x, prob=0.7, training=False), x)
            inner_logits = utils.de_interleave(inner_logits, 2 * uratio + 1)
            inner_logits_weak, _ = tf.split(inner_logits[batch:], 2)
            del inner_logits

            raw_predictions.append(tf.expand_dims(tf.stop_gradient(tf.nn.softmax(inner_logits_weak)), axis=0))

        output_probs = tf.concat(raw_predictions, axis=0)
        output_probs_std = tf.math.reduce_std(output_probs, axis=0)
        output_probs_mean = tf.reduce_mean(output_probs, axis=0)
        output_probs_argmax = tf.math.argmax(output_probs_mean, axis=-1)
        uncertainty = tf.gather(output_probs_std, output_probs_argmax, axis=1, batch_dims=1)

        # Calculate the unlabeled loss
        loss_xeu = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(pseudo_labels, axis=1),
                                                                  logits=logits_strong)
        # Mask out instances that to not meet the confidence or uncertainty threshold
        pseudo_mask = tf.to_float(tf.logical_and(tf.reduce_max(pseudo_labels, axis=1) >= conf_p,
                                                 uncertainty <= uncertainty_p))

        tf.summary.scalar('monitors/mask', tf.reduce_mean(pseudo_mask))
        loss_xeu = tf.reduce_mean(loss_xeu * pseudo_mask)
        tf.summary.scalar('losses/xeu', loss_xeu)

        # L2 regularization
        loss_wd = sum(tf.nn.l2_loss(v) for v in utils.model_vars('classify') if 'kernel' in v.name)
        tf.summary.scalar('losses/wd', loss_wd)

        ema = tf.train.ExponentialMovingAverage(decay=ema)
        ema_op = ema.apply(utils.model_vars())
        ema_getter = functools.partial(utils.getter_ema, ema)
        post_ops.append(ema_op)

        train_op = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True).minimize(
            loss_xe + wu * loss_xeu + wd * loss_wd, colocate_gradients_with_ops=True)
        with tf.control_dependencies([train_op]):
            train_op = tf.group(*post_ops)

        return utils.EasyDict(
            xt=xt_in, x=x_in, y=y_in, label=l_in, train_op=train_op,
            classify_raw=tf.nn.softmax(classifier(x_in, training=False)),
            classify_op=tf.nn.softmax(classifier(x_in, getter=ema_getter, training=False)))


def main(argv):
    utils.setup_main()
    del argv
    dataset = data.PAIR_DATASETS()[FLAGS.dataset]()
    log_width = utils.ilog2(dataset.width)
    model = UPSMatch(
        os.path.join(FLAGS.train_dir, dataset.name, UPSMatch.cta_name()),
        dataset,
        lr=FLAGS.lr,
        wd=FLAGS.wd,
        arch=MultiModel.MODEL_MCDROPRESNET,
        batch=FLAGS.batch,
        nclass=dataset.nclass,
        wu=FLAGS.wu,
        conf_p=FLAGS.conf_p,
        uncertainty_p=FLAGS.uncertainty_p,
        uratio=FLAGS.uratio,
        scales=FLAGS.scales or (log_width - 2),
        filters=FLAGS.filters,
        repeat=FLAGS.repeat)
    model.train(FLAGS.train_kimg << 10, FLAGS.report_kimg << 10)


if __name__ == '__main__':
    utils.setup_tf()
    flags.DEFINE_float('conf_p', 0.7, 'Confidence threshold (lower bound).')
    flags.DEFINE_float('uncertainty_p', 0.05, 'Positive uncertainty threshold (upper bound).')
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
