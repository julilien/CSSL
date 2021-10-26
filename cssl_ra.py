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
import os

import tensorflow.compat.v1 as tf
from absl import app
from absl import flags

from cssl import CSSL
from libml import data, utils, models, layers

FLAGS = flags.FLAGS


class CSSLRA(models.MultiModel):
    """
    Credal Self-Supervised Learning using RandAugment (rather than CTAugment as in the default version). Reuses most of
    the CTAugment implementation.
    """

    def distribution_summary(self, p_data, p_model, p_target=None):
        def kl(p, q):
            p /= tf.reduce_sum(p)
            q /= tf.reduce_sum(q)
            return -tf.reduce_sum(p * tf.log(q / p))

        tf.summary.scalar('metrics/kld', kl(p_data, p_model))
        if p_target is not None:
            tf.summary.scalar('metrics/kld_target', kl(p_data, p_target))

        for i in range(self.nclass):
            tf.summary.scalar('matching/class%d_ratio' % i, p_model[i] / p_data[i])
        for i in range(self.nclass):
            tf.summary.scalar('matching/val%d' % i, p_model[i])

    def train(self, train_nimg, report_nimg):
        CSSL.cssl_train(self, train_nimg, report_nimg)

    def model(self, batch, lr, wd, wu, confidence, alpha_bound, uratio, ema=0.999, dbuf=128, **kwargs):
        return CSSL.cssl_model(self, batch, lr, wd, wu, confidence, alpha_bound, uratio, ema, dbuf, **kwargs)


def main(argv):
    utils.setup_main()
    del argv
    dataset = data.PAIR_DATASETS()[FLAGS.dataset]()
    log_width = utils.ilog2(dataset.width)
    model = CSSLRA(
        os.path.join(FLAGS.train_dir, dataset.name),
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
    FLAGS.set_default('augment', 'd.d.rac')
    FLAGS.set_default('dataset', 'cifar10.3@250-1')
    FLAGS.set_default('batch', 64)
    FLAGS.set_default('lr', 0.03)
    FLAGS.set_default('train_kimg', 1 << 16)
    app.run(main)
