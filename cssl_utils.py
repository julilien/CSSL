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

import numpy as np


def calculate_ece(predicted, labels, num_bins=15):
    predicted_cls = predicted.argmax(1)

    interval_step = 1. / num_bins
    bin_sizes = np.zeros(num_bins)

    confidences = np.max(predicted, axis=-1)
    bin_indices = np.minimum(confidences // interval_step, num_bins - 1).astype(np.int32)
    bin_accs = np.zeros(num_bins)
    bin_confs = np.zeros(num_bins)

    for i in range(predicted.shape[0]):
        bin_idx = bin_indices[i]
        bin_sizes[bin_idx] += 1

        if predicted_cls[i] == labels[i]:
            bin_accs[bin_idx] += 1
        bin_confs[bin_idx] += confidences[i]

    for i in range(num_bins):
        bin_accs[i] /= max(bin_sizes[i], 1)
        bin_confs[i] /= max(bin_sizes[i], 1)

    ece = 0.
    for i in range(num_bins):
        ece += (bin_sizes[i] / predicted.shape[0]) * np.abs(bin_accs[i] - bin_confs[i])

    return ece
