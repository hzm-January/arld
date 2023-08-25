# ==============================================================================
# Copyright (c) 2022 The PersFormer Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
from config import persformer_openlane, persformer_once, persformer_apollo, arld_apollo
from utils.utils import *
# from experiments.ddp import *
from experiments.runner import *


def main():
    parser = define_args() # args in utils.py
    args = parser.parse_args()

    # specify dataset and model config
    arld_apollo.config(args)
    # persformer_once.config(args)
    # persformer_openlane.config(args)

    # initialize distributed data parallel set
    # ddp_init(args)

    # define runner to begin training or evaluation
    runner = Runner(args)
    # args.evaluate = True
    if not args.evaluate:
        runner.train()
    else:
        runner.eval()

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
    main()