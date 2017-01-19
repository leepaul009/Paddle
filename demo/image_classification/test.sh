#!/bin/bash
# Copyright (c) 2016 Baidu, Inc. All Rights Reserved
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
set -e
config=vgg_16_cifar.py
log=log_test.log
model=cifar_vgg_model/pass-00001/

paddle train \
--config=$config \
--log_period=2 \
--init_model_path=$model \
--job=test \
--use_gpu=0 \
--trainer_count=1 \
2>&1 | tee $log
