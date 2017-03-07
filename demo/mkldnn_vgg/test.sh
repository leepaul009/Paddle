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

unset OMP_NUM_THREADS MKL_NUM_THREADS
num=$((`nproc`-2))
use_num=$(($num>0?$num:1))
export OMP_NUM_THREADS=$use_num
export MKL_NUM_THREADS=$use_num

config=vgg_19.py
log=log_test.log
model=models_vgg_19/pass-00001/
train_list="data/train.list"
test_list="data/test.list"
use_dummy=1
is_test=1
use_mkldnn_wgt=0
if [ ! -d "data" ]; then
    mkdir -p data
fi
if [ ! -f $train_list ]; then
    if [ $use_dummy -eq 1 ]; then
        echo " " > $train_list
    else
        echo "$train_list does not exist!"
        exit 0
    fi
fi
if [ ! -f $test_list ]; then
    if [ $use_dummy -eq 1 ]; then
        echo " " > $test_list
    else
        echo "$test_list does not exist!"
        exit 0
    fi
fi
if [ $is_test -eq 1 ] ; then
    if [ ! -d $model ]; then
      echo "model does not exist!"
    fi
fi
# run paddle
paddle train \
--config=$config \
--log_period=1 \
--init_model_path=$model \
--job=test \
--use_gpu=0 \
--config_args="use_mkldnn=1,use_mkldnn_wgt=${use_mkldnn_wgt},is_test=${is_test},use_dummy=${use_dummy},batch_size=64" \
2>&1 | tee $log
