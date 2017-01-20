/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"
#include "MkldnnFcLayer.h"

// ex fc
#include "paddle/math/SparseMatrix.h"
#include <vector>
#include <algorithm>

using namespace mkldnn;  // NOLINT

namespace paddle {

REGISTER_LAYER(mkldnn_fc, MkldnnFcLayer);

bool MkldnnFcLayer::initDnn(const LayerMap &layerMap,
                           const ParameterMap &parameterMap) {
  // only support 1 input layer by now
  CHECK_EQ(config_.inputs_size(), 1);
  CHECK(inputLayers_.size() == parameters_.size());

  bs_ = 0;
  oc_ = getSize();
  has_spatial_ = false;
  // TODO(TJ): should get this flag from layer proto , default true
  usePaddleFmt_ = true;
  for (size_t i = 0; i < inputLayers_.size(); i++) {
    // Option the parameters
    ic_.push_back(0);
    iw_.push_back(0);
    ih_.push_back(0);
    ow_.push_back(0);
    oh_.push_back(0);
    inputSizeByBS_.push_back(inputLayers_[i]->getSize());  // == ic*ih*iw
    // create a new weight
    size_t height, width;
    if (parameters_[i]->isSparse()) {
      CHECK_LE(parameters_[i]->getSize(), oc_ * inputSizeByBS_[i]);
    } else {
      CHECK_EQ(parameters_[i]->getSize(), oc_ * inputSizeByBS_[i]);
    }
    selfWgtData_.push_back(nullptr);
    selfWgtDiff_.push_back(nullptr);
    if (usePaddleFmt_) {
      height = inputSizeByBS_[i];
      width = oc_;
      selfWgtData_[i] = Matrix::create(width, height, false, false);
      selfWgtDiff_[i] = Matrix::create(width, height, false, false);
      selfWgtData_[i]->zeroMem();
      selfWgtDiff_[i]->zeroMem();
    } else {  // TODO(TJ): never tested this case
      height = oc_;
      width = inputSizeByBS_[i];
    }
    Weight* w = new Weight(height, width, parameters_[i]);
    weights_.emplace_back(w);
  }

  // initialize biases_
  if (biasParameter_.get() != NULL) {
    biases_ = std::unique_ptr<Weight>(new Weight(1, oc_, biasParameter_));
    hasBias_ = true;
  }
  return true;
}

// keep for paddle
void MkldnnFcLayer::prefetch() {
  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    auto* sparseParam =
        dynamic_cast<SparsePrefetchRowCpuMatrix*>(weights_[i]->getW().get());
    if (sparseParam) {
      MatrixPtr input = getInputValue(i);
      sparseParam->addRows(input);
    }
  }
}

void MkldnnFcLayer::clearDataDiff() {
  reserveOutput(bs_, getSize());
}

void MkldnnFcLayer::reshape() {
  // reshape input and output size
  CHECK_NE(inputLayers_.size(), 0UL);
  size_t layerSize = 0;
  oc_ = getSize();
  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    int height = inputLayers_[i]->getOutput().getFrameHeight();
    int width = inputLayers_[i]->getOutput().getFrameWidth();
    if (height > 0 && width > 0) {
      has_spatial_ = true;
      ih_[i] = height;
      iw_[i] = width;
    } else {
      has_spatial_ = false;
      ih_[i] = 1;
      iw_[i] = 1;
    }
    ic_[i] = inputSizeByBS_[i] / (iw_[i] * ih_[i]);
    oh_[i] = 1;
    ow_[i] = 1;
    CHECK(ih_[i] * iw_[i]);
    CHECK(layerSize == 0 || size_t(oh_[i] * ow_[i] * oc_) == layerSize);
    layerSize = oh_[i] * ow_[i] * oc_;
  }

  printInfo();
}

void MkldnnFcLayer::resetDnnFwd(PassType passType) {
  CHECK(bs_ == getInput(0).getBatchSize())
    << "Assert batchsize of input layers are equal";
  mkldnn::engine eg = CpuEngine::Instance().getEngine();
  hasBias_ = (biases_ && biases_->getW()) ? true : false;
  // create dim structure that describes user data.
  memory::dims botDims, wgtDims, biasDims, topDims;
  memory::format botFmt, wgtFmt, biasFmt, topFmt;
  if (!has_spatial_) {
    botDims = {bs_, ic_[0]};
    wgtDims = {oc_, ic_[0]};  // transpose from paddle weight
    // TODO(TJ): when backward done
    // maybe we donot need reley on weight format of paddle
    botFmt = memory::format::nc;
    wgtFmt = memory::format::oi;
  } else {
    botDims = {bs_, ic_[0], ih_[0], iw_[0]};
    wgtDims = {oc_, ic_[0], ih_[0], iw_[0]};
    botFmt = memory::format::nchw;  // perfect fmt is or nChw8c
    wgtFmt = memory::format::oihw;  // perfect fmt is or oIhw8i
  }
  // no matter what inputs
  topDims = {bs_, oc_};
  topFmt = memory::format::nc;
  biasDims = {oc_};
  biasFmt = memory::format::x;

  dataBot_.reset(new MkldnnBuffer());
  dataTop_.reset(new MkldnnBuffer());
  dataWgt_.reset(new MkldnnBuffer());
  if (hasBias_) {
    dataBias_.reset(new MkldnnBuffer());
  }

  // init user memory of bottom, weights and bias
  real *botData = getPrev(0)->getOutputValue()->getData();
  real *topData = getOutputValue()->getData();
  const std::shared_ptr<memory::desc> prvMD = getPrev(0)->getTopDataMD();
  if (prvMD) {
    dataBot_->initUser(botData, *prvMD, eg);
    LOG(INFO) << "use prev format: " << DNN_FORMAT[dataBot_->getUserFmt()];
  } else {
    dataBot_->initUser(botData, botDims, botFmt, eg);
  }

  // create fc desc from internal desc
  std::shared_ptr<inner_product_forward::desc> fwdDesc;
  if (hasBias_) {
    real *biasData = biases_->getW()->getData();
    dataBias_->initUser(biasData, biasDims, biasFmt, eg);
    fwdDesc.reset(new inner_product_forward::desc(
        prop_kind::forward_training,
        prvMD ? dataBot_->getUserMD() : getAnyMD(botDims),
        getAnyMD(wgtDims), getAnyMD(biasDims), getAnyMD(topDims)));
  } else {
    fwdDesc.reset(new inner_product_forward::desc(
        prop_kind::forward_training,
        prvMD ? dataBot_->getUserMD() : getAnyMD(botDims),
        getAnyMD(wgtDims), getAnyMD(topDims)));
  }
  std::shared_ptr<mkldnn::inner_product_forward::primitive_desc> fwdPD;
  fwdPD.reset(new inner_product_forward::primitive_desc(*fwdDesc, eg));
  // init cvt
  if (dataBot_->initIntlCvt(
    fwdPD->src_primitive_desc(), dnnCvtUser2Internal)) {
    LOG(INFO) << "need reorder --- bottom data: "
      << DNN_FORMAT[dataBot_->getUserFmt()]
      << " >>>>> "
      << DNN_FORMAT[dataBot_->getIntlFmt()];
  }
  if (usePaddleFmt_) {
    weights_[0]->getW()->transpose(selfWgtData_[0], false);
    real *wgtData = selfWgtData_[0]->getData();
    dataWgt_->initUser(wgtData, wgtDims, wgtFmt, eg);
    if (dataWgt_->initIntlCvt(
      fwdPD->weights_primitive_desc(), dnnCvtUser2Internal)) {
      LOG(INFO) << "need reorder --- weight data: "
        << DNN_FORMAT[dataWgt_->getUserFmt()]
        << " >>>>> "
        << DNN_FORMAT[dataWgt_->getIntlFmt()];
    }
    if (passType == PASS_TEST) {
      std::vector<primitive> cvtWgt;
      dataWgt_->submitCvt(cvtWgt, wgtData);
      stream(stream::kind::eager).submit(cvtWgt).wait();
    }
  } else {
    // TODO(TJ): initial wgt data with input format
    real *wgtData = weights_[0]->getW()->getData();
    dataWgt_->initUser(wgtData, fwdPD->weights_primitive_desc());
    dataWgt_->initIntlCvt(dataWgt_->getUserPD(), dnnCvtNoNeed);
  }
  if (hasBias_) {
    if (dataBias_->initIntlCvt(
      fwdPD->bias_primitive_desc(), dnnCvtUser2Internal)) {
      LOG(INFO) << "need reorder --- bias data: "
        << DNN_FORMAT[dataBias_->getUserFmt()]
        << " >>>>> "
        << DNN_FORMAT[dataBias_->getIntlFmt()];
    }
  }
  // init top user memory and cvt
  if (setDnnTopDataFmt_) {
    dataTop_->initUser(topData, fwdPD->dst_primitive_desc());
    setTopDataMD(dataTop_->getUserMD());
    LOG(INFO) << "set next format: " << DNN_FORMAT[dataTop_->getUserFmt()];
  } else {
    dataTop_->initUser(topData, topDims, topFmt, eg);
  }
  if (dataTop_->initIntlCvt
    (fwdPD->dst_primitive_desc(), dnnCvtInternal2User)) {
    LOG(INFO) << "need reorder --- top data: "
      << DNN_FORMAT[dataTop_->getIntlFmt()]
      << " >>>>> "
      << DNN_FORMAT[dataTop_->getUserFmt()];
  }
  if (hasBias_) {
    fwd_.reset(new inner_product_forward(*fwdPD,
      *(dataBot_->getIntlMem()), *(dataWgt_->getIntlMem()),
      *(dataBias_->getIntlMem()), *(dataTop_->getIntlMem())));
  } else {
    fwd_.reset(new inner_product_forward(*fwdPD,
      *(dataBot_->getIntlMem()), *(dataWgt_->getIntlMem()),
      *(dataTop_->getIntlMem())));
  }
  LOG(INFO) << "data format flow --- "
    << DNN_FORMAT[dataBot_->getUserFmt()] << " >>> ("
    << DNN_FORMAT[dataBot_->getIntlFmt()] << " >>> "
    << DNN_FORMAT[dataTop_->getIntlFmt()] << ") >>> "
    << DNN_FORMAT[dataTop_->getUserFmt()];
}

void MkldnnFcLayer::resetDnnBwd() {
  /*LOG(INFO) << "init or reset fc backward of layer: " << config_.name();
  */
}

void MkldnnFcLayer::myFwd(PassType passType) {
  /// all sumbit cvt should be clear
  clearAllCvtFlags();
  CHECK(getInput(0).value) << "The input of 'fc' layer must be matrix";
  real *botdata = getPrev(0)->getOutputValue()->getData();
  real *topdata = getOutputValue()->getData();
  std::vector<primitive> pipeline;
  dataBot_->submitCvt(pipeline, botdata);

  if (usePaddleFmt_ && passType == PASS_TRAIN) {
    weights_[0]->getW()->transpose(selfWgtData_[0], false);
    real *wgtdata = selfWgtData_[0]->getData();
    dataWgt_->submitCvt(pipeline, wgtdata);
  }  // else do not need cvt wgt
  pipeline.push_back(*fwd_);

  dataTop_->submitCvt(pipeline, topdata);

  // start forward
  REGISTER_TIMER_INFO("mkldnn_FcFwd", getName().c_str());
  stream(stream::kind::eager).submit(pipeline).wait();
//  LOG(INFO) << "my-" << topdata[0] << "," << topdata[1] << "," << topdata[2];

  // activation
  REGISTER_TIMER_INFO("mkldnn_FcFwAtvTimer", getName().c_str());
  forwardActivation();
}

void MkldnnFcLayer::exFwd(PassType passType) {
  /* malloc memory for the output_ if necessary
  //int batchSize = getInput(0).getBatchSize();
  //int size = getSize();
  //reserveOutput(batchSize, size);
  //MatrixPtr outV = getOutputValue()*/;

  MatrixPtr outV = Matrix::create(bs_, oc_, false, false);
  outV->zeroMem();
  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    auto input = getInput(i);
    CHECK(input.value) << "The input of 'fc' layer must be matrix";
    i == 0 ? outV->mul(input.value, weights_[i]->getW(), 1, 0)
           : outV->mul(input.value, weights_[i]->getW(), 1, 1);
  }
  /* add the bias-vector */
  if (biases_.get() != NULL) {
    outV->addBias(*(biases_->getW()), 1);
  }
/*  real *topdata = outV->getData();
  LOG(INFO) << "ex ------------" << topdata[0] << "," << topdata[1] << "," << topdata[2];*/
  // forwardActivation();
}

void MkldnnFcLayer::submitDnnFwd(PassType passType) {
  myFwd(passType);
//  exFwd(passType);
}

void MkldnnFcLayer::exBwd(const UpdateCallback &callback) {
  /* Do derivation */ {
    REGISTER_TIMER_INFO("BpAvtTimer", getName().c_str());
    backwardActivation();
  }

  if (biases_ && biases_->getWGrad()) {
    REGISTER_TIMER_INFO("BpBiasTimer", getName().c_str());
    biases_->getWGrad()->collectBias(*getOutputGrad(), 1);

    /* Increasing the number of gradient */
    biases_->getParameterPtr()->incUpdate(callback);
  }

  bool syncFlag = hl_get_sync_flag();

  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    /* Calculate the W-gradient for the current layer */
    if (weights_[i]->getWGrad()) {
      MatrixPtr input_T = getInputValue(i)->getTranspose();
      MatrixPtr oGrad = getOutputGrad();
      {
        REGISTER_TIMER_INFO("GradMulTimer", getName().c_str());
        weights_[i]->getWGrad()->mul(input_T, oGrad, 1, 1);
      }
    }

    // If callback does not change value, backprop error asynchronously so that
    // we can do the callback concurrently.
    hl_set_sync_flag(false);

    /* Calculate the input layers error */
    MatrixPtr preGrad = getInputGrad(i);
    if (NULL != preGrad) {
      MatrixPtr weights_T = weights_[i]->getW()->getTranspose();
      REGISTER_TIMER_INFO("BpMulTimer", getName().c_str());
      preGrad->mul(getOutputGrad(), weights_T, 1, 1);
    }
    hl_set_sync_flag(syncFlag);
    {
      REGISTER_TIMER_INFO("WeightUpdate", getName().c_str());
      weights_[i]->getParameterPtr()->incUpdate(callback);
    }
  }
}

void MkldnnFcLayer::submitDnnBwd(const UpdateCallback &callback) {
  exBwd(callback);
  // dnn backward
  /*
  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    // backward weights before data, since may have not botdiff in some layer
    if (weights_[i]->getWGrad()) {
      submitBwdWgts(i, getPrev(i)->getOutputValue(), getOutputGrad());
      // Increasing the number of gradient 
      weights_[i]->getParameterPtr()->incUpdate(callback);
    }
    submitBwdData(i, getOutputGrad(), getPrev(i)->getOutputGrad());
  }
  if (biases_ && biases_->getWGrad()) {
    // Increasing the number of gradient 
    biases_->getParameterPtr()->incUpdate(callback);
  }
  */
}

}  // namespace paddle
