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

void MkldnnFcLayer::resetDnn(PassType passType) {
  CHECK(bs_ == getInput(0).getBatchSize())
    << "Assert batchsize of input layers are equal";
  mkldnn::engine eg = CpuEngine::Instance().getEngine();
  prop_kind pk = prop_kind::forward;
  bool hasBias = (biases_ && biases_->getW());
  // create dim structure that describes user data.
  memory::dims botDims, wgtDims, biasDims, topDims;
  memory::format botFmt, wgtFmt, biasFmt, topFmt;
  if (!has_spatial_) {
    botDims = {bs_, ic_[0]};
    wgtDims = {oc_, ic_[0]};  // transpose from paddle weight
    botFmt = memory::format::nc;
    wgtFmt = memory::format::oi;
  } else {
    botDims = {bs_, ic_[0], ih_[0], iw_[0]};
    wgtDims = {oc_, ic_[0], ih_[0], iw_[0]};
    botFmt = memory::format::nchw;  // perfect fmt is or nChw8c
    wgtFmt = memory::format::oihw;  // perfect fmt is or oIhw8i
  }
  topDims = {bs_, oc_};
  topFmt = memory::format::nc;
  biasDims = {oc_};
  biasFmt = memory::format::x;
  hasCvtTopData_ = false;
  hasCvtTopDiff_ = false;
  hasCvtBiasData_ = false;
  hasCvtBiasDiff_ = false;
  // 1. create mkldnn buffer, only have one output and bias buffer
  dataTop_.reset(new MkldnnBuffer());
  if (hasBias) {
    dataBias_.reset(new MkldnnBuffer());
  }
  if (passType != PASS_TEST) {  // for backward
    diffTop_.reset(new MkldnnBuffer());
    if (hasBias) {
      CHECK(biases_->getWGrad()) << "assert have grad";
      diffBias_.reset(new MkldnnBuffer());
    }
  }
  // 2. init user top and bias
  real *topData = getOutputValue()->getData();
  dataTop_->initUser(topData, topDims, topFmt, eg);
  if (hasBias) {
    real *biasData = biases_->getW()->getData();
    dataBias_->initUser(biasData, biasDims, biasFmt, eg);
  }
  if (passType != PASS_TEST) {
    real *topDiff = getOutputGrad()->getData();
    diffTop_->initUser(topDiff, topDims, topFmt, eg);
    if (hasBias) {
      real* biasDiff = biases_->getWGrad()->getData();
      diffBias_->initUser(biasDiff, biasDims, biasFmt, eg);
    }
    // use internal top diff if use dnn input
    const std::shared_ptr<mkldnn::memory::desc> inputDiffMD = getTopDiffMD();
    if (inputDiffMD) {
      diffTop_->resetUser(topDiff, *inputDiffMD, eg);
      LOG(INFO) << "keep prev diff fmt: " << DNN_FMTS[diffTop_->getUserFmt()];
    }
  }
  // TODO(TJ): only care about i==0 yet
  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    CHECK(bs_ == getInput(i).getBatchSize()) << "batchsize should equal";
    /// 1. create buffer, could be vector later
    dataBot_.reset(new MkldnnBuffer());
    dataWgt_.reset(new MkldnnBuffer());
    // 2. init user memory of bottom, weights and bias
    real *botData = getPrev(i)->getOutputValue()->getData();
    real *wgtData = usePaddleFmt_ ? selfWgtData_[i]->getData()
        : weights_[i]->getW()->getData();
    dataBot_->initUser(botData, botDims, botFmt, eg);
    dataWgt_->initUser(wgtData, wgtDims, wgtFmt, eg);
    // 3. create fc desc
    std::shared_ptr<inner_product_forward::desc> fwdDesc;
    std::shared_ptr<mkldnn::inner_product_forward::primitive_desc> fwdPD;
    const std::shared_ptr<memory::desc> prvMD = getPrev(i)->getTopDataMD();
    if (prvMD) {
      dataBot_->resetUser(botData, *prvMD, eg);
      LOG(INFO) << "use prev format: " << DNN_FMTS[dataBot_->getUserFmt()];
    }
    if (hasBias) {
      fwdDesc.reset(new inner_product_forward::desc(pk,
          prvMD ? dataBot_->getUserMD() : getAnyMD(botDims),
          getAnyMD(wgtDims), getAnyMD(biasDims), getAnyMD(topDims)));
    } else {
      fwdDesc.reset(new inner_product_forward::desc(pk,
          prvMD ? dataBot_->getUserMD() : getAnyMD(botDims),
          getAnyMD(wgtDims), getAnyMD(topDims)));
    }
    fwdPD.reset(new inner_product_forward::primitive_desc(*fwdDesc, eg));
    // 4. init cvt
    dataBot_->initCvt(fwdPD->src_primitive_desc(), dnnCvtUser2Intl);
    if (usePaddleFmt_) {
      if (dataWgt_->initCvt(
        fwdPD->weights_primitive_desc(), dnnCvtUser2Intl)) {
        LOG(INFO) << "need reorder --- weight data: "
          << DNN_FMTS[dataWgt_->getUserFmt()]
          << " >>> "
          << DNN_FMTS[dataWgt_->getIntlFmt()];
      }
      if (passType == PASS_TEST) {
        weights_[i]->getW()->transpose(selfWgtData_[i], false);
        std::vector<primitive> cvtWgt;
        dataWgt_->submitCvt(cvtWgt, wgtData);
        stream(stream::kind::eager).submit(cvtWgt).wait();
      }
    } else {
      // TODO(TJ): never tested
      wgtData = weights_[i]->getW()->getData();
      dataWgt_->resetUser(wgtData, fwdPD->weights_primitive_desc());
      dataWgt_->initCvt(dataWgt_->getUserPD(), dnnCvtNoNeed);
    }
    if (hasBias) {
      // only cvt once
      if (!hasCvtBiasData_) {
        hasCvtBiasData_ = true;
        CHECK(dataBias_->getUserPD() == fwdPD->bias_primitive_desc())
          << "should always be format::x, or changed in new mkldnn version";
        dataBias_->initCvt(dataBias_->getUserPD(), dnnCvtNoNeed);
      } else {
        CHECK(dataBias_->getIntlPD() == fwdPD->bias_primitive_desc())
          << "all bias formats should equal";
      }
    }
    // cvt topdata buffer only once, set dnn MemDesc if next is also mkldnn
    if (!hasCvtTopData_) {
      hasCvtTopData_ = true;
      if (setDnnTopDataFmt_) {
        dataTop_->resetUser(topData, fwdPD->dst_primitive_desc());
        setTopDataMD(dataTop_->getUserMD());
        LOG(INFO) << "set next format: " << DNN_FMTS[dataTop_->getUserFmt()];
      }
      dataTop_->initCvt(fwdPD->dst_primitive_desc(), dnnCvtIntl2User);
    } else {
      CHECK(dataTop_->getIntlPD() == fwdPD->dst_primitive_desc())
        << "all output formats should equal";
    }
    // 5. create fwd handle
    if (hasBias) {
      fwd_.reset(new inner_product_forward(*fwdPD,
        *(dataBot_->getIntlMem()), *(dataWgt_->getIntlMem()),
        *(dataBias_->getIntlMem()), *(dataTop_->getIntlMem())));
    } else {
      fwd_.reset(new inner_product_forward(*fwdPD,
        *(dataBot_->getIntlMem()), *(dataWgt_->getIntlMem()),
        *(dataTop_->getIntlMem())));
    }
    LOG(INFO) << "data format flow --- "
      << DNN_FMTS[dataBot_->getUserFmt()] << " >>> ("
      << DNN_FMTS[dataBot_->getIntlFmt()] << " >>> "
      << DNN_FMTS[dataTop_->getIntlFmt()] << ") >>> "
      << DNN_FMTS[dataTop_->getUserFmt()];

    /// init mkldnn backward ***************************************************
    if (passType == PASS_TEST)
      continue;
    if (hasBias) {
      CHECK(biases_->getWGrad()) << "assert has bias grad since has bias data";
    }
    // 1. create mkldnn buffer and init user
    CHECK(weights_[i]->getWGrad()) << "should have weight anyway";
    diffWgt_.reset(new MkldnnBuffer());
    real *wgtDiff = usePaddleFmt_ ? selfWgtDiff_[i]->getData()
      : weights_[i]->getWGrad()->getData();
    diffWgt_->initUser(wgtDiff, wgtDims, wgtFmt, eg);
    // 2. prepare backward weight and bias
    std::shared_ptr<inner_product_forward::desc> bwdFwdDesc;
    std::shared_ptr<inner_product_forward::primitive_desc> bwdFwdPD;
    std::shared_ptr<inner_product_backward_weights::desc> bwdWgtDesc;
    std::shared_ptr<inner_product_backward_weights::primitive_desc> bwdWgtPD;
    bwdFwdDesc.reset(new inner_product_forward::desc(pk,
      dataBot_->getIntlMD(), dataWgt_->getIntlMD(), dataTop_->getIntlMD()));
    bwdFwdPD.reset(new inner_product_forward::primitive_desc(
      *bwdFwdDesc, eg));
    CHECK(hasBias) << "only support with bias in mkldnn";
    bwdWgtDesc.reset(new inner_product_backward_weights::desc(
      dataBot_->getIntlMD(), dataWgt_->getIntlMD(),
      dataBias_->getIntlMD(), dataTop_->getIntlMD()));
    bwdWgtPD.reset(new inner_product_backward_weights::primitive_desc(
      *bwdWgtDesc, eg, *bwdFwdPD));
    CHECK(dataBot_->getIntlPD() == bwdWgtPD->src_primitive_desc());
    CHECK(dataWgt_->getIntlPD() == bwdWgtPD->diff_weights_primitive_desc());
    CHECK(dataBias_->getIntlPD() == bwdWgtPD->diff_bias_primitive_desc());
    // 3. init conversion    
    if (usePaddleFmt_) {
      if (diffWgt_->initCvt(dataWgt_->getIntlPD(), dnnCvtIntl2User)) {
        LOG(INFO) << "need reorder --- weight diff: "
          << DNN_FMTS[diffWgt_->getIntlFmt()]
          << " >>>>> "
          << DNN_FMTS[diffWgt_->getUserFmt()];
      }
    } else {
      wgtDiff = weights_[i]->getWGrad()->getData();
      diffWgt_->resetUser(wgtDiff, dataWgt_->getIntlPD());
      diffWgt_->initCvt(diffWgt_->getUserPD(), dnnCvtNoNeed);
    }
    if (hasBias) {
      if (!hasCvtBiasDiff_) {
        hasCvtBiasDiff_ = true;
        CHECK(diffBias_->getUserPD() == bwdWgtPD->diff_bias_primitive_desc())
          << "should always be format::x, or changed in new mkldnn version";
        diffBias_->initCvt(diffBias_->getUserPD(), dnnCvtNoNeed);
      } else {
        CHECK(diffBias_->getIntlPD() == bwdWgtPD->diff_bias_primitive_desc())
          << "all bias formats should equal";
      }
    }
    if (!hasCvtTopDiff_) {
      hasCvtTopDiff_ = true;
      diffTop_->initCvt(bwdWgtPD->diff_dst_primitive_desc(), dnnCvtUser2Intl);
    } else {
      CHECK(diffTop_->getIntlPD() == bwdWgtPD->diff_dst_primitive_desc())
        << "all topdiff formats should equal";
    }
    // 4. bias backward can only be executed in weight backward with MKL-DNN
    bwdWgt_.reset(new inner_product_backward_weights(*bwdWgtPD,
      *(dataBot_->getIntlMem()), *(diffTop_->getIntlMem()),
      *(diffWgt_->getIntlMem()), *(diffBias_->getIntlMem())));

    // then prepare backward data ----------------------------------------------
    LayerPtr prevLayer = getPrev(i);
    if (NULL == prevLayer->getOutputGrad()) {
      continue; // data layer has not diff
    }
    // 1. create buffer and init user
    real* botDiff = prevLayer->getOutputGrad()->getData();
    diffBot_.reset(new MkldnnBuffer());
    diffBot_->initUser(botDiff, botDims, botFmt, eg);
    // 2. init backward data primitive desc
    std::shared_ptr<inner_product_backward_data::desc> bwdDataDesc;
    std::shared_ptr<inner_product_backward_data::primitive_desc> bwdDataPD;
    bwdDataDesc.reset(new inner_product_backward_data::desc(
      dataBot_->getIntlMD(), dataWgt_->getIntlMD(), dataTop_->getIntlMD()));
    bwdDataPD.reset(new inner_product_backward_data::primitive_desc(
      *bwdDataDesc, eg, *bwdFwdPD));
    CHECK(dataWgt_->getIntlPD() == bwdDataPD->weights_primitive_desc());
    CHECK(diffTop_->getIntlPD() == bwdDataPD->diff_dst_primitive_desc());
    // 3. init conversion
    if (setDnnBotDiffFmt_[i]) {
      diffBot_->resetUser(botDiff, bwdDataPD->diff_src_primitive_desc());
      prevLayer->setTopDiffMD(diffBot_->getUserMD());
    }
    diffBot_->initCvt(dataBot_->getIntlPD(), dnnCvtIntl2User);
    // 4. create bwd data handle
    bwdData_.reset(new inner_product_backward_data(
      *bwdDataPD, *(diffTop_->getIntlMem()),
      *(dataWgt_->getIntlMem()), *(diffBot_->getIntlMem())));
    LOG(INFO) << "diff format flow --- "
      << DNN_FMTS[diffBot_->getUserFmt()] << " <<< ("
      << DNN_FMTS[diffBot_->getIntlFmt()] << " <<< "
      << DNN_FMTS[diffTop_->getIntlFmt()] << ") <<< "
      << DNN_FMTS[diffTop_->getUserFmt()];
  }
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

}

void MkldnnFcLayer::submitDnnFwd(PassType passType) {
  real *topdata = getOutputValue()->getData();
  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    CHECK(getInput(i).value) << "The input of 'fc' layer must be matrix";
    real *botdata = getPrev(0)->getOutputValue()->getData();
    std::vector<primitive> pipeline;
    dataBot_->submitCvt(pipeline, botdata);
    if (usePaddleFmt_ && passType != PASS_TEST) {
      weights_[i]->getW()->transpose(selfWgtData_[i], false);
      real *wgtdata = selfWgtData_[i]->getData();
      dataWgt_->submitCvt(pipeline, wgtdata);
    }  // else do not need cvt wgt
    pipeline.push_back(*fwd_);
    dataTop_->submitCvt(pipeline, topdata);
    stream(stream::kind::eager).submit(pipeline).wait();
    //  LOG(INFO) << "my-" << topdata[0] << "," << topdata[1] << "," << topdata[2];
  }

//  exFwd(passType);
// activation
  forwardActivation();
}

void MkldnnFcLayer::exBwd(const UpdateCallback &callback) {
  real* biasdiff = biases_->getWGrad()->getData();
  
  real* wgtdiff = weights_[0]->getWGrad()->getData();

  LOG(INFO) << "--------------------ex before wgt, bias diff: "<< wgtdiff[0] << "," << wgtdiff[3] << ","<<biasdiff[0]<< ","<<biasdiff[2];
      
  if (biases_ && biases_->getWGrad()) {
    biases_->getWGrad()->collectBias(*getOutputGrad(), 1);

    /* Increasing the number of gradient */
//    biases_->getParameterPtr()->incUpdate(callback);
  }


  bool syncFlag = hl_get_sync_flag();

  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    /* Calculate the W-gradient for the current layer */
    if (weights_[i]->getWGrad()) {
      MatrixPtr input_T = getInputValue(i)->getTranspose();
      MatrixPtr oGrad = getOutputGrad();
      weights_[i]->getWGrad()->mul(input_T, oGrad, 1, 1);
      LOG(INFO) << "--------------------ex wgt, bias diff: "<< wgtdiff[0] << "," << wgtdiff[3] << ","<<biasdiff[0]<< ","<<biasdiff[2];
            
    }

    // If callback does not change value, backprop error asynchronously so that
    // we can do the callback concurrently.
    hl_set_sync_flag(false);

    /* Calculate the input layers error */
    
    
    MatrixPtr preGrad = getInputGrad(i);
    real* botdiff = preGrad->getData();
    LOG(INFO) << "--------------------ex before data diff: "<< botdiff[0] << "," << botdiff[10];
    if (NULL != preGrad) {
      MatrixPtr weights_T = weights_[i]->getW()->getTranspose();
      preGrad->mul(getOutputGrad(), weights_T, 1, 1);
    }
    hl_set_sync_flag(syncFlag);
    LOG(INFO) << "--------------------ex data diff: "<< botdiff[0] << "," << botdiff[10];


//      weights_[i]->getParameterPtr()->incUpdate(callback);

  }
}

void MkldnnFcLayer::submitBwdData(int idx, const MatrixPtr& botGrad) {
  if (botGrad == NULL) {
    return;
  }
  real* botdiff = botGrad->getData();
  real* topdiff = getOutputGrad()->getData();
  std::vector<primitive> pipeline;
  if (usePaddleFmt_) {  // no need cvt wgt without usePaddleFmt_
    CHECK(selfWgtData_[idx]);
    real* wgtdata = selfWgtData_[idx]->getData();
    dataWgt_->submitCvt(pipeline, wgtdata);
  }
  diffTop_->submitCvt(pipeline, topdiff);
  pipeline.push_back(*bwdData_);
  diffBot_->submitCvt(pipeline, botdiff);
  stream(stream::kind::eager).submit(pipeline).wait();
//  LOG(INFO) << "--------------------my data diff: "<< botdiff[0] << "," << botdiff[10];
}

void MkldnnFcLayer::submitBwdWgts(int idx, const MatrixPtr& botVal) {
  real* botdata = botVal->getData();  
  real* topdiff = getOutputGrad()->getData();
  real* wgtdiff = weights_[idx]->getWGrad()->getData();
  if (usePaddleFmt_) {
    CHECK(selfWgtDiff_[idx]);
    wgtdiff = selfWgtDiff_[idx]->getData();
  }
  std::vector<primitive> pipeline;
  diffTop_->submitCvt(pipeline, topdiff);
  dataBot_->submitCvt(pipeline, botdata);
  pipeline.push_back(*bwdWgt_);
  diffWgt_->submitCvt(pipeline, wgtdiff);
  if (biases_ && biases_->getWGrad()) {
    // bias backward can only execute in filter backward with MKL-DNN
    real* biasdiff = biases_->getWGrad()->getData();
    diffBias_->submitCvt(pipeline, biasdiff);
  }
//  LOG(INFO) << "size:" << pipeline.size();
  stream(stream::kind::eager).submit(pipeline).wait();
  
  if (usePaddleFmt_) {
    // save to actual weight param
    selfWgtDiff_[idx]->transpose(weights_[idx]->getWGrad_mutable(), false);
  }
}

void MkldnnFcLayer::submitDnnBwd(const UpdateCallback &callback) {
  backwardActivation();

//  exBwd(nullptr);

//  real* wgtdiff = weights_[0]->getWGrad()->getData();
//  real* biasdiff = biases_->getWGrad()->getData();


  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    submitBwdData(i, getPrev(i)->getOutputGrad());
    
    if (weights_[i]->getWGrad()) {
 //     LOG(INFO) << "--------------------ex wgt, bias diff: "<< wgtdiff[0] << "," << wgtdiff[3] << ","<<biasdiff[0]<< ","<<biasdiff[2];
      
      submitBwdWgts(i, getPrev(i)->getOutputValue());
 //     LOG(INFO) << "--------------------my wgt, bias diff: "<< wgtdiff[0] << "," << wgtdiff[3] << ","<<biasdiff[0]<< ","<<biasdiff[2];
      weights_[i]->getParameterPtr()->incUpdate(callback);   
    }
  }
  if (biases_ && biases_->getWGrad()) {
    biases_->getParameterPtr()->incUpdate(callback);
  }


}

}  // namespace paddle