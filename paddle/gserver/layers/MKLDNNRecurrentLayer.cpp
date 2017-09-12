/* Copyright (c) 2017 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "MKLDNNRecurrentLayer.h"
#include "paddle/utils/Logging.h"

using namespace mkldnn;  // NOLINT
typedef memory::format format;
typedef inner_product_forward fc_fwd;

namespace paddle {

REGISTER_LAYER(mkldnn_rnn, MKLDNNRecurrentLayer);

bool MKLDNNRecurrentLayer::init(const LayerMap& layerMap,
                                const ParameterMap& parameterMap) {
  if (!MKLDNNLayer::init(layerMap, parameterMap)) {
    return false;
  }

  CHECK_EQ(inputLayers_.size(), 1U) << "Only support one input layer yet";
  CHECK_EQ(parameters_.size(), 1U);

  // output size and input size cat not be changed
  oc_ = getSize();
  iLayerSize_ = inputLayers_[0]->getSize();
  CHECK_EQ(iLayerSize_, getSize()) << "input and output size should equal";
  CHECK_EQ(parameters_[0]->getSize(), iLayerSize_ * oc_);
  // image sizes always be 1
  oh_ = 1;
  ow_ = 1;
  ih_ = 1;
  iw_ = 1;
  reversed_ = config_.reversed();

  // create weight and bias
  weight_.reset(new Weight(oc_, iLayerSize_, parameters_[0], 0));
  if (biasParameter_.get() != NULL) {
    biases_.reset(new Weight(1, oc_, biasParameter_));
  }
  return true;
}

void MKLDNNRecurrentLayer::convertWeightsFromPaddle() {
  if (hasInitedWgt_) {
    return;
  }
  // TODO(TJ): maybe need tranpose one by one?
  hasInitedWgt_ = true;
}

void MKLDNNRecurrentLayer::convertWeightsToPaddle() {
  // TODO(TJ): maybe need tranpose one by one?
}

void MKLDNNRecurrentLayer::reshape(
    int& bs, int& ic, int& ih, int& iw, int oc, int& oh, int& ow) {
  const Argument& input = inputLayers_[0]->getOutput();

  // TODO(TJ): add check only support input tensor shape: (seqlen, bs, dim)
  seqLen_ = getSeqLen(input);
  CHECK_GE(seqLen_, 1U);
  size_t totalBatchSize = input.getBatchSize();
  bs = totalBatchSize / seqLen_;
  CHECK_EQ(size_t(bs), input.getNumSequences());
  CHECK_EQ(size_t(bs * seqLen_), totalBatchSize) << "not divisible";
  ih = 1;
  iw = 1;
  oh = 1;
  ow = 1;

  ic = input.value->getWidth();
  CHECK_EQ(size_t(ic), iLayerSize_) << "Can not change input size";
  CHECK_EQ(size_t(oc), getSize());

  resizeOutput(totalBatchSize, oc);

  printSizeInfo();
}

void MKLDNNRecurrentLayer::resetFwd(std::vector<primitive>& pipeline,
                                    MKLDNNMatrixPtr& in,
                                    MKLDNNMatrixPtr& wgt,
                                    MKLDNNMatrixPtr& bias,
                                    MKLDNNMatrixPtr& out) {
  pipeline.clear();
  CHECK_GE(seqLen_, 1U);
  seqMul_.resize(seqLen_ - 1);
  seqSum_.resize(seqLen_);
  seqAct_.resize(seqLen_);

  // weight and bias always use the same buffer
  bool hasBias = biases_ && biases_->getW();
  const MatrixPtr& wgtVal = weight_->getW();
  const MatrixPtr& biasVal = hasBias ? biases_->getW() : nullptr;
  wgt =
      MKLDNNMatrix::create(wgtVal, memory::dims{oc_, ic_}, format::oi, engine_);
  bias = hasBias ? MKLDNNMatrix::create(biasVal, {oc_}, format::x, engine_)
                 : nullptr;

  // reset all seqVals
  resetSeqValue(seqInVal_, seqOutVal_);

  // reset pipeline
  size_t start = 0;
  size_t end = seqLen_ - 1;
  if (reversed_) {
    // out_end = act(in_end)
    resetAct(seqAct_[end], seqOutVal_[end], seqInVal_[end]);
    pipeline.push_back(*seqAct_[end]);
    for (size_t i = end - 1; i >= start; --i) {
      // out_i = W * out_(i+1)
      resetMul(seqMul_[i], seqOutVal_[i], wgt, seqOutVal_[i + 1]);

      // out_i = out_i + in_i + bias
      resetSum(seqSum_[i], seqOutVal_[i], {seqOutVal_[i], seqInVal_[i], bias});

      // out_i = act(out_i)
      resetAct(seqAct_[i], seqOutVal_[i], seqOutVal_[i]);

      // push back pipeline
      pipeline.push_back(*seqMul_[i]);
      pipeline.push_back(*seqSum_[i]);
      pipeline.push_back(*seqAct_[i]);
    }
  } else {
    // out_start = act(in_start)
    resetAct(seqAct_[start], seqOutVal_[start], seqInVal_[start]);
    pipeline.push_back(*seqAct_[start]);
    for (size_t i = start + 1; i <= end; ++i) {
      // out_i = W * out_(i-1)
      resetMul(seqMul_[i], seqOutVal_[i], wgt, seqOutVal_[i - 1]);

      // out_i = out_i + in_i + bias
      // TODO: check inplace sum, ok?
      resetSum(seqSum_[i], seqOutVal_[i], {seqOutVal_[i], seqInVal_[i], bias});

      // out_i = act(out_i)
      resetAct(seqAct_[i], seqOutVal_[i], seqOutVal_[i]);

      // push back pipeline
      pipeline.push_back(*seqMul_[i]);
      pipeline.push_back(*seqSum_[i]);
      pipeline.push_back(*seqAct_[i]);
    }
  }

  // TODO(TJ): Can not cast to MKLDNNMatrix, next mkldnn layer should not check
  // it when prev is rnn
  // output_.value = std::dynamic_pointer_cast<Matrix>(out);
  if (!outputIsOnlyMKLDNN()) {
    // fc cpu output value do not need create convert
    // just share point
    getOutput(CPU_DEVICE).value->setData(output_.value->getData());
  }
}

void MKLDNNRecurrentLayer::resetBwd(std::vector<mkldnn::primitive>& pipeline,
                                    MKLDNNMatrixPtr& in,
                                    MKLDNNMatrixPtr& wgt,
                                    MKLDNNMatrixPtr& bias,
                                    MKLDNNMatrixPtr& out) {
  LOG(FATAL) << "not implemented";
}

void MKLDNNRecurrentLayer::updateInputData() {
  // maybe needed
  //  inVal_->setData(getInputValue(0, CPU_DEVICE)->getData());
}

void MKLDNNRecurrentLayer::resetSeqValue(std::vector<MKLDNNMatrixPtr>& seqIn,
                                         std::vector<MKLDNNMatrixPtr>& seqOut) {
  const MatrixPtr& inVal = inputLayers_[0]->getOutput().value;
  const MatrixPtr& outVal = output_.value;
  CHECK_EQ(inVal->getElementCnt(), seqLen_ * bs_ * ic_);
  CHECK_EQ(outVal->getElementCnt(), seqLen_ * bs_ * oc_);
  // input and output shape: (seqlen, bs, dim) with matrix(seqlen*bs, dim)
  seqIn.clear();
  seqOut.clear();
  seqIn.resize(seqLen_);
  seqOut.resize(seqLen_);
  for (size_t i = 0; i < seqLen_; ++i) {
    real* inData = inVal->getData() + i * ic_;
    real* outData = outVal->getData() + i * oc_;
    const MatrixPtr& in = Matrix::create(inData, 1, ic_, false, false);
    const MatrixPtr& out = Matrix::create(outData, 1, oc_, false, false);

    // MKLDNNMatrix
    seqIn[i] = MKLDNNMatrix::create(in, {bs_, ic_}, format::nc, engine_);
    seqOut[i] = MKLDNNMatrix::create(out, {bs_, oc_}, format::nc, engine_);
  }
}

void MKLDNNRecurrentLayer::resetMul(std::shared_ptr<primitive>& prim,
                                    MKLDNNMatrixPtr& dst,
                                    MKLDNNMatrixPtr& wgt,
                                    MKLDNNMatrixPtr& src) {
  // no bias, only mul
  fc_fwd::desc mulDesc = fc_fwd::desc(prop_kind::forward,
                                      src->getMemoryDesc(),
                                      wgt->getMemoryDesc(),
                                      dst->getMemoryDesc());
  fc_fwd::primitive_desc mulPD = fc_fwd::primitive_desc(mulDesc, engine_);
  prim.reset(new fc_fwd(mulPD, *src, *wgt, *dst));
}

void MKLDNNRecurrentLayer::resetAct(std::shared_ptr<primitive>& prim,
                                    MKLDNNMatrixPtr& dst,
                                    MKLDNNMatrixPtr& src,
                                    std::string actType) {}

void MKLDNNRecurrentLayer::resetSum(std::shared_ptr<primitive>& prim,
                                    MKLDNNMatrixPtr& dst,
                                    std::vector<MKLDNNMatrixPtr> srcs) {
  for (size_t i = 0; i < srcs.size(); ++i) {
    if (srcs[i] == nullptr) {
      continue;
    }
  }
}

}  // namespace paddle
