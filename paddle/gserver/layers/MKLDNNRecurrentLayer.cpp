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

DECLARE_bool(rnn_use_batch);

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

  // TODO(TJ): double check it
  CHECK(!FLAGS_rnn_use_batch)
      << "MKLDNN Recurrent Layer do not support the batch method";

  // create weight and bias
  weight_.reset(new Weight(oc_, iLayerSize_, parameters_[0], 0));
  if (biasParameter_.get() != NULL) {
    CHECK_EQ(biasParameter_->getSize(), (size_t)oc_);
    biases_.reset(new Weight(1, oc_, biasParameter_, 0));
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
  seqFc_.clear();
  seqSum_.clear();
  seqAct_.clear();
  CHECK_GE(seqLen_, 1U);
  seqFc_.resize(seqLen_);  // only used seqLen_ - 1, but keep size enough
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
  std::string actType = config_.active_type();
  if (reversed_) {
    // out_end = act(in_end)
    addActOp(pipeline, seqAct_[end], seqOutVal_[end], seqInVal_[end], actType);
    for (int i = end - 1; i >= (int)start; --i) {
      // out_i = W * out_(i+1) + bias
      addFcOp(pipeline, seqFc_[i], seqOutVal_[i], wgt, seqOutVal_[i + 1], bias);

      // out_i = out_i + in_i
      addSumOp(
          pipeline, seqSum_[i], seqOutVal_[i], {seqOutVal_[i], seqInVal_[i]});

      // out_i = act(out_i)
      addActOp(pipeline, seqAct_[i], seqOutVal_[i], seqOutVal_[i], actType);
    }
  } else {
    // out_start = act(in_start)
    addActOp(
        pipeline, seqAct_[start], seqOutVal_[start], seqInVal_[start], actType);
    for (size_t i = start + 1; i <= end; ++i) {
      // out_i = W * out_(i-1) + bias
      addFcOp(pipeline, seqFc_[i], seqOutVal_[i], wgt, seqOutVal_[i - 1], bias);

      // out_i = out_i + in_i
      // TODO: check inplace sum, ok?
      addSumOp(
          pipeline, seqSum_[i], seqOutVal_[i], {seqOutVal_[i], seqInVal_[i]});

      // out_i = act(out_i)
      addActOp(pipeline, seqAct_[i], seqOutVal_[i], seqOutVal_[i], actType);
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
    real* inData = inVal->getData() + i * bs_ * ic_;
    real* outData = outVal->getData() + i * bs_ * oc_;
    const MatrixPtr& in = Matrix::create(inData, bs_, ic_, false, false);
    const MatrixPtr& out = Matrix::create(outData, bs_, oc_, false, false);

    // MKLDNNMatrix
    seqIn[i] = MKLDNNMatrix::create(in, {bs_, ic_}, format::nc, engine_);
    seqOut[i] = MKLDNNMatrix::create(out, {bs_, oc_}, format::nc, engine_);
  }
}

void MKLDNNRecurrentLayer::addFcOp(std::vector<primitive>& pipeline,
                                   std::shared_ptr<primitive>& prim,
                                   MKLDNNMatrixPtr& dst,
                                   MKLDNNMatrixPtr& wgt,
                                   MKLDNNMatrixPtr& src,
                                   MKLDNNMatrixPtr& bias) {
  auto fwdDesc = bias != nullptr ? fc_fwd::desc(prop_kind::forward,
                                                src->getMemoryDesc(),
                                                wgt->getMemoryDesc(),
                                                bias->getMemoryDesc(),
                                                dst->getMemoryDesc())
                                 : fc_fwd::desc(prop_kind::forward,
                                                src->getMemoryDesc(),
                                                wgt->getMemoryDesc(),
                                                dst->getMemoryDesc());

  fc_fwd::primitive_desc fwdPD = fc_fwd::primitive_desc(fwdDesc, engine_);
  if (bias) {
    prim.reset(new fc_fwd(fwdPD, *src, *wgt, *bias, *dst));
  } else {
    prim.reset(new fc_fwd(fwdPD, *src, *wgt, *dst));
  }
  pipeline.push_back(*prim);
}

void MKLDNNRecurrentLayer::addActOp(std::vector<primitive>& pipeline,
                                    std::shared_ptr<primitive>& prim,
                                    MKLDNNMatrixPtr& dst,
                                    MKLDNNMatrixPtr& src,
                                    std::string actType) {
  if (actType == "brelu") {
    LOG(WARNING) << "Do not support brelu yet, will use relu instead";
    addReluOp(pipeline, prim, dst, src);
  } else if (actType == "relu") {
    addReluOp(pipeline, prim, dst, src);
  } else {
    LOG(FATAL) << "Do not support " << actType << " yet";
  }
}

void MKLDNNRecurrentLayer::addReluOp(std::vector<primitive>& pipeline,
                                     std::shared_ptr<primitive>& prim,
                                     MKLDNNMatrixPtr& dst,
                                     MKLDNNMatrixPtr& src,
                                     float negativeSlope) {
  // TODO(TJ): double check the negativeSlope = -0.f right?
  CHECK(src->getPrimitiveDesc() == dst->getPrimitiveDesc());
  auto reluDesc = relu_forward::desc(
      prop_kind::forward_training, src->getMemoryDesc(), negativeSlope);
  auto pd = relu_forward::primitive_desc(reluDesc, engine_);
  prim.reset(new relu_forward(pd, *src, *dst));
  pipeline.push_back(*prim);
}

void MKLDNNRecurrentLayer::addSumOp(std::vector<primitive>& pipeline,
                                    std::shared_ptr<primitive>& prim,
                                    MKLDNNMatrixPtr& dst,
                                    std::vector<MKLDNNMatrixPtr> srcs) {
  std::vector<double> scales;
  std::vector<memory::primitive_desc> srcPDs;
  std::vector<primitive::at> inputs;
  for (size_t i = 0; i < srcs.size(); ++i) {
    if (srcs[i] == nullptr) {
      continue;
    }
    CHECK(dst->getPrimitiveDesc() == srcs[i]->getPrimitiveDesc())
        << "all PrimitiveDesc should be the same";
    scales.push_back(1.0);  // do not need scale here
    srcPDs.push_back(srcs[i]->getPrimitiveDesc());
    inputs.push_back(*srcs[i]);
  }
  auto pd = sum::primitive_desc(dst->getMemoryDesc(), scales, srcPDs);
  prim.reset(new sum(pd, inputs, *dst));
  pipeline.push_back(*prim);
}

}  // namespace paddle
