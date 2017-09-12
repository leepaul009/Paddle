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

#pragma once

#include "MKLDNNLayer.h"
#include "mkldnn.hpp"

namespace paddle {

/**
 * @brief A subclass of MKLDNNLayer recurrent layer.
 *
 * The config file api is mkldnn_rnn
 */
class MKLDNNRecurrentLayer : public MKLDNNLayer {
protected:
  std::vector<std::shared_ptr<mkldnn::primitive>> seqMul_;
  std::vector<std::shared_ptr<mkldnn::primitive>> seqSum_;
  std::vector<std::shared_ptr<mkldnn::primitive>> seqAct_;
  std::vector<MKLDNNMatrixPtr> seqInVal_;
  std::vector<MKLDNNMatrixPtr> seqOutVal_;

  // input layer size, can not be change after init
  size_t iLayerSize_;  // == ic * ih * iw

  bool reversed_;

  size_t seqLen_;

  // if has already init the weight
  bool hasInitedWgt_;

  // rnn weight and bias
  std::unique_ptr<Weight> weight_;
  std::unique_ptr<Weight> biases_;

public:
  explicit MKLDNNRecurrentLayer(const LayerConfig& config)
      : MKLDNNLayer(config), seqLen_(0), hasInitedWgt_(false) {}

  ~MKLDNNRecurrentLayer() {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void reshape(
      int& bs, int& ic, int& ih, int& iw, int oc, int& oh, int& ow) override;

  void resetFwd(std::vector<mkldnn::primitive>& pipeline,
                MKLDNNMatrixPtr& in,
                MKLDNNMatrixPtr& wgt,
                MKLDNNMatrixPtr& bias,
                MKLDNNMatrixPtr& out) override;

  void resetBwd(std::vector<mkldnn::primitive>& pipeline,
                MKLDNNMatrixPtr& in,
                MKLDNNMatrixPtr& wgt,
                MKLDNNMatrixPtr& bias,
                MKLDNNMatrixPtr& out) override;

  void updateInputData() override;

  void convertWeightsFromPaddle() override;

  void convertWeightsToPaddle() override;

protected:
  /**
   * reset the MKLDNNMatrix
   */
  void resetSeqValue(std::vector<MKLDNNMatrixPtr>& seqIn,
                     std::vector<MKLDNNMatrixPtr>& seqOut);

  /**
   * out_i = act(in_i)
   * out_i+1 = act(in_i+1 + W * out_i + bias)
   */

  /**
   * dst = Wgt * src
   */
  void resetMul(std::shared_ptr<mkldnn::primitive>& prim,
                MKLDNNMatrixPtr& dst,
                MKLDNNMatrixPtr& wgt,
                MKLDNNMatrixPtr& src);

  /**
   * dst = sum(srcs)
   */
  void resetSum(std::shared_ptr<mkldnn::primitive>& prim,
                MKLDNNMatrixPtr& dst,
                std::vector<MKLDNNMatrixPtr> srcs);

  /**
   * dst = act(src)
   */
  void resetAct(std::shared_ptr<mkldnn::primitive>& prim,
                MKLDNNMatrixPtr& dst,
                MKLDNNMatrixPtr& src,
                std::string actType = "relu");

  void printSizeInfo() override {
    MKLDNNLayer::printSizeInfo();
    VLOG(MKLDNN_SIZES) << getName() << ": seqLen: " << seqLen_;
  }

  /**
   * get the aligned seq length from paddle sequence info
   * and the length among batchsize should be the same
   */
  int getSeqLen(const Argument& arg) {
    CHECK(arg.sequenceStartPositions);
    int sampleSize = arg.getBatchSize();  // seqlen * bs
    size_t numSequences = arg.getNumSequences();
    const int* starts = arg.sequenceStartPositions->getData(false);
    CHECK_EQ(starts[numSequences], sampleSize);
    int len = 0;
    for (size_t i = 0; i < numSequences; ++i) {
      int tmp = starts[i + 1] - starts[i];
      CHECK(len == 0 || len == tmp) << "all seq length should be equal," << len
                                    << " vs " << tmp;
      len = tmp;
    }
    return len;
  }
};

}  // namespace paddle
