/* Copyright (c) 2016 */

#pragma once

#include "MkldnnLayer.h"
#include "paddle/math/Matrix.h"
#include <vector>
#include "mkldnn.hpp"
#include "MkldnnMemory.h"

namespace paddle {

/**
 * @brief A subclass of MkldnnLayer fc layer.
 *
 * The config file api is 
 */
class MkldnnFcLayer : public MkldnnLayer {
protected:
  std::shared_ptr<mkldnn::inner_product_forward> fwd_;
  std::shared_ptr<mkldnn::inner_product_backward_data> bwdData_;
  std::shared_ptr<mkldnn::inner_product_backward_weights> bwdWgt_;

  // if image width and height !=0
  bool hasSpatial_;
  /// data buffers
  MkldnnBufferPtr dataWgt_;
  MkldnnBufferPtr dataBias_;
  /// diff buffer
  MkldnnBufferPtr diffWgt_;
  MkldnnBufferPtr diffBias_;

  // fc
  WeightList weights_;
  std::unique_ptr<Weight> biases_;

  // dnn self wgt, only create if use paddle fmt
  std::vector<MatrixPtr> selfWgtData_;
  std::vector<MatrixPtr> selfWgtDiff_;

  // use paddle weight format
  bool usePaddleFmt_;
  bool hasInited_;

  // input size (== ic*ih*iw) by batch size
  std::vector<size_t> inputSizeByBS_;

public:
  explicit MkldnnFcLayer(const LayerConfig& config)
    : MkldnnLayer(config),
      fwd_(nullptr),
      hasSpatial_(false),
      dataWgt_(nullptr),
      dataBias_(nullptr),
      diffWgt_(nullptr),
      diffBias_(nullptr),
      hasInited_(false)
    {}

  ~MkldnnFcLayer() {}

  bool initDnn(const LayerMap& layerMap, const ParameterMap& parameterMap);

  virtual void clearAllDnnCvtFlags() {
    MkldnnLayer::clearAllDnnCvtFlags();
    if (dataBias_) dataBias_->clearCvtFlag();
    if (dataWgt_) dataWgt_->clearCvtFlag();
    if (diffBias_) diffBias_->clearCvtFlag();
    if (diffWgt_) diffWgt_->clearCvtFlag();
  }

  void reshape();

  void clearDataDiff();

  void resetDnnFwd(PassType passType);

  void resetDnnBwd();

  void submitDnnFwd(PassType passType);

  void submitBwdData(int idx);

  void submitBwdWgts(int idx);

  void submitDnnBwd(const UpdateCallback& callback);

  // keep as paddle did
  void prefetch();

  // keep as paddle did
  Weight& getWeight(int idx) { return *weights_[idx]; }

  void printInfo() {
    for (size_t i = 0; i < iw_.size(); ++i) {
      VLOG(2)
        << "ih: " << ih_[i] << ", iw: " << iw_[i]
        << ", ic: " << ic_[i]
        << ", oh: " << oh_[i] << ", ow: " << ow_[i]
        << ", oc: " << oc_;
    }
  }
};

}  // namespace paddle
