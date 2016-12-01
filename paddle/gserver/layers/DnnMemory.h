/* 
 */

#pragma once

#include <vector>
#include "paddle/utils/Logging.h"
#include "mkldnn.hpp"

using namespace mkldnn;

namespace paddle {

/**
 * @brief A DNN memory buffer class
 *
 * 
 */
 typedef enum {
    dnnCvtNone             = 0,
    dnnCvtUser2Internal    = 1,
    dnnCvtInternal2User    = 2,
    dnnCvtNoNeed           = 3,
    dnnCvtNumber           = 4
} dnnCvtType_t;

class DnnBuffer {

protected:
  /// user and internal layout
//  memory::desc userMD;
//  memory::desc intlMD_;
  std::shared_ptr<memory::dims> pDims_;
  
  std::shared_ptr<memory> pUser_;
  std::shared_ptr<memory> pIntl_;
  
  /// conversion handle and type
  std::shared_ptr<primitive> pCvt_;

  int   type_;
  
public:
  explicit DnnBuffer() :
    pDims_(NULL),
    pUser_(NULL),
    pIntl_(NULL),
    pCvt_(NULL),
    type_(dnnCvtNone){}

  ~DnnBuffer() {}
  
  void initUser(void *pd, memory::dims &dm, memory::format fmt, engine eg,
                   memory::data_type tp = memory::data_type::f32) {
    this->pDims_.reset(new memory::dims(dm));
    memory::desc desc = memory::desc({dm}, tp, fmt);
    this->pUser_.reset(new memory(memory::primitive_desc(desc, eg), pd));
  }

  memory::desc getInitIntlMD(memory::data_type tp = memory::data_type::f32) {
    CHECK(pDims_);
    return memory::desc({*pDims_}, tp, memory::format::any);
  }

  std::shared_ptr<memory> getIntlMem() {
     return this->pIntl_;
  }

  std::shared_ptr<memory> getUserMem() {
     return this->pUser_;
  }
  
  // init conversion(reorder) return true if need cvt.
  bool initCvt(memory::primitive_desc intlPD, int cvtType) {
    CHECK(cvtType == dnnCvtUser2Internal || cvtType == dnnCvtInternal2User) <<
      "please specify one type of conversion";
    CHECK(pUser_) << "need create user layout before init conversion";
    // CHECK(pIntl_) << "need create internal layout before init conversion";
    pIntl_ = pUser_;
    type_ = cvtType;
    if (intlPD != pUser_->get_primitive_desc()) {
      // allocate internal src memory from user
      this->pIntl_.reset(new mkldnn::memory(intlPD));

      // create a reorder 
      if (cvtType == dnnCvtUser2Internal) {
        this->pCvt_.reset(new reorder(*pUser_, *pIntl_));
      } else {
        this->pCvt_.reset(new reorder(*pIntl_, *pUser_));
      }
      return true;
    } else {
      type_ = dnnCvtNoNeed;
      return false;
    }
  }
  
  bool needCvt() {
    CHECK(type_) << "init conversion firstly";
    if (type_ == dnnCvtNoNeed) {
      return false;
    } else {
      return pCvt_==NULL ? false : true;
    }
  }
  void setUserData(void* userData) {
    if (userData && (userData != pUser_->get_data_handle())) {
      pUser_->set_data_handle(userData);
    }
  }
  /**
   * submit reorder conversion.
   */
  void submitCvt(std::vector<primitive> &net, void* userData = NULL) {
    CHECK(type_) << "init conversion firstly";
    // set user data handle, whether need reorder or not
    setUserData(userData);
    if (type_ == dnnCvtNoNeed) {
      return;
    } else {
      CHECK(pCvt_) << "init conversion firstly";
      net.push_back(*pCvt_);
    }
  }

};

typedef std::shared_ptr<DnnBuffer> DnnBufferPtr;

}  // namespace paddle