#pragma once

#include <stdio.h>
#include <cassert>

#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

typedef c10::Half __half;
#define TILE 134217728

#define STEP(SPAN)                                \
    void Step_##SPAN(float* _params,              \
                     float* grads,                \
                     float* _exp_avg,             \
                     float* _exp_avg_sq,          \
                     size_t _param_size,          \
                     __half* dev_param = nullptr, \
                     bool half_precision = false);

class Adam_Optimizer {
public:
    explicit Adam_Optimizer(
        float alpha = 1e-3,
        float betta1 = 0.9,
        float betta2 = 0.999,
        float eps = 1e-8,
        float weight_decay = 0,
        bool adamw_mode = true
    ):
        _alpha(alpha),
        _betta1(betta1),
        _betta2(betta2),
        _eps(eps),
        _weight_decay(weight_decay),
        _betta1_t(1.0),
        _betta2_t(1.0),
        _step(0),
        _buf_index(false),
        _adamw_mode(adamw_mode)
    {
        int res1 = aclrtMallocHost((void**)_doubled_buffer, TILE * sizeof(__half));
        int res2 = aclrtMallocHost((void**)(_doubled_buffer + 1), TILE * sizeof(__half));
        if(res1 != 0 || res2 != 0){
            throw std::runtime_error("Malloc host memory error.");
        }
    }

    ~Adam_Optimizer()
    {
        aclrtFreeHost(_doubled_buffer[0]);
        aclrtFreeHost(_doubled_buffer[1]);
    }

    STEP(1)
    STEP(4)
    STEP(8)

    inline void SynchronizeStreams()
    {
        c10_npu::npuSynchronizeDevice();
    }

    inline void IncrementStep(size_t step, float beta1, float beta2)
    {
        if (beta1 != _betta1 || beta2 != _betta2) {
            _step = step;
            _betta1 = beta1;
            _betta2 = beta2;
            _betta1_t = std::pow(_betta1, step);
            _betta2_t = std::pow(_betta2, step);
        } else {
            _step++;
            if (_step != step) {
                _betta1_t = std::pow(_betta1, step);
                _betta2_t = std::pow(_betta2, step);
                _step = step;
            } else {
                _betta1_t *= _betta1;
                _betta2_t *= _betta2;
            }
        }
    }

    inline void update_state(float lr, float epsilon, float weight_decay, bool bias_correction)
    {
        _alpha = lr;
        _eps = epsilon;
        _weight_decay = weight_decay;

        _bias_correction1 = 1.0f;
        _bias_correction2 = 1.0f;
        if (bias_correction == 1) {
            _bias_correction1 = 1 - _betta1_t;
            _bias_correction2 = 1 / sqrt(1 - _betta2_t);
        }
    }

private:
    float _alpha;
    float _betta1;
    float _betta2;
    float _eps;
    float _weight_decay;

    float _betta1_t;
    float _betta2_t;
    size_t _step;

    float _bias_correction1;
    float _bias_correction2;

    __half* _doubled_buffer[2];
    bool _buf_index;
    bool _adamw_mode;

    c10_npu::NPUStream _streams[2] = { c10_npu::getCurrentNPUStream(), c10_npu::getNPUStreamFromPool() };
};
