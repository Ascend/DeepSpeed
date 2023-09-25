import math
import torch
import torch_npu
import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.ops.adam import FusedAdam
from deepspeed.ops.op_builder import UtilsBuilder


def FusedAdamInit(self, params, lr=1e-3, bias_correction=True, betas=(0.9, 0.999), eps=1e-8, adam_w_mode=True,
                  weight_decay=0., amsgrad=False, set_grad_none=True):
    if amsgrad:
        raise RuntimeError('FusedAdam does not support the AMSGrad variant.')
    defaults = dict(lr=lr, bias_correction=bias_correction, betas=betas, eps=eps, weight_decay=weight_decay)
    super(FusedAdam, self).__init__(params, defaults)
    self.adam_w_mode = 1 if adam_w_mode else 0
    self.set_grad_none = set_grad_none
    self.amsgrad = amsgrad  # its possible with npu_apply_adam_w operator

    # Skip buffer
    self._dummy_overflow_buf = get_accelerator().IntTensor([0])

    util_ops = UtilsBuilder().load()
    self.flatten = util_ops.flatten

    def unflatten(flatted_tensor, tensor_list):
        for buf, flatted in zip(tensor_list, util_ops.unflatten(flatted_tensor, tensor_list)):
            buf.copy_(flatted)
        return tensor_list
    self.unflatten = unflatten


def step(self, closure=None, grads=None, output_params=None, scale=None, grad_norms=None):
    if any(p is not None for p in [grads, output_params, scale, grad_norms]):
        raise RuntimeError(
            'FusedAdam has been updated.  Simply initialize it identically to torch.optim.Adam, and call step() with no arguments.'
        )
    loss = None
    if closure is not None:
        loss = closure()

    for group in self.param_groups:
        bias_correction = 1 if group['bias_correction'] else 0
        beta1, beta2 = group['betas']

        if 'step' not in group:
            group['step'] = 0

        # create lists for multi-tensor apply
        g_16, p_16, m_16, v_16 = [], [], [], []
        g_32, p_32, m_32, v_32 = [], [], [], []

        for p in group['params']:
            if p.grad is None:
                continue
            if p.grad.data.is_sparse:
                raise RuntimeError(
                    'FusedAdam does not support sparse gradients, please consider SparseAdam instead')

            state = self.state[p]
            # State initialization
            if len(state) == 0:
                # DeepSpeed ZeRO 3 processes each subgroup a time, so we need to keep tracking step count for each tensor separately.
                # While this is not an issue for ZeRO 1 & 2, since they apply a single optimizatin step to the whole param group at the same time.
                # In order to keep backward compatibility for the existing checkpoints, we use group['state'] to initialize state['step'] if it exists.
                state['step'] = group.get('step', 0)
                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(p.data)
                # Exponential moving average of squared gradient values
                state['exp_avg_sq'] = torch.zeros_like(p.data)

            if p.dtype == torch.float16:
                g_16.append(p.grad.data)
                p_16.append(p.data)
                m_16.append(state['exp_avg'])
                v_16.append(state['exp_avg_sq'])
            elif p.dtype == torch.float32:
                g_32.append(p.grad.data)
                p_32.append(p.data)
                m_32.append(state['exp_avg'])
                v_32.append(state['exp_avg_sq'])
            else:
                raise RuntimeError('FusedAdam only support fp16 and fp32.')

        has_param_to_flatten = True
        if len(g_16) > 0:
            grad_flat = self.flatten(g_16)
            param_flat = self.flatten(p_16)
            m_flat = self.flatten(m_16)
            v_flat = self.flatten(v_16)
        elif len(g_32) > 0:
            grad_flat = self.flatten(g_32)
            param_flat = self.flatten(p_32)
            m_flat = self.flatten(m_32)
            v_flat = self.flatten(v_32)
        else:
            has_param_to_flatten = False

        if has_param_to_flatten:
            state['step'] += 1
            bias_correction1 = beta1 ** state['step']
            bias_correction2 = beta2 ** state['step']

            param_flat.data, m_flat, v_flat = torch_npu.npu_apply_adam_w(
                bias_correction1,
                bias_correction2,
                group['lr'],
                group['weight_decay'],
                beta1,
                beta2,
                group['eps'],
                grad_flat,
                None,
                self.amsgrad,
                False,
                out=(param_flat.data, m_flat, v_flat)
            )

            if len(g_16) > 0:
                self.unflatten(param_flat, p_16)
                self.unflatten(m_flat, m_16)
                self.unflatten(v_flat, v_16)
            if len(g_32) > 0:
                self.unflatten(param_flat, p_32)
                self.unflatten(m_flat, m_32)
                self.unflatten(v_flat, v_32)

    return loss


deepspeed.ops.adam.FusedAdam.__init__ = FusedAdamInit
deepspeed.ops.adam.FusedAdam.step = step
