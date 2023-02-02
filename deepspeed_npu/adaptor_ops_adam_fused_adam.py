import math
import torch
import torch_npu
import deepspeed
from collections import defaultdict
from apex.contrib.combine_tensors import combine_npu


class FusedAdamNPU(torch.optim.Optimizer):
    def __init__(self,
                 params,
                 lr=1e-3,
                 bias_correction=True,
                 betas=(0.9,
                        0.999),
                 eps=1e-8,
                 adam_w_mode=True,
                 weight_decay=0.01,
                 amsgrad=False,
                 set_grad_none=True):

        if amsgrad:
            raise RuntimeError('FusedAdam does not support the AMSGrad variant.')
        defaults = dict(lr=lr,
                        bias_correction=bias_correction,
                        betas=betas,
                        eps=eps,
                        weight_decay=weight_decay)
        super(FusedAdamNPU, self).__init__(params, defaults)
        self.adam_w_mode = 1 if adam_w_mode else 0

        self.combined_states = []
        # Skip buffer
        self._dummy_overflow_buf = torch.npu.IntTensor([0])

    def _init_combined_states(self):
        for group in self.param_groups:
            amsgrad = group['amsgrad'] if 'amsgrad' in group else False

            params_list = []
            grads_list = []
            step_list = []
            exp_avg_list = []
            exp_avg_sq_list = []
            max_exp_avg_sq_list = []

            # get params states list
            for param in group['params']:
                if param.grad is None:
                    param.grad = torch.zeros_like(param, memory_format=torch.preserve_format)
                if param.grad.is_sparse:
                    raise RuntimeError('NpuFusedAdamW does not support sparse gradients, '
                                       'please consider SparseAdam instead')

                params_list.append(param.data)
                grads_list.append(param.grad.data)

                self._init_param_state(param, amsgrad)

                step_list.append(self.state[param]['step'])
                exp_avg_list.append(self.state[param]['exp_avg'])
                exp_avg_sq_list.append(self.state[param]['exp_avg_sq'])
                if amsgrad:
                    max_exp_avg_sq_list.append(self.state[param]['max_exp_avg_sq'])

            combined_step = 0
            combined_params = None
            combined_grads = None
            combined_exp_avg = None
            combined_exp_avg_sq = None
            combined_max_exp_avg_sq = None

            if len(exp_avg_list) > 0:
                combined_step = step_list[0]
                combined_params = combine_npu(params_list)
                combined_grads = combine_npu(grads_list)
                combined_exp_avg = combine_npu(exp_avg_list)
                combined_exp_avg_sq = combine_npu(exp_avg_sq_list)
                combined_max_exp_avg_sq = combine_npu(max_exp_avg_sq_list)

            combined_state = defaultdict(dict)
            combined_state['params'] = combined_params
            combined_state['grads'] = combined_grads
            combined_state['step'] = combined_step
            combined_state['exp_avg'] = combined_exp_avg
            combined_state['exp_avg_sq'] = combined_exp_avg_sq
            combined_state['max_exp_avg_sq'] = combined_max_exp_avg_sq
            self.combined_states.append(combined_state)

    def _init_param_state(self, p, amsgrad):
        state = self.state[p]
        # State initialization
        if len(state) == 0:
            state['step'] = 0
            # Exponential moving average of gradient values
            state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
            # Exponential moving average of squared gradient values
            state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
            if amsgrad:
                # Maintains max of all exp. moving avg. of sq. grad. values
                state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
        else:
            exp_avg_tmp = torch.zeros_like(p, memory_format=torch.preserve_format)
            exp_avg_tmp.copy_(state['exp_avg'])
            state['exp_avg'] = exp_avg_tmp

            exp_avg_sq_tmp = torch.zeros_like(p, memory_format=torch.preserve_format)
            exp_avg_sq_tmp.copy_(state['exp_avg_sq'])
            state['exp_avg_sq'] = exp_avg_sq_tmp

            if amsgrad:
                max_exp_avg_sq_tmp = torch.zeros_like(p, memory_format=torch.preserve_format)
                max_exp_avg_sq_tmp.copy_(state['max_exp_avg_sq'])
                state['max_exp_avg_sq'] = max_exp_avg_sq_tmp

    def zero_grad(self, set_to_none: bool = False):
        for combined_state in self.combined_states:
            combined_state['grads'].zero_()

    def _group_step(self, group_index, group):
        amsgrad = group['amsgrad'] if 'amsgrad' in group else False
        beta1, beta2 = group['betas']

        combined_params = self.combined_states[group_index]['params']
        combined_grads = self.combined_states[group_index]['grads']

        if combined_params is None or combined_grads is None:
            return

        exp_avg = self.combined_states[group_index]['exp_avg']
        exp_avg_sq = self.combined_states[group_index]['exp_avg_sq']
        if amsgrad:
            max_exp_avg_sq = self.combined_states[group_index]['max_exp_avg_sq']

        self.combined_states[group_index]['step'] += 1

        # Perform stepweight decay. The fused method is used here to speed up the calculation
        if self.adam_w_mode:
            combined_params.mul_(1 - group['lr'] * group['weight_decay'])

        bias_correction1 = 1 - beta1 ** self.combined_states[group_index]['step']
        bias_correction2 = 1 - beta2 ** self.combined_states[group_index]['step']

        if not self.adam_w_mode and group['weight_decay'] != 0:
            combined_grads = combined_grads.add(combined_params, alpha=group['weight_decay'])

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(combined_grads, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(combined_grads, combined_grads, value=1 - beta2)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

        step_size = group['lr'] / bias_correction1

        combined_params.addcdiv_(exp_avg, denom, value=-step_size)

    def step(self,
             closure=None,
             grads=None,
             output_params=None,
             scale=None,
             grad_norms=None):
        if any(p is not None for p in [grads, output_params, scale, grad_norms]):
            raise RuntimeError(
                'FusedAdam has been updated.  Simply initialize it identically to torch.optim.Adam, and call step() with no arguments.'
            )
        loss = None
        if closure is not None:
            loss = closure()

        if len(self.combined_states) == 0:
            self._init_combined_states()

        for i, group in enumerate(self.param_groups):
            self._group_step(i, group)

        return loss


deepspeed.ops.adam.fused_adam.FusedAdam = FusedAdamNPU