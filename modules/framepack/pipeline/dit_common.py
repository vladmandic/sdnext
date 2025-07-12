import torch
import accelerate.accelerator

from diffusers.models.normalization import RMSNorm, LayerNorm, FP32LayerNorm, AdaLayerNormContinuous


accelerate.accelerator.convert_outputs_to_fp32 = lambda x: x


def LayerNorm_forward(self, x):
    return torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps).to(x)


LayerNorm.forward = LayerNorm_forward
torch.nn.LayerNorm.forward = LayerNorm_forward


def FP32LayerNorm_forward(self, x):
    origin_dtype = x.dtype
    return torch.nn.functional.layer_norm(
        x.float(),
        self.normalized_shape,
        self.weight.float() if self.weight is not None else None,
        self.bias.float() if self.bias is not None else None,
        self.eps,
    ).to(origin_dtype)


FP32LayerNorm.forward = FP32LayerNorm_forward


def RMSNorm_forward(self, hidden_states):
    input_dtype = hidden_states.dtype
    variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

    if self.weight is None:
        return hidden_states.to(input_dtype)

    return hidden_states.to(input_dtype) * self.weight.to(input_dtype)


RMSNorm.forward = RMSNorm_forward


def AdaLayerNormContinuous_forward(self, x, conditioning_embedding):
    emb = self.linear(self.silu(conditioning_embedding))
    scale, shift = emb.chunk(2, dim=1)
    x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
    return x


AdaLayerNormContinuous.forward = AdaLayerNormContinuous_forward
