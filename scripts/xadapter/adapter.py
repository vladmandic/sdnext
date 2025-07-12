import torch
import torch.nn as nn
from collections import OrderedDict
from diffusers.models.embeddings import (
    TimestepEmbedding,
    Timesteps,
)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def get_parameter_dtype(parameter: torch.nn.Module):
    try:
        params = tuple(parameter.parameters())
        if len(params) > 0:
            return params[0].dtype

        buffers = tuple(parameter.buffers())
        if len(buffers) > 0:
            return buffers[0].dtype

    except StopIteration:
        # For torch.nn.DataParallel compatibility in PyTorch 1.5

        def find_tensor_attributes(module: torch.nn.Module) -> List[Tuple[str, Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].dtype


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=padding)
        else:
            assert self.channels == self.out_channels
            from torch.nn import MaxUnpool2d
            self.op = MaxUnpool2d(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = nn.ConvTranspose2d(self.channels, self.out_channels, 3, stride=stride, padding=1)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x, output_size):
        assert x.shape[1] == self.channels
        return self.op(x, output_size)


class Linear(nn.Module):
    def __init__(self, temb_channels, out_channels):
        super(Linear, self).__init__()
        self.linear = nn.Linear(temb_channels, out_channels)

    def forward(self, x):
        return self.linear(x)



class ResnetBlock(nn.Module):

    def __init__(self, in_c, out_c, down, up, ksize=3, sk=False, use_conv=True, enable_timestep=False, temb_channels=None, use_norm=False):
        super().__init__()
        self.use_norm = use_norm
        self.enable_timestep = enable_timestep
        ps = ksize // 2
        if in_c != out_c or sk == False:
            self.in_conv = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            self.in_conv = None
        self.block1 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.act = nn.ReLU()
        if use_norm:
            self.norm1 = nn.GroupNorm(num_groups=32, num_channels=out_c, eps=1e-6, affine=True)
        self.block2 = nn.Conv2d(out_c, out_c, ksize, 1, ps)
        if sk == False:
            self.skep = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            self.skep = None

        self.down = down
        self.up = up
        if self.down:
            self.down_opt = Downsample(in_c, use_conv=use_conv)
        if self.up:
            self.up_opt = Upsample(in_c, use_conv=use_conv)
        if enable_timestep:
            self.timestep_proj = Linear(temb_channels, out_c)


    def forward(self, x, output_size=None, temb=None):
        if self.down == True:
            x = self.down_opt(x)
        if self.up == True:
            x = self.up_opt(x, output_size)
        if self.in_conv is not None:  # edit
            x = self.in_conv(x)

        h = self.block1(x)
        if temb is not None:
            temb = self.timestep_proj(temb)[:, :, None, None]
            h = h + temb
        if self.use_norm:
            h = self.norm1(h)
        h = self.act(h)
        h = self.block2(h)
        if self.skep is not None:
            return h + self.skep(x)
        else:
            return h + x


class Adapter_XL(nn.Module):

    def __init__(self, in_channels=[1280, 640, 320], out_channels=[1280, 1280, 640], nums_rb=3, ksize=3, sk=True, use_conv=False, use_zero_conv=True,
                 enable_timestep=False, use_norm=False, temb_channels=None, fusion_type='ADD'):
        super(Adapter_XL, self).__init__()
        self.channels = in_channels
        self.nums_rb = nums_rb
        self.body = []
        self.out = []
        self.use_zero_conv = use_zero_conv
        self.fusion_type = fusion_type
        self.gamma = []
        self.beta = []
        self.norm = []
        if fusion_type == "SPADE":
            self.use_zero_conv = False
        for i in range(len(self.channels)):
            if self.fusion_type == 'SPADE':
                # Corresponding to SPADE <Semantic Image Synthesis with Spatially-Adaptive Normalization>
                self.gamma.append(nn.Conv2d(out_channels[i], out_channels[i], 1, padding=0))
                self.beta.append(nn.Conv2d(out_channels[i], out_channels[i], 1, padding=0))
                self.norm.append(nn.BatchNorm2d(out_channels[i]))
            elif use_zero_conv:
                self.out.append(self.make_zero_conv(out_channels[i]))
            else:
                self.out.append(nn.Conv2d(out_channels[i], out_channels[i], 1, padding=0))
            for j in range(nums_rb):
                if i==0:
                    # 1280, 32, 32 -> 1280, 32, 32
                    self.body.append(
                        ResnetBlock(in_channels[i], out_channels[i], down=False, up=False, ksize=ksize, sk=sk, use_conv=use_conv,
                                    enable_timestep=enable_timestep, temb_channels=temb_channels, use_norm=use_norm))
                    # 1280, 32, 32 -> 1280, 32, 32
                elif i==1:
                    # 640, 64, 64 -> 1280, 64, 64
                    if j==0:
                        self.body.append(
                            ResnetBlock(in_channels[i], out_channels[i], down=False, up=False, ksize=ksize, sk=sk,
                                        use_conv=use_conv, enable_timestep=enable_timestep, temb_channels=temb_channels, use_norm=use_norm))
                    else:
                        self.body.append(
                            ResnetBlock(out_channels[i], out_channels[i], down=False, up=False, ksize=ksize,sk=sk,
                                        use_conv=use_conv, enable_timestep=enable_timestep, temb_channels=temb_channels, use_norm=use_norm))
                else:
                    # 320, 64, 64 -> 640, 128, 128
                    if j==0:
                        self.body.append(
                            ResnetBlock(in_channels[i], out_channels[i], down=False, up=True, ksize=ksize, sk=sk,
                                        use_conv=True, enable_timestep=enable_timestep, temb_channels=temb_channels, use_norm=use_norm))
                        # use convtranspose2d
                    else:
                        self.body.append(
                            ResnetBlock(out_channels[i], out_channels[i], down=False, up=False, ksize=ksize, sk=sk,
                                        use_conv=use_conv,  enable_timestep=enable_timestep, temb_channels=temb_channels, use_norm=use_norm))


        self.body = nn.ModuleList(self.body)
        if self.use_zero_conv:
            self.zero_out = nn.ModuleList(self.out)

        # if self.fusion_type == 'SPADE':
        #     self.norm = nn.ModuleList(self.norm)
        #     self.gamma = nn.ModuleList(self.gamma)
        #     self.beta = nn.ModuleList(self.beta)
        # else:
        #     self.zero_out = nn.ModuleList(self.out)


        # if enable_timestep:
        #     a = 320
        #
        #     time_embed_dim = a * 4
        #     self.time_proj = Timesteps(a, True, 0)
        #     timestep_input_dim = a
        #
        #     self.time_embedding = TimestepEmbedding(
        #         timestep_input_dim,
        #         time_embed_dim,
        #         act_fn='silu',
        #         post_act_fn=None,
        #         cond_proj_dim=None,
        #     )


    def make_zero_conv(self, channels):

        return zero_module(nn.Conv2d(channels, channels, 1, padding=0))

    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)

    def forward(self, x, t=None):
        # extract features
        features = []
        b, c, _, _ = x[-1].shape
        if t is not None:
            if not torch.is_tensor(t):
                is_mps = x[0].device.type == "mps"
                if isinstance(timestep, float):
                    dtype = torch.float32 if is_mps else torch.float64
                else:
                    dtype = torch.int32 if is_mps else torch.int64
                t = torch.tensor([t], dtype=dtype, device=x[0].device)
            elif len(t.shape) == 0:
                t = t[None].to(x[0].device)

            t = t.expand(b)
            t = self.time_proj(t) # b, 320
            t = t.to(dtype=x[0].dtype)
            t = self.time_embedding(t)  # b, 1280
        output_size = (b, 640, 128, 128)  # last CA layer output
        for i in range(len(self.channels)):
            for j in range(self.nums_rb):
                idx = i * self.nums_rb + j
                if j == 0:
                    if i < 2:
                        out = self.body[idx](x[i], temb=t)
                    else:
                        out = self.body[idx](x[i], output_size=output_size, temb=t)
                else:
                    out = self.body[idx](out, temb=t)
            if self.fusion_type == 'SPADE':
                out_gamma = self.gamma[i](out)
                out_beta = self.beta[i](out)
                out = [out_gamma, out_beta]
            else:
                out = self.zero_out[i](out)
            features.append(out)

        return features


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module
