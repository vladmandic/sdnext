import torch
from modules.rife.model_ifnet import IFNet
from modules import devices


class RifeModel:
    def __init__(self, local_rank=-1):
        self.flownet = IFNet()
        self.device()
        self.version = 4.25
        self.tenFlow_div_cache = {}
        self.backwarp_tenGrid_cache = {}
        if local_rank != -1:
            from torch.nn.parallel import DistributedDataParallel as DDP
            self.flownet = DDP(self.flownet, device_ids=[local_rank], output_device=local_rank)

    def train(self):
        self.flownet.train()

    def eval(self):
        self.flownet.eval()

    def device(self):
        self.flownet.to(devices.device)
        self.flownet.to(torch.float32)  # bfloat16 produces visible checkerboard artifacts at the new IFNet's depth

    def load_model(self, model_file, rank=0):
        def convert(param):
            if rank == -1:
                return { k.replace("module.", ""): v for k, v in param.items() if "module." in k }
            else:
                return param
        if rank <= 0:
            if torch.cuda.is_available():
                self.flownet.load_state_dict(convert(torch.load(model_file)), False)
            else:
                self.flownet.load_state_dict(convert(torch.load(model_file, map_location='cpu')), False)

    def save_model(self, model_file, rank=0):
        if rank == 0:
            torch.save(self.flownet.state_dict(), model_file)

    def grid_for(self, h, w, device, dtype):
        key = (h, w, str(device), str(dtype))
        grid = self.backwarp_tenGrid_cache.get(key)
        if grid is None:
            tenHorizontal = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype).view(1, 1, 1, w).expand(1, -1, h, -1)
            tenVertical = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype).view(1, 1, h, 1).expand(1, -1, -1, w)
            grid = torch.cat([tenHorizontal, tenVertical], 1)
            self.backwarp_tenGrid_cache[key] = grid
        div = self.tenFlow_div_cache.get(key)
        if div is None:
            div = torch.tensor([(w - 1.0) / 2.0, (h - 1.0) / 2.0], device=device, dtype=dtype)
            self.tenFlow_div_cache[key] = div
        return grid, div

    def inference(self, img0, img1, timestep=0.5, scale=1.0):
        in_dtype = img0.dtype
        img0 = img0.float()
        img1 = img1.float()
        n, _c, h, w = img0.shape
        device = img0.device
        backwarp_tenGrid, tenFlow_div = self.grid_for(h, w, device, torch.float32)
        self.flownet.scale_list = [16 / scale, 8 / scale, 4 / scale, 2 / scale, 1 / scale]
        f0 = self.flownet.encode(img0)
        f1 = self.flownet.encode(img1)
        timestep_t = torch.full((n, 1, h, w), timestep, device=device, dtype=torch.float32)
        out = self.flownet(img0, img1, timestep_t, tenFlow_div, backwarp_tenGrid, f0, f1)
        return out.to(in_dtype)
