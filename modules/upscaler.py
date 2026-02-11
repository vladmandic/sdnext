import os
from abc import abstractmethod
from PIL import Image
from modules import modelloader, shared


models = None


class Upscaler:
    name = None
    folder = None
    model_path = None
    model_name = None
    model_url = None
    enable = True
    filter = None
    model = None
    user_path = None
    scalers = []
    tile = True

    def __init__(self, create_dirs=True):
        global models # pylint: disable=global-statement
        if models is None:
            models_file = os.path.join('data', 'upscalers.json')
            models = shared.readfile(models_file, as_type="dict")
        self.mod_pad_h = None
        self.tile_size = shared.opts.upscaler_tile_size
        self.tile_pad = shared.opts.upscaler_tile_overlap
        self.device = shared.device
        self.img = None
        self.output = None
        self.scale = 1
        self.half = not shared.cmd_opts.no_half
        self.pre_pad = 0
        self.mod_scale = None
        self.model_download_path = None
        if self.user_path is not None and len(self.user_path) > 0 and not os.path.exists(self.user_path):
            shared.log.info(f'Upscaler create: folder="{self.user_path}"')
        if self.model_path is None and self.name:
            self.model_path = os.path.join(shared.models_path, self.name)
        try:
            if self.model_path and create_dirs:
                os.makedirs(self.model_path, exist_ok=True)
        except Exception:
            pass
        try:
            import cv2  # pylint: disable=unused-import
            self.can_tile = True
        except Exception:
            pass

    def find_folder(self, folder, scalers, loaded):
        for fn in os.listdir(folder): # from folder
            file_name = os.path.join(folder, fn)
            if os.path.isdir(file_name):
                self.find_folder(file_name, scalers, loaded)
                continue
            if not file_name.endswith('.pth') and not file_name.endswith('.pt'):
                continue
            if file_name not in loaded:
                model_name = os.path.splitext(fn)[0]
                scaler = UpscalerData(name=f'{self.name} {model_name}', path=file_name, upscaler=self)
                scaler.custom = True
                scalers.append(scaler)
                loaded.append(file_name)
                # shared.log.debug(f'Upscaler type={self.name} folder="{folder}" model="{model_name}" path="{file_name}"')

    def find_scalers(self):
        scalers = []
        loaded = []
        for k, v in models.items(): # from config
            if k != self.name:
                continue
            for model in v:
                local_name = os.path.join(self.user_path, modelloader.friendly_fullname(model[1]))
                model_path = local_name if os.path.exists(local_name) else model[1]
                scaler = UpscalerData(name=f'{k} {model[0]}', path=model_path, upscaler=self)
                scalers.append(scaler)
                loaded.append(model_path)
                # shared.log.debug(f'Upscaler type={self.name} folder="{self.user_path}" model="{model[0]}" path="{model_path}"')
        if self.user_path is None or not os.path.exists(self.user_path):
            return scalers
        self.find_folder(self.user_path, scalers, loaded)
        return scalers

    @abstractmethod
    def do_upscale(self, img: Image, selected_model: str):
        return img

    def upscale(self, img: Image, scale, selected_model: str = None):
        jobid = shared.state.begin('Upscale')
        self.scale = scale
        if isinstance(img, Image.Image):
            dest_w = int(img.width * scale)
            dest_h = int(img.height * scale)
        else:
            dest_w = int(img.shape[-1] * scale)
            dest_h = int(img.shape[-2] * scale)
        if self.name.lower().startswith('latent'):
            img = self.do_upscale(img, selected_model)
        else:
            for _ in range(3):
                shape = (img.width, img.height)
                img = self.do_upscale(img, selected_model)
                if shape == (img.width, img.height):
                    break
                if img.width >= dest_w and img.height >= dest_h:
                    break
            if img.width != dest_w or img.height != dest_h:
                from modules import images_sharpfin
                img = images_sharpfin.resize(img, (int(dest_w), int(dest_h)))
        shared.state.end(jobid)
        return img

    @abstractmethod
    def load_model(self, path: str):
        pass

    def find_models(self, ext_filter=None) -> list: # pylint: disable=unused-argument
        return modelloader.load_models(model_path=self.model_path, model_url=self.model_url, command_path=self.user_path)

    def update_status(self, prompt):
        shared.log.info(f'Upscaler: type={self.name} model="{prompt}"')

    def find_model(self, path):
        info = None
        for scaler in self.scalers:
            if (scaler.data_path == path) or (scaler.name == path):
                info = scaler
                break
        if info is None:
            shared.log.error(f'Upscaler cannot match model: type={self.name} model="{path}"')
            return None
        if info.local_data_path.startswith("http"):
            from modules.modelloader import load_file_from_url
            info.local_data_path = load_file_from_url(url=info.data_path, model_dir=self.model_download_path, progress=True)
        if not os.path.isfile(info.local_data_path):
            shared.log.error(f'Upscaler cannot find model: type={self.name} model="{info.local_data_path}"')
            return None
        return info


class UpscalerData:
    custom: bool = False
    name = None
    data_path = None
    scale: int = 4
    scaler: Upscaler = None
    model: None

    def __init__(self, name: str, path: str = None, upscaler: Upscaler = None, scale: int = 4, model=None):
        self.name = name
        self.data_path = path
        self.local_data_path = path
        self.scaler = upscaler
        self.scale = scale
        self.model = model


def compile_upscaler(model):
    if "Upscaler" in shared.opts.ipex_optimize:
        try:
            from modules.sd_models_compile import ipex_optimize
            model = ipex_optimize(model, apply_to_components=False, op="Upscaler")
        except Exception as e:
            shared.log.warning(f"Upscaler IPEX Optimize: error: {e}")

    if "Upscaler" in shared.opts.cuda_compile:
        try:
            from modules.sd_models_compile import compile_torch
            model = compile_torch(model, apply_to_components=False, op="Upscaler")
        except Exception as e:
            shared.log.warning(f"Upscaler compile error: {e}")
    return model
