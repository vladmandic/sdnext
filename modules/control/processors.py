import os
import time
import numpy as np
from PIL import Image
from modules.logger import log
from modules.errors import display
from modules import devices, images


models = {}
cache_dir = 'models/control/processors'
debug = log.trace if os.environ.get('SD_CONTROL_DEBUG', None) is not None else lambda *args, **kwargs: None
debug('Trace: CONTROL')
config = {
    # placeholder
    'None': {},
    # pose models
    'OpenPose': {'class': None, 'group': 'Pose', 'checkpoint': True, 'params': {'include_body': True, 'include_hand': False, 'include_face': False}},
    'MediaPipe Face': {'class': None, 'group': 'Pose', 'checkpoint': False, 'params': {'max_faces': 1, 'min_confidence': 0.5}},
    'DWPose (ONNX)': {'class': None, 'group': 'Pose', 'checkpoint': False, 'params': {'min_confidence': 0.3}},
    'RTMW': {'class': None, 'group': 'Pose', 'checkpoint': False, 'params': {'min_confidence': 0.3, 'draw_body_pose': True, 'draw_hand_pose': True, 'draw_face_pose': True}},
    'RTMO': {'class': None, 'group': 'Pose', 'checkpoint': False, 'params': {'min_confidence': 0.3}},
    'ViTPose': {'class': None, 'group': 'Pose', 'checkpoint': True, 'load_config': {'pretrained_model_or_path': 'usyd-community/vitpose-plus-base'}, 'params': {'min_confidence': 0.3}},
    # edge models
    'Canny': {'class': None, 'group': 'Edge', 'checkpoint': False, 'params': {'low_threshold': 100, 'high_threshold': 200}},
    'Edge': {'class': None, 'group': 'Edge', 'checkpoint': False, 'params': {'pf': True, 'mode': 'edge'}},
    'LineArt Realistic': {'class': None, 'group': 'Edge', 'checkpoint': True, 'params': {'coarse': False}},
    'LineArt Anime': {'class': None, 'group': 'Edge', 'checkpoint': True, 'params': {}},
    'HED': {'class': None, 'group': 'Edge', 'checkpoint': True, 'params': {'scribble': False, 'safe': False}},
    'PidiNet': {'class': None, 'group': 'Edge', 'checkpoint': True, 'params': {'scribble': False, 'safe': False, 'apply_filter': False}},
    'MLSD': {'class': None, 'group': 'Edge', 'checkpoint': True, 'params': {'thr_v': 0.1, 'thr_d': 0.1}},
    'TEED': {'class': None, 'group': 'Edge', 'checkpoint': True, 'load_config': {'pretrained_model_or_path': 'fal/teed'}, 'params': {}},
    'Anyline': {'class': None, 'group': 'Edge', 'checkpoint': True, 'load_config': {'pretrained_model_or_path': 'TheMistoAI/MistoLine'}, 'params': {}},
    # depth models
    'Midas Depth Hybrid': {'class': None, 'group': 'Depth', 'checkpoint': True, 'params': {'bg_th': 0.1, 'depth_and_normal': False}},
    'Leres Depth': {'class': None, 'group': 'Depth', 'checkpoint': True, 'params': {'boost': False, 'thr_a': 0, 'thr_b': 0}},
    'Zoe Depth': {'class': None, 'group': 'Depth', 'checkpoint': True, 'params': {'gamma_corrected': False}, 'load_config': {'pretrained_model_or_path': 'halffried/gyre_zoedepth', 'filename': 'ZoeD_M12_N.safetensors', 'model_type': "zoedepth"}},
    'Marigold Depth': {'class': None, 'group': 'Depth', 'checkpoint': True, 'params': {'denoising_steps': 4, 'ensemble_size': 4, 'processing_res': 768, 'match_input_res': True, 'color_map': 'None'}, 'load_config': {'pretrained_model_or_path': 'prs-eth/marigold-depth-v1-1'}},
    'DPT Depth Hybrid': {'class': None, 'group': 'Depth', 'checkpoint': False, 'params': {}},
    'GLPN Depth': {'class': None, 'group': 'Depth', 'checkpoint': False, 'params': {}},
    'Depth Anything': {'class': None, 'group': 'Depth', 'checkpoint': True, 'load_config': {'pretrained_model_or_path': 'LiheYoung/depth_anything_vitl14'}, 'params': {'color_map': 'inferno'}},
    'Depth Pro': {'class': None, 'group': 'Depth', 'checkpoint': True, 'load_config': {'pretrained_model_or_path': 'apple/DepthPro-hf'}, 'params': {'color_map': 'inferno'}},
    'Depth Anything V2 Small': {'class': None, 'group': 'Depth', 'checkpoint': True, 'load_config': {'pretrained_model_or_path': 'depth-anything/Depth-Anything-V2-Small-hf'}, 'params': {'color_map': 'inferno'}},
    'Depth Anything V2 Large': {'class': None, 'group': 'Depth', 'checkpoint': True, 'load_config': {'pretrained_model_or_path': 'depth-anything/Depth-Anything-V2-Large-hf'}, 'params': {'color_map': 'inferno'}},
    'Marigold Depth LCM': {'class': None, 'group': 'Depth', 'checkpoint': True, 'params': {'denoising_steps': 1, 'ensemble_size': 1, 'processing_res': 768, 'match_input_res': True, 'color_map': 'None'}, 'load_config': {'pretrained_model_or_path': 'prs-eth/marigold-depth-lcm-v1-0'}},
    'Lotus Depth': {'class': None, 'group': 'Depth', 'checkpoint': True, 'load_config': {'pretrained_model_or_path': 'jingheya/lotus-depth-g-v2-1-disparity'}, 'params': {'color_map': 'inferno'}},
    # normal models
    'Normal Bae': {'class': None, 'group': 'Normal', 'checkpoint': True, 'params': {}},
    'DSINE': {'class': None, 'group': 'Normal', 'checkpoint': True, 'load_config': {'pretrained_model_or_path': 'hugoycj/DSINE-hub'}, 'params': {}},
    'StableNormal': {'class': None, 'group': 'Normal', 'checkpoint': True, 'load_config': {'pretrained_model_or_path': 'Stable-X/StableNormal'}, 'params': {}},
    'Marigold Normals': {'class': None, 'group': 'Normal', 'checkpoint': True, 'params': {'denoising_steps': 4, 'ensemble_size': 4, 'processing_res': 768, 'match_input_res': True}, 'load_config': {'pretrained_model_or_path': 'prs-eth/marigold-normals-v1-1'}},
    # segmentation models
    'SegmentAnything': {'class': None, 'group': 'Segmentation', 'checkpoint': True, 'model': 'Base', 'params': {}},
    'SAM 2.1': {'class': None, 'group': 'Segmentation', 'checkpoint': True, 'model': 'Large', 'load_config': {'pretrained_model_or_path': 'facebook/sam2.1-hiera-large'}, 'params': {}},
    'OneFormer': {'class': None, 'group': 'Segmentation', 'checkpoint': True, 'load_config': {'pretrained_model_or_path': 'shi-labs/oneformer_ade20k_swin_large'}, 'params': {}},
    # other models
    'Shuffle': {'class': None, 'group': 'Other', 'checkpoint': False, 'params': {}},
}


def delay_load_config():
    global config # pylint: disable=global-statement
    from modules.control.proc.hed import HEDdetector
    from modules.control.proc.canny import CannyDetector
    from modules.control.proc.edge import EdgeDetector
    from modules.control.proc.lineart import LineartDetector
    from modules.control.proc.lineart_anime import LineartAnimeDetector
    from modules.control.proc.pidi import PidiNetDetector
    from modules.control.proc.mediapipe_face import MediapipeFaceDetector
    from modules.control.proc.shuffle import ContentShuffleDetector
    from modules.control.proc.leres import LeresDetector
    from modules.control.proc.midas import MidasDetector
    from modules.control.proc.mlsd import MLSDdetector
    from modules.control.proc.openpose import OpenposeDetector
    from modules.control.proc.segment_anything import SamDetector
    from modules.control.proc.zoe import ZoeDetector
    from modules.control.proc.marigold import MarigoldDetector
    from modules.control.proc.dpt import DPTDetector
    from modules.control.proc.glpn import GLPNDetector
    from modules.control.proc.depth_anything import DepthAnythingDetector
    from modules.control.proc.depth_pro import DepthProDetector
    from modules.control.proc.depth_anything_v2 import DepthAnythingV2Detector
    from modules.control.proc.teed import TEEDDetector
    from modules.control.proc.anyline import AnylineDetector
    from modules.control.proc.rtmlib_pose import RtmlibPoseDetector
    from modules.control.proc.vitpose import ViTPoseDetector
    from modules.control.proc.sam2 import Sam2Detector
    from modules.control.proc.oneformer import OneFormerDetector
    from modules.control.proc.dsine import DSINEDetector
    from modules.control.proc.stablenormal import StableNormalDetector
    from modules.control.proc.marigold_normals import MarigoldNormalsDetector
    from modules.control.proc.lotus import LotusDetector
    config = {
        # placeholder
        'None': {},
        # pose models
        'OpenPose': {'class': OpenposeDetector, 'group': 'Pose', 'checkpoint': True, 'params': {'include_body': True, 'include_hand': False, 'include_face': False}},
        'MediaPipe Face': {'class': MediapipeFaceDetector, 'group': 'Pose', 'checkpoint': False, 'params': {'max_faces': 1, 'min_confidence': 0.5}},
        'DWPose (ONNX)': {'class': RtmlibPoseDetector, 'group': 'Pose', 'checkpoint': False, 'params': {'min_confidence': 0.3}},
        'RTMW': {'class': RtmlibPoseDetector, 'group': 'Pose', 'checkpoint': False, 'params': {'min_confidence': 0.3, 'draw_body_pose': True, 'draw_hand_pose': True, 'draw_face_pose': True}},
        'RTMO': {'class': RtmlibPoseDetector, 'group': 'Pose', 'checkpoint': False, 'params': {'min_confidence': 0.3}},
        'ViTPose': {'class': ViTPoseDetector, 'group': 'Pose', 'checkpoint': True, 'load_config': {'pretrained_model_or_path': 'usyd-community/vitpose-plus-base'}, 'params': {'min_confidence': 0.3}},
        # edge models
        'Canny': {'class': CannyDetector, 'group': 'Edge', 'checkpoint': False, 'params': {'low_threshold': 100, 'high_threshold': 200}},
        'Edge': {'class': EdgeDetector, 'group': 'Edge', 'checkpoint': False, 'params': {'pf': True, 'mode': 'edge'}},
        'LineArt Realistic': {'class': LineartDetector, 'group': 'Edge', 'checkpoint': True, 'params': {'coarse': False}},
        'LineArt Anime': {'class': LineartAnimeDetector, 'group': 'Edge', 'checkpoint': True, 'params': {}},
        'HED': {'class': HEDdetector, 'group': 'Edge', 'checkpoint': True, 'params': {'scribble': False, 'safe': False}},
        'PidiNet': {'class': PidiNetDetector, 'group': 'Edge', 'checkpoint': True, 'params': {'scribble': False, 'safe': False, 'apply_filter': False}},
        'MLSD': {'class': MLSDdetector, 'group': 'Edge', 'checkpoint': True, 'params': {'thr_v': 0.1, 'thr_d': 0.1}},
        'TEED': {'class': TEEDDetector, 'group': 'Edge', 'checkpoint': True, 'load_config': {'pretrained_model_or_path': 'fal/teed'}, 'params': {}},
        'Anyline': {'class': AnylineDetector, 'group': 'Edge', 'checkpoint': True, 'load_config': {'pretrained_model_or_path': 'TheMistoAI/MistoLine'}, 'params': {}},
        # depth models
        'Midas Depth Hybrid': {'class': MidasDetector, 'group': 'Depth', 'checkpoint': True, 'params': {'bg_th': 0.1, 'depth_and_normal': False}},
        'Leres Depth': {'class': LeresDetector, 'group': 'Depth', 'checkpoint': True, 'params': {'boost': False, 'thr_a': 0, 'thr_b': 0}},
        'Zoe Depth': {'class': ZoeDetector, 'group': 'Depth', 'checkpoint': True, 'params': {'gamma_corrected': False}, 'load_config': {'pretrained_model_or_path': 'halffried/gyre_zoedepth', 'filename': 'ZoeD_M12_N.safetensors', 'model_type': "zoedepth"}},
        'Marigold Depth': {'class': MarigoldDetector, 'group': 'Depth', 'checkpoint': True, 'params': {'denoising_steps': 4, 'ensemble_size': 4, 'processing_res': 768, 'match_input_res': True, 'color_map': 'None'}, 'load_config': {'pretrained_model_or_path': 'prs-eth/marigold-depth-v1-1'}},
        'DPT Depth Hybrid': {'class': DPTDetector, 'group': 'Depth', 'checkpoint': False, 'params': {}},
        'GLPN Depth': {'class': GLPNDetector, 'group': 'Depth', 'checkpoint': False, 'params': {}},
        'Depth Anything': {'class': DepthAnythingDetector, 'group': 'Depth', 'checkpoint': True, 'load_config': {'pretrained_model_or_path': 'LiheYoung/depth_anything_vitl14'}, 'params': {'color_map': 'inferno'}},
        'Depth Pro': {'class': DepthProDetector, 'group': 'Depth', 'checkpoint': True, 'load_config': {'pretrained_model_or_path': 'apple/DepthPro-hf'}, 'params': {'color_map': 'inferno'}},
        'Depth Anything V2 Small': {'class': DepthAnythingV2Detector, 'group': 'Depth', 'checkpoint': True, 'load_config': {'pretrained_model_or_path': 'depth-anything/Depth-Anything-V2-Small-hf'}, 'params': {'color_map': 'inferno'}},
        'Depth Anything V2 Large': {'class': DepthAnythingV2Detector, 'group': 'Depth', 'checkpoint': True, 'load_config': {'pretrained_model_or_path': 'depth-anything/Depth-Anything-V2-Large-hf'}, 'params': {'color_map': 'inferno'}},
        'Marigold Depth LCM': {'class': MarigoldDetector, 'group': 'Depth', 'checkpoint': True, 'params': {'denoising_steps': 1, 'ensemble_size': 1, 'processing_res': 768, 'match_input_res': True, 'color_map': 'None'}, 'load_config': {'pretrained_model_or_path': 'prs-eth/marigold-depth-lcm-v1-0'}},
        'Lotus Depth': {'class': LotusDetector, 'group': 'Depth', 'checkpoint': True, 'load_config': {'pretrained_model_or_path': 'jingheya/lotus-depth-g-v2-1-disparity'}, 'params': {'color_map': 'inferno'}},
        # normal models
        'Normal Bae': {'class': None, 'group': 'Normal', 'checkpoint': True, 'params': {}},
        'DSINE': {'class': DSINEDetector, 'group': 'Normal', 'checkpoint': True, 'load_config': {'pretrained_model_or_path': 'hugoycj/DSINE-hub'}, 'params': {}},
        'StableNormal': {'class': StableNormalDetector, 'group': 'Normal', 'checkpoint': True, 'load_config': {'pretrained_model_or_path': 'Stable-X/StableNormal'}, 'params': {}},
        'Marigold Normals': {'class': MarigoldNormalsDetector, 'group': 'Normal', 'checkpoint': True, 'params': {'denoising_steps': 4, 'ensemble_size': 4, 'processing_res': 768, 'match_input_res': True}, 'load_config': {'pretrained_model_or_path': 'prs-eth/marigold-normals-v1-1'}},
        # segmentation models
        'SegmentAnything': {'class': SamDetector, 'group': 'Segmentation', 'checkpoint': True, 'model': 'Base', 'params': {}},
        'SAM 2.1': {'class': Sam2Detector, 'group': 'Segmentation', 'checkpoint': True, 'model': 'Large', 'load_config': {'pretrained_model_or_path': 'facebook/sam2.1-hiera-large'}, 'params': {}},
        'OneFormer': {'class': OneFormerDetector, 'group': 'Segmentation', 'checkpoint': True, 'load_config': {'pretrained_model_or_path': 'shi-labs/oneformer_ade20k_swin_large'}, 'params': {}},
        # other models
        'Shuffle': {'class': ContentShuffleDetector, 'group': 'Other', 'checkpoint': False, 'params': {}},
    }


def list_models(refresh=False):
    global models # pylint: disable=global-statement
    if not refresh and len(models) > 0:
        return models
    models = list(config)
    debug(f'Control list processors: path={cache_dir} models={models}')
    return models


def update_settings(*settings):
    debug(f'Control settings: {settings}')
    def update(what, val):
        processor_id = what[0]
        if len(what) == 2 and config[processor_id][what[1]] != val:
            config[processor_id][what[1]] = val
            config[processor_id]['dirty'] = True
            log.debug(f'Control settings: id="{processor_id}" {what[-1]}={val}')
        elif len(what) == 3 and config[processor_id][what[1]][what[2]] != val:
            config[processor_id][what[1]][what[2]] = val
            config[processor_id]['dirty'] = True
            log.debug(f'Control settings: id="{processor_id}" {what[-1]}={val}')
        elif len(what) == 4 and config[processor_id][what[1]][what[2]][what[3]] != val:
            config[processor_id][what[1]][what[2]][what[3]] = val
            config[processor_id]['dirty'] = True
            log.debug(f'Control settings: id="{processor_id}" {what[-1]}={val}')

    update(['HED', 'params', 'scribble'], settings[0])
    update(['Midas Depth Hybrid', 'params', 'bg_th'], settings[1])
    update(['Midas Depth Hybrid', 'params', 'depth_and_normal'], settings[2])
    update(['MLSD', 'params', 'thr_v'], settings[3])
    update(['MLSD', 'params', 'thr_d'], settings[4])
    update(['OpenPose', 'params', 'include_body'], settings[5])
    update(['OpenPose', 'params', 'include_hand'], settings[6])
    update(['OpenPose', 'params', 'include_face'], settings[7])
    update(['PidiNet', 'params', 'scribble'], settings[8])
    update(['PidiNet', 'params', 'apply_filter'], settings[9])
    update(['LineArt Realistic', 'params', 'coarse'], settings[10])
    update(['Leres Depth', 'params', 'boost'], settings[11])
    update(['Leres Depth', 'params', 'thr_a'], settings[12])
    update(['Leres Depth', 'params', 'thr_b'], settings[13])
    update(['MediaPipe Face', 'params', 'max_faces'], settings[14])
    update(['MediaPipe Face', 'params', 'min_confidence'], settings[15])
    update(['Canny', 'params', 'low_threshold'], settings[16])
    update(['Canny', 'params', 'high_threshold'], settings[17])
    update(['DWPose', 'model'], settings[18])
    update(['DWPose', 'params', 'min_confidence'], settings[19])
    update(['SegmentAnything', 'model'], settings[20])
    update(['Edge', 'params', 'pf'], settings[21])
    update(['Edge', 'params', 'mode'], settings[22])
    update(['Zoe Depth', 'params', 'gamma_corrected'], settings[23])
    update(['Marigold Depth', 'params', 'color_map'], settings[24])
    update(['Marigold Depth', 'params', 'denoising_steps'], settings[25])
    update(['Marigold Depth', 'params', 'ensemble_size'], settings[26])
    update(['Depth Anything', 'params', 'color_map'], settings[27])
    update(['Depth Pro', 'params', 'color_map'], settings[28])


class Processor:
    def __init__(self, processor_id: str | None = None, resize = True):
        self.model = None
        self.processor_id: str | None = None
        self.override: Image.Image | None = None
        self.resize = resize
        self.reset()
        self.config(processor_id)
        if processor_id is not None:
            self.load()

    def __str__(self):
        return f' Processor(id={self.processor_id} model={self.model.__class__.__name__})' if self.processor_id and self.model else ''

    def reset(self, processor_id: str | None = None):
        if self.model is not None:
            debug(f'Control Processor unloaded: id="{self.processor_id}"')
            self.model = None
            self.processor_id = processor_id
            devices.torch_gc(force=True, reason='processor')
        self.load_config = { 'cache_dir': cache_dir }
        from modules.shared import opts
        if opts.offline_mode:
            self.load_config["local_files_only"] = True
            os.environ['HF_HUB_OFFLINE'] = '1'
        else:
            os.environ.pop('HF_HUB_OFFLINE', None)
            os.unsetenv('HF_HUB_OFFLINE')


    def config(self, processor_id = None):
        if processor_id is not None:
            self.processor_id = processor_id
        from_config = config.get(self.processor_id, {}).get('load_config', None)
        """
        if load_config is not None:
            for k, v in load_config.items():
                self.load_config[k] = v
        """
        if from_config is not None:
            for k, v in from_config.items():
                self.load_config[k] = v

    def load(self, processor_id: str | None = None, force: bool = True) -> str:
        from modules.shared import state
        try:
            t0 = time.time()
            processor_id = processor_id or self.processor_id
            if processor_id is None or processor_id == 'None':
                self.reset()
                return ''
            if self.processor_id != processor_id:
                self.reset()
                self.config(processor_id)
            else:
                if not force and self.model is not None:
                    # log.debug(f'Control Processor: id={processor_id} already loaded')
                    return ''
            if processor_id not in config:
                log.error(f'Control Processor unknown: id="{processor_id}" available={list(config)}')
                return f'Processor failed to load: {processor_id}'
            cls = config[processor_id]['class']
            if cls is None:
                delay_load_config()
                cls = config[processor_id]['class']
            # log.debug(f'Control Processor loading: id="{processor_id}" class={cls.__name__}')
            debug(f'Control Processor config={self.load_config}')
            jobid = state.begin('Load processor')
            if processor_id == 'DWPose':
                det_ckpt = 'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth'
                if 'Tiny' == config['DWPose']['model']:
                    pose_config = 'config/rtmpose-t_8xb64-270e_coco-ubody-wholebody-256x192.py'
                    pose_ckpt = 'https://huggingface.co/yzd-v/DWPose/resolve/main/dw-tt_ucoco.pth'
                elif 'Medium' == config['DWPose']['model']:
                    pose_config = 'config/rtmpose-m_8xb64-270e_coco-ubody-wholebody-256x192.py'
                    pose_ckpt = 'https://huggingface.co/yzd-v/DWPose/resolve/main/dw-mm_ucoco.pth'
                elif 'Large' == config['DWPose']['model']:
                    pose_config = 'config/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py'
                    pose_ckpt = 'https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.pth'
                else:
                    log.error(f'Control Processor load failed: id="{processor_id}" error=unknown model type')
                    return f'Processor failed to load: {processor_id}'
                self.model = cls(det_ckpt=det_ckpt, pose_config=pose_config, pose_ckpt=pose_ckpt, device="cpu")
            elif processor_id in ('DWPose (ONNX)', 'RTMW', 'RTMO'):
                model_type = {'DWPose (ONNX)': 'DWPose', 'RTMW': 'RTMW-l', 'RTMO': 'RTMO-l'}[processor_id]
                self.model = cls.from_pretrained(model_type, **self.load_config)
            elif 'SegmentAnything' in processor_id:
                if 'Base' == config['SegmentAnything']['model']:
                    self.model = cls.from_pretrained(model_path = 'segments-arnaud/sam_vit_b', filename='sam_vit_b_01ec64.pth', model_type='vit_b', **self.load_config)
                elif 'Large' == config['SegmentAnything']['model']:
                    self.model = cls.from_pretrained(model_path = 'segments-arnaud/sam_vit_l', filename='sam_vit_l_0b3195.pth', model_type='vit_l', **self.load_config)
                else:
                    log.error(f'Control Processor load failed: id="{processor_id}" error=unknown model type')
                    return f'Processor failed to load: {processor_id}'
            elif config[processor_id].get('load_config', None) is not None:
                self.model = cls.from_pretrained(**self.load_config)
            elif config[processor_id]['checkpoint']:
                self.model = cls.from_pretrained("lllyasviel/Annotators", **self.load_config)
            else:
                self.model = cls() # class instance only
            t1 = time.time()
            state.end(jobid)
            self.processor_id = processor_id
            log.debug(f'Control Processor loaded: id="{processor_id}" class={self.model.__class__.__name__} time={t1-t0:.2f}')
            return f'Processor loaded: {processor_id}'
        except Exception as e:
            log.error(f'Control Processor load failed: id="{processor_id}" error={e}')
            display(e, 'Control Processor load')
            return f'Processor load filed: {processor_id}'

    def __call__(self, image_input: Image, mode: str = 'RGB', width: int = 0, height: int = 0, resize_mode: int = 0, resize_name: str = 'None', scale_tab: int = 1, scale_by: float = 1.0, local_config: dict | None = None):
        """Run the preprocessor on an input image and return the processed control map.

        Args:
            image_input: Source image to preprocess.
            mode: Output color mode ('RGB', 'L', etc.).
            width, height: Target dimensions for resize when an override image is provided.
            resize_mode: Resize strategy index (0 = no resize).
            resize_name: Resize algorithm name ('None' to skip).
            scale_tab: Scale mode selector (1 = scale by multiplier).
            scale_by: Scale multiplier when scale_tab is 1.
            local_config: Per-call parameter overrides merged on top of the processor's global
                config[processor_id]['params']. Keys must match the processor's accepted kwargs
                (e.g. {'low_threshold': 50, 'high_threshold': 150} for Canny). Passed from
                the API via Unit.process_params.
        """
        if local_config is None:
            local_config = {}
        if self.override is not None:
            debug(f'Control Processor: id="{self.processor_id}" override={self.override}')
            width = image_input.width if image_input is not None else width
            height = image_input.height if image_input is not None else height
            if (width != self.override.width) or (height != self.override.height):
                debug(f'Control resize: op=override image={self.override} width={width} height={height} mode={resize_mode} name={resize_name}')
                image_input = images.resize_image(resize_mode, self.override, width, height, resize_name)
            else:
                image_input = self.override
            if resize_mode != 0 and resize_name != 'None':
                if scale_tab == 1:
                    width_before, height_before = int(image_input.width * scale_by), int(image_input.height * scale_by)
                    debug(f'Control resize: op=before image={image_input} width={width_before} height={height_before} mode={resize_mode} name={resize_name}')
                    image_input = images.resize_image(resize_mode, image_input, width_before, height_before, resize_name)
        if self.processor_id is None or self.processor_id == 'None':
            return image_input
        image_process = image_input
        if image_input is None:
            # log.error('Control Processor: no input')
            return image_process
        if isinstance(image_input, list):
            image_input = image_input[0]
        if self.processor_id not in config:
            return image_process
        if config[self.processor_id].get('dirty', False):
            processor_id = self.processor_id
            config[processor_id].pop('dirty')
            self.reset()
            self.load(processor_id)
        if self.model is None:
            # log.error('Control Processor: model not loaded')
            return image_process
        try:
            t0 = time.time()
            kwargs = dict(config.get(self.processor_id, {}).get('params', {}))
            if local_config:
                kwargs.update(local_config)
            if self.resize:
                image_resized = image_input.resize((512, 512), Image.Resampling.LANCZOS)
            else:
                image_resized = image_input
            with devices.inference_context():
                image_process = self.model(image_resized, **kwargs)
            if image_process is None:
                log.error(f'Control Processor: id="{self.processor_id}" no image')
                return image_input
            if isinstance(image_process, np.ndarray):
                if np.max(image_process) < 2:
                    image_process = (255.0 * image_process).astype(np.uint8)
                image_process = Image.fromarray(image_process, 'L')
            if self.resize and image_process.size != image_input.size:
                image_process = image_process.resize(image_input.size, Image.Resampling.LANCZOS)
            t1 = time.time()
            log.debug(f'Control Processor: id="{self.processor_id}" mode={mode} args={kwargs} time={t1-t0:.2f}')
        except Exception as e:
            log.error(f'Control Processor failed: id="{self.processor_id}" error={e}')
            display(e, 'Control Processor')
        if mode != 'RGB':
            image_process = image_process.convert(mode)
        return image_process

    def preview(self):
        import modules.ui_control_helpers as helpers
        input_image = helpers.input_source
        if isinstance(input_image, list):
            input_image = input_image[0]
        debug('Control process preview')
        return self.__call__(input_image)
