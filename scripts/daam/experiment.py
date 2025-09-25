from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass
import json

from transformers import PreTrainedTokenizer, AutoTokenizer
import PIL.Image
import numpy as np
import torch

from .utils import auto_autocast
from .evaluate import load_mask


__all__ = ['GenerationExperiment', 'COCO80_LABELS', 'COCOSTUFF27_LABELS', 'COCO80_INDICES', 'build_word_list_coco80']


COCO80_LABELS: List[str] = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

COCO80_INDICES: Dict[str, int] = {x: i for i, x in enumerate(COCO80_LABELS)}

UNUSED_LABELS: List[str] = [f'__unused_{i}__' for i in range(1, 200)]

COCOSTUFF27_LABELS: List[str] = [
    'electronic', 'appliance', 'food', 'furniture', 'indoor', 'kitchen', 'accessory', 'animal', 'outdoor', 'person',
    'sports', 'vehicle', 'ceiling', 'floor', 'food', 'furniture', 'rawmaterial', 'textile', 'wall', 'window',
    'building', 'ground', 'plant', 'sky', 'solid', 'structural', 'water'
]

COCO80_ONTOLOGY = {
    'two-wheeled vehicle': ['bicycle', 'motorcycle'],
    'vehicle': ['two-wheeled vehicle', 'four-wheeled vehicle'],
    'four-wheeled vehicle': ['bus', 'truck', 'car'],
    'four-legged animals': ['livestock', 'pets', 'wild animals'],
    'livestock': ['cow', 'horse', 'sheep'],
    'pets': ['cat', 'dog'],
    'wild animals': ['elephant', 'bear', 'zebra', 'giraffe'],
    'bags': ['backpack', 'handbag', 'suitcase'],
    'sports boards': ['snowboard', 'surfboard', 'skateboard'],
    'utensils': ['fork', 'knife', 'spoon'],
    'receptacles': ['bowl', 'cup'],
    'fruits': ['banana', 'apple', 'orange'],
    'foods': ['fruits', 'meals', 'desserts'],
    'meals': ['sandwich', 'hot dog', 'pizza'],
    'desserts': ['cake', 'donut'],
    'furniture': ['chair', 'couch', 'bench'],
    'electronics': ['monitors', 'appliances'],
    'monitors': ['tv', 'cell phone', 'laptop'],
    'appliances': ['oven', 'toaster', 'refrigerator']
}

COCO80_TO_27 = {
    'bicycle': 'vehicle', 'car': 'vehicle', 'motorcycle': 'vehicle', 'airplane': 'vehicle', 'bus': 'vehicle',
    'train': 'vehicle', 'truck': 'vehicle', 'boat': 'vehicle', 'traffic light': 'accessory', 'fire hydrant': 'accessory',
    'stop sign': 'accessory', 'parking meter': 'accessory', 'bench': 'furniture', 'bird': 'animal', 'cat': 'animal',
    'dog': 'animal', 'horse': 'animal', 'sheep': 'animal', 'cow': 'animal', 'elephant': 'animal', 'bear': 'animal',
    'zebra': 'animal', 'giraffe': 'animal', 'backpack': 'accessory', 'umbrella': 'accessory', 'handbag': 'accessory',
    'tie': 'accessory', 'suitcase': 'accessory', 'frisbee': 'sports', 'skis': 'sports', 'snowboard': 'sports',
    'sports ball': 'sports', 'kite': 'sports', 'baseball bat': 'sports', 'baseball glove': 'sports',
    'skateboard': 'sports', 'surfboard': 'sports', 'tennis racket': 'sports', 'bottle': 'food', 'wine glass': 'food',
    'cup': 'food', 'fork': 'food', 'knife': 'food', 'spoon': 'food', 'bowl': 'food', 'banana': 'food', 'apple': 'food',
    'sandwich': 'food', 'orange': 'food', 'broccoli': 'food', 'carrot': 'food', 'hot dog': 'food', 'pizza': 'food',
    'donut': 'food', 'cake': 'food', 'chair': 'furniture', 'couch': 'furniture', 'potted plant': 'plant',
    'bed': 'furniture', 'dining table': 'furniture', 'toilet': 'furniture', 'tv': 'electronic', 'laptop': 'electronic',
    'mouse': 'electronic', 'remote': 'electronic', 'keyboard': 'electronic', 'cell phone': 'electronic',
    'microwave': 'appliance', 'oven': 'appliance', 'toaster': 'appliance', 'sink': 'appliance',
    'refrigerator': 'appliance', 'book': 'indoor', 'clock': 'indoor', 'vase': 'indoor', 'scissors': 'indoor',
    'teddy bear': 'indoor', 'hair drier': 'indoor', 'toothbrush': 'indoor'
}


def build_word_list_coco80() -> Dict[str, List[str]]:
    words_map = COCO80_ONTOLOGY.copy()
    words_map = {k: v for k, v in words_map.items() if not any(item in COCO80_ONTOLOGY for item in v)}

    return words_map


def _add_mask(masks: Dict[str, torch.Tensor], word: str, mask: torch.Tensor, simplify80: bool = False) -> Dict[str, torch.Tensor]:
    if simplify80:
        word = COCO80_TO_27.get(word, word)

    if word in masks:
        masks[word] = masks[word.lower()] + mask
        masks[word].clamp_(0, 1)
    else:
        masks[word] = mask

    return masks


@dataclass
class GenerationExperiment:
    """Class to hold experiment parameters. Pickleable."""
    image: PIL.Image.Image
    global_heat_map: torch.Tensor
    prompt: str

    seed: int = None
    id: str = '.'
    path: Optional[Path] = None

    truth_masks: Optional[Dict[str, torch.Tensor]] = None
    prediction_masks: Optional[Dict[str, torch.Tensor]] = None
    annotations: Optional[Dict[str, Any]] = None
    subtype: Optional[str] = '.'
    tokenizer: AutoTokenizer = None

    def __post_init__(self):
        if isinstance(self.path, str):
            self.path = Path(self.path)

        self.path = None if self.path is None else self.path / self.id

    def nsfw(self) -> bool:
        return np.sum(np.array(self.image)) == 0

    def heat_map(self, tokenizer: AutoTokenizer = None):
        if tokenizer is None:
            tokenizer = self.tokenizer

        from daam import GlobalHeatMap
        return GlobalHeatMap(tokenizer, self.prompt, self.global_heat_map)

    def clear_checkpoint(self):
        path = self if isinstance(self, Path) else self.path

        (path / 'generation.pt').unlink(missing_ok=True)

    def save(self, path: str = None, heat_maps: bool = True, tokenizer: AutoTokenizer = None):
        if path is None:
            path = self.path
        else:
            path = Path(path) / self.id

        if tokenizer is None:
            tokenizer = self.tokenizer

        (path / self.subtype).mkdir(parents=True, exist_ok=True)
        torch.save(self, path / self.subtype / 'generation.pt')
        self.image.save(path / self.subtype / 'output.png')

        with (path / 'prompt.txt').open('w') as f:
            f.write(self.prompt)

        with (path / 'seed.txt').open('w') as f:
            f.write(str(self.seed))

        if self.truth_masks is not None:
            for name, mask in self.truth_masks.items():
                im = PIL.Image.fromarray((mask * 255).unsqueeze(-1).expand(-1, -1, 4).byte().numpy())
                im.save(path / f'{name.lower()}.gt.png')

        if heat_maps and tokenizer is not None:
            self.save_all_heat_maps(tokenizer)

        self.save_annotations()

    def save_annotations(self, path: Path = None):
        if path is None:
            path = self.path

        if self.annotations is not None:
            with (path / 'annotations.json').open('w') as f:
                json.dump(self.annotations, f)

    def _load_truth_masks(self, simplify80: bool = False) -> Dict[str, torch.Tensor]:
        masks = {}

        for mask_path in self.path.glob('*.gt.png'):
            word = mask_path.name.split('.gt.png')[0].lower()
            mask = load_mask(str(mask_path))
            _add_mask(masks, word, mask, simplify80)

        return masks

    def _load_pred_masks(self, pred_prefix, composite=False, simplify80=False, vocab=None):
        # type: (str, bool, bool, List[str] | None) -> Dict[str, torch.Tensor]
        masks = {}

        if vocab is None:
            vocab = UNUSED_LABELS

        if composite:
            try:
                im = PIL.Image.open(self.path / self.subtype / f'composite.{pred_prefix}.pred.png')
                im = np.array(im)

                for mask_idx in np.unique(im):
                    mask = torch.from_numpy((im == mask_idx).astype(np.float32))
                    _add_mask(masks, vocab[mask_idx], mask, simplify80)
            except FileNotFoundError:
                pass
        else:
            for mask_path in (self.path / self.subtype).glob(f'*.{pred_prefix}.pred.png'):
                mask = load_mask(str(mask_path))
                word = mask_path.name.split(f'.{pred_prefix}.pred')[0].lower()
                _add_mask(masks, word, mask, simplify80)

        return masks

    def clear_prediction_masks(self, name: str):
        path = self if isinstance(self, Path) else self.path
        path = path / self.subtype

        for mask_path in path.glob(f'*.{name}.pred.png'):
            mask_path.unlink()

    def save_prediction_mask(self, mask: torch.Tensor, word: str, name: str):
        path = self if isinstance(self, Path) else self.path
        im = PIL.Image.fromarray((mask * 255).unsqueeze(-1).expand(-1, -1, 4).cpu().byte().numpy())
        im.save(path / self.subtype / f'{word.lower()}.{name}.pred.png')

    def save_heat_map(
            self,
            word: str,
            tokenizer: PreTrainedTokenizer = None,
            crop: int = None,
            output_prefix: str = '',
            absolute: bool = False
    ) -> Path:
        from .trace import GlobalHeatMap  # because of cyclical import

        if tokenizer is None:
            tokenizer = self.tokenizer

        with auto_autocast(dtype=torch.float32):
            path = self.path / self.subtype / f'{output_prefix}{word.lower()}.heat_map.png'
            heat_map = GlobalHeatMap(tokenizer, self.prompt, self.global_heat_map)
            heat_map.compute_word_heat_map(word).expand_as(self.image, color_normalize=not absolute, out_file=path, plot=True)

        return path

    def save_all_heat_maps(self, tokenizer: PreTrainedTokenizer = None, crop: int = None) -> Dict[str, Path]:
        path_map = {}

        if tokenizer is None:
            tokenizer = self.tokenizer

        for word in self.prompt.split(' '):
            try:
                path = self.save_heat_map(word, tokenizer, crop=crop)
                path_map[word] = path
            except:
                pass

        return path_map

    @staticmethod
    def contains_truth_mask(path: Union[str, Path], prompt_id: str = None) -> bool:
        if prompt_id is None:
            return any(Path(path).glob('*.gt.png'))
        else:
            return any((Path(path) / prompt_id).glob('*.gt.png'))

    @staticmethod
    def read_seed(path: Union[str, Path], prompt_id: str = None) -> int:
        if prompt_id is None:
            return int(Path(path).joinpath('seed.txt').read_text())
        else:
            return int(Path(path).joinpath(prompt_id).joinpath('seed.txt').read_text())

    @staticmethod
    def has_annotations(path: Union[str, Path]) -> bool:
        return Path(path).joinpath('annotations.json').exists()

    @staticmethod
    def has_experiment(path: Union[str, Path], prompt_id: str) -> bool:
        return (Path(path) / prompt_id / 'generation.pt').exists()

    @staticmethod
    def read_prompt(path: Union[str, Path], prompt_id: str = None) -> str:
        if prompt_id is None:
            prompt_id = '.'

        with (Path(path) / prompt_id / 'prompt.txt').open('r') as f:
            return f.read().strip()

    def _try_load_annotations(self):
        if not (self.path / 'annotations.json').exists():
            return None

        return json.load((self.path / 'annotations.json').open())

    def annotate(self, key: str, value: Any) -> 'GenerationExperiment':
        if self.annotations is None:
            self.annotations = {}

        self.annotations[key] = value

        return self

    @classmethod
    def load(
            cls,
            path,
            pred_prefix='daam',
            composite=False,
            simplify80=False,
            vocab=None,
            subtype='.',
            all_subtypes=False
    ):
        # type: (str, str, bool, bool, List[str] | None, str, bool) -> GenerationExperiment | List[GenerationExperiment]
        if all_subtypes:
            experiments = []

            for directory in Path(path).iterdir():
                if not directory.is_dir():
                    continue

                try:
                    experiments.append(cls.load(
                        path,
                        pred_prefix=pred_prefix,
                        composite=composite,
                        simplify80=simplify80,
                        vocab=vocab,
                        subtype=directory.name
                    ))
                except:
                    pass

            return experiments

        path = Path(path)
        exp = torch.load(path / subtype / 'generation.pt')
        exp.subtype = subtype
        exp.path = path
        exp.truth_masks = exp._load_truth_masks(simplify80=simplify80)
        exp.prediction_masks = exp._load_pred_masks(pred_prefix, composite=composite, simplify80=simplify80, vocab=vocab)
        exp.annotations = exp._try_load_annotations()

        return exp
