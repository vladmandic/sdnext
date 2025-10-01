from collections import defaultdict
from typing import List, Union

from scipy.optimize import linear_sum_assignment
import PIL.Image as Image
import numpy as np
import torch
import torch.nn.functional as F


__all__ = ['compute_iou', 'MeanEvaluator', 'load_mask', 'compute_ioa']


def compute_iou(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.shape[0] != b.shape[0]:
        a = F.interpolate(a.unsqueeze(0).unsqueeze(0).float(), size=b.shape, mode='bicubic').squeeze()
        a[a < 1] = 0
        a[a >= 1] = 1

    intersection = (a * b).sum()
    union = a.sum() + b.sum() - intersection

    return (intersection / (union + 1e-8)).item()


def compute_ioa(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.shape[0] != b.shape[0]:
        a = F.interpolate(a.unsqueeze(0).unsqueeze(0).float(), size=b.shape, mode='bicubic').squeeze()
        a[a < 1] = 0
        a[a >= 1] = 1

    intersection = (a * b).sum()
    area = a.sum()

    return (intersection / (area + 1e-8)).item()


def load_mask(path: str) -> torch.Tensor:
    mask = np.array(Image.open(path))
    mask = torch.from_numpy(mask).float()[:, :, 3]  # use alpha channel
    mask = (mask > 0).float()

    return mask


class UnsupervisedEvaluator:
    def __init__(self, name: str = 'UnsupervisedEvaluator'):
        self.name = name
        self.ious = defaultdict(list)
        self.num_samples = 0

    def log_iou(self, preds: Union[torch.Tensor, List[torch.Tensor]], truth: torch.Tensor, gt_idx: int = 0, pred_idx: int = 0):
        if not isinstance(preds, list):
            preds = [preds]

        iou = max(compute_iou(pred, truth) for pred in preds)
        self.ious[gt_idx].append((pred_idx, iou))

    @property
    def mean_iou(self) -> float:
        n = max(max(self.ious), max([y[0] for x in self.ious.values() for y in x])) + 1
        iou_matrix = np.zeros((n, n))
        count_matrix = np.zeros((n, n))

        for gt_idx, ious in self.ious.items():
            for pred_idx, iou in ious:
                iou_matrix[gt_idx, pred_idx] += iou
                count_matrix[gt_idx, pred_idx] += 1

        row_ind, col_ind = linear_sum_assignment(iou_matrix, maximize=True)
        return iou_matrix[row_ind, col_ind].sum() / count_matrix[row_ind, col_ind].sum()

    def increment(self):
        self.num_samples += 1

    def __len__(self) -> int:
        return self.num_samples

    def __str__(self):
        return f'{self.name}<{self.mean_iou:.4f} (mIoU) {len(self)} samples>'


class MeanEvaluator:
    def __init__(self, name: str = 'MeanEvaluator'):
        self.ious: List[float] = []
        self.intensities: List[float] = []
        self.name = name

    def log_iou(self, preds: Union[torch.Tensor, List[torch.Tensor]], truth: torch.Tensor):
        if not isinstance(preds, list):
            preds = [preds]

        self.ious.append(max(compute_iou(pred, truth) for pred in preds))
        return self

    def log_intensity(self, pred: torch.Tensor):
        self.intensities.append(pred.mean().item())
        return self

    @property
    def mean_iou(self) -> float:
        return np.mean(self.ious)

    @property
    def mean_intensity(self) -> float:
        return np.mean(self.intensities)

    @property
    def ci95_miou(self) -> float:
        return 1.96 * np.std(self.ious) / np.sqrt(len(self.ious))

    def __len__(self) -> int:
        return max(len(self.ious), len(self.intensities))

    def __str__(self):
        return f'{self.name}<{self.mean_iou:.4f} (±{self.ci95_miou:.3f} mIoU) {self.mean_intensity:.4f} (mInt) {len(self)} samples>'


if __name__ == '__main__':
    mask = load_mask('truth/output/452/sink.gt.png')

    print(MeanEvaluator().log_iou(mask, mask))
