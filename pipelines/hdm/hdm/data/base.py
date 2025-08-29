import random

import numpy as np
import torch
import torch.utils.data as Data
from transformers import PreTrainedTokenizer
from tqdm import tqdm, trange


class BaseDataset(Data.Dataset):
    def collate(self, batch):
        samples = torch.stack([x["sample"] for x in batch])
        caption = [x["caption"] for x in batch]
        tokenizer_outs = [x["tokenizer_out"] for x in batch]
        # TODO: change to stack and reduce dim?
        add_time_ids = [x["add_time_ids"] for x in batch]
        tokenizer_outputs = []
        for tokenizer_out in zip(*tokenizer_outs):
            input_ids = torch.concat([x["input_ids"] for x in tokenizer_out])
            attention_mask = torch.concat([x["attention_mask"] for x in tokenizer_out])
            tokenizer_outputs.append(
                {"input_ids": input_ids, "attention_mask": attention_mask}
            )
        return (
            samples,
            caption,
            tokenizer_outputs,
            {"time_ids": torch.concat(add_time_ids).float()},
        )


class DummyDataset(BaseDataset):
    def __init__(
        self,
        # (4, 128, 128) for latent
        sample_size: tuple[int] = (3, 1024, 1024),
        n_samples: int = 100,
        tokenizers: list[PreTrainedTokenizer] = [],
        **kwargs,
    ):
        if not isinstance(sample_size, tuple):
            sample_size = tuple(sample_size)
        self.samples = [torch.randn(sample_size) for _ in range(n_samples)]
        if isinstance(tokenizers, list):
            self.tokenizers = tokenizers
        else:
            self.tokenizers = [tokenizers]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        caption = "DUMMY TEST"
        return {
            "sample": sample,
            "caption": caption,
            "tokenizer_out": [
                tokenizer(
                    caption,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                for tokenizer in self.tokenizers
            ],
            # org_h, org_w, crop_top, crop_left, target_h, target_w
            "add_time_ids": torch.tensor([[1024, 1024, 0, 0, 1024, 1024]]),
        }


class CombineDataset(Data.Dataset):
    def __init__(
        self,
        datasets: list[Data.Dataset],
        latent_scale: float = 1.0,
        latent_shift: float = 0.0,
        tokenizers: list[PreTrainedTokenizer] = [],
        shuffle=True,
        arb_mode=False,
    ):
        self.shuffle = shuffle
        self.datasets_ref = datasets
        self.datasets = datasets
        self.shard_string = sum(
            ([chr(i).encode()] * len(dataset) for i, dataset in enumerate(datasets)),
            [],
        )
        self.tokenizers = tokenizers
        self.latent_scale = latent_scale
        self.latent_shift = latent_shift

        if shuffle:
            random.shuffle(self.shard_string)
        self.arb_mode = arb_mode

        dataset_ids = [0] * len(datasets)
        for i, data in tqdm(
            enumerate(self.shard_string),
            total=len(self.shard_string),
            smoothing=0.01,
            desc="Dataset Indexing...",
        ):
            index = dataset_ids[data[0]]
            dataset_ids[data[0]] += 1
            self.shard_string[i] = data + np.base_repr(index, 36).encode()
        self.shard_string = np.array(self.shard_string)

    @torch.no_grad()
    def collate(self, batch):
        if self.arb_mode:
            assert len(batch) == 1
            latents = batch[0]["latent"]
            caption = batch[0]["caption"]
            pos_map = batch[0]["pos_map"]
            tokenizer_outs = batch[0]["tokenizer_out"]
            if "aspect_ratio" in batch[0]:
                return (
                    latents,
                    caption,
                    tokenizer_outs,
                    pos_map,
                    {"addon_info": batch[0]["aspect_ratio"]},
                )
            return latents, caption, tokenizer_outs, pos_map
        latents = torch.stack([x["latent"] for x in batch])
        caption = [x["caption"] for x in batch]
        pos_map = torch.stack([x["pos_map"] for x in batch])
        tokenizer_outs = [x["tokenizer_out"] for x in batch]

        tokenizer_outputs = []
        for tokenizer_out in zip(*tokenizer_outs):
            input_ids = torch.concat([x["input_ids"] for x in tokenizer_out])
            attention_mask = torch.concat([x["attention_mask"] for x in tokenizer_out])
            tokenizer_outputs.append(
                {"input_ids": input_ids, "attention_mask": attention_mask}
            )
        if "aspect_ratio" in batch[0]:
            aspect_ratio = torch.tensor([x["aspect_ratio"] for x in batch])
            return (
                latents,
                caption,
                tokenizer_outputs,
                pos_map,
                {"addon_info": aspect_ratio},
            )
        return latents, caption, tokenizer_outputs, pos_map

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

    @torch.no_grad()
    def __getitem__(self, index):
        choosed = self.shard_string[index]
        dataset = self.datasets[choosed[0]]
        index = int(choosed[1:], 36)
        latent, caption, pos_map, *ar = dataset[index]
        if self.arb_mode:
            tokenizer_out = [
                [
                    tokenizer(
                        c,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    )
                    for tokenizer in self.tokenizers
                ]
                for c in caption
            ]
            data = {
                "latent": (latent * self.latent_scale + self.latent_shift),
                "caption": caption,
                "pos_map": pos_map,
                "tokenizer_out": tokenizer_out,
            }
            if len(ar) > 0:
                aspect_ratio = ar[0]
                data["aspect_ratio"] = aspect_ratio
            return data
        tokenizer_out = [
            tokenizer(
                caption,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            for tokenizer in self.tokenizers
        ]
        data = {
            "latent": (latent * self.latent_scale + self.latent_shift),
            "caption": caption,
            "pos_map": pos_map,
            "tokenizer_out": tokenizer_out,
        }
        if len(ar) > 0:
            aspect_ratio = ar[0]
            data["aspect_ratio"] = aspect_ratio
        return data


if __name__ == "__main__":
    from transformers import Qwen2Tokenizer
    from .kohya import *

    tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    dataset = KohyaDataset(
        dataset_folder="/mp34-1/danbooru2023",
        keep_token_seperator="|||",
        tag_seperator="$$",
        seperator=", ",
        group_seperator="%%",
        tag_shuffle=True,
        group_shuffle=True,
        tag_dropout_rate=0.0,
        group_dropout_rate=0.0,
        use_cached_meta=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ]
        ),
        use_arb=True,
        arb_config={
            "batch_size": 32,
            "target_res": 1024,
            "res_step": 16,
            "seed": 0,
        },
        meta_postfix="_filtered",
    )
    combine = CombineDataset(
        [dataset], tokenizers=[tokenizer], shuffle=True, arb_mode=True
    )

    print(len(combine))

    dataloader = Data.DataLoader(
        combine, batch_size=1, num_workers=0, shuffle=True, collate_fn=combine.collate
    )
    for batch in tqdm(dataloader):
        latent, caption, tokenizer_out, pos_map, *ar = batch
        print(latent.size(), pos_map.size())
        print(len(caption), caption[0])
        print(
            len(tokenizer_out), len(tokenizer_out[0]), tokenizer_out[0][0]["input_ids"]
        )
        print(len(ar))
        if len(ar) > 0:
            print(ar[0])
        break
