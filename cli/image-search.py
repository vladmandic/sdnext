#!/usr/bin/env python

from typing import Union
import os
import re
import logging
from tqdm.rich import tqdm
import torch
import PIL
import faiss
import numpy as np
import pandas as pd
import transformers


class ImageDB:
    # TODO index: quantize and train faiss index
    # TODO index: clip batch processing
    def __init__(self,
                 name:str='db',
                 fmt:str='json',
                 cache_dir:str=None,
                 dtype:torch.dtype=torch.float16,
                 device:torch.device=torch.device('cpu'),
                 model:str='openai/clip-vit-large-patch14', # 'facebook/dinov2-small'
                 debug:bool=False,
                 pbar:bool=True,
                ):
        self.format = fmt
        self.name = name
        self.cache_dir = cache_dir
        self.processor: transformers.AutoImageProcessor = None
        self.model: transformers.AutoModel = None
        self.tokenizer = transformers.AutoTokenizer = None
        self.device: torch.device = device
        self.dtype: torch.dtype = dtype
        self.dimension = 768 if 'clip' in model else 384
        self.debug = debug
        self.pbar = pbar
        self.repo = model
        self.df = pd.DataFrame([], columns=['filename', 'timestamp', 'metadata']) # image/metadata database
        self.index = faiss.IndexFlatL2(self.dimension) # embed database
        self.log = logging.getLogger(__name__)
        self.err = logging.getLogger(__name__).error
        self.log = logging.getLogger(__name__).info if self.debug else logging.getLogger(__name__).debug
        # self.init()
        # self.load()

    def __str__(self):
        return f'db: name="{self.name}" format={self.format} device={self.device} dtype={self.dtype} dimension={self.dimension} model="{self.repo}" records={len(self.df)} index={self.index.ntotal}'

    def init(self): # initialize models
        if self.processor is None or self.model is None:
            if 'clip' in self.repo:
                self.processor = transformers.CLIPImageProcessor.from_pretrained(self.repo, cache_dir=self.cache_dir)
                self.tokenizer = transformers.CLIPTokenizer.from_pretrained(self.repo, cache_dir=self.cache_dir)
                self.model = transformers.CLIPModel.from_pretrained(self.repo, cache_dir=self.cache_dir).to(device=self.device, dtype=self.dtype)
            elif 'dino' in self.repo:
                self.processor = transformers.AutoImageProcessor.from_pretrained(self.repo, cache_dir=self.cache_dir)
                self.model = transformers.AutoModel.from_pretrained(self.repo, cache_dir=self.cache_dir).to(device=self.device, dtype=self.dtype)
            else:
                self.err(f'db: model="{self.repo}" unknown')
            self.log(f'db: load model="{self.repo}" cache="{self.cache_dir}" device={self.device} dtype={self.dtype}')

    def load(self): # load db to disk
        if self.format == 'json' and os.path.exists(f'{self.name}.json'):
            self.df = pd.read_json(f'{self.name}.json')
        elif self.format == 'csv' and os.path.exists(f'{self.name}.csv'):
            self.df = pd.read_csv(f'{self.name}.csv')
        elif self.format == 'pickle' and os.path.exists(f'{self.name}.pkl'):
            self.df = pd.read_pickle(f'{self.name}.parquet')
        if os.path.exists(f'{self.name}.index'):
            self.index = faiss.read_index(f'{self.name}.index')
        if self.index.ntotal != len(self.df):
            self.err(f'db: index={self.index.ntotal} data={len(self.df)} mismatch')
            self.index = faiss.IndexFlatL2(self.dimension)
            self.df = pd.DataFrame([], columns=['filename', 'timestamp', 'metadata'])
        self.log(f'db: load data={len(self.df)} name={self.name} format={self.format} name={self.name}')

    def save(self): # save db to disk
        if self.format == 'json':
            self.df.to_json(f'{self.name}.json')
        elif self.format == 'csv':
            self.df.to_csv(f'{self.name}.csv')
        elif self.format == 'pickle':
            self.df.to_pickle(f'{self.name}.pkl')
        faiss.write_index(self.index, f'{self.name}.index')
        self.log(f'db: save data={len(self.df)} name={self.name} format={self.format} name={self.name}')

    def normalize(self, embed) -> np.ndarray: # normalize embed before using it
        embed = embed.detach().float().cpu().numpy()
        faiss.normalize_L2(embed)
        return embed

    def embedding(self, query: Union[PIL.Image.Image | str]) -> np.ndarray: # calculate embed for prompt or image
        if self.processor is None or self.model is None:
            self.err('db: model not loaded')
        if isinstance(query, str) and os.path.exists(query):
            query = PIL.Image.open(query).convert('RGB')
        self.model = self.model.to(self.device)
        with torch.no_grad():
            if 'clip' in self.repo:
                if isinstance(query, str):
                    processed = self.tokenizer(text=query, padding=True, return_tensors="pt").to(device=self.device)
                    results = self.model.get_text_features(**processed)
                else:
                    processed = self.processor(images=query, return_tensors="pt").to(device=self.device, dtype=self.dtype)
                    results = self.model.get_image_features(**processed)
            elif 'dino' in self.repo:
                processed = self.processor(images=query, return_tensors="pt").to(device=self.device, dtype=self.dtype)
                results = self.model(**processed)
                results = results.last_hidden_state.mean(dim=1)
            else:
                self.err(f'db: model="{self.repo}" unknown')
                return None
        return self.normalize(results)

    def add(self, embed, filename=None, metadata=None): # add embed to db
        rec = pd.DataFrame([{'filename': filename, 'timestamp': pd.Timestamp.now(), 'metadata': metadata}])
        if len(self.df) > 0:
            self.df = pd.concat([self.df, rec], ignore_index=True)
        else:
            self.df = rec
        self.index.add(embed)

    def search(self, filename: str = None, metadata: str = None, embed: np.ndarray = None, k=10, d=1.0): # search by filename/metadata/prompt-embed/image-embed
        def dct(record: pd.DataFrame, mode: str, distance: float = None):
            if distance is not None:
                return {'type': mode, 'filename': record[1]['filename'], 'metadata': record[1]['metadata'], 'distance': round(distance, 2)}
            else:
                return {'type': mode, 'filename': record[1]['filename'], 'metadata': record[1]['metadata']}

        if self.index.ntotal == 0:
            return
        self.log(f'db: search k={k} d={d}')
        if embed is not None:
            distances, indexes = self.index.search(embed, k)
            records = self.df.iloc[indexes[0]]
            for record, distance in zip(records.iterrows(), distances[0]):
                if d <= 0 or distance <= d:
                    yield dct(record, distance=distance, mode='embed')
        if filename is not None:
            records = self.df[self.df['filename'].str.contains(filename, na=False, case=False)]
            for record in records.iterrows():
                yield dct(record, mode='filename')
        if metadata is not None:
            records = self.df[self.df['metadata'].str.contains(filename, na=False, case=False)]
            for record in records.iterrows():
                yield dct(record, mode='metadata')

    def decode(self, s: bytes): # decode byte-encoded exif metadata
        remove_prefix = lambda text, prefix: text[len(prefix):] if text.startswith(prefix) else text # pylint: disable=unnecessary-lambda-assignment
        for encoding in ['utf-8', 'utf-16', 'ascii', 'latin_1', 'cp1252', 'cp437']: # try different encodings
            try:
                s = remove_prefix(s, b'UNICODE')
                s = remove_prefix(s, b'ASCII')
                s = remove_prefix(s, b'\x00')
                val = s.decode(encoding, errors="strict")
                val = re.sub(r'[\x00-\x09\n\s\s+]', '', val).strip() # remove remaining special characters, new line breaks, and double empty spaces
                if len(val) == 0: # remove empty strings
                    val = None
                return val
            except Exception:
                pass
        return None

    def metadata(self, image: PIL.Image.Image): # get exif metadata from image
        exif = image._getexif() # pylint: disable=protected-access
        if exif is None:
            return ''
        for k, v in exif.items():
            if k == 37510: # comment
                return self.decode(v)
        return ''

    def image(self, filename: str, image=None): # add file/image to db
        try:
            if image is None:
                image = PIL.Image.open(filename)
                image.load()
            embed = self.embedding(image.convert('RGB'))
            metadata = self.metadata(image)
            image.close()
            self.add(embed, filename=filename, metadata=metadata)
        except Exception as _e:
            # self.err(f'db: {str(_e)}')
            pass

    def folder(self, folder: str): # add all files from folder to db
        files = []
        for root, _subdir, _files in os.walk(folder):
            for f in _files:
                files.append(os.path.join(root, f))
        if self.pbar:
            for f in tqdm(files):
                self.image(filename=f)
        else:
            for f in files:
                self.image(filename=f)

    def offload(self): # offload model to cpu
        if self.model is not None:
            self.model = self.model.to('cpu')


if __name__ == '__main__':
    import time
    import argparse
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description = 'image-search')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--search', action='store_true', help='run search')
    group.add_argument('--index', action='store_true', help='run indexing')
    parser.add_argument('--db', default='db', help='database name')
    parser.add_argument('--model', default='openai/clip-vit-large-patch14', help='huggingface model')
    parser.add_argument('--cache', default='/mnt/models/huggingface', help='cache folder')
    parser.add_argument('input', nargs='*', default=os.getcwd())
    args = parser.parse_args()

    db = ImageDB(
        name=args.db,
        model=args.model, # 'facebook/dinov2-small'
        cache_dir=args.cache,
        dtype=torch.bfloat16,
        device=torch.device('cuda'),
        debug=True,
        pbar=True,
    )
    db.init()
    db.load()
    print(db)

    if args.index:
        t0 = time.time()
        if len(args.input) > 0:
            for fn in args.input:
                if os.path.isfile(fn):
                    db.image(filename=fn)
                elif os.path.isdir(fn):
                    db.folder(folder=fn)
        t1 = time.time()
        print('index', t1-t0)
        db.save()
        db.offload()

    if args.search:
        for ref in args.input:
            emb = db.embedding(ref)
            res = db.search(filename=ref, metadata=ref, embed=emb, k=10, d=0)
            for r in res:
                print(ref, r)
