from typing import List, Union
import os
import time
import torch
import safetensors.torch
from modules import shared, devices, errors
from modules.files_cache import directory_files, directory_mtime, extension_filter


debug = shared.log.trace if os.environ.get('SD_TI_DEBUG', None) is not None else lambda *args, **kwargs: None
debug('Trace: TEXTUAL INVERSION')
supported_models = ['ldm', 'sd', 'sdxl']


def list_embeddings(*dirs):
    is_ext = extension_filter(['.SAFETENSORS', '.PT' ])
    is_not_preview = lambda fp: not next(iter(os.path.splitext(fp))).upper().endswith('.PREVIEW') # pylint: disable=unnecessary-lambda-assignment
    return list(filter(lambda fp: is_ext(fp) and is_not_preview(fp) and os.stat(fp).st_size > 0, directory_files(*dirs)))


def open_embeddings(filename):
    """
    Load Embedding files from drive. Image embeddings not currently supported.
    """
    embeddings = []
    skipped = []
    if filename is None:
        return None
    filenames = list(filename)
    exts = [".SAFETENSORS", '.BIN', '.PT']
    for _filename in filenames:
        # debug(f'Embedding check: {filename}')
        fullname = _filename
        _filename = os.path.basename(fullname)
        fn, ext = os.path.splitext(_filename)
        name = os.path.basename(fn)
        embedding = Embedding(vec=[], name=name, filename=fullname)
        try:
            if ext.upper() not in exts:
                debug(f'extension `{ext}` is invalid, expected one of: {exts}')
                skipped.append(name)
                continue
            if ext.upper() in ['.SAFETENSORS']:
                with safetensors.torch.safe_open(embedding.filename, framework="pt") as f:  # type: ignore
                    for k in f.keys():
                        embedding.vec.append(f.get_tensor(k))
            else:  # fallback for sd1.5 pt embeddings
                vectors = torch.load(fullname, map_location=devices.device)["string_to_param"]["*"]
                embedding.vec.append(vectors)
            embedding.tokens = [embedding.name if i == 0 else f"{embedding.name}_{i}" for i in range(len(embedding.vec[0]))]
        except Exception as e:
            debug(f"Could not load embedding file {fullname} {e}")
        if embedding.vec:
            embeddings.append(embedding)
        else:
            skipped.append(embedding)
    return embeddings, skipped


def convert_bundled(data):
    """
    Bundled embeddings are passed as a dict from lora loading, convert to Embedding objects and pass back as list.
    """
    embeddings = []
    for key in data.keys():
        embedding = Embedding(vec=[], name=key, filename=None)
        for vector in data[key].values():
            embedding.vec.append(vector)
        embedding.tokens = [embedding.name if i == 0 else f"{embedding.name}_{i}" for i in range(len(embedding.vec[0]))]
        embeddings.append(embedding)
    return embeddings, []


def get_text_encoders():
    """
    Select all text encoder and tokenizer pairs from known pipelines, and index them based on the dimensionality of
    their embedding layers.
    """
    pipe = shared.sd_model
    te_names = ["text_encoder", "text_encoder_2", "text_encoder_3"]
    tokenizers_names = ["tokenizer", "tokenizer_2", "tokenizer_3"]
    text_encoders = []
    tokenizers = []
    hidden_sizes = []
    for te, tok in zip(te_names, tokenizers_names):
        text_encoder = getattr(pipe, te, None)
        if text_encoder is None:
            continue
        tokenizer = getattr(pipe, tok, None)
        hidden_size = text_encoder.get_input_embeddings().weight.data.shape[-1] or None
        if all([text_encoder, tokenizer, hidden_size]):
            text_encoders.append(text_encoder)
            tokenizers.append(tokenizer)
            hidden_sizes.append(hidden_size)
    return text_encoders, tokenizers, hidden_sizes


def deref_tokenizers(tokens, tokenizers):
    """
    Bundled embeddings may have the same name as a seperately loaded embedding, or there may be multiple LoRA with
    differing numbers of vectors. By editing the AddedToken objects, and deleting the dict keys pointing to them,
    we can ensure that a smaller embedding will not get tokenized as itself, plus the remaining vectors of the previous.
    """
    for tokenizer in tokenizers:
        if len(tokens) > 1:
            last_token = tokens[-1]
            suffix = int(last_token.split("_")[-1])
            newsuffix = suffix + 1
            while last_token.replace(str(suffix), str(newsuffix)) in tokenizer.get_vocab():
                idx = tokenizer.convert_tokens_to_ids(last_token.replace(str(suffix), str(newsuffix)))
                debug(f"Textual inversion: deref idx={idx}")
                del tokenizer._added_tokens_encoder[last_token.replace(str(suffix), str(newsuffix))] # pylint: disable=protected-access
                tokenizer._added_tokens_decoder[idx].content = str(time.time()) # pylint: disable=protected-access
                newsuffix += 1


def insert_tokens(embeddings: list, tokenizers: list):
    """
    Add all tokens to each tokenizer in the list, with one call to each.
    """
    tokens = []
    for embedding in embeddings:
        if embedding is not None:
            tokens += embedding.tokens
    for tokenizer in tokenizers:
        tokenizer.add_tokens(tokens)


def insert_vectors(embedding, tokenizers, text_encoders, hiddensizes):
    """
    Insert embeddings into the input embedding layer of a list of text encoders, matched based on embedding size,
    not by name.
    Future warning, if another text encoder becomes available with embedding dimensions in [768,1280,4096]
    this may cause collisions.
    """
    with devices.inference_context():
        for vector, size in zip(embedding.vec, embedding.vector_sizes):
            if size not in hiddensizes:
                continue
            idx = hiddensizes.index(size)
            unk_token_id = tokenizers[idx].convert_tokens_to_ids(tokenizers[idx].unk_token)
            if text_encoders[idx].get_input_embeddings().weight.data.shape[0] != len(tokenizers[idx]):
                text_encoders[idx].resize_token_embeddings(len(tokenizers[idx]))
            for token, v in zip(embedding.tokens, vector.unbind()):
                token_id = tokenizers[idx].convert_tokens_to_ids(token)
                if token_id > unk_token_id:
                    text_encoders[idx].get_input_embeddings().weight.data[token_id] = v


class Embedding:
    def __init__(self, vec, name, filename=None, step=None):
        self.vec = vec
        self.name = name
        self.tag = name
        self.step = step
        self.filename = filename
        self.basename = os.path.relpath(filename, shared.opts.embeddings_dir) if filename is not None else None
        self.shape = None
        self.vectors = 0
        self.cached_checksum = None
        self.sd_checkpoint = None
        self.sd_checkpoint_name = None
        self.optimizer_state_dict = None
        self.tokens = None

    def save(self, filename):
        embedding_data = {
            "string_to_token": {"*": 265},
            "string_to_param": {"*": self.vec},
            "name": self.name,
            "step": self.step,
            "sd_checkpoint": self.sd_checkpoint,
            "sd_checkpoint_name": self.sd_checkpoint_name,
        }
        torch.save(embedding_data, filename)
        if shared.opts.save_optimizer_state and self.optimizer_state_dict is not None:
            optimizer_saved_dict = {
                'hash': self.checksum(),
                'optimizer_state_dict': self.optimizer_state_dict,
            }
            torch.save(optimizer_saved_dict, f"{filename}.optim")

    def checksum(self):
        if self.cached_checksum is not None:
            return self.cached_checksum
        def const_hash(a):
            r = 0
            for v in a:
                r = (r * 281 ^ int(v) * 997) & 0xFFFFFFFF
            return r
        self.cached_checksum = f'{const_hash(self.vec.reshape(-1) * 100) & 0xffff:04x}'
        return self.cached_checksum


class DirWithTextualInversionEmbeddings:
    def __init__(self, path):
        self.path = path
        self.mtime = None

    def has_changed(self):
        if not os.path.isdir(self.path):
            return False
        return directory_mtime(self.path) != self.mtime

    def update(self):
        if not os.path.isdir(self.path):
            return
        self.mtime = directory_mtime(self.path)


def convert_embedding(tensor, text_encoder, text_encoder_2):
    """
    Given a tensor of shape (b, embed_dim) and two text encoders whose tokenizers match, return a tensor with
    approximately mathcing meaning, or padding if the input tensor is dissimilar to any frozen text embed
    """
    with torch.no_grad():
        vectors = []
        clip_l_embeds = text_encoder.get_input_embeddings().weight.data.clone().to(device=devices.device)
        tensor = tensor.to(device=devices.device)
        for vec in tensor:
            values, indices = torch.max(torch.nan_to_num(torch.cosine_similarity(vec.unsqueeze(0), clip_l_embeds)), 0)
            if values < 0.707:  # Arbitrary similarity to cutoff, here 45 degrees
                indices *= 0  # Use SDXL padding vector 0
            vectors.append(indices)
        vectors = torch.stack(vectors).to(text_encoder_2.device)
        output = text_encoder_2.get_input_embeddings().weight.data[vectors]
    return output


class EmbeddingDatabase:
    def __init__(self):
        self.ids_lookup = {}
        self.word_embeddings = {}
        self.skipped_embeddings = {}
        self.embedding_dirs = {}
        self.previously_displayed_embeddings = ()
        self.embeddings_used = []

    def add_embedding_dir(self, path):
        self.embedding_dirs[path] = DirWithTextualInversionEmbeddings(path)

    def register_embedding(self, embedding, model):
        self.word_embeddings[embedding.name] = embedding
        if hasattr(model, 'cond_stage_model'):
            ids = model.cond_stage_model.tokenize([embedding.name])[0]
        elif hasattr(model, 'tokenizer'):
            ids = model.tokenizer.convert_tokens_to_ids(embedding.name)
        if type(ids) != list:
            ids = [ids]
        first_id = ids[0]
        if first_id not in self.ids_lookup:
            self.ids_lookup[first_id] = []
        self.ids_lookup[first_id] = sorted(self.ids_lookup[first_id] + [(ids, embedding)], key=lambda x: len(x[0]), reverse=True)
        return embedding

    def load_diffusers_embedding(self, filename: Union[str, List[str]] = None, data: dict = None):
        """
        File names take precidence over bundled embeddings passed as a dict.
        Bundled embeddings are automatically set to overwrite previous embeddings.
        """
        overwrite = bool(data)
        if not shared.sd_loaded:
            return
        if not shared.opts.diffusers_enable_embed:
            return
        embeddings, skipped = open_embeddings(filename) or convert_bundled(data)
        for skip in skipped:
            self.skipped_embeddings[skip.name] = skipped
        if not embeddings:
            return
        text_encoders, tokenizers, hiddensizes = get_text_encoders()
        if not all([text_encoders, tokenizers, hiddensizes]):
            return
        for embedding in embeddings:
            try:
                embedding.vector_sizes = [v.shape[-1] for v in embedding.vec]
                if shared.opts.diffusers_convert_embed and 768 in hiddensizes and 1280 in hiddensizes and 1280 not in embedding.vector_sizes and 768 in embedding.vector_sizes:
                    embedding.vec.append(convert_embedding(embedding.vec[embedding.vector_sizes.index(768)], text_encoders[hiddensizes.index(768)], text_encoders[hiddensizes.index(1280)]))
                    embedding.vector_sizes.append(1280)
                if (not all(vs in hiddensizes for vs in embedding.vector_sizes) or  # Skip SD2.1 in SD1.5/SDXL/SD3 vis versa
                        len(embedding.vector_sizes) > len(hiddensizes) or  # Skip SDXL/SD3 in SD1.5
                        (len(embedding.vector_sizes) < len(hiddensizes) and len(embedding.vector_sizes) != 2)):  # SD3 no T5
                    embedding.tokens = []
                    self.skipped_embeddings[embedding.name] = embedding
            except Exception as e:
                shared.log.error(f'Load embedding invalid: name="{embedding.name}" fn="{filename}" {e}')
                self.skipped_embeddings[embedding.name] = embedding
        if overwrite:
            shared.log.info(f"Load bundled embeddings: {list(data.keys())}")
            for embedding in embeddings:
                if embedding.name not in self.skipped_embeddings:
                    deref_tokenizers(embedding.tokens, tokenizers)
        insert_tokens(embeddings, tokenizers)
        for embedding in embeddings:
            if embedding.name not in self.skipped_embeddings:
                try:
                    insert_vectors(embedding, tokenizers, text_encoders, hiddensizes)
                    self.register_embedding(embedding, shared.sd_model)
                except Exception as e:
                    shared.log.error(f'Load embedding: name="{embedding.name}" file="{embedding.filename}" {e}')
                    errors.display(e, f'Load embedding: name="{embedding.name}" file="{embedding.filename}"')
        return

    def load_from_dir(self, embdir):
        if not shared.sd_loaded:
            shared.log.info('Skipping embeddings load: model not loaded')
            return
        if not os.path.isdir(embdir.path):
            return
        file_paths = list_embeddings(embdir.path)
        self.load_diffusers_embedding(file_paths)

    def load_textual_inversion_embeddings(self, force_reload=False):
        if not shared.sd_loaded:
            return
        if shared.sd_model_type not in supported_models:
            return
        t0 = time.time()
        if not force_reload:
            need_reload = False
            for embdir in self.embedding_dirs.values():
                if embdir.has_changed():
                    need_reload = True
                    break
            if not need_reload:
                return
        self.ids_lookup.clear()
        self.word_embeddings.clear()
        self.skipped_embeddings.clear()
        self.embeddings_used.clear()
        for embdir in self.embedding_dirs.values():
            self.load_from_dir(embdir)
            embdir.update()

        # re-sort word_embeddings because load_from_dir may not load in alphabetic order.
        # using a temporary copy so we don't reinitialize self.word_embeddings in case other objects have a reference to it.
        sorted_word_embeddings = {e.name: e for e in sorted(self.word_embeddings.values(), key=lambda e: e.name.lower())}
        self.word_embeddings.clear()
        self.word_embeddings.update(sorted_word_embeddings)

        displayed_embeddings = (tuple(self.word_embeddings.keys()), tuple(self.skipped_embeddings.keys()))
        if self.previously_displayed_embeddings != displayed_embeddings and shared.opts.diffusers_enable_embed:
            self.previously_displayed_embeddings = displayed_embeddings
            t1 = time.time()
            shared.log.info(f"Network load: type=embeddings loaded={len(self.word_embeddings)} skipped={len(self.skipped_embeddings)} time={t1-t0:.2f}")
