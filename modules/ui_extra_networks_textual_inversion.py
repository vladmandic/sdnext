import json
import os
from modules import shared, ui_extra_networks, files_cache, modelstats
from modules.logger import log
from modules.textual_inversion import Embedding


class ExtraNetworksPageTextualInversion(ui_extra_networks.ExtraNetworksPage):
    def __init__(self):
        super().__init__('Embedding')
        self.allow_negative_prompt = True
        self.embeddings = []

    def refresh(self):
        if not shared.sd_loaded:
            return
        if hasattr(shared.sd_model, 'embedding_db'):
            shared.sd_model.embedding_db.load_textual_inversion_embeddings(force_reload=True)

    def create_item(self, embedding: Embedding):
        record = None
        try:
            tags = {}
            if embedding.tag is not None:
                tags[embedding.tag]=1
            name = os.path.splitext(embedding.basename)[0]
            size, mtime = modelstats.stat(embedding.filename)
            info = self.find_info(embedding.filename)
            record = {
                "type": 'Embedding',
                "name": name,
                "filename": embedding.filename,
                "alias": os.path.splitext(os.path.basename(embedding.filename))[0],
                "prompt": json.dumps(f" {os.path.splitext(embedding.name)[0]}"),
                "tags": tags,
                "mtime": mtime,
                "size": size,
                "info": info,
                "description": self.find_description(embedding.filename, info),
            }
        except Exception as e:
            log.debug(f'Networks error: type=embedding file="{embedding.filename}" {e}')
        return record

    def list_items(self):
        if not shared.sd_loaded:
            candidates = list(files_cache.list_files(shared.opts.embeddings_dir, ext_filter=['.pt', '.safetensors'], recursive=files_cache.not_hidden))
            self.embeddings = [
                Embedding(vec=0, name=os.path.basename(embedding_path), filename=embedding_path)
                for embedding_path
                in candidates
            ]
        elif hasattr(shared.sd_model, 'embedding_db'):
            self.embeddings = list(shared.sd_model.embedding_db.word_embeddings.values())
        else:
            self.embeddings = []
        self.embeddings = sorted(self.embeddings, key=lambda emb: emb.filename)

        items = [self.create_item(embedding) for embedding in self.embeddings]
        self.update_all_previews(items)
        return items

    def allowed_directories_for_previews(self):
        return [shared.opts.embeddings_dir]
