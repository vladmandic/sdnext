import re
import torch
import transformers
from modules import devices, shared
from modules.logger import log


safe_words = [ # Safe fallback vocabulary (neutral, high-probability)
    "person", "human", "agent", "actor", "being", "creature",
    "entity", "object", "item", "thing", "unit", "part", "piece",
    "form", "body", "figure", "shape", "model", "sample", "type",
    "kind", "sort", "role", "case", "instance", "element", "aspect",
    "factor", "point", "idea", "concept", "matter", "material",
    "stuff", "device", "tool", "instrument", "machine", "mechanism",
    "system", "module", "component", "substance", "resource",
    "asset", "product", "article"
]


class LogitsParser(transformers.LogitsProcessor):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, words: str, semantic_threshold=0.75, embedding_similarity=0.75):
        self.replacements = []
        self.semantic_threshold = semantic_threshold
        self.embedding_similarity = embedding_similarity
        self.tokenizer = tokenizer
        self.embedder = None
        self.window = 64

        if self.embedding_similarity > 0:
            if LogitsParser._embedder is None:
                try:
                    from installer import install
                    install('sentence_transformers')
                    from sentence_transformers import SentenceTransformer
                    embedder_repo = "sentence-transformers/all-MiniLM-L6-v2"
                    model_kwargs = { 'torch_dtype': devices.dtype, 'cache_dir': shared.opts.hfcache_dir }
                    LogitsParser._embedder = SentenceTransformer(embedder_repo, device=devices.device, model_kwargs=model_kwargs)
                    log.debug(f'LogitsParser: embedding_similarity={self.embedding_similarity} semantic_threshold={self.semantic_threshold} model="{embedder_repo}"')
                except Exception as e:
                    log.warning(f'LogitsParser: failed to initialize embedder: {e}')
                    LogitsParser._embedder = None
            self.embedder = LogitsParser._embedder
        self.banned_pairs = self.parse_banned(words) # Parse banned into list of (variant, replacement, original_banned)
        self.block_map, self.block_info = self.compute_block_map() # Compute first-subword blocklist + replacement IDs

        if self.semantic_threshold > 0 and self.embedder is not None:
            banned_words = [variant for variant, _, _ in self.banned_pairs]
            self.banned_embeddings = self.embedder.encode(banned_words, convert_to_tensor=True)
        else:
            self.banned_embeddings = None

        decoded = {self.tokenizer.decode([k]): self.tokenizer.decode([v]) for k, v in self.block_map.items()}
        log.debug(f'LogitsParser: pairs={self.banned_pairs} map={self.block_map} decoded={decoded}')

    _vocab_cache = None # Cache vocabulary embeddings for fast semantic expansion; keyed by tokenizer
    _embedder = None # Shared embedder instance across LogitsParser instances

    def get_replacements(self):
        seen = set()
        unique = []
        for replacement in self.replacements:
            key = (replacement.get("search"), replacement.get("match"), replacement.get("replace"))
            if key not in seen:
                seen.add(key)
                unique.append(replacement)
        return unique

    def get_vocab_embeddings(self):
        if self.embedder is None:
            return ([], [], None)
        if LogitsParser._vocab_cache is not None:
            cache_tokenizer, clean_tokens, clean_ids, embs = LogitsParser._vocab_cache
            if cache_tokenizer is self.tokenizer:
                return clean_tokens, clean_ids, embs
            LogitsParser._vocab_cache = None
        vocab = self.tokenizer.get_vocab()
        tokens = list(vocab.keys())
        decoded = [self.tokenizer.decode([vocab[t]]).strip().lower() for t in tokens] # Decode tokens to readable text
        clean_tokens = [] # Filter out empty / punctuation / symbols
        clean_ids = []
        for t, d in zip(tokens, decoded):
            if d and any(c.isalpha() for c in d):
                clean_tokens.append(d)
                clean_ids.append(vocab[t])
        embs = self.embedder.encode(clean_tokens, convert_to_tensor=True) # Embed all clean tokens
        LogitsParser._vocab_cache = (self.tokenizer, clean_tokens, clean_ids, embs)
        return clean_tokens, clean_ids, embs

    def expand_morphology(self, word): # Embedding-based morphological/semantic expansion
        word = self.normalize(word)
        if self.embedder is None: # If no embedder, fallback to trivial expansion
            return {word}
        clean_tokens, _clean_ids, embs = self.get_vocab_embeddings()
        if not clean_tokens or embs is None:
            return {word}
        w_emb = self.embedder.encode([word], convert_to_tensor=True) # Embed the banned word
        sims = torch.nn.functional.cosine_similarity(w_emb, embs) # Compute cosine similarity
        top_k = 20 # Pick top-N nearest neighbors
        idxs = torch.topk(sims, k=min(top_k, len(sims))).indices.tolist()
        out = {word}
        threshold = self.embedding_similarity if self.embedding_similarity > 0 else self.semantic_threshold
        for i in idxs:
            sim = sims[i].item()
            if sim < threshold: # Filter out anything below similarity threshold
                continue
            candidate = clean_tokens[i]
            if 2 <= len(candidate) <= 20: # Filter out overly long or weird tokens
                out.add(candidate)
        return out

    def parse_banned(self, banned): # Parse banned into list of (variant, replacement, original_banned)
        raw_terms = re.split(r"[,\n;]+", banned)
        pairs = []
        seen_pairs = set()
        for term in raw_terms:
            term = term.strip()
            if not term:
                continue
            match = re.match(r"^(.*?)\s*[:=]\s*(.*)$", term)
            if match:
                original = match.group(1).strip().lower()
                repl = match.group(2).strip()
            else:
                original = term.lower()
                repl = self.guess_replacement(original)
            normalized = self.normalize(original)
            expanded = self.expand_morphology(normalized)
            for variant in expanded:
                pair = (variant, repl, normalized)
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    pairs.append(pair)
        return pairs

    def normalize(self, s): # Unicode normalization + homoglyph cleanup
        import unicodedata
        s = unicodedata.normalize("NFKC", s)
        s = s.replace("\u00AD", "")  # soft hyphen
        s = s.replace("\u200B", "")  # zero-width space
        s = s.replace("0", "o")      # simple homoglyph fix
        return s

    def guess_replacement(self, banned): # Best-guess fallback replacement (embedding-based)
        b = self.normalize(banned)
        if self.embedder is None: # If no embedder, fallback to "entity"
            return "entity"
        filtered_safe = []
        for w in safe_words:
            ids = self.tokenizer.encode(" " + w, add_special_tokens=False)
            if len(ids) == 1:
                filtered_safe.append(w)
        if not filtered_safe:
            filtered_safe = ["entity"]
        banned_emb = self.embedder.encode([b], convert_to_tensor=True) # Embed banned word
        safe_embs = self.embedder.encode(filtered_safe, convert_to_tensor=True) # Embed safe words
        sims = torch.nn.functional.cosine_similarity(banned_emb, safe_embs) # Compute cosine similarity
        idx = torch.argmax(sims).item() # Pick the nearest safe word
        return filtered_safe[idx]

    def compute_block_map(self): # Compute first-subword blocklist with redirection targets block_map: {blocked_token_id: replacement_token_id}
        block_map = {}
        block_info = {}
        for variant, repl, original in self.banned_pairs:
            repl_ids = self.tokenizer.encode(" " + repl, add_special_tokens=False) # Encode replacement
            if not repl_ids:
                continue
            repl_id = repl_ids[0]
            spaced = " " + variant # Encode banned variant (leading space)
            ids = self.tokenizer.encode(spaced, add_special_tokens=False)
            if ids:
                block_map[ids[0]] = repl_id
                block_info[ids[0]] = {
                    "search": original,
                    "match": self.tokenizer.decode([ids[0]]).strip(),
                    "replace": repl,
                }
            sp_variant = "▁" + variant # Encode SP underline variant
            ids2 = self.tokenizer.encode(sp_variant, add_special_tokens=False)
            if ids2:
                block_map[ids2[0]] = repl_id
                block_info[ids2[0]] = {
                    "search": original,
                    "match": self.tokenizer.decode([ids2[0]]).strip(),
                    "replace": repl,
                }
        return block_map, block_info

    def semantic_match(self, text): # Optional semantic detection
        if self.semantic_threshold <= 0 or self.embedder is None or self.banned_embeddings is None:
            return False
        emb = self.embedder.encode([text], convert_to_tensor=True)
        sim = torch.nn.functional.cosine_similarity(emb, self.banned_embeddings)
        return torch.any(sim > self.semantic_threshold).item()

    def redirect_scores(self, scores): # Redirect banned tokens to replacement tokens
        if not self.block_map:
            return scores
        if scores.ndim == 2:
            for bad_id, repl_id in self.block_map.items(): # Batch mode
                if bad_id in self.block_info:
                    self.replacements.append(self.block_info[bad_id].copy())
                scores[:, repl_id] = torch.maximum(scores[:, repl_id], scores[:, bad_id])
                scores[:, bad_id] = float("-inf")
        else:
            for bad_id, repl_id in self.block_map.items(): # Single sequence
                if bad_id in self.block_info:
                    self.replacements.append(self.block_info[bad_id].copy())
                scores[repl_id] = max(scores[repl_id], scores[bad_id])
                scores[bad_id] = float("-inf")
        return scores

    def __call__(self, input_ids, scores): # Main processor
        text = self.tokenizer.decode(input_ids[0][-self.window:], skip_special_tokens=True).lower()
        if self.semantic_match(text): # Semantic detection → redirect continuation
            return self.redirect_scores(scores)
        return self.redirect_scores(scores) # Normal blocking (prevent starting banned words)
