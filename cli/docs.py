#!/usr/bin/env python
import os
import sys
import time
import logging


logging.basicConfig(level = logging.INFO, format = '%(asctime)s %(levelname)s: %(message)s')
log = logging.getLogger(__name__)


class Page():
    def __init__(self, fn, full: bool = True):
        self.fn = fn
        self.title = ''
        self.size = 0
        self.mtime = 0
        self.h1 = []
        self.h2 = []
        self.h3 = []
        self.lines = []
        self.read(full=full)

    def read(self, full: bool = True):
        try:
            self.title = ' ' + os.path.basename(self.fn).replace('.md', '').replace('-', ' ') + ' '
            self.mtime = int(os.path.getmtime(self.fn))
            with open(self.fn, 'r', encoding='utf-8') as f:
                content = f.read()
            self.size = len(content)
            self.lines = [line.strip().lower() + ' ' for line in content.splitlines() if len(line)>1]
            self.h1 = [line[1:] for line in self.lines if line.startswith('# ')]
            self.h2 = [line[2:] for line in self.lines if line.startswith('## ')]
            self.h3 = [line[3:] for line in self.lines if line.startswith('### ')]
            if not full:
                self.lines.clear()
        except Exception as e:
            log.error(f'Wiki: page="{self.fn}" {e}')

    def search(self, text):
        if not text or len(text) < 2:
            return []
        text = text.lower()
        if text.strip() == self.title.lower().strip():
            return 1.0
        if self.title.lower().startswith(f'{text} '):
            return 0.99
        if f' {text} ' in self.title.lower():
            return 0.98
        if f' {text}' in self.title.lower():
            return 0.97

        if any(f' {text} ' in h for h in self.h1):
            return 0.89
        if any(f' {text}' in h for h in self.h1):
            return 0.88

        if any(f' {text} ' in h for h in self.h2):
            return 0.79
        if any(f' {text}' in h for h in self.h2):
            return 0.78

        if any(f' {text} ' in h for h in self.h3):
            return 0.69
        if any(f' {text}' in h for h in self.h3):
            return 0.68

        if f'{text}' in self.title.lower():
            return 0.59
        if any(f'{text}' in h for h in self.h1):
            return 0.58
        if any(f'{text}' in h for h in self.h2):
            return 0.57
        if any(f'{text}' in h for h in self.h3):
            return 0.56

        if any(text in line for line in self.lines):
            return 0.50

        return 0.0

    def get(self):
        try:
            with open(self.fn, 'r', encoding='utf-8') as f:
                content = f.read()
                return content
        except Exception as e:
            log.error(f'Wiki: page="{self.fn}" {e}')
        return ''

    def __str__(self):
        return f'Page(title="{self.title.strip()}" fn="{self.fn}" mtime={self.mtime} h1={[h.strip() for h in self.h1]} h2={len(self.h2)} h3={len(self.h3)} lines={len(self.lines)} size={self.size})'


class Pages():
    def __init__(self):
        self.time = time.time()
        self.size = 0
        self.full = None
        self.pages: list[Page] = []

    def build(self, full: bool = True):
        self.pages.clear()
        self.full = full
        with os.scandir('wiki') as entries:
            for entry in entries:
                if entry.is_file() and entry.name.endswith('.md'):
                    page = Page(entry.path, full=full)
                    self.pages.append(page)
        self.size = sum(page.size for page in self.pages)

    def search(self, text: str, topk: int = 10, full: bool = True) -> list[Page]:
        if not text:
            return []
        if len(self.pages) == 0:
            self.build(full=full)
        text = text.lower()
        scores = [page.search(text) for page in self.pages]
        mtimes = [page.mtime for page in self.pages]
        found = sorted(zip(scores, mtimes, self.pages), key=lambda x: (x[0], x[1]), reverse=True)
        found = [item for item in found if item[0] > 0]
        return [(item[0], item[2]) for item in found][:topk]


index = Pages()


if __name__ == "__main__":
    sys.argv.pop(0)
    if len(sys.argv) < 1:
        log.error("Usage: python cli/docs.py <search_term>")
    text = ' '.join(sys.argv)
    topk = 10
    full = True
    log.info(f'Search: "{text}" topk={topk}, full={full}')
    t0 = time.time()
    results = index.search(text, topk=topk, full=full)
    t1 = time.time()
    log.info(f'Results: pages={len(results)} size={index.size} time={t1-t0:.3f}')
    for score, page in results:
        log.info(f'Score: {score:.2f} {page}')
    # if len(results) > 0:
    #     log.info('Top result:')
    #     log.info(results[0][1].get())
