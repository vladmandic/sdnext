import os
import time
import gradio as gr
from installer import install
from modules import ui_symbols, ui_components
from modules.logger import log


class Page:
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
            self.mtime = time.localtime(os.path.getmtime(self.fn))
            with open(self.fn, encoding='utf-8') as f:
                content = f.read()
            self.size = len(content)
            self.lines = [line.strip().lower() + ' ' for line in content.splitlines() if len(line)>1]
            self.h1 = [line[1:] for line in self.lines if line.startswith('# ')]
            self.h2 = [line[2:] for line in self.lines if line.startswith('## ')]
            self.h3 = [line[3:] for line in self.lines if line.startswith('### ')]
            if not full:
                self.lines.clear()
        except Exception as e:
            log.error(f'Search docs: page="{self.fn}" {e}')

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
        if self.fn is None or not os.path.exists(self.fn):
            log.error(f'Search docs: page="{self.fn}" does not exist')
            return f'page="{self.fn}" does not exist'
        try:
            with open(self.fn, encoding='utf-8') as f:
                content = f.read()
                return content
        except Exception as e:
            log.error(f'Search docs: page="{self.fn}" {e}')
        return ''

    def __str__(self):
        return f'Page(title="{self.title.strip()}" fn="{self.fn}" mtime={self.mtime} h1={[h.strip() for h in self.h1]} h2={len(self.h2)} h3={len(self.h3)} lines={len(self.lines)} size={self.size})'


class Pages:
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
        if not text or len(text) < 2:
            return []
        if len(self.pages) == 0:
            self.build(full=full)
        try:
            text = text.lower()
            scores = [page.search(text) for page in self.pages]
            mtimes = [page.mtime for page in self.pages]
            found = sorted(zip(scores, mtimes, self.pages, strict=False), key=lambda x: (x[0], x[1]), reverse=True)
            found = [item for item in found if item[0] > 0]
            return [(item[0], item[2]) for item in found][:topk]
        except Exception as e:
            log.error(f'Search docs: text="{text}" {e}')
            return []

    def get(self, title: str) -> Page:
        if len(self.pages) == 0:
            self.build(full=self.full)
        for page in self.pages:
            if page.title.lower().strip() == title.lower().strip():
                return page
        return Page('')


index = Pages()


def get_docs_page(page_title: str) -> str:
    if len(index.pages) == 0:
        index.build(full=True)
    page = index.get(page_title)
    log.debug(f'Search docs: title="{page_title}" {page}')
    content = page.get()
    return content


def search_html(pages: list[Page]) -> str:
    html = ''
    for score, page in pages:
        if score > 0.0:
            html += f'''
                <div class="docs-card" onclick="clickDocsPage('{page.title}')">
                    <div class="docs-card-title">{page.title.strip()}</div>
                    <div class="docs-card-h1">Heading | {' | '.join([h.strip() for h in page.h1])}</div>
                    <div class="docs-card-h2"><b>Topics</b> | {' | '.join([h.strip() for h in page.h2])}</div>
                    <div class="docs-card-footer">
                        <span class="docs-card-score">Score | {score}</span>
                        <span class="docs-card-mtime">Last modified | {time.strftime('%c', page.mtime)}</span>
                    </div>
                </div>'''
    return html


def search_docs(search_term):
    topk = 10
    full = True
    t0 = time.time()
    results = index.search(search_term, topk=topk, full=full)
    t1 = time.time()
    log.debug(f'Search results: search="{search_term}" topk={topk}, full={full} pages={len(results)} size={index.size} time={t1-t0:.3f}')
    for score, page in results:
        log.trace(f'Search results: score={score:.2f} {page}')
    html = search_html(results)
    return html


def get_github_page(page):
    try:
        with open(os.path.join('wiki', f'{page}.md'), encoding='utf-8') as f:
            content = f.read()
            log.debug(f'Search wiki: page="{page}" size={len(content)}')
    except Exception as e:
        log.error(f'Search wiki: page="{page}" {e}')
        content = f'Error: {e}'
    return content


def search_github(search_term):
    import requests
    from urllib.parse import quote
    install('beautifulsoup4')
    from bs4 import BeautifulSoup

    url = f'https://github.com/search?q=repo%3Avladmandic%2Fsdnext+{quote(search_term)}&type=wikis'
    res = requests.get(url, timeout=10)
    pages = []
    if res.status_code == 200:
        html = res.content
        soup = BeautifulSoup(html, 'html.parser')

        # remove header links
        tags = soup.find_all(attrs={"data-hovercard-url": "/vladmandic/sdnext/hovercard"})
        for tag in tags:
            tag.extract()

        # replace relative links with full links
        tags = soup.find_all('a')
        for tag in tags:
            if tag.has_attr('href'):
                if tag['href'].startswith('/vladmandic/sdnext/wiki/'):
                    page = tag['href'].replace('/vladmandic/sdnext/wiki/', '')
                    tag.name = 'div'
                    tag['class'] = 'github-page'
                    tag['onclick'] = f'clickGitHubWikiPage("{page}")'
                    pages.append(page)
                elif tag['href'].startswith('/'):
                    tag['href'] = 'https://github.com' + tag['href']

        # find result only
        result = soup.find(attrs={"data-testid": "results-list"})
        if result is None:
            return 'No results found'
        html = str(result)
    else:
        html = f'Error: {res.status_code}'
    log.debug(f'Search wiki: code={res.status_code} text="{search_term}" pages={pages}')
    return html


def create_ui_logs():
    def get_changelog():
        with open('CHANGELOG.md', encoding='utf-8') as f:
            content = f.read()
            content = content.replace('# Change Log for SD.Next', '  ')
        return content

    with gr.Column():
        get_changelog_btn = gr.Button(value='Get Changelog', elem_id="get_changelog")
    with gr.Column():
        _changelog_search = gr.Textbox(label="Search Changelog", elem_id="changelog_search", elem_classes="docs-search")
        _changelog_result = gr.HTML(elem_id="changelog_result")

    changelog_markdown = gr.Markdown('', elem_id="changelog_markdown")
    get_changelog_btn.click(fn=get_changelog, outputs=[changelog_markdown], show_progress='full')


def create_ui_github():
    with gr.Row():
        github_search = gr.Textbox(label="Search GitHub Wiki Pages", elem_id="github_search", elem_classes="docs-search")
        github_search_btn = ui_components.ToolButton(value=ui_symbols.search, elem_id="github_btn_search")
    with gr.Row():
        github_result = gr.HTML(elem_id="github_result", value='', elem_classes="github-result")
    with gr.Row():
        github_md_btn = gr.Button(value='html2md', elem_id="github_md_btn", visible=False)
        github_md = gr.Markdown(elem_id="github_md", value='', elem_classes="github-md")
    github_search.submit(fn=search_github, inputs=[github_search], outputs=[github_result], show_progress='full')
    github_search_btn.click(fn=search_github, inputs=[github_search], outputs=[github_result], show_progress='full')
    github_md_btn.click(fn=get_github_page, _js='getGitHubWikiPage', inputs=[github_search], outputs=[github_md], show_progress='full')


def create_ui_docs():
    with gr.Row():
        docs_search = gr.Textbox(label="Search Docs", elem_id="github_search", elem_classes="docs-search")
        docs_search_btn = ui_components.ToolButton(value=ui_symbols.search, elem_id="docs_btn_search")
    with gr.Row():
        docs_result = gr.HTML(elem_id="docs_result", value='', elem_classes="docs-result")
    with gr.Row():
        docs_md_btn = gr.Button(value='html2md', elem_id="docs_md_btn", visible=False)
        docs_md = gr.Markdown(elem_id="docs_md", value='', elem_classes="docs-md")
    docs_search.submit(fn=search_docs, inputs=[docs_search], outputs=[docs_result], show_progress='hidden')
    docs_search.change(fn=search_docs, inputs=[docs_search], outputs=[docs_result], show_progress='hidden')
    docs_search_btn.click(fn=search_docs, inputs=[docs_search], outputs=[docs_result], show_progress='hidden')
    docs_md_btn.click(fn=get_docs_page, _js='getDocsPage', inputs=[docs_search], outputs=[docs_md], show_progress='hidden')


def create_ui():
    log.debug('UI initialize: tab=info')
    with gr.Tabs(elem_id="tabs_info"):
        with gr.TabItem("Docs", id="docs", elem_id="system_tab_docs"):
            create_ui_docs()
        with gr.TabItem("Wiki", id="wiki", elem_id="system_tab_wiki"):
            create_ui_github()
        with gr.TabItem("Change log", id="change_log", elem_id="system_tab_changelog"):
            create_ui_logs()
