import gradio as gr
from modules import ui_symbols, ui_components, shared


def create_ui_logs():
    def get_changelog():
        with open('CHANGELOG.md', 'r', encoding='utf-8') as f:
            content = f.read()
            content = content.replace('# Change Log for SD.Next', '  ')
        return content

    with gr.Column():
        get_changelog_btn = gr.Button(value='Get Changelog', elem_id="get_changelog")
    with gr.Column():
        _changelog_search = gr.Textbox(label="Search Changelog", elem_id="changelog_search")
        _changelog_result = gr.HTML(elem_id="changelog_result")

    changelog_markdown = gr.Markdown('', elem_id="changelog_markdown")
    get_changelog_btn.click(fn=get_changelog, outputs=[changelog_markdown], show_progress=True)


def create_ui_wiki():
    def search_github(search_term):
        import requests
        from urllib.parse import quote
        from installer import install

        install('beautifulsoup4')
        from bs4 import BeautifulSoup

        url = f'https://github.com/search?q=repo%3Avladmandic%2Fsdnext+{quote(search_term)}&type=wikis'
        res = requests.get(url, timeout=10)
        shared.log.debug(f'Search: wiki="{search_term}" code={res.status_code}')
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
                if tag.has_attr('href') and tag['href'].startswith('/'):
                    tag['href'] = 'https://github.com' + tag['href']

            # find result only
            result = soup.find(attrs={"data-testid": "results-list"})
            if result is None:
                return 'No results found'
            html = str(result)
            return html
        else:
            return f'Error: {res.status_code}'

    with gr.Row():
        wiki_search = gr.Textbox(label="Search Wiki Pages", elem_id="wiki_search")
        wiki_search_btn = ui_components.ToolButton(value=ui_symbols.search, elem_id="wiki_search_btn")
    with gr.Row():
        wiki_result = gr.HTML(elem_id="wiki_result", value='')
    wiki_search.submit(_js="wikiSearch", fn=search_github, inputs=[wiki_search], outputs=[wiki_result])
    wiki_search_btn.click(_js="wikiSearch", fn=search_github, inputs=[wiki_search], outputs=[wiki_result])
