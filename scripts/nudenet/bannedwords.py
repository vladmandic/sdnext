import re


def check_banned(words:str='', prompt:str='') -> list:
    words = [a.lower().strip() for a in words.split(',')] if words else [] if isinstance(words, str) else words
    prompt = prompt.lower().replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').replace('_', ' ').replace('  ', ' ').replace(',', ' ').replace('.', ' ')
    found = [word for word in words if re.search(r'\b' + re.escape(word) + r'\b', prompt)]
    return found
