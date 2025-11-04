let changelogElements = [];

const getAllChildren = (el) => {
  const elements = [];
  for (let i = 0; i < el.children.length; i++) {
    elements.push(el.children[i]);
    if (el.children[i].children.length) elements.push(...getAllChildren(el.children[i]));
  }
  return elements;
};

function getText(el) {
  let text = '';
  el.childNodes.forEach((node) => {
    if (node.nodeType === Node.TEXT_NODE) text += node.nodeValue;
  });
  return text.trim();
}

let currentElement = -1;

function changelogNavigate(found) {
  const result = gradioApp().getElementById('changelog_result');
  result.innerHTML = '';
  const text = document.createElement('p');

  const onPrev = () => {
    if (currentElement > 0) {
      currentElement--;
      found[currentElement].scrollIntoView();
      text.innerHTML = ` &nbsp search item ${currentElement + 1} of ${found.length}`;
    }
  };
  const onNext = () => {
    if (currentElement < found.length - 1) {
      currentElement++;
      found[currentElement].scrollIntoView();
      text.innerHTML = ` &nbsp search item ${currentElement + 1} of ${found.length}`;
    }
  };

  const prev = document.createElement('p');
  prev.innerHTML = ' ⇦ ';
  prev.className = 'changelog_arrow';
  prev.onclick = onPrev;
  prev.title = 'Search previous';
  result.appendChild(prev);

  const next = document.createElement('p');
  next.innerHTML = ' ⇨ ';
  next.className = 'changelog_arrow';
  next.title = 'Search next';
  next.onclick = onNext;
  result.appendChild(next);

  text.innerHTML = ` &nbsp found ${found.length} items`;
  result.appendChild(text);
}

async function initChangelog() {
  const search = gradioApp().querySelector('#changelog_search > label> textarea');
  const md = gradioApp().getElementById('changelog_markdown');
  if (!search || !md) {
    // error('initChangelog', 'Missing search or markdown elements');
    return;
  }
  const searchChangelog = async (e) => {
    if (changelogElements.length < 100) changelogElements = getAllChildren(md);
    const found = [];
    for (const el of changelogElements) {
      if (search.value.length > 1 && getText(el).toLowerCase().includes(search.value.toLowerCase())) {
        el.classList.add('changelog_highlight');
        found.push(el);
      } else {
        el.classList.remove('changelog_highlight');
      }
    }
    changelogNavigate(found);
  };
  search.addEventListener('keyup', searchChangelog);
}
