const contextMenuInit = () => {
  let eventListenerApplied = false;
  const menuSpecs = new Map();

  const uid = () => Date.now().toString(36) + Math.random().toString(36).substring(2);

  function showContextMenu(event, element, menuEntries) {
    const posx = event.clientX + document.body.scrollLeft + document.documentElement.scrollLeft;
    const posy = event.clientY + document.body.scrollTop + document.documentElement.scrollTop;
    const oldMenu = gradioApp().querySelector('#context-menu');
    if (oldMenu) oldMenu.remove();
    const contextMenu = document.createElement('nav');
    contextMenu.id = 'context-menu';
    contextMenu.style.top = `${posy}px`;
    contextMenu.style.left = `${posx}px`;
    const contextMenuList = document.createElement('ul');
    contextMenuList.className = 'context-menu-items';
    contextMenu.append(contextMenuList);
    menuEntries.forEach((entry) => {
      const contextMenuEntry = document.createElement('a');
      contextMenuEntry.innerHTML = entry.name;
      contextMenuEntry.addEventListener('click', (e) => entry.func());
      contextMenuList.append(contextMenuEntry);
    });
    gradioApp().appendChild(contextMenu);
    const menuWidth = contextMenu.offsetWidth + 4;
    const menuHeight = contextMenu.offsetHeight + 4;
    const windowWidth = window.innerWidth;
    const windowHeight = window.innerHeight;
    if ((windowWidth - posx) < menuWidth) contextMenu.style.left = `${windowWidth - menuWidth}px`;
    if ((windowHeight - posy) < menuHeight) contextMenu.style.top = `${windowHeight - menuHeight}px`;
  }

  function appendContextMenuOption(targetElementSelector, entryName, entryFunction, primary = false) {
    let currentItems = menuSpecs.get(targetElementSelector);
    if (!currentItems) {
      currentItems = [];
      menuSpecs.set(targetElementSelector, currentItems);
    }
    const newItem = {
      id: `${targetElementSelector}_${uid()}`,
      name: entryName,
      func: entryFunction,
      primary,
      // isNew: true,
    };
    currentItems.push(newItem);
    return newItem.id;
  }

  function removeContextMenuOption(id) {
    menuSpecs.forEach((v, k) => {
      let index = -1;
      v.forEach((e, ei) => {
        if (e.id === id) { index = ei; }
      });
      if (index >= 0) v.splice(index, 1);
    });
  }

  async function addContextMenuEventListener() {
    if (eventListenerApplied) return;
    log('initContextMenu');
    gradioApp().addEventListener('click', (e) => {
      if (!e.isTrusted) return;
      const oldMenu = gradioApp().querySelector('#context-menu');
      if (oldMenu) oldMenu.remove();
      menuSpecs.forEach((v, k) => {
        const items = v.filter((item) => item.primary);
        if (items.length > 0 && e.composedPath()[0].matches(k)) {
          showContextMenu(e, e.composedPath()[0], items);
          e.preventDefault();
        }
      });
    });
    gradioApp().addEventListener('contextmenu', (e) => {
      const oldMenu = gradioApp().querySelector('#context-menu');
      if (oldMenu) oldMenu.remove();
      menuSpecs.forEach((v, k) => {
        const items = v.filter((item) => !item.primary);
        if (items.length > 0 && e.composedPath()[0].matches(k)) {
          showContextMenu(e, e.composedPath()[0], items);
          e.preventDefault();
        }
      });
    });
    eventListenerApplied = true;
  }
  return [appendContextMenuOption, removeContextMenuOption, addContextMenuEventListener];
};

const initContextResponse = contextMenuInit();
const appendContextMenuOption = initContextResponse[0];
const removeContextMenuOption = initContextResponse[1];
const addContextMenuEventListener = initContextResponse[2];

const generateForever = (genbuttonid) => {
  if (window.generateOnRepeatInterval) {
    log('generateForever: cancel');
    clearInterval(window.generateOnRepeatInterval);
    window.generateOnRepeatInterval = null;
  } else {
    log('generateForever: start');
    const genbutton = gradioApp().querySelector(genbuttonid);
    const busy = document.getElementById('progressbar')?.style.display === 'block';
    if (!busy) genbutton.click();
    window.generateOnRepeatInterval = setInterval(() => {
      const pbBusy = document.getElementById('progressbar')?.style.display === 'block';
      if (!pbBusy) genbutton.click();
    }, 500);
  }
};

const reprocessClick = (tabId, state) => {
  const btn = document.getElementById(`${tabId}_${state}`);
  window.submit_state = state;
  if (btn) btn.click();
};

const getStatus = async () => {
  const headers = new Headers();
  const body = JSON.stringify({ id_task: -1, id_live_preview: false });
  headers.set('Content-Type', 'application/json');
  const tab = getUICurrentTabContent()?.id.replace('tab_', '') || '';
  const el = gradioApp().querySelector(`#html_log_${tab} .performance p`);

  let res;
  let data;
  res = await fetch('./internal/progress', { method: 'POST', headers, body });
  if (res?.ok) {
    data = await res.json();
    log('progressInternal:', data);
    if (el) el.innerText += '\nProgress internal:\n' + JSON.stringify(data, null, 2); // eslint-disable-line prefer-template
  }
  res = await fetch('./sdapi/v1/progress?skip_current_image=true', { method: 'GET', headers });
  if (res?.ok) {
    data = await res.json();
    log('progressAPI:', data);
    if (el) el.innerText += '\nProgress API:\n' + JSON.stringify(data, null, 2); // eslint-disable-line prefer-template
  }
};

async function initContextMenu() {
  let id = '';
  for (const tab of ['txt2img', 'img2img', 'control', 'video']) {
    id = `#${tab}_generate`;
    appendContextMenuOption(id, 'Get server status', getStatus);
    appendContextMenuOption(id, 'Copy prompt to clipboard', () => navigator.clipboard.writeText(document.querySelector(`#${tab}_prompt > label > textarea`).value));
    appendContextMenuOption(id, 'Generate forever', () => generateForever(`#${tab}_generate`));
    appendContextMenuOption(id, 'Apply selected style', quickApplyStyle);
    appendContextMenuOption(id, 'Quick save style', quickSaveStyle);
    id = `#${tab}_reprocess`;
    appendContextMenuOption(id, 'Decode full quality', () => reprocessClick(`${tab}`, 'reprocess_decode'), true);
    appendContextMenuOption(id, 'Refine & HiRes pass', () => reprocessClick(`${tab}`, 'reprocess_refine'), true);
    appendContextMenuOption(id, 'Detailer pass', () => reprocessClick(`${tab}`, 'reprocess_detail'), true);
  }
  addContextMenuEventListener();
}
