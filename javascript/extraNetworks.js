const activePromptTextarea = {};
let sortVal = -1;
let totalCards = -1;
let lastTab = 'control';

// helpers

const getENActiveTab = () => {
  let tabName = '';
  if (gradioApp().getElementById('txt2img_prompt')?.checkVisibility() || gradioApp().getElementById('txt2img_generate')?.checkVisibility()) tabName = 'txt2img';
  else if (gradioApp().getElementById('img2img_prompt')?.checkVisibility() || gradioApp().getElementById('img2img_generate')?.checkVisibility()) tabName = 'img2img';
  else if (gradioApp().getElementById('control_prompt')?.checkVisibility() || gradioApp().getElementById('control_generate')?.checkVisibility()) tabName = 'control';
  else if (gradioApp().getElementById('video_prompt')?.checkVisibility() || gradioApp().getElementById('video_generate')?.checkVisibility()) tabName = 'video';
  else if (gradioApp().getElementById('extras_image')?.checkVisibility()) tabName = 'process';
  else if (gradioApp().getElementById('interrogate_image')?.checkVisibility()) tabName = 'caption';
  else if (gradioApp().getElementById('tab-gallery-search')?.checkVisibility()) tabName = 'gallery';

  if (['process', 'caption', 'gallery'].includes(tabName)) {
    tabName = lastTab;
  } else if (tabName !== '') {
    lastTab = tabName;
  }

  if (tabName !== '') return tabName;
  // legacy method
  if (gradioApp().getElementById('tab_txt2img')?.style.display === 'block') tabName = 'txt2img';
  else if (gradioApp().getElementById('tab_img2img')?.style.display === 'block') tabName = 'img2img';
  else if (gradioApp().getElementById('tab_control')?.style.display === 'block') tabName = 'control';
  else if (gradioApp().getElementById('tab_video')?.style.display === 'block') tabName = 'video';
  else tabName = 'control';
  // log('getENActiveTab', tabName);
  return tabName;
};

const getENActivePage = () => {
  const tabName = getENActiveTab();
  let page = gradioApp().querySelector(`#${tabName}_extra_networks > .tabs > .tab-nav > .selected`);
  if (!page) page = gradioApp().querySelector(`#${tabName}_extra_tabs > .tab-nav > .selected`);
  const pageName = page ? page.innerText : '';
  const btnApply = gradioApp().getElementById(`${tabName}_extra_apply`);
  if (btnApply) btnApply.style.display = pageName === 'Style' ? 'inline-flex' : 'none';
  // log('getENActivePage', pageName);
  return pageName;
};

const setENState = (state) => {
  if (!state) return;
  state.tab = getENActiveTab();
  state.page = getENActivePage();
  // log('setENState', state);
  const el = gradioApp().querySelector(`#${state.tab}_extra_state  > label > textarea`);
  if (el) {
    el.value = JSON.stringify(state);
    updateInput(el);
  }
};

// methods

function showCardDetails(event) {
  // log('showCardDetails', event);
  const tabName = getENActiveTab();
  const btn = gradioApp().getElementById(`${tabName}_extra_details_btn`);
  btn.click();
  event.stopPropagation();
  event.preventDefault();
}

function getCardDetails(...args) {
  // log('getCardDetails', args);
  const el = event?.target?.parentElement?.parentElement;
  if (el?.classList?.contains('card')) setENState({ op: 'getCardDetails', item: el.dataset.name });
  else setENState({ op: 'getCardDetails', item: null });
  return [...args];
}

function readCardTags(el, tags) {
  const replaceOutsideBrackets = (input, target, replacement) => input.split(/(<[^>]*>|\{[^}]*\})/g).map((part, i) => {
    if (i % 2 === 0) return part.split(target).join(replacement); // Only replace in the parts that are not inside brackets (which are at even indices)
    return part;
  }).join('');

  const clickTag = (e, tag) => {
    e.preventDefault();
    e.stopPropagation();
    const textarea = activePromptTextarea[getENActiveTab()];
    let new_prompt = textarea.value;
    new_prompt = replaceOutsideBrackets(new_prompt, ` ${tag}`, ''); // try to remove tag
    new_prompt = replaceOutsideBrackets(new_prompt, `${tag} `, '');
    if (new_prompt === textarea.value) new_prompt += ` ${tag}`; // if not removed, then append it
    textarea.value = new_prompt;
    updateInput(textarea);
  };

  if (tags.length === 0) return;
  const cardTags = tags.split('|');
  if (!cardTags || cardTags.length === 0) return;
  const tagsEl = el.getElementsByClassName('tags')[0];
  if (!tagsEl?.children || tagsEl.children.length > 0) return;
  for (const tag of cardTags) {
    const span = document.createElement('span');
    span.classList.add('tag');
    span.textContent = tag;
    span.onclick = (e) => clickTag(e, tag);
    tagsEl.appendChild(span);
  }
}

function readCardDescription(page, item) {
  xhrGet('/sdapi/v1/network/desc', { page, item }, (data) => {
    const tabName = getENActiveTab();
    const description = gradioApp().querySelector(`#${tabName}_description > label > textarea`);
    if (description) {
      description.value = data?.description?.trim() || '';
      updateInput(description);
    }
    setENState({ op: 'readCardDescription', page, item });
  });
}

function getCardsForActivePage() {
  const pageName = getENActivePage();
  if (!pageName) return [];
  let allCards = Array.from(gradioApp().querySelectorAll('.extra-network-cards > .card'));
  allCards = allCards.filter((el) => el.dataset.page?.toLowerCase().includes(pageName.toLowerCase()));
  // log('getCardsForActivePage', pagename, cards.length);
  return allCards;
}

async function filterExtraNetworksForTab(searchTerm) {
  let items = 0;
  let found = 0;
  searchTerm = searchTerm.toLowerCase().trim();
  const t0 = performance.now();
  const pagename = getENActivePage();
  if (!pagename) return;
  const allPages = Array.from(gradioApp().querySelectorAll('.extra-network-cards'));
  const pages = allPages.filter((el) => el.id.toLowerCase().includes(pagename.toLowerCase()));
  for (const pg of pages) {
    const cards = Array.from(pg.querySelectorAll('.card') || []);
    items += cards.length;
    if (searchTerm === '' || searchTerm === 'all/') {
      cards.forEach((elem) => { elem.style.display = ''; });
    } else if (searchTerm === 'reference/') {
      cards.forEach((elem) => {
        elem.style.display = elem.dataset.name
          .toLowerCase()
          .includes('reference/') && elem.dataset.tags === '' ? '' : 'none';
      });
    } else if (searchTerm === 'distilled/') {
      cards.forEach((elem) => {
        elem.style.display = elem.dataset.tags
          .toLowerCase()
          .includes('distilled') ? '' : 'none';
      });
    } else if (searchTerm === 'community/') {
      cards.forEach((elem) => {
        elem.style.display = elem.dataset.tags
          .toLowerCase()
          .includes('community') ? '' : 'none';
      });
    } else if (searchTerm === 'cloud/') {
      cards.forEach((elem) => {
        elem.style.display = elem.dataset.tags
          .toLowerCase()
          .includes('cloud') ? '' : 'none';
      });
    } else if (searchTerm === 'quantized/') {
      cards.forEach((elem) => {
        elem.style.display = elem.dataset.tags
          .toLowerCase()
          .includes('quantized') ? '' : 'none';
      });
    } else if (searchTerm === 'nunchaku/') {
      cards.forEach((elem) => {
        elem.style.display = elem.dataset.tags
          .toLowerCase()
          .includes('nunchaku') ? '' : 'none';
      });
    } else if (searchTerm === 'local/') {
      cards.forEach((elem) => {
        elem.style.display = elem.dataset.name
          .toLowerCase()
          .includes('reference/') ? 'none' : '';
      });
    } else if (searchTerm === 'diffusers/') {
      cards.forEach((elem) => {
        elem.style.display = elem.dataset.name
          .toLowerCase().replace('models--', 'diffusers').replaceAll('\\', '/')
          .includes('diffusers/') ? '' : 'none';
      });
    } else if (searchTerm.startsWith('r#')) {
      searchTerm = searchTerm.substring(2);
      const re = new RegExp(searchTerm, 'i');
      cards.forEach((elem) => {
        elem.style.display = re.test(`filename: ${elem.dataset.filename}|name: ${elem.dataset.name}|tags: ${elem.dataset.tags}`) ? '' : 'none';
      });
    } else {
      const searchList = searchTerm.split('|').filter((s) => s !== '' && !s.startsWith('-')).map((s) => s.trim());
      const excludeList = searchTerm.split('|').filter((s) => s !== '' && s.trim().startsWith('-')).map((s) => s.trim().substring(1).trim());
      const searchListAll = searchList.map((s) => s.split('&').map((t) => t.trim()));
      const excludeListAll = excludeList.map((s) => s.split('&').map((t) => t.trim()));
      cards.forEach((elem) => {
        let text = '';
        if (elem.dataset.filename) text += `${elem.dataset.filename} `;
        if (elem.dataset.name) text += `${elem.dataset.name} `;
        if (elem.dataset.tags) text += `${elem.dataset.tags} `;
        text = text.toLowerCase().replace('models--', 'diffusers').replaceAll('\\', '/');
        if (searchListAll.some((sl) => sl.every((st) => text.includes(st))) && !excludeListAll.some((el) => el.every((et) => text.includes(et)))) {
          elem.style.display = '';
        } else {
          elem.style.display = 'none';
        }
      });
    }
    found += cards.filter((elem) => elem.style.display === '').length;
  }
  const t1 = performance.now();
  log(`filterExtraNetworks: text="${searchTerm}" items=${items} match=${found} time=${Math.round(t1 - t0)}`);
}

function tryToRemoveExtraNetworkFromPrompt(textarea, text) {
  const re_extranet = /<([^:]+:[^:]+):[\d.]+>/;
  const re_extranet_g = /\s+<([^:]+:[^:]+):[\d.]+>/g;
  let m = text.match(re_extranet);
  let replaced = false;
  let newTextareaText;
  if (m) {
    const partToSearch = m[1];
    newTextareaText = textarea.value.replaceAll(re_extranet_g, (found) => {
      m = found.match(re_extranet);
      if (m[1] === partToSearch) {
        replaced = true;
        return '';
      }
      return found;
    });
  } else {
    newTextareaText = textarea.value.replaceAll(new RegExp(text, 'g'), (found) => {
      if (found === text) {
        replaced = true;
        return '';
      }
      return found;
    });
  }
  if (replaced) {
    textarea.value = newTextareaText;
    return true;
  }
  return false;
}

function sortExtraNetworks(fixed = 'no') {
  const t0 = performance.now();
  const sortDesc = ['Default', 'Name [A-Z]', 'Name [Z-A]', 'Date [Newest]', 'Date [Oldest]', 'Size [Largest]', 'Size [Smallest]'];
  const pagename = getENActivePage();
  if (!pagename) return 'sort error: unknown page';
  const allPages = Array.from(gradioApp().querySelectorAll('.extra-network-cards'));
  const pages = allPages.filter((el) => el.id.toLowerCase().includes(pagename.toLowerCase()));
  let num = 0;
  if (sortVal === -1) sortVal = sortDesc.indexOf(opts.extra_networks_sort);
  if (fixed !== 'fixed') sortVal = (sortVal + 1) % sortDesc.length;
  for (const pg of pages) {
    const cards = Array.from(pg.querySelectorAll('.card') || []);
    if (cards.length === 0) return 'sort: no cards';
    num += cards.length;
    cards.sort((a, b) => {
      switch (sortVal) {
        case 0: return 0;
        case 1: return a.dataset.name ? a.dataset.name.localeCompare(b.dataset.name) : 0;
        case 2: return b.dataset.name ? b.dataset.name.localeCompare(a.dataset.name) : 0;
        case 3: return a.dataset.mtime ? (new Date(b.dataset.mtime)).getTime() - (new Date(a.dataset.mtime)).getTime() : 0;
        case 4: return b.dataset.mtime ? (new Date(a.dataset.mtime)).getTime() - (new Date(b.dataset.mtime)).getTime() : 0;
        case 5: return a.dataset.size && !isNaN(a.dataset.size) ? parseFloat(b.dataset.size) - parseFloat(a.dataset.size) : 0;
        case 6: return b.dataset.size && !isNaN(b.dataset.size) ? parseFloat(a.dataset.size) - parseFloat(b.dataset.size) : 0;
      }
      return 0;
    });
    for (const card of cards) pg.appendChild(card);
  }
  const desc = sortDesc[sortVal];
  const t1 = performance.now();
  log('sortNetworks', { name: pagename, val: sortVal, order: desc, fixed: fixed === 'fixed', items: num, time: Math.round(t1 - t0) });
  return desc;
}

function refreshENInput(tabName) {
  log('refreshNetworks', tabName, gradioApp().querySelector(`#${tabName}_extra_networks textarea`)?.value);
  gradioApp().querySelector(`#${tabName}_extra_networks textarea`)?.dispatchEvent(new Event('input'));
}

async function markSelectedCards(selected, page = '') {
  log('markSelectedCards', selected, page);
  gradioApp().querySelectorAll('.extra-network-cards .card').forEach((el) => {
    if (page.length > 0 && el.dataset.page !== page) return; // filter by page
    if (selected.includes(el.dataset.name) || selected.includes(el.dataset.short)) el.classList.add('card-selected');
    else el.classList.remove('card-selected');
  });
}

function extractLoraNames(prompt) {
  const regex = /<lora:([^:>]+)(?::[\d.]+)?>/g;
  const names = [];
  let match;
  while ((match = regex.exec(prompt)) !== null) names.push(match[1]); // eslint-disable-line no-cond-assign
  return names;
}

function cardClicked(textToAdd) {
  const tabName = getENActiveTab();
  log('cardClicked', tabName, textToAdd);
  const textarea = activePromptTextarea[tabName];
  if (textarea.value.indexOf(textToAdd) !== -1) textarea.value = textarea.value.replace(textToAdd, '');
  else textarea.value += textToAdd;
  updateInput(textarea);
  markSelectedCards(extractLoraNames(textarea.value), 'lora');
}

function extraNetworksSearchButton(event) {
  // log('extraNetworksSearchButton', event);
  const tabName = getENActiveTab();
  const searchTextarea = gradioApp().querySelector(`#${tabName}_extra_search textarea`);
  const button = event.target;
  if (searchTextarea) {
    searchTextarea.value = `${button.textContent.trim()}/`;
    updateInput(searchTextarea);
  } else {
    console.error(`Could not find the search textarea for the tab: ${tabName}`);
  }
}

function extraNetworksFilterVersion(event) {
  const version = event.target.textContent.trim();
  const activePage = getENActivePage().toLowerCase();
  const cardContainers = gradioApp().querySelectorAll('.extra-network-cards');
  log('extraNetworksFilterVersion', { activePage, version });
  for (const cardContainer of cardContainers) {
    if (!cardContainer.id.includes(activePage)) continue;
    if (cardContainer.dataset.activeVersion === version) {
      cardContainer.dataset.activeVersion = '';
      cardContainer.querySelectorAll('.card').forEach((card) => { card.style.display = ''; });
    } else {
      cardContainer.dataset.activeVersion = version;
      cardContainer.querySelectorAll('.card').forEach((card) => {
        if (card.dataset.version === version) card.style.display = '';
        else card.style.display = 'none';
      });
    }
  }
}

let desiredStyle = '';
function selectStyle(name) {
  desiredStyle = name;
  const tabName = getENActiveTab();
  const button = gradioApp().querySelector(`#${tabName}_styles_select`);
  button.click();
}

function applyStyles(styles) {
  let newStyles = [];
  if (styles) {
    newStyles = Array.isArray(styles) ? styles : [styles];
  } else {
    const tabName = getENActiveTab();
    styles = gradioApp().querySelectorAll(`#${tabName}_styles .token span`);
    newStyles = Array.from(styles).map((el) => el.textContent).filter((el) => el.length > 0);
  }
  const index = newStyles.indexOf(desiredStyle);
  if (index > -1) newStyles.splice(index, 1);
  else newStyles.push(desiredStyle);
  markSelectedCards(newStyles, 'style');
  return newStyles.join('|');
}

function quickApplyStyle() {
  const tabName = getENActiveTab();
  const btnApply = gradioApp().getElementById(`${tabName}_extra_apply`);
  if (btnApply) btnApply.click();
}

function quickSaveStyle() {
  const tabName = getENActiveTab();
  const btnSave = gradioApp().getElementById(`${tabName}_extra_quicksave`);
  if (btnSave) btnSave.click();
  const btnRefresh = gradioApp().getElementById(`${tabName}_extra_refresh`);
  if (btnRefresh) {
    setTimeout(() => btnRefresh.click(), 100);
    // setTimeout(() => sortExtraNetworks('fixed'), 500);
  }
}

function selectHistory(id) {
  const headers = new Headers();
  headers.set('Content-Type', 'application/json');
  const init = { method: 'POST', body: { name: id }, headers };
  authFetch(`${window.api}/history`, { method: 'POST', body: JSON.stringify({ name: id }), headers });
}

let enDirty = false;
function closeDetailsEN(...args) {
  // log('closeDetailsEN');
  enDirty = true;
  const tabName = getENActiveTab();
  const btnClose = gradioApp().getElementById(`${tabName}_extra_details_close`);
  if (btnClose) setTimeout(() => btnClose.click(), 100);
  const btnRefresh = gradioApp().getElementById(`${tabName}_extra_refresh`);
  if (btnRefresh && enDirty) setTimeout(() => btnRefresh.click(), 100);
  return [...args];
}

function refeshDetailsEN(args) {
  // log(`refeshDetailsEN: ${enDirty}`);
  const tabName = getENActiveTab();
  const btnRefresh = gradioApp().getElementById(`${tabName}_extra_refresh`);
  if (btnRefresh && enDirty) setTimeout(() => btnRefresh.click(), 100);
  enDirty = false;
  return args;
}

// refresh on en show
function refreshENpage() {
  if (getCardsForActivePage().length === 0) {
    // log('refreshENpage');
    const tabName = getENActiveTab();
    const btnRefresh = gradioApp().getElementById(`${tabName}_extra_refresh`);
    if (btnRefresh) btnRefresh.click();
  }
}

// init
function setupExtraNetworksForTab(tabName) {
  let tabs = gradioApp().querySelector(`#${tabName}_extra_tabs`);
  if (tabs) tabs.classList.add('extra-networks');
  const en = gradioApp().getElementById(`${tabName}_extra_networks`);
  tabs = gradioApp().querySelector(`#${tabName}_extra_tabs > div`);
  if (!tabs) return;

  // buttons
  const btnShow = gradioApp().getElementById(`${tabName}_extra_networks_btn`);
  const btnRefresh = gradioApp().getElementById(`${tabName}_extra_refresh`);
  const btnScan = gradioApp().getElementById(`${tabName}_extra_scan`);
  const btnSave = gradioApp().getElementById(`${tabName}_extra_save`);
  const btnClose = gradioApp().getElementById(`${tabName}_extra_close`);
  const btnSort = gradioApp().getElementById(`${tabName}_extra_sort`);
  const btnView = gradioApp().getElementById(`${tabName}_extra_view`);
  const btnModel = gradioApp().getElementById(`${tabName}_extra_model`);
  const btnApply = gradioApp().getElementById(`${tabName}_extra_apply`);
  const buttons = document.createElement('span');
  buttons.classList.add('buttons');
  if (btnRefresh) buttons.appendChild(btnRefresh);
  if (btnModel) buttons.appendChild(btnModel);
  if (btnApply) buttons.appendChild(btnApply);
  if (btnScan) buttons.appendChild(btnScan);
  if (btnSave) buttons.appendChild(btnSave);
  if (btnSort) buttons.appendChild(btnSort);
  if (btnView) buttons.appendChild(btnView);
  if (btnClose) buttons.appendChild(btnClose);
  btnModel.onclick = () => btnModel.classList.toggle('toolbutton-selected');
  // btnRefresh.onclick = () => setTimeout(() => sortExtraNetworks('fixed'), 500);
  tabs.appendChild(buttons);

  // details
  const detailsImg = gradioApp().getElementById(`${tabName}_extra_details_img`);
  const detailsClose = gradioApp().getElementById(`${tabName}_extra_details_close`);
  if (detailsImg && detailsClose) {
    detailsImg.title = 'Close details';
    detailsImg.onclick = () => detailsClose.click();
  }

  // search and description
  const div = document.createElement('div');
  div.classList.add('second-line');
  tabs.appendChild(div);
  const txtSearch = gradioApp().querySelector(`#${tabName}_extra_search`);
  const txtSearchValue = gradioApp().querySelector(`#${tabName}_extra_search textarea`);
  const txtDescription = gradioApp().getElementById(`${tabName}_description`);
  txtSearch.classList.add('search');
  txtDescription.classList.add('description');
  div.appendChild(txtSearch);
  div.appendChild(txtDescription);
  let searchTimer = null;
  txtSearchValue.addEventListener('input', (evt) => {
    if (searchTimer) clearTimeout(searchTimer);
    searchTimer = setTimeout(async () => {
      await filterExtraNetworksForTab(txtSearchValue.value.toLowerCase());
      searchTimer = null;
    }, 100);
  });

  // card hover
  let hoverTimer = null;
  let previousCard = null;
  if (window.opts.extra_networks_fetch) {
    gradioApp().getElementById(`${tabName}_extra_tabs`).onmouseover = async (e) => {
      const el = e.target.closest('.card'); // bubble-up to card
      if (!el || (el.title === previousCard)) return;
      if (!hoverTimer) {
        hoverTimer = setTimeout(() => {
          readCardDescription(el.dataset.page, el.dataset.name);
          readCardTags(el, el.dataset.tags);
          previousCard = el.title;
        }, 300);
      }
      el.onmouseout = () => {
        clearTimeout(hoverTimer);
        hoverTimer = null;
      };
    };
  }

  // auto-resize networks sidebar
  const resizeObserver = new ResizeObserver((entries) => {
    for (const entry of entries) {
      for (const el of Array.from(gradioApp().getElementById(`${tabName}_extra_tabs`).querySelectorAll('.extra-networks-page'))) {
        const h = Math.trunc(entry.contentRect.height);
        if (h <= 0) return;
        const vh = opts.logmonitor_show ? '55vh' : '68vh';
        if (window.opts.extra_networks_card_cover === 'sidebar' && window.opts.theme_type === 'Standard') el.style.height = `max(${vh}, ${h - 90}px)`;
        else if (window.opts.extra_networks_card_cover === 'inline' && window.opts.theme_type === 'Standard') el.style.height = '25vh';
        else if (window.opts.extra_networks_card_cover === 'cover' && window.opts.theme_type === 'Standard') el.style.height = '50vh';
        else el.style.height = 'unset';
        // log(`${tabName} height: ${entry.target.id}=${h} ${el.id}=${el.clientHeight}`);
      }
    }
  });
  const settingsEl = gradioApp().getElementById(`${tabName}_settings`);
  const interfaceEl = gradioApp().getElementById(`${tabName}_interface`);
  if (settingsEl) resizeObserver.observe(settingsEl);
  if (interfaceEl) resizeObserver.observe(interfaceEl);

  // en style
  if (!en) return;
  let lastView;
  let heightInitialized = false;
  const intersectionObserver = new IntersectionObserver((entries) => {
    if (!heightInitialized) {
      heightInitialized = true;
      let h = 0;
      const target = window.opts.extra_networks_card_cover === 'sidebar' ? 0 : window.opts.extra_networks_height;
      if (window.opts.theme_type === 'Standard') h = target > 0 ? target : 55;
      else h = target > 0 ? target : 87;
      for (const el of Array.from(gradioApp().getElementById(`${tabName}_extra_tabs`).querySelectorAll('.extra-networks-page'))) {
        if (h > 0) el.style.height = `${h}vh`;
        el.parentElement.style.width = '-webkit-fill-available';
      }
    }
    const cards = Array.from(gradioApp().querySelectorAll('.extra-network-cards > .card'));
    if (cards.length > 0 && cards.length !== totalCards) {
      totalCards = cards.length;
      sortExtraNetworks('fixed');
    }
    if (lastView !== entries[0].intersectionRatio > 0) {
      lastView = entries[0].intersectionRatio > 0;
      if (lastView) {
        refreshENpage();
        if (window.opts.extra_networks_card_cover === 'cover') {
          en.style.position = 'absolute';
          en.style.height = 'unset';
          en.style.width = 'unset';
          en.style.right = 'unset';
          en.style.maxWidth = 'unset';
          en.style.maxHeight = '58vh';
          en.style.top = '13em';
          en.style.transition = '';
          en.style.zIndex = 100;
          gradioApp().getElementById(`${tabName}_settings`).parentNode.style.width = 'unset';
        } else if (window.opts.extra_networks_card_cover === 'sidebar') {
          en.style.position = 'absolute';
          en.style.height = 'auto';
          en.style.width = `${window.opts.extra_networks_sidebar_width}vw`;
          en.style.maxWidth = '50vw';
          en.style.maxHeight = 'unset';
          en.style.right = '0';
          en.style.top = '13em';
          en.style.transition = 'width 0.3s ease';
          en.style.zIndex = 100;
          gradioApp().getElementById(`${tabName}_settings`).parentNode.style.width = `calc(100vw - 2em - min(${window.opts.extra_networks_sidebar_width}vw, 50vw))`;
        } else {
          en.style.position = 'relative';
          en.style.height = 'unset';
          en.style.width = 'unset';
          en.style.right = 'unset';
          en.style.maxWidth = 'unset';
          en.style.maxHeight = '33vh';
          en.style.top = 0;
          en.style.transition = '';
          en.style.zIndex = 0;
          gradioApp().getElementById(`${tabName}_settings`).parentNode.style.width = 'unset';
        }
      } else {
        if (window.opts.extra_networks_card_cover === 'sidebar') en.style.width = 0;
        gradioApp().getElementById(`${tabName}_settings`).parentNode.style.width = 'unset';
      }
      if (tabName === 'video') {
        gradioApp().getElementById('framepack_settings').parentNode.style.width = gradioApp().getElementById(`${tabName}_settings`).parentNode.style.width;
        gradioApp().getElementById('ltx_settings').parentNode.style.width = gradioApp().getElementById(`${tabName}_settings`).parentNode.style.width;
      }
    }
  });
  intersectionObserver.observe(en); // monitor visibility
}

async function showNetworks() {
  for (const tabName of ['txt2img', 'img2img', 'control', 'video']) {
    const btn = gradioApp().getElementById(`${tabName}_extra_networks_btn`);
    if (window.opts.extra_networks_show && btn) btn.click();
  }
  log('showNetworks');
}

async function setupExtraNetworks() {
  setupExtraNetworksForTab('txt2img');
  setupExtraNetworksForTab('img2img');
  setupExtraNetworksForTab('control');
  setupExtraNetworksForTab('video');

  function registerPrompt(tabName, id) {
    const textarea = gradioApp().querySelector(`#${id} > label > textarea`);
    if (!textarea) return;
    if (!activePromptTextarea[tabName]) activePromptTextarea[tabName] = textarea;
    textarea.addEventListener('focus', () => { activePromptTextarea[tabName] = textarea; });
  }

  registerPrompt('txt2img', 'txt2img_prompt');
  registerPrompt('txt2img', 'txt2img_neg_prompt');
  registerPrompt('img2img', 'img2img_prompt');
  registerPrompt('img2img', 'img2img_neg_prompt');
  registerPrompt('control', 'control_prompt');
  registerPrompt('control', 'control_neg_prompt');
  registerPrompt('video', 'video_prompt');
  registerPrompt('video', 'video_neg_prompt');
  log('initNetworks', window.opts.extra_networks_card_size);
  document.documentElement.style.setProperty('--card-size', `${window.opts.extra_networks_card_size}px`);
}
