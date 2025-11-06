const allLocales = ['en', 'de', 'es', 'fr', 'it', 'ja', 'ko', 'pt', 'hr', 'ru', 'zh'];
const localeData = {
  prev: null,
  locale: null,
  data: [],
  timeout: null,
  finished: false,
  initial: true,
  type: 2,
  hint: null,
  btn: null,
  expandTimeout: null, // Property for expansion timeout
  currentElement: null, // Track current element for expansion
  observer: null, // MutationObserver for DOM changes
};
let localeTimeout = null;
const isTouchDevice = 'ontouchstart' in window;

async function cycleLocale() {
  clearTimeout(localeTimeout);
  localeTimeout = setTimeout(() => {
    log('cycleLocale', localeData.prev, localeData.locale);
    const index = allLocales.indexOf(localeData.prev);
    localeData.locale = allLocales[(index + 1) % allLocales.length];
    localeData.btn.innerText = localeData.locale;
    // localeData.btn.style.backgroundColor = localeData.locale !== 'en' ? 'var(--primary-500)' : '';
    localeData.finished = false;
    localeData.data = [];
    localeData.prev = localeData.locale;
    window.opts.ui_locale = localeData.locale;
    setHints(); // eslint-disable-line no-use-before-define
  }, 250);
}

async function resetLocale() {
  clearTimeout(localeTimeout); // Prevent the single click logic
  localeData.locale = 'en';
  log('resetLocale', localeData.locale);
  const index = allLocales.indexOf(localeData.locale);
  localeData.locale = allLocales[(index) % allLocales.length];
  localeData.btn.innerText = localeData.locale;
  localeData.finished = false;
  localeData.data = [];
  window.opts.ui_locale = localeData.locale;
  setHints(); // eslint-disable-line no-use-before-define
}

async function tooltipCreate() {
  localeData.hint = document.createElement('div');
  localeData.hint.className = 'tooltip';
  localeData.hint.id = 'tooltip-container';
  localeData.hint.innerText = 'this is a hint';
  gradioApp().appendChild(localeData.hint);
  localeData.btn = gradioApp().getElementById('locale-container');
  if (!localeData.btn) {
    localeData.btn = document.createElement('div');
    localeData.btn.className = 'locale';
    localeData.btn.id = 'locale-container';
    gradioApp().appendChild(localeData.btn);
  }
  localeData.btn.innerText = localeData.locale;
  localeData.btn.ondblclick = resetLocale;
  localeData.btn.onclick = cycleLocale;
  if (window.opts.tooltips === 'None') localeData.type = 0;
  if (window.opts.tooltips === 'Browser default') localeData.type = 1;
  if (window.opts.tooltips === 'UI tooltips') localeData.type = 2;

  if (localeData.type === 2) { // setup event delegation for tooltips instead of individual listeners
    if (isTouchDevice) {
      gradioApp().addEventListener('touchstart', tooltipShowDelegated); // eslint-disable-line no-use-before-define
      gradioApp().addEventListener('touchend', tooltipHideDelegated); // eslint-disable-line no-use-before-define
    }
    gradioApp().addEventListener('pointerover', tooltipShowDelegated); // eslint-disable-line no-use-before-define
    gradioApp().addEventListener('pointerout', tooltipHideDelegated); // eslint-disable-line no-use-before-define
  }
  if (!localeData.observer) initializeDOMObserver(); // eslint-disable-line no-use-before-define
}

async function expandTooltip(element, longHint) {
  if (localeData.currentElement === element && localeData.hint.classList.contains('tooltip-show')) {
    const ring = localeData.hint.querySelector('.tooltip-progress-ring');
    if (ring) ring.style.opacity = '0';
    localeData.hint.classList.add('tooltip-expanded');
    setTimeout(() => {
      const longContent = localeData.hint.querySelector('.long-content');
      if (longContent) longContent.classList.add('show');
    }, 100);
  }
}

async function tooltipShowDelegated(e) { // use event delegation to handle dynamically created elements
  if (e.target.dataset && e.target.dataset.hint) tooltipShow(e); // eslint-disable-line no-use-before-define
}

async function tooltipHideDelegated(e) {
  if (e.target.dataset && e.target.dataset.hint) tooltipHide(e); // eslint-disable-line no-use-before-define
}

async function tooltipShow(e) {
  if (localeData.expandTimeout) { // clear any existing expansion timeout
    clearTimeout(localeData.expandTimeout);
    localeData.expandTimeout = null;
  }

  localeData.hint.classList.remove('tooltip-expanded'); // remove expanded class and reset current element
  localeData.currentElement = e.target;

  if (e.target.dataset.hint) {
    const progressRing = ` // create progress ring SVG
      <div class="tooltip-progress-ring">
        <svg viewBox="0 0 12 12">
          <circle class="ring-background" cx="6" cy="6" r="5"></circle>
          <circle class="ring-progress" cx="6" cy="6" r="5"></circle>
        </svg>
      </div>
    `;
    // set up the complete content structure from the start
    let content = `
      <div class="tooltip-header">
        <b>${e.target.textContent}</b>
        ${e.target.dataset.longHint ? progressRing : ''}
      </div>
      <div class="separator"></div>
      ${e.target.dataset.hint}
    `;
    if (e.target.dataset.longHint) content += `<div class="long-content"><div class="separator"></div>${e.target.dataset.longHint}</div>`; // add long content if available, but keep it hidden
    if (e.target.dataset.reload) { // add reload notice if needed
      const reloadType = e.target.dataset.reload;
      let reloadText = '';
      if (reloadType === 'model') reloadText = 'Requires model reload';
      else if (reloadType === 'server') reloadText = 'Requires server restart';
      if (reloadText) {
        content += `
          <div class="tooltip-reload-notice">
            <div class="separator"></div>
            <span class="tooltip-reload-text">${reloadText}</span>
          </div>
        `;
      }
    }

    localeData.hint.innerHTML = content;
    localeData.hint.classList.add('tooltip-show');

    if (e.clientX > window.innerWidth / 2) localeData.hint.classList.add('tooltip-left');
    else localeData.hint.classList.remove('tooltip-left');

    if (e.target.dataset.longHint) { // set up expansion timer if long hint is available
      const ring = localeData.hint.querySelector('.tooltip-progress-ring'); // start progress ring animation
      const ringProgress = localeData.hint.querySelector('.ring-progress');
      if (ring && ringProgress) {
        setTimeout(() => {
          ring.classList.add('active');
          ringProgress.classList.add('animate');
        }, 100);
      }
      localeData.expandTimeout = setTimeout(() => expandTooltip(e.target, e.target.dataset.longHint), 3000);
    }
  }
}

async function tooltipHide(e) {
  if (localeData.expandTimeout) {
    clearTimeout(localeData.expandTimeout);
    localeData.expandTimeout = null;
  }
  localeData.hint.classList.remove('tooltip-show', 'tooltip-expanded');
  localeData.currentElement = null;
}

async function validateHints(json, elements, tab) {
  json.missing = [];
  const data = Object.values(json).flat().filter((e) => e.hint.length > 0);
  for (const e of data) e.label = e.label.trim();
  if (tab) {
    elements = elements.filter((el) => el.closest(`#${tab}_tabitem`)); // include only elements within specified tab
    elements = elements.filter((el) => !el.closest(`#${tab}_scripts_tabitem`));
  }
  let original = elements.map((e) => e.textContent.toLowerCase().trim()).sort(); // should be case sensitive
  let duplicateUI = original.filter((e, i, a) => a.indexOf(e.toLowerCase()) !== i).sort();
  original = [...new Set(original)]; // remove duplicates
  duplicateUI = [...new Set(duplicateUI)]; // remove duplicates
  const current = data.map((e) => e.label.toLowerCase().trim()).sort(); // should be case sensitive
  // log('all elements:', original);
  // log('all hints:', current);
  log('hints-differences', { elements: original.length, hints: current.length });
  const missingHints = original.filter((e) => !current.includes(e.toLowerCase())).sort();
  const orphanedHints = current.filter((e) => !original.includes(e.toLowerCase())).sort();
  const duplicateHints = current.filter((e, i, a) => a.indexOf(e.toLowerCase()) !== i).sort();
  log('duplicate hints:', duplicateHints);
  log('duplicate labels:', duplicateUI);
  return [missingHints, orphanedHints];
}

async function addMissingHints(json, missingHints) {
  if (missingHints.length === 0) return;
  json.missing = [];
  for (const h of missingHints.sort()) {
    if (h.length <= 1) continue;
    json.missing.push({ id: '', label: h, localized: '', hint: h, longHint: '' }); // Add longHint property
  }
  log('missing hints', missingHints);
  log('added missing hints:', { missing: json.missing });
}

async function removeOrphanedHints(json, orphanedHints) {
  const data = Object.values(json).flat().filter((e) => e.hint.length > 0);
  for (const e of data) e.label = e.label.trim();
  const orphaned = data.filter((e) => orphanedHints.includes(e.label.toLowerCase()));
  log('orphaned hints:', { orphaned });
}

async function replaceButtonText(el) {
  // https://www.nerdfonts.com/cheat-sheet
  // use unicode of icon with format nf-md-<icon>_circle
  const textIcons = {
    Generate: '\uf144',
    Enqueue: '\udb81\udc17',
    Stop: '\udb81\ude66',
    Skip: '\udb81\ude61',
    Pause: '\udb80\udfe5',
    Restore: '\udb82\udd9b',
    Clear: '\udb80\udd59',
    Networks: '\uf261',
  };
  if (textIcons[el.innerText]) {
    el.classList.add('button-icon');
    el.innerText = textIcons[el.innerText];
  }
}

async function getLocaleData(desiredLocale = null) {
  if (desiredLocale) desiredLocale = desiredLocale.split(':')[0];
  if (desiredLocale === 'Auto') {
    try {
      localeData.locale = navigator.languages && navigator.languages.length ? navigator.languages[0] : navigator.language;
      localeData.locale = localeData.locale.split('-')[0];
      localeData.prev = localeData.locale;
    } catch (e) {
      localeData.locale = 'en';
      log('getLocale', e);
    }
  } else {
    localeData.locale = desiredLocale || 'en';
    localeData.prev = localeData.locale;
  }
  log('getLocale', desiredLocale, localeData.locale);
  // primary
  let json = {};
  try {
    let res = await fetch(`${window.subpath}/file=html/locale_${localeData.locale}.json`);
    if (!res || !res.ok) {
      localeData.locale = 'en';
      res = await fetch(`${window.subpath}/file=html/locale_${localeData.locale}.json`);
    }
    json = await res.json();
  } catch { /**/ }

  try {
    const res = await fetch(`${window.subpath}/file=html/override_${localeData.locale}.json`);
    if (res && res.ok) json.override = await res.json();
  } catch { /**/ }

  return json;
}

async function replaceTextContent(el, text) {
  if (el.children.length === 1 && el.firstElementChild.classList.contains('mask-icon')) return;
  if (el.querySelector('span')) el = el.querySelector('span');
  if (el.querySelector('div')) el = el.querySelector('div');
  if (el.classList.contains('mask-icon')) return; // skip icon buttons
  if (el.dataset.selector) { // replace on rehosted child if exists
    el = el.firstElementChild || el.querySelector(el.dataset.selector);
    replaceTextContent(el, text);
    return;
  }
  el.textContent = text;
}

async function setHint(el, entry) {
  if (localeData.type === 1) {
    el.title = entry.hint;
  } else if (localeData.type === 2) {
    el.dataset.hint = entry.hint;
    if (entry.longHint && entry.longHint.length > 0) el.dataset.longHint = entry.longHint;
    if (entry.reload && entry.reload.length > 0) el.dataset.reload = entry.reload;
  } else {
    // tooltips disabled
  }
}

async function setHints(analyze = false) {
  let json = {};
  let overrideData = [];
  if (localeData.finished) return;
  if (Object.keys(opts).length === 0) return;
  const elements = [
    ...Array.from(gradioApp().querySelectorAll('button')),
    ...Array.from(gradioApp().querySelectorAll('h2')),
    ...Array.from(gradioApp().querySelectorAll('label > span')),
    ...Array.from(gradioApp().querySelectorAll('.label-wrap > span')),
  ];
  if (elements.length === 0) return;
  if (localeData.data.length === 0) {
    json = await getLocaleData(window.opts.ui_locale);
    overrideData = Object.values(json.override || {}).flat().filter((e) => e.hint.length > 0);
    const jsonData = Object.values(json).flat().filter((e) => e.hint.length > 0);
    localeData.data = [...overrideData, ...jsonData];
  }
  if (!localeData.hint) tooltipCreate();
  let localized = 0;
  let hints = 0;
  const t0 = performance.now();
  for (const el of elements) {
    // localize elements text
    let found;
    if (el.dataset.original) found = localeData.data.find((l) => l.label.toLowerCase().trim() === el.dataset.original.toLowerCase().trim());
    else found = localeData.data.find((l) => l.label.toLowerCase().trim() === el.textContent.toLowerCase().trim());
    if (found?.localized?.length > 0) {
      if (!el.dataset.original) el.dataset.original = el.textContent;
      localized++;
      replaceTextContent(el, found.localized);
    } else if (found?.label && !localeData.initial && (localeData.locale === 'en')) { // reset to english
      replaceTextContent(el, found.label);
    }
    // set hints
    if (found?.hint?.length > 0) {
      hints++;
      setHint(el, found);
    }
  }
  localeData.finished = true;
  localeData.initial = false;
  const t1 = performance.now();
  // localeData.btn.style.backgroundColor = localeData.locale !== 'en' ? 'var(--primary-500)' : '';
  log('touchDevice', isTouchDevice);
  log('setHints', { type: localeData.type, locale: localeData.locale, elements: elements.length, localized, hints, data: localeData.data.length, override: overrideData.length, time: Math.round(t1 - t0) });
  // sortUIElements();
  if (analyze) {
    log('analyzing hints', 'control_tabitem');
    const [missingHints, orphanedHints] = await validateHints(json, elements);
    await addMissingHints(json, missingHints);
    await removeOrphanedHints(json, orphanedHints);
  }
}

const analyzeHints = async () => {
  localeData.finished = false;
  localeData.data = [];
  await setHints(true);
};

// Apply hints to a single element immediately
async function applyHintToElement(el) {
  if (!localeData.data || localeData.data.length === 0) return;
  if (!el.textContent) return;

  // check if element matches our selector criteria
  const isValidElement = el.tagName === 'BUTTON'
    || el.tagName === 'H2'
    || (el.tagName === 'SPAN' && (el.parentElement?.tagName === 'LABEL' || el.parentElement?.classList.contains('label-wrap')));
  if (!isValidElement) return;

  let found; // find matching hint data
  if (el.dataset.original) found = localeData.data.find((l) => l.label.toLowerCase().trim() === el.dataset.original.toLowerCase().trim());
  else found = localeData.data.find((l) => l.label.toLowerCase().trim() === el.textContent.toLowerCase().trim());

  if (found?.localized?.length > 0) { // apply localization if found
    if (!el.dataset.original) el.dataset.original = el.textContent;
    replaceTextContent(el, found.localized);
  }

  if (found?.hint?.length > 0) setHint(el, found); // apply hint if found
}

// Initialize MutationObserver for immediate hint application
function initializeDOMObserver() {
  if (localeData.observer) {
    localeData.observer.disconnect();
  }

  localeData.observer = new MutationObserver((mutations) => {
    // Process added nodes immediately
    for (const mutation of mutations) {
      if (mutation.type === 'childList') {
        for (const node of mutation.addedNodes) {
          if (node.nodeType === Node.ELEMENT_NODE) {
            // Apply hints to the node itself
            applyHintToElement(node);

            // Apply hints to all relevant children
            const elements = [
              ...Array.from(node.querySelectorAll('button')),
              ...Array.from(node.querySelectorAll('h2')),
              ...Array.from(node.querySelectorAll('label > span')),
              ...Array.from(node.querySelectorAll('.label-wrap > span')),
            ];

            // Include the node itself if it matches
            if (node.matches && (
              node.matches('button')
              || node.matches('h2')
              || node.matches('label > span')
              || node.matches('.label-wrap > span')
            )) {
              elements.push(node);
            }

            // Apply hints immediately to all found elements
            elements.forEach((el) => applyHintToElement(el));
          }
        }
      }
    }
  });

  // Start observing the entire gradio app for changes
  const targetNode = gradioApp();
  if (targetNode) {
    localeData.observer.observe(targetNode, {
      childList: true,
      subtree: true,
    });
  }
}

// Export for external use if needed
const forceReapplyHints = () => setHints();
