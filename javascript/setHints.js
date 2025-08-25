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
  expandTimeout: null, // New property for expansion timeout
  currentElement: null, // Track current element for expansion
  elementHintMap: new WeakMap(), // Store hints separately from DOM
  delegationSetup: false, // Track if global delegation is setup
};
let localeTimeout = null;

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
    localeData.elementHintMap = new WeakMap();
    // Don't reset delegationSetup as it should persist
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
  localeData.elementHintMap = new WeakMap();
  // Don't reset delegationSetup as it should persist
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
}

async function expandTooltip(element, hintData) {
  if (localeData.currentElement === element && localeData.hint.classList.contains('tooltip-show')) {
    // Hide the progress ring
    const ring = localeData.hint.querySelector('.tooltip-progress-ring');
    if (ring) {
      ring.style.opacity = '0';
    }

    // Expand the container
    localeData.hint.classList.add('tooltip-expanded');

    // After container starts expanding, reveal the long content
    setTimeout(() => {
      const longContent = localeData.hint.querySelector('.long-content');
      if (longContent) {
        longContent.classList.add('show');
      }
    }, 100);
  }
}

async function tooltipShow(e) {
  // For event delegation, ensure we have the right target
  const target = e.target || e;

  // Get hint data from WeakMap first, then fall back to dataset
  const hintData = localeData.elementHintMap.get(target) || {
    hint: target.dataset?.hint,
    longHint: target.dataset?.longHint,
    reload: target.dataset?.reload,
  };

  if (!hintData.hint) return;

  // Clear any existing expansion timeout
  if (localeData.expandTimeout) {
    clearTimeout(localeData.expandTimeout);
    localeData.expandTimeout = null;
  }

  // Remove expanded class and reset current element
  localeData.hint.classList.remove('tooltip-expanded');
  localeData.currentElement = target;

  // Create progress ring SVG
  const progressRing = `
    <div class="tooltip-progress-ring">
      <svg viewBox="0 0 12 12">
        <circle class="ring-background" cx="6" cy="6" r="5"></circle>
        <circle class="ring-progress" cx="6" cy="6" r="5"></circle>
      </svg>
    </div>
  `;

  // Set up the complete content structure from the start
  let content = `
    <div class="tooltip-header">
      <b>${target.textContent}</b>
      ${hintData.longHint ? progressRing : ''}
    </div>
    <div class="separator"></div>
    ${hintData.hint}
  `;

  // Add long content if available, but keep it hidden
  if (hintData.longHint) {
    content += `<div class="long-content"><div class="separator"></div>${hintData.longHint}</div>`;
  }

  // Add reload notice if needed
  if (hintData.reload) {
    const reloadType = hintData.reload;
    let reloadText = '';
    if (reloadType === 'model') {
      reloadText = 'Requires model reload';
    } else if (reloadType === 'server') {
      reloadText = 'Requires server restart';
    }
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

  if (e.clientX > window.innerWidth / 2) {
    localeData.hint.classList.add('tooltip-left');
  } else {
    localeData.hint.classList.remove('tooltip-left');
  }

  // Set up expansion timer if long hint is available
  if (hintData.longHint) {
    // Start progress ring animation
    const ring = localeData.hint.querySelector('.tooltip-progress-ring');
    const ringProgress = localeData.hint.querySelector('.ring-progress');

    if (ring && ringProgress) {
      // Show the ring and start animation
      setTimeout(() => {
        ring.classList.add('active');
        ringProgress.classList.add('animate');
      }, 100);
    }

    localeData.expandTimeout = setTimeout(() => {
      expandTooltip(target, hintData);
    }, 3000);
  }
}

async function tooltipHide(e) {
  // Clear expansion timeout when hiding
  if (localeData.expandTimeout) {
    clearTimeout(localeData.expandTimeout);
    localeData.expandTimeout = null;
  }

  localeData.hint.classList.remove('tooltip-show', 'tooltip-expanded');
  localeData.currentElement = null;
}

// Setup global event delegation that persists through DOM changes
function setupGlobalDelegation() {
  if (localeData.delegationSetup) return;

  // Use event delegation on document level for maximum persistence
  document.addEventListener('mouseover', (e) => {
    const target = e.target;
    if (target && (target.dataset?.hasHint === 'true' || localeData.elementHintMap.has(target))) {
      tooltipShow({ target, clientX: e.clientX, clientY: e.clientY });
    }
  }, true);

  document.addEventListener('mouseout', (e) => {
    const target = e.target;
    if (target && (target.dataset?.hasHint === 'true' || localeData.elementHintMap.has(target))) {
      if (!target.contains(e.relatedTarget)) {
        tooltipHide({ target });
      }
    }
  }, true);

  document.addEventListener('click', (e) => {
    if (localeData.hint && localeData.hint.classList.contains('tooltip-show')) {
      tooltipHide({ target: localeData.currentElement });
    }
  }, true);

  localeData.delegationSetup = true;
  log('Global event delegation setup for tooltips');
}

// Setup global event delegation for tooltips (only once)
if (localeData.type === 2 && !localeData.delegationSetup) {
  setupGlobalDelegation();
}

async function validateHints(json, elements) {
  json.missing = [];
  const data = Object.values(json).flat().filter((e) => e.hint.length > 0);
  for (const e of data) e.label = e.label.trim();
  let original = elements.map((e) => e.textContent.toLowerCase().trim()).sort(); // should be case sensitive
  let duplicateUI = original.filter((e, i, a) => a.indexOf(e.toLowerCase()) !== i).sort();
  original = [...new Set(original)]; // remove duplicates
  duplicateUI = [...new Set(duplicateUI)]; // remove duplicates
  const current = data.map((e) => e.label.toLowerCase().trim()).sort(); // should be case sensitive
  log('all elements:', original);
  log('all hints:', current);
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
  // Store hint data in WeakMap for persistence
  const hintData = {
    hint: entry.hint,
    longHint: entry.longHint || null,
    reload: entry.reload || null,
  };
  localeData.elementHintMap.set(el, hintData);

  if (localeData.type === 1) {
    el.title = entry.hint;
  } else if (localeData.type === 2) {
    // Also set dataset attributes for compatibility
    el.dataset.hint = entry.hint;
    if (entry.longHint && entry.longHint.length > 0) el.dataset.longHint = entry.longHint;
    if (entry.reload && entry.reload.length > 0) el.dataset.reload = entry.reload;

    // Mark element as having hints for event delegation
    el.dataset.hasHint = 'true';

    // Don't add individual listeners here - we'll use global delegation
  } else {
    // tooltips disabled
  }
}

async function setHints(analyze = false) {
  let json = {};
  let overrideData = [];
  // Remove early return to allow re-initialization after tab changes
  // if (localeData.finished) return;
  if (Object.keys(opts).length === 0) return;
  const elements = [
    ...Array.from(gradioApp().querySelectorAll('button')),
    ...Array.from(gradioApp().querySelectorAll('h2')),
    ...Array.from(gradioApp().querySelectorAll('label > span')),
    ...Array.from(gradioApp().querySelectorAll('.label-wrap > span')),
    // Include all tab buttons specifically
    ...Array.from(gradioApp().querySelectorAll('.tab-nav > button')),
    ...Array.from(gradioApp().querySelectorAll('#settings .tab-nav > button')),
    ...Array.from(gradioApp().querySelectorAll('#system .tab-nav > button')),
  ];
  if (elements.length === 0) return;

  // Load data only if not already loaded
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

    // set hints - always re-apply to handle DOM changes
    if (found?.hint?.length > 0) {
      hints++;
      setHint(el, found);
    }
  }

  localeData.finished = true;
  localeData.initial = false;
  const t1 = performance.now();
  // localeData.btn.style.backgroundColor = localeData.locale !== 'en' ? 'var(--primary-500)' : '';
  log('setHints', { type: localeData.type, locale: localeData.locale, elements: elements.length, localized, hints, data: localeData.data.length, override: overrideData.length, time: Math.round(t1 - t0) });
  // sortUIElements();

  if (analyze) {
    const [missingHints, orphanedHints] = await validateHints(json, elements);
    await addMissingHints(json, missingHints);
    await removeOrphanedHints(json, orphanedHints);
  }
}

// Force refresh hints on tab changes (main tabs)
onUiTabChange(() => {
  // Small delay to let DOM settle
  setTimeout(() => {
    localeData.finished = false; // Allow setHints to run again
    setHints();
  }, 25);
});

// Also handle any click on tab buttons directly
onUiUpdate(() => {
  // Find all tab buttons
  const tabButtons = gradioApp().querySelectorAll('.tab-nav > button, #settings .tab-nav > button, #system .tab-nav > button');

  tabButtons.forEach((btn) => {
    // Check if this button already has our click handler
    if (!btn.dataset.hintClickHandlerAdded) {
      btn.dataset.hintClickHandlerAdded = 'true';

      // Add click handler to force hint refresh
      btn.addEventListener('click', () => {
        setTimeout(() => {
          localeData.finished = false;
          setHints();
        }, 25); // Slightly longer delay for tab content to load
      });
    }
  });
});

const analyzeHints = async () => {
  localeData.finished = false;
  localeData.data = [];
  // Don't recreate WeakMap unless necessary to preserve existing mappings
  await setHints(true);
};
