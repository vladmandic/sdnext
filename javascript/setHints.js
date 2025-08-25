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
  expandTimeout: null,
  currentElement: null,
  elementHintMap: new WeakMap(), // Store hints separately from DOM
};
let localeTimeout = null;

async function cycleLocale() {
  clearTimeout(localeTimeout);
  localeTimeout = setTimeout(() => {
    log('cycleLocale', localeData.prev, localeData.locale);
    const index = allLocales.indexOf(localeData.prev);
    localeData.locale = allLocales[(index + 1) % allLocales.length];
    localeData.btn.innerText = localeData.locale;
    localeData.finished = false;
    localeData.data = [];
    localeData.prev = localeData.locale;
    localeData.elementHintMap = new WeakMap();
    window.opts.ui_locale = localeData.locale;
    setHints();
  }, 250);
}

async function resetLocale() {
  clearTimeout(localeTimeout);
  localeData.locale = 'en';
  log('resetLocale', localeData.locale);
  const index = allLocales.indexOf(localeData.locale);
  localeData.locale = allLocales[(index) % allLocales.length];
  localeData.btn.innerText = localeData.locale;
  localeData.finished = false;
  localeData.data = [];
  localeData.elementHintMap = new WeakMap();
  window.opts.ui_locale = localeData.locale;
  setHints();
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
    const ring = localeData.hint.querySelector('.tooltip-progress-ring');
    if (ring) {
      ring.style.opacity = '0';
    }

    localeData.hint.classList.add('tooltip-expanded');

    setTimeout(() => {
      const longContent = localeData.hint.querySelector('.long-content');
      if (longContent) {
        longContent.classList.add('show');
      }
    }, 100);
  }
}

async function tooltipShow(e) {
  // Get hint data from WeakMap or dataset
  const hintData = localeData.elementHintMap.get(e.target) || {
    hint: e.target.dataset?.hint,
    longHint: e.target.dataset?.longHint,
    reload: e.target.dataset?.reload,
  };

  if (!hintData.hint) return;

  // Clear any existing expansion timeout
  if (localeData.expandTimeout) {
    clearTimeout(localeData.expandTimeout);
    localeData.expandTimeout = null;
  }

  // Remove expanded class and reset current element
  localeData.hint.classList.remove('tooltip-expanded');
  localeData.currentElement = e.target;

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
      <b>${e.target.textContent}</b>
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
    const ring = localeData.hint.querySelector('.tooltip-progress-ring');
    const ringProgress = localeData.hint.querySelector('.ring-progress');

    if (ring && ringProgress) {
      setTimeout(() => {
        ring.classList.add('active');
        ringProgress.classList.add('animate');
      }, 100);
    }

    localeData.expandTimeout = setTimeout(() => {
      expandTooltip(e.target, hintData);
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

async function validateHints(json, elements) {
  json.missing = [];
  const data = Object.values(json).flat().filter((e) => e.hint.length > 0);
  for (const e of data) e.label = e.label.trim();
  let original = elements.map((e) => e.textContent.toLowerCase().trim()).sort();
  let duplicateUI = original.filter((e, i, a) => a.indexOf(e.toLowerCase()) !== i).sort();
  original = [...new Set(original)];
  duplicateUI = [...new Set(duplicateUI)];
  const current = data.map((e) => e.label.toLowerCase().trim()).sort();
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
    json.missing.push({ id: '', label: h, localized: '', hint: h, longHint: '' });
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

async function setHints(analyze = false) {
  let json = {};
  let overrideData = [];

  if (Object.keys(opts).length === 0) return;

  const elements = [
    ...Array.from(gradioApp().querySelectorAll('button')),
    ...Array.from(gradioApp().querySelectorAll('h2')),
    ...Array.from(gradioApp().querySelectorAll('label > span')),
    ...Array.from(gradioApp().querySelectorAll('.label-wrap > span')),
    // Include tab buttons specifically
    ...Array.from(gradioApp().querySelectorAll('.tab-nav > button')),
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

  for (const possible of elements) {
    let el = possible;
    if (possible.querySelector('span')) el = possible.querySelector('span');
    if (el.children.length === 1 && el.firstElementChild.classList.contains('mask-icon')) continue;

    // Get text to match against
    const elementText = el.dataset?.original || el.textContent;

    let found;
    if (elementText) {
      found = localeData.data.find((l) => l.label.toLowerCase().trim() === elementText.toLowerCase().trim());
    }

    if (found?.localized?.length > 0) {
      if (!el.dataset.original) el.dataset.original = el.textContent;
      localized++;
      el.textContent = found.localized;
    } else if (found?.label && !localeData.initial && (localeData.locale === 'en')) {
      el.textContent = found.label;
    }

    if (found?.hint?.length > 0) {
      hints++;

      if (localeData.type === 1) {
        el.title = found.hint;
      } else if (localeData.type === 2) {
        // Store hint data in both dataset and WeakMap
        const hintData = {
          hint: found.hint,
          longHint: found.longHint || null,
          reload: found.reload || null,
        };

        // Store in WeakMap for persistence
        localeData.elementHintMap.set(el, hintData);

        // Also set dataset attributes for compatibility
        el.dataset.hint = found.hint;
        if (found.longHint) el.dataset.longHint = found.longHint;
        if (found.reload) el.dataset.reload = found.reload;

        // Remove old listeners if any
        el.removeEventListener('mouseover', tooltipShow);
        el.removeEventListener('mouseout', tooltipHide);
        el.removeEventListener('click', tooltipHide);

        // Add new listeners
        el.addEventListener('mouseover', tooltipShow);
        el.addEventListener('mouseout', tooltipHide);
        el.addEventListener('click', tooltipHide);
      }
    }
  }

  localeData.finished = true;
  localeData.initial = false;
  const t1 = performance.now();

  log('setHints', {
    type: localeData.type,
    locale: localeData.locale,
    elements: elements.length,
    localized,
    hints,
    data: localeData.data.length,
    override: overrideData.length,
    time: Math.round(t1 - t0),
  });

  if (analyze) {
    const [missingHints, orphanedHints] = await validateHints(json, elements);
    await addMissingHints(json, missingHints);
    await removeOrphanedHints(json, orphanedHints);
  }
}

// Force refresh hints on tab changes
onUiTabChange(() => {
  // Small delay to let DOM settle
  setTimeout(() => {
    localeData.finished = false; // Allow setHints to run again
    setHints();
  }, 100);
});

const analyzeHints = async () => {
  localeData.finished = false;
  localeData.data = [];
  localeData.elementHintMap = new WeakMap();
  await setHints(true);
};
