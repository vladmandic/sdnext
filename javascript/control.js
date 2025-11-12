function controlInputMode(inputMode, ...args) {
  const updateEl = gradioApp().getElementById('control_update');
  if (updateEl) updateEl.click();
  const tab = gradioApp().querySelector('#control-tab-input button.selected');
  if (!tab) return ['Image', ...args];
  let inputTab = tab.innerText;
  log('controlInputMode', { mode: inputMode, tab: inputTab, kanvas: typeof Kanvas });
  if ((inputTab === 'Image') && (typeof 'Kanvas' !== 'undefined')) {
    inputTab = 'Kanvas';
    const imageData = window.kanvas.getImage();
    args[0] = imageData;
  }
  return [inputTab, ...args];
}

async function setupControlUI() {
  const tabs = ['input', 'output', 'preview'];
  for (const tab of tabs) {
    const btn = gradioApp().getElementById(`control-${tab}-button`);
    if (!btn) continue; // eslint-disable-line no-continue
    btn.style.cursor = 'pointer';
    btn.onclick = () => {
      const t = gradioApp().getElementById(`control-tab-${tab}`);
      t.style.display = t.style.display === 'none' ? 'block' : 'none';
      const c = gradioApp().getElementById(`control-${tab}-column`);
      c.style.flexGrow = c.style.flexGrow === '0' ? '9' : '0';
    };
  }

  const el = gradioApp().getElementById('control-input-column');
  if (!el) return;
  const intersectionObserver = new IntersectionObserver((entries) => {
    if (entries[0].intersectionRatio > 0) {
      const allTabs = Array.from(gradioApp().querySelectorAll('#control-tabs > .tab-nav > .selected'));
      for (const tab of allTabs) {
        const name = tab.innerText.toLowerCase();
        for (let i = 0; i < 10; i += 1) {
          const btn = gradioApp().getElementById(`refresh_${name}_models_${i}`);
          if (btn) btn.click();
        }
      }
    }
  });
  intersectionObserver.observe(el); // monitor visibility of tab

  log('initControlUI');
}
