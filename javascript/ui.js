window.opts = {};
window.localization = {};
window.titles = {};
let tabSelected = '';
let txt2img_textarea;
let img2img_textarea;
const wait_time = 800;
const token_timeouts = {};
let uiLoaded = false;
let promptsInitialized = false;
window.args_to_array = Array.from; // Compatibility with e.g. extensions that may expect this to be around

function rememberGallerySelection(name) {
  // dummy
}

function set_theme(theme) {
  const gradioURL = window.location.href;
  if (!gradioURL.includes('?__theme=')) window.location.replace(`${gradioURL}?__theme=${theme}`);
}

function update_token_counter(button_id) {
  if (token_timeouts[button_id]) clearTimeout(token_timeouts[button_id]);
  token_timeouts[button_id] = setTimeout(() => gradioApp().getElementById(button_id)?.click(), wait_time);
}

function clip_gallery_urls(gallery) {
  const files = gallery.map((v) => v.data);
  navigator.clipboard.writeText(JSON.stringify(files)).then(
    () => log('clipboard:', files),
    (err) => error(`clipboard: ${files} ${err}`),
  );
}

function isVisible(el) {
  const rect = el.getBoundingClientRect();
  if (rect.width === 0 && rect.height === 0) return false;
  return rect.top >= 0 && rect.left >= 0 && rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) && rect.right <= (window.innerWidth || document.documentElement.clientWidth);
}

function all_gallery_buttons() {
  let allGalleryButtons = gradioApp().querySelectorAll('[style="display: block;"].tabitem div[id$=_gallery].gradio-gallery .thumbnails > .thumbnail-item.thumbnail-small');
  if (allGalleryButtons.length === 0) allGalleryButtons = gradioApp().querySelectorAll('.gradio-gallery .thumbnails > .thumbnail-item.thumbnail-small');
  const visibleGalleryButtons = [];
  allGalleryButtons.forEach((elem) => {
    if (elem.parentElement.offsetParent) visibleGalleryButtons.push(elem);
  });
  return visibleGalleryButtons;
}

function selected_gallery_button() {
  let allCurrentButtons = gradioApp().querySelectorAll('[style="display: block;"].tabitem div[id$=_gallery].gradio-gallery .thumbnail-item.thumbnail-small.selected');
  if (allCurrentButtons.length === 0) allCurrentButtons = gradioApp().querySelectorAll('.gradio-gallery .thumbnails > .thumbnail-item.thumbnail-small.selected');
  let visibleCurrentButton = null;
  allCurrentButtons.forEach((elem) => {
    if (elem.parentElement.offsetParent) visibleCurrentButton = elem;
  });
  return visibleCurrentButton;
}

function selected_gallery_index() {
  const buttons = all_gallery_buttons();
  const button = selected_gallery_button();
  let result = -1;
  buttons.forEach((v, i) => { if (v === button) { result = i; } });
  return result;
}

function selected_gallery_files() {
  let allImages = [];
  try {
    let allCurrentButtons = gradioApp().querySelectorAll('[style="display: block;"].tabitem div[id$=_gallery].gradio-gallery .thumbnail-item.thumbnail-small');
    if (allCurrentButtons.length === 0) allCurrentButtons = gradioApp().querySelectorAll('.gradio-gallery .thumbnails > .thumbnail-item.thumbnail-small');
    allImages = Array.from(allCurrentButtons).map((v) => v.querySelector('img')?.src);
    allImages = allImages.filter((el) => isVisible(el));
  } catch { /**/ }
  const selectedIndex = selected_gallery_index();
  return [allImages, selectedIndex];
}

function extract_image_from_gallery(gallery) {
  if (gallery.length === 0) return [null];
  if (gallery.length === 1) return [gallery[0]];
  let index = selected_gallery_index();
  if (index < 0 || index >= gallery.length) index = 0;
  return [gallery[index]];
}

async function setTheme(val, old) {
  if (!old || val === old) return;
  old = old.replace('modern/', '');
  val = val.replace('modern/', '');
  const links = Array.from(document.getElementsByTagName('link')).filter((l) => l.href.includes(old));
  if (links.length === 0) {
    log('setTheme: current theme not matched', old);
    return;
  }
  for (const link of links) {
    const href = link.href.replace(old, val);
    const res = await fetch(href);
    if (res.ok) {
      log('setTheme:', old, val);
      link.href = link.href.replace(old, val);
    } else {
      log('setTheme: CSS not found', val);
    }
  }
}

function setFontSize(val, old) {
  const size = val || opts.font_size;
  if (size === old) return;
  document.documentElement.style.setProperty('--font-size', `${size}px`);
  gradioApp().style.setProperty('--font-size', `${size}px`);
  gradioApp().style.setProperty('--text-xxs', `${size - 3}px`);
  gradioApp().style.setProperty('--text-xs', `${size - 2}px`);
  gradioApp().style.setProperty('--text-sm', `${size - 1}px`);
  gradioApp().style.setProperty('--text-md', `${size}px`);
  gradioApp().style.setProperty('--text-lg', `${size + 1}px`);
  gradioApp().style.setProperty('--text-xl', `${size + 2}px`);
  gradioApp().style.setProperty('--text-xxl', `${size + 3}px`);
  log('setFontSize', size);
}

function switchToTab(tab) {
  const tabs = Array.from(gradioApp().querySelectorAll('#tabs > .tab-nav > button'));
  const btn = tabs?.find((t) => t.innerText === tab);
  log('switchToTab', tab);
  if (btn) btn.click();
}

function switch_to_txt2img(...args) {
  switchToTab('Text');
  return Array.from(arguments);
}

function switch_to_img2img_tab(no) {
  switchToTab('Image');
  gradioApp().getElementById('mode_img2img').querySelectorAll('button')[no].click();
}

function switch_to_img2img(...args) {
  switchToTab('Image');
  switch_to_img2img_tab(0);
  return Array.from(arguments);
}

function switch_to_inpaint(...args) {
  switchToTab('Image');
  switch_to_img2img_tab(1);
  return Array.from(arguments);
}

function switch_to_sketch(...args) {
  switchToTab('Image');
  switch_to_img2img_tab(2);
  return Array.from(arguments);
}

function switch_to_composite(...args) {
  switchToTab('Image');
  switch_to_img2img_tab(3);
  return Array.from(arguments);
}

function switch_to_extras(...args) {
  switchToTab('Process');
  return Array.from(arguments);
}

function switch_to_control(...args) {
  switchToTab('Control');
  return Array.from(arguments);
}

function switch_to_video(...args) {
  switchToTab('Video');
  return Array.from(arguments);
}

function switch_to_caption(...args) {
  switchToTab('Caption');
  return Array.from(arguments);
}

function get_tab_index(tabId) {
  let res = 0;
  gradioApp().getElementById(tabId)?.querySelector('div').querySelectorAll('button')
    .forEach((button, i) => {
      if (button.className.indexOf('selected') !== -1) res = i;
    });
  return res;
}

function create_tab_index_args(tabId, args) {
  const res = Array.from(args);
  res[0] = get_tab_index(tabId);
  return res;
}

function get_img2img_tab_index(...args) {
  const res = Array.from(arguments);
  res.splice(-2);
  res[0] = get_tab_index('mode_img2img');
  return res;
}

function create_submit_args(args) {
  const res = Array.from(args);
  if (Array.isArray(res[res.length - 3])) res[res.length - 3] = null;
  return res;
}

function showSubmitButtons(tabname, show) {}

function clearGallery(tabname) {
  const gallery = gradioApp().getElementById(`${tabname}_gallery`);
  gallery.classList.remove('logo');
  // gallery.style.height = window.innerHeight - gallery.getBoundingClientRect().top - 200 + 'px'
  const footer = gradioApp().getElementById(`${tabname}_footer`);
  footer.style.display = 'flex';
}

function submit_txt2img(...args) {
  log('submitTxt');
  clearGallery('txt2img');
  const id = randomId();
  requestProgress(id, null, gradioApp().getElementById('txt2img_gallery'));
  const res = create_submit_args(args);
  res[0] = id;
  res[1] = window.submit_state;
  window.submit_state = '';
  return res;
}

function submit_img2img(...args) {
  log('submitImg');
  clearGallery('img2img');
  const id = randomId();
  requestProgress(id, null, gradioApp().getElementById('img2img_gallery'));
  const res = create_submit_args(args);
  res[0] = id;
  res[1] = window.submit_state;
  res[2] = get_tab_index('mode_img2img');
  window.submit_state = '';
  return res;
}

function submit_control(...args) {
  log('submitControl');
  clearGallery('control');
  const id = randomId();
  requestProgress(id, null, gradioApp().getElementById('control_gallery'));
  const res = create_submit_args(args);
  res[0] = id;
  res[1] = window.submit_state;
  res[2] = gradioApp().querySelector('#control-tabs > .tab-nav > .selected')?.innerText.toLowerCase() || ''; // selected tab name
  window.submit_state = '';
  return res;
}

function submit_video(...args) {
  log('submitVideo');
  clearGallery('video');
  const id = randomId();
  requestProgress(id, null, gradioApp().getElementById('video_gallery'));
  const res = create_submit_args(args);
  res[0] = id;
  res[1] = window.submit_state;
  window.submit_state = '';
  return res;
}

function submit_framepack(...args) {
  const id = randomId();
  log('submitFramepack', id);
  requestProgress(id, null, null);
  window.submit_state = '';
  args[0] = id;
  return args;
}

function submit_ltx(...args) {
  const id = randomId();
  log('submitFramepack', id);
  requestProgress(id, null, null);
  window.submit_state = '';
  args[0] = id;
  return args;
}

function submit_video_wrapper(...args) {
  const modernEl = gradioApp().querySelector('.video_output.fade-in');
  let id = modernEl ? modernEl.id : args[0];
  id = id.replace('video-selector-', '');
  log('submitVideoWrapper', id);
  const btn = gradioApp().getElementById(`${id}_generate_btn`);
  if (btn) btn.click();
  else console.log('submit_video_wrapper: no button found', id);
}

function submit_postprocessing(...args) {
  log('SubmitExtras');
  clearGallery('extras');
  return args;
}

window.submit = submit_txt2img;
window.submit_state = '';

function modelmerger(...args) {
  const id = randomId();
  const res = create_submit_args(args);
  res[0] = id;
  return res;
}

function clearPrompts(prompt, negative_prompt) {
  prompt = '';
  negative_prompt = '';
  return [prompt, negative_prompt];
}

const promptTokenCountUpdateFuncs = {};

function recalculatePromptTokens(name) {
  if (promptTokenCountUpdateFuncs[name]) {
    promptTokenCountUpdateFuncs[name]();
  }
}

function recalculate_prompts_txt2img(...args) {
  recalculatePromptTokens('txt2img_prompt');
  recalculatePromptTokens('txt2img_neg_prompt');
  return Array.from(arguments);
}

function recalculate_prompts_img2img(...args) {
  recalculatePromptTokens('img2img_prompt');
  recalculatePromptTokens('img2img_neg_prompt');
  return Array.from(arguments);
}

function recalculate_prompts_inpaint(...args) {
  recalculatePromptTokens('img2img_prompt');
  recalculatePromptTokens('img2img_neg_prompt');
  return Array.from(arguments);
}

function recalculate_prompts_control(...args) {
  recalculatePromptTokens('control_prompt');
  recalculatePromptTokens('control_neg_prompt');
  return Array.from(arguments);
}

function registerDragDrop() {
  const qs = gradioApp().getElementById('quicksettings');
  if (!qs) return;
  qs.addEventListener('dragover', (evt) => {
    evt.preventDefault();
    evt.dataTransfer.dropEffect = 'copy';
  });
  qs.addEventListener('drop', (evt) => {
    evt.preventDefault();
    evt.dataTransfer.dropEffect = 'copy';
    for (const f of evt.dataTransfer.files) {
      log('QuickSettingsDrop', f);
    }
  });
}

function sortUIElements() {
  // sort top-level tabs
  const currSelected = gradioApp()?.querySelector('.tab-nav > .selected')?.innerText;
  if (currSelected === tabSelected || !opts.ui_tab_reorder) return;
  tabSelected = currSelected;
  const tabs = gradioApp().getElementById('tabs')?.children[0];
  if (!tabs) return;
  let tabsOrder = opts.ui_tab_reorder?.split(',').map((el) => el.trim().toLowerCase()) || [];
  for (const el of Array.from(tabs.children)) {
    const elIndex = tabsOrder.indexOf(el.innerText.toLowerCase());
    if (elIndex > -1) el.style.order = elIndex - 50; // default is 0 so setting to negative values
  }
  // sort always-on scripts
  const find = (el, ordered) => {
    for (const i in ordered) {
      if (el.innerText.toLowerCase().startsWith(ordered[i])) return i;
    }
    return 99;
  };

  tabsOrder = opts.ui_scripts_reorder?.split(',').map((el) => el.trim().toLowerCase()) || [];

  const scriptsTxt = gradioApp().getElementById('scripts_alwayson_txt2img').children;
  for (const el of Array.from(scriptsTxt)) el.style.order = find(el, tabsOrder);

  const scriptsImg = gradioApp().getElementById('scripts_alwayson_img2img').children;
  for (const el of Array.from(scriptsImg)) el.style.order = find(el, tabsOrder);
  log('sortUIElements');
}

onAfterUiUpdate(async () => {
  async function registerTextarea(id, id_counter, id_button) {
    const prompt = gradioApp().getElementById(id);
    if (!prompt) return;
    const counter = gradioApp().getElementById(id_counter);
    const localTextarea = gradioApp().querySelector(`#${id} > label > textarea`);
    if (counter.parentElement === prompt.parentElement) return;
    prompt.parentElement.insertBefore(counter, prompt);
    prompt.parentElement.style.position = 'relative';
    promptTokenCountUpdateFuncs[id] = () => { update_token_counter(id_button); };
    localTextarea.addEventListener('input', promptTokenCountUpdateFuncs[id]);
  }

  // sortUIElements();
  if (promptsInitialized) return;
  log('initPrompts');
  registerTextarea('txt2img_prompt', 'txt2img_token_counter', 'txt2img_token_button');
  registerTextarea('txt2img_neg_prompt', 'txt2img_negative_token_counter', 'txt2img_negative_token_button');
  registerTextarea('img2img_prompt', 'img2img_token_counter', 'img2img_token_button');
  registerTextarea('img2img_neg_prompt', 'img2img_negative_token_counter', 'img2img_negative_token_button');
  registerTextarea('control_prompt', 'control_token_counter', 'control_token_button');
  registerTextarea('control_neg_prompt', 'control_negative_token_counter', 'control_negative_token_button');
  promptsInitialized = true;
});

function update_txt2img_tokens(...args) {
  update_token_counter('txt2img_token_button');
  if (args.length === 2) return args[0];
  return args;
}

function update_img2img_tokens(...args) {
  update_token_counter('img2img_token_button');
  if (args.length === 2) return args[0];
  return args;
}

function getTranslation(...args) {
  return null;
}

function monitorServerStatus() {
  document.open();
  document.write(`
    <html>
      <head><title>SD.Next</title></head>
      <body style="background: #222222; font-size: 1rem; font-family:monospace; margin-top:20%; color:lightgray; text-align:center">
        <h1>Waiting for server...</h1>
        <script>
          function monitorServerStatus() {
            fetch('${window.api}/progress?skip_current_image=true')
              .then((res) => { !res?.ok ? setTimeout(monitorServerStatus, 1000) : location.reload(); })
              .catch((e) => setTimeout(monitorServerStatus, 1000))
          }
          window.onload = () => monitorServerStatus();
        </script>
      </body>
    </html>
  `);
  document.close();
}

function restartReload() {
  document.body.style = 'background: #222222; font-size: 1rem; font-family:monospace; margin-top:20%; color:lightgray; text-align:center';
  document.body.innerHTML = '<h1>Server shutdown in progress...</h1>';
  fetch(`${window.api}/progress?skip_current_image=true`)
    .then((res) => setTimeout(restartReload, 1000))
    .catch((e) => setTimeout(monitorServerStatus, 500));
  return [];
}

function updateInput(target) {
  const e = new Event('input', { bubbles: true });
  Object.defineProperty(e, 'target', { value: target });
  target.dispatchEvent(e);
}

let desiredCheckpointName = null;
function selectCheckpoint(name) {
  desiredCheckpointName = name;
  const tabName = getENActiveTab();
  const btnModel = gradioApp().getElementById(`${tabName}_extra_model`);
  const isRefiner = btnModel && btnModel.classList.contains('toolbutton-selected');
  if (isRefiner) gradioApp().getElementById('change_refiner').click();
  else gradioApp().getElementById('change_checkpoint').click();
  log(`selectCheckpoint ${isRefiner ? 'refiner' : 'model'}: ${desiredCheckpointName}`);
  markSelectedCards([desiredCheckpointName], 'model');
}

let desiredVAEName = null;
function selectVAE(name) {
  desiredVAEName = name;
  gradioApp().getElementById('change_vae').click();
  log(`selectVAE: ${desiredVAEName}`);
  markSelectedCards([desiredVAEName], 'vae');
}

function selectReference(name) {
  log(`selectReference: ${name}`);
  desiredCheckpointName = name;
  gradioApp().getElementById('change_reference').click();
  markSelectedCards([desiredCheckpointName], 'model');
}

function currentImageResolutionimg2img(_a, _b, scaleBy) {
  const img = gradioApp().querySelector('#mode_img2img > div[style="display: block;"] img');
  return img ? [img.naturalWidth, img.naturalHeight, scaleBy] : [0, 0, scaleBy];
}

function currentImageResolutioncontrol(_a, _b, scaleBy) {
  const img = gradioApp().querySelector('#control-tab-input > div[style="display: block;"] img');
  return img ? [img.naturalWidth, img.naturalHeight, scaleBy] : [0, 0, scaleBy];
}

function updateImg2imgResizeToTextAfterChangingImage() {
  const el = gradioApp().getElementById('img2img_update_resize_to');
  if (el) setTimeout(() => gradioApp().getElementById('img2img_update_resize_to').click(), 500);
  return [];
}

function createThemeElement() {
  const el = document.createElement('img');
  el.id = 'theme-preview';
  el.className = 'theme-preview';
  el.onclick = () => { el.style.display = 'none'; };
  document.body.appendChild(el);
  return el;
}

function toggleCompact(val, old) {
  if (val === old) return;
  log('toggleCompact', val, old);
  if (val) {
    gradioApp().style.setProperty('--layout-gap', 'var(--spacing-md)');
    gradioApp().querySelectorAll('input[type=range]').forEach((el) => el.classList.add('hidden'));
    gradioApp().querySelectorAll('div .form').forEach((el) => el.classList.add('form-compact'));
    gradioApp().querySelectorAll('.small-accordion .label-wrap').forEach((el) => el.classList.add('accordion-compact'));
  } else {
    gradioApp().style.setProperty('--layout-gap', 'var(--spacing-xxl)');
    gradioApp().querySelectorAll('input[type=range]').forEach((el) => el.classList.remove('hidden'));
    gradioApp().querySelectorAll('div .form').forEach((el) => el.classList.remove('form-compact'));
    gradioApp().querySelectorAll('.small-accordion .label-wrap').forEach((el) => el.classList.remove('accordion-compact'));
  }
}

function previewTheme() {
  let name = gradioApp().getElementById('setting_gradio_theme').querySelectorAll('input')?.[0].value || '';
  fetch(`${window.subpath}/file=html/themes.json`)
    .then((res) => {
      res.json()
        .then((themes) => {
          const theme = Array.isArray(themes) ? themes.find((t) => t.id === name) : null;
          if (theme) {
            window.open(theme.subdomain, '_blank');
          } else {
            const el = document.getElementById('theme-preview') || createThemeElement();
            el.style.display = el.style.display === 'block' ? 'none' : 'block';
            name = name.replace('/', '-');
            el.src = `/file=html/${name}.jpg`;
          }
        })
        .catch((e) => error(`previewTheme: ${e}`));
    })
    .catch((e) => error(`previewTheme: ${e}`));
}

async function browseFolder() {
  const f = await window.showDirectoryPicker();
  if (f && f.kind === 'directory') return f.name;
  return null;
}

async function reconnectUI() {
  const gallery = gradioApp().getElementById('txt2img_gallery');
  const task_id = localStorage.getItem('task');
  const api_logo = Array.from(gradioApp().querySelectorAll('img')).filter((el) => el?.src?.endsWith('api-logo.svg'));
  if (api_logo.length > 0) api_logo[0].remove();
  if (task_id) {
    debug('task check:', task_id);
    requestProgress(task_id, null, gallery, null, null, true);
  }
  uiLoaded = true;

  const sd_model = gradioApp().getElementById('setting_sd_model_checkpoint');
  let loadingStarted = 0;
  let loadingMonitor = 0;

  const sd_model_callback = () => {
    const loading = sd_model.querySelector('.eta-bar');
    if (!loading) {
      loadingStarted = 0;
      clearInterval(loadingMonitor);
    } else if (loadingStarted === 0) {
      loadingStarted = Date.now();
      loadingMonitor = setInterval(() => {
        const elapsed = Date.now() - loadingStarted;
        if (elapsed > 3000 && loading) loading.style.display = 'none';
      }, 5000);
    }
  };
  const sd_model_observer = new MutationObserver(sd_model_callback);
  sd_model_observer.observe(sd_model, { attributes: true, childList: true, subtree: true });
  log('reconnectUI');
}
