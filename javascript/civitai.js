String.prototype.format = function (args) { // eslint-disable-line no-extend-native, func-names
  let thisString = '';
  for (let charPos = 0; charPos < this.length; charPos++) thisString += this[charPos];
  for (const key in args) {
    const stringKey = `{${key}}`;
    thisString = thisString.replace(new RegExp(stringKey, 'g'), args[key]);
  }
  return thisString;
};

let selectedURL = '';
let selectedName = '';
let selectedType = '';
let selectedBase = '';
let selectedModelId = '';
let selectedVersionId = '';

function clearModelDetails() {
  const el = gradioApp().getElementById('model-details') || gradioApp().getElementById('civitai_models_output') || gradioApp().getElementById('models_outcome');
  if (!el) return;
  el.innerHTML = '';
}

const modelDetailsHTML = `
  <div>
    <img src="{image}" alt="model image" class="preview" style="display: none">
    <button style="float: right" class="lg secondary gradio-button tool extra-details-close" id="model_details_close" data-hint="Close" onclick="clearModelDetails()"> ✕</button>
    <table id="model-details-table" class="model-details simple-table">
      <tr><td>Name</td><td>{name}</td></tr>
      <tr><td>Type</td><td>{type}</td></tr>
      <tr><td>Tags</td><td><div>{tags}</div></td></tr>
      <tr><td>NSFW</td><td>{nsfw} | {level}</td></tr>
      <tr><td>Availability</td><td>{availability}</td></tr>
      <tr><td>Downloads</td><td>{downloads}</td></tr>
      <tr><td>Author</td><td>{creator}</td></tr>
      <tr><td>Description</td><td><div>{desc}</div></td></tr>
      <tr><td>Download</td><td><div class="div-link" onclick="startCivitAllDownload(event)">All variants</div></td></tr>
    </table>
    <br>
    <table id="model-versions-table" class="model-versions simple-table">
      <thead>
        <tr>
          <th> </th>
          <th>Version</th>
          <th>Type</th>
          <th>Base</th>
          <th>File</th>
          <th>Updated</th>
          <th>Size</th>
          <th>Availability</th>
          <th>Description</th>
        </tr>
      </thead>
      <tbody>
        {versions}
      </tbody>
    </table>
  </div>
`;

const modelVersionsHTML = `
  <tr>
    <td>{url}</td>
    <td>{name}</td>
    <td>{type}</td>
    <td>{base}</td>
    <td>{file}</td>
    <td>{mtime}</td>
    <td>{size}</td>
    <td>{availability}</td>
    <td><div>{desc}</div></td>
  </tr>
`;

async function modelCardClick(id) {
  log('modelCardClick id', id);
  const el = gradioApp().getElementById('model-details') || gradioApp().getElementById('civitai_models_output') || gradioApp().getElementById('models_outcome');
  if (!el) return;
  const res = await authFetch(`${window.api}/civitai?model_id=${encodeURI(id)}`);
  if (!res || res.status !== 200) {
    error(`modelCardClick: id=${id} status=${res ? res.status : 'unknown'}`);
    return;
  }
  let data = await res.json();
  log('modelCardClick data', data);
  if (!data || data.length === 0) return;
  data = data[0]; // assuming the first item is the one we want

  const versionsHTML = data.versions.map((v) => modelVersionsHTML.format({
    url: `<div class="link" onclick="startCivitDownload('${v.files[0]?.url}', '${v.files[0]?.name}', '${data.type}', '${v.base || ''}', ${data.id}, ${v.id})"> \udb80\uddda </div>`,
    name: v.name || 'unknown',
    type: v.files[0]?.type || 'unknown',
    base: v.base || 'unknown',
    mtime: (new Date(v.mtime)).toLocaleDateString(),
    availability: v.availability || 'unknown',
    size: v.files[0]?.size ? `${(v.files[0].size / 1024 / 1024).toFixed(2)} MB` : 'unknown',
    file: `<a href=${v.files[0]?.url} target="_blank" rel="noopener noreferrer">${v.files[0]?.name || 'unknown'}</a>`,
    desc: v.desc || 'no description available',
  })).join('');
  const url = `<a href="${data.url}" target="_blank" rel="noopener noreferrer">${data.name || 'unknown'}</a>`;
  const creator = `<a href="https://civitai.com/user/${data.creator}" target="_blank" rel="noopener noreferrer">${data.creator || 'unknown'}</a>`;
  const images = data.versions.map((v) => v.images).flat().map((i) => i.url); // TODO image gallery
  const modelHTML = modelDetailsHTML.format({
    name: url,
    type: data.type || 'unknown',
    tags: data.tags?.join(', ') || '',
    nsfw: data.nsfw ? 'yes' : 'no',
    level: data.level?.toString() || '',
    availability: data.availability || 'unknown',
    downloads: data.downloads?.toString() || '',
    creator,
    desc: data.desc || 'no description available',
    image: images.length > 0 ? images[0] : '/sdapi/v1/network/thumb?filename=html/missing.png',
    versions: versionsHTML || '',
  });
  el.innerHTML = modelHTML;
}

function startCivitDownload(url, name, type, base, modelId, versionId) {
  log('startCivitDownload', { url, name, type, base, modelId, versionId });
  selectedURL = [url];
  selectedName = [name];
  selectedType = [type];
  selectedBase = [base || ''];
  selectedModelId = [modelId || 0];
  selectedVersionId = [versionId || 0];
  const civitDownloadBtn = gradioApp().getElementById('civitai_download_btn');
  if (civitDownloadBtn) civitDownloadBtn.click();
}

function startCivitAllDownload(evt) {
  log('startCivitAllDownload', evt);
  const versions = gradioApp().getElementById('model-versions-table').querySelectorAll('tr');
  selectedURL = [];
  selectedName = [];
  selectedType = [];
  selectedBase = [];
  selectedModelId = [];
  selectedVersionId = [];
  for (const version of versions) {
    const parsed = version.querySelector('td:nth-child(1) div')?.getAttribute('onclick')?.match(/startCivitDownload\('([^']+)', '([^']+)', '([^']+)', '([^']*)', (\d+), (\d+)\)/);
    if (!parsed || parsed.length < 7) continue;
    selectedURL.push(parsed[1]);
    selectedName.push(parsed[2]);
    selectedType.push(parsed[3]);
    selectedBase.push(parsed[4]);
    selectedModelId.push(parseInt(parsed[5], 10));
    selectedVersionId.push(parseInt(parsed[6], 10));
  }
  const civitDownloadBtn = gradioApp().getElementById('civitai_download_btn');
  if (civitDownloadBtn) civitDownloadBtn.click();
}

function downloadCivitModel(modelUrl, modelName, modelType, modelBase, mId, vId, modelPath, civitToken, innerHTML) {
  log('downloadCivitModel', { modelUrl, modelName, modelType, modelBase, mId, vId, modelPath, civitToken });
  const el = gradioApp().getElementById('civitai_models_output') || gradioApp().getElementById('models_outcome');
  const currentHTML = el?.innerHTML || '';
  return [selectedURL, selectedName, selectedType, selectedBase, selectedModelId, selectedVersionId, modelPath, civitToken, currentHTML];
}

let civitMutualExcludeBound = false;

function civitaiMutualExclude() {
  if (civitMutualExcludeBound) return;
  const searchEl = gradioApp().querySelector('#civit_search_text textarea');
  const tagEl = gradioApp().querySelector('#civit_search_tag textarea');
  if (!searchEl || !tagEl) return;
  civitMutualExcludeBound = true;
  searchEl.addEventListener('input', () => {
    tagEl.closest('.gradio-textbox')?.classList.toggle('disabled-look', !!searchEl.value.trim());
  });
  tagEl.addEventListener('input', () => {
    searchEl.closest('.gradio-textbox')?.classList.toggle('disabled-look', !!tagEl.value.trim());
  });
}

onUiLoaded(civitaiMutualExclude);
