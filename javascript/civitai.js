String.prototype.format = function (args) { // eslint-disable-line no-extend-native, func-names
  let thisString = '';
  for (let charPos = 0; charPos < this.length; charPos++) thisString += this[charPos];
  for (const key in args) { // eslint-disable-line guard-for-in
    const stringKey = `{${key}}`;
    thisString = thisString.replace(new RegExp(stringKey, 'g'), args[key]);
  }
  return thisString;
};

let selectedURL = '';
let selectedName = '';
let selectedType = '';

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
  const res = await fetch(`${window.api}/civitai?model_id=${encodeURI(id)}`);
  if (!res || res.status !== 200) {
    error(`modelCardClick: id=${id} status=${res ? res.status : 'unknown'}`);
    return;
  }
  let data = await res.json();
  log('modelCardClick data', data);
  if (!data || data.length === 0) return;
  data = data[0]; // assuming the first item is the one we want

  const versionsHTML = data.versions.map((v) => modelVersionsHTML.format({
    url: `<div class="link" onclick="startCivitDownload('${v.files[0]?.url}', '${v.files[0]?.name}', '${data.type}')"> \udb80\uddda </div>`,
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
    image: images.length > 0 ? images[0] : '/sdapi/v1/network/thumb?filename=html/card-no-preview.png',
    versions: versionsHTML || '',
  });
  el.innerHTML = modelHTML;
}

function startCivitDownload(url, name, type) {
  log('startCivitDownload', { url, name, type });
  selectedURL = [url];
  selectedName = [name];
  selectedType = [type];
  const civitDownloadBtn = gradioApp().getElementById('civitai_download_btn');
  if (civitDownloadBtn) civitDownloadBtn.click();
}

function startCivitAllDownload(evt) {
  log('startCivitAllDownload', evt);
  const versions = gradioApp().getElementById('model-versions-table').querySelectorAll('tr');
  selectedURL = [];
  selectedName = [];
  selectedType = [];
  for (const version of versions) {
    const parsed = version.querySelector('td:nth-child(1) div')?.getAttribute('onclick')?.match(/startCivitDownload\('([^']+)', '([^']+)', '([^']+)'\)/);
    if (!parsed || parsed.length < 4) continue;
    selectedURL.push(parsed[1]);
    selectedName.push(parsed[2]);
    selectedType.push(parsed[3]);
  }
  const civitDownloadBtn = gradioApp().getElementById('civitai_download_btn');
  if (civitDownloadBtn) civitDownloadBtn.click();
}

function downloadCivitModel(modelUrl, modelName, modelType, modelPath, civitToken, innerHTML) {
  log('downloadCivitModel', { modelUrl, modelName, modelType, modelPath, civitToken });
  const el = gradioApp().getElementById('civitai_models_output') || gradioApp().getElementById('models_outcome');
  const currentHTML = el?.innerHTML || '';
  return [selectedURL, selectedName, selectedType, modelPath, civitToken, currentHTML];
}
