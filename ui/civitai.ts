import { gradioApp, onUiLoaded } from './script';
import { log, error } from './logger';
import { authFetch } from './authWrap';

interface CivitFileMetadata {
  fp?: string | null;
  format?: string | null;
  size?: string | null;
  quantType?: string | null;
}

interface CivitFile {
  id: number;
  url?: string;
  name?: string;
  type?: string;
  size?: number;
  primary?: boolean;
  metadata?: CivitFileMetadata;
}

interface CivitImage {
  url: string;
}

interface CivitVersion {
  id: number;
  name?: string;
  base?: string;
  mtime: string;
  availability?: string;
  desc?: string;
  files: CivitFile[];
  images: CivitImage[];
}

interface CivitModel {
  id: number;
  url: string;
  name?: string;
  type?: string;
  tags?: string[];
  nsfw?: boolean;
  level?: number;
  availability?: string;
  downloads?: number;
  creator?: string;
  desc?: string;
  versions: CivitVersion[];
}

interface QueuedFile {
  version: CivitVersion;
  file: CivitFile;
}

// eslint-disable-next-line no-extend-native
String.prototype.format = function format(this: string, args: Record<string, string | number>): string {
  let thisString = '';
  for (let charPos = 0; charPos < this.length; charPos++) thisString += this[charPos];
  for (const key in args) {
    const stringKey = `{${key}}`;
    thisString = thisString.replace(new RegExp(stringKey, 'g'), String(args[key]));
  }
  return thisString;
};

let selectedURL: string[] = [];
let selectedName: string[] = [];
let selectedType: string[] = [];
let selectedBase: string[] = [];
let selectedModelId: number[] = [];
let selectedVersionId: number[] = [];
let currentModel: CivitModel | null = null;

const precisionOrder = ['fp32', 'bf16', 'fp16', 'fp8', 'int8', 'int4'];
const companionTypes = ['VAE', 'Text Encoder'];

export function clearModelDetails() {
  const el = gradioApp().getElementById('model-details') || gradioApp().getElementById('civitai_models_output') || gradioApp().getElementById('models_outcome');
  if (!el) return;
  el.innerHTML = '';
}
window.clearModelDetails = clearModelDetails;

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
          <th>Variant</th>
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

function fileVariant(file: CivitFile): string | null {
  return file.metadata?.fp || file.metadata?.quantType || null;
}

// Model files before companions, then by precision, then larger first
function sortFiles(files: CivitFile[]): CivitFile[] {
  const isModel = (f: CivitFile) => f.type === 'Model' || f.type === 'Pruned Model';
  const rank = (f: CivitFile) => {
    const index = precisionOrder.indexOf((fileVariant(f) || '').toLowerCase());
    return index < 0 ? precisionOrder.length : index;
  };
  return [...files].sort((a, b) => Number(isModel(b)) - Number(isModel(a)) || rank(a) - rank(b) || (b.size || 0) - (a.size || 0));
}

function insertNameSuffix(name: string, suffix: string): string {
  const dot = name.lastIndexOf('.');
  return dot > 0 ? `${name.slice(0, dot)}-${suffix}${name.slice(dot)}` : `${name}-${suffix}`;
}

// Precision suffix, then full/pruned and the file id only as far as needed to stay unique within the version
function fileSaveName(file: CivitFile, siblings: CivitFile[]): string {
  const tier1 = (f: CivitFile) => {
    const variant = fileVariant(f);
    return variant ? insertNameSuffix(f.name || '', variant) : f.name || '';
  };
  const tier2 = (f: CivitFile) => (f.metadata?.size ? insertNameSuffix(tier1(f), f.metadata.size) : tier1(f));
  const others = siblings.filter((s) => s.id !== file.id);
  const name = tier1(file);
  if (!others.some((s) => tier1(s) === name)) return name;
  const sized = tier2(file);
  if (!others.some((s) => tier2(s) === sized)) return sized;
  return insertNameSuffix(name, String(file.id));
}

function escapeHTML(text: string): string {
  return text.replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
}

function versionRows(version: CivitVersion, divider: boolean): string {
  const files = sortFiles(version.files);
  const entries: (CivitFile | null)[] = files.length > 0 ? files : [null];
  const border = divider ? ' style="border-top: 1px solid var(--sd-panel-border-color, #555)"' : '';
  const span = entries.length > 1 ? ` rowspan="${entries.length}"` : '';
  const versionCell = (content: string) => `<td${span}${border}>${content}</td>`;
  return entries.map((file, i) => {
    const first = i === 0;
    const cell = (content: string) => `<td${first ? border : ''}>${content}</td>`;
    const link = file ? `<div class="link" onclick="startCivitFileDownload(${version.id}, ${file.id})"> \udb80\uddda </div>` : '';
    const name = file ? `<a href="${escapeHTML(file.url || '')}" target="_blank" rel="noopener noreferrer">${escapeHTML(file.name || 'unknown')}</a>${file.primary ? ' <span title="Primary file">★</span>' : ''}` : 'unknown';
    const variant = file ? [fileVariant(file), file.metadata?.size].filter(Boolean).join(' · ') : '';
    const size = file?.size ? `${(file.size / 1024 / 1024 / 1024).toFixed(2)} GB` : 'unknown';
    const cells = [
      cell(link),
      first ? versionCell(escapeHTML(version.name || 'unknown')) : '',
      cell(escapeHTML(file?.type || 'unknown')),
      first ? versionCell(escapeHTML(version.base || 'unknown')) : '',
      cell(name),
      cell(escapeHTML(variant || '-')),
      first ? versionCell((new Date(version.mtime)).toLocaleDateString()) : '',
      cell(size),
      first ? versionCell(escapeHTML(version.availability || 'unknown')) : '',
      first ? versionCell(`<div>${version.desc || 'no description available'}</div>`) : '',
    ];
    return `<tr>${cells.join('')}</tr>`;
  }).join('');
}

export async function modelCardClick(id) {
  log('modelCardClick id', id);
  const el = gradioApp().getElementById('model-details') || gradioApp().getElementById('civitai_models_output') || gradioApp().getElementById('models_outcome');
  if (!el) return;
  const res = await authFetch(`${window.api}/civitai?model_id=${encodeURI(id)}`);
  if (!res || res.status !== 200) {
    error(`modelCardClick: id=${id} status=${res ? res.status : 'unknown'}`);
    return;
  }
  const dataArray = await res.json();
  log('modelCardClick data', dataArray);
  if (!dataArray || dataArray.length === 0) return;
  const data: any = dataArray[0]; // assuming the first item is the one we want
  currentModel = data;

  const versionsHTML = data.versions.map((v: CivitVersion, i: number) => versionRows(v, i > 0)).join('');
  const url = `<a href="${data.url}" target="_blank" rel="noopener noreferrer">${data.name || 'unknown'}</a>`;
  const creator = `<a href="https://civitai.com/user/${data.creator}" target="_blank" rel="noopener noreferrer">${data.creator || 'unknown'}</a>`;
  const images = data.versions.map((v: CivitVersion) => v.images).flat().map((i: CivitImage) => i.url); // TODO image gallery
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
    image: images.length > 0 ? images[0] : '/sdapi/v1/network/thumb?filename=ui/assets/missing.png',
    versions: versionsHTML || '',
  });
  el.innerHTML = modelHTML;
}
window.modelCardClick = modelCardClick;

function queueFiles(model: CivitModel, queued: QueuedFile[]) {
  selectedURL = queued.map(({ file }) => file.url || '');
  selectedName = queued.map(({ version, file }) => fileSaveName(file, version.files));
  selectedType = queued.map(({ file }) => (companionTypes.includes(file.type || '') ? file.type : model.type) || '');
  selectedBase = queued.map(({ version }) => version.base || '');
  selectedModelId = queued.map(() => model.id || 0);
  selectedVersionId = queued.map(({ version }) => version.id || 0);
  const civitDownloadBtn = gradioApp().getElementById('civitai_download_btn');
  if (civitDownloadBtn) civitDownloadBtn.click();
}

export function startCivitFileDownload(versionId: number, fileId: number) {
  log('startCivitFileDownload', { versionId, fileId });
  const version = currentModel?.versions.find((v) => v.id === versionId);
  const file = version?.files.find((f) => f.id === fileId);
  if (!currentModel || !version || !file) return;
  queueFiles(currentModel, [{ version, file }]);
}
window.startCivitFileDownload = startCivitFileDownload;

export function startCivitAllDownload(evt) {
  log('startCivitAllDownload', evt);
  if (!currentModel) return;
  const queued = currentModel.versions
    .map((version) => ({ version, file: version.files.find((f) => f.primary) || version.files[0] }))
    .filter((entry): entry is QueuedFile => !!entry.file);
  queueFiles(currentModel, queued);
}
window.startCivitAllDownload = startCivitAllDownload;

export function downloadCivitModel(modelUrl, modelName, modelType, modelBase, mId, vId, modelPath, civitToken, innerHTML) {
  log('downloadCivitModel', { modelUrl, modelName, modelType, modelBase, mId, vId, modelPath, civitToken });
  const el = gradioApp().getElementById('civitai_models_output') || gradioApp().getElementById('models_outcome');
  const currentHTML = el?.innerHTML || '';
  return [selectedURL, selectedName, selectedType, selectedBase, selectedModelId, selectedVersionId, modelPath, civitToken, currentHTML];
}
window.downloadCivitModel = downloadCivitModel;

let civitMutualExcludeBound = false;

export function civitaiMutualExclude() {
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
