// hack to get pythons str.format in js
String.prototype.format = function (arguments) { // eslint-disable-line no-extend-native, func-names
  let thisString = '';
  for (let charPos = 0; charPos < this.length; charPos++) thisString += this[charPos];
  for (const key in arguments) { // eslint-disable-line guard-for-in
    error(key, arguments[key]);
    const stringKey = `{${key}}`;
    thisString = thisString.replace(new RegExp(stringKey, 'g'), arguments[key]);
  }
  return thisString;
};

const modelDetailsHTML = `
  <div id="model-details" class="model-details">
    <h3>{name}</h3>
    <p>Type: {type}</p>
    <p>Tags: {tags}</p>
    <p>NSFW: {nsfw}/{level}</p>
    <p>Availability: {availability}</p>
    <p>Downloads: {downloads}</p>
    <p>Author: {creator}</p>
    <div>{versions}</div>
  </div>
`;

async function modelCardClick(id) {
  log('modelCardClick id', id);
  const el = gradioApp().getElementById('model-details');
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
  const obj = {
    name: data.name || 'unknown',
    type: data.type || 'unknown',
    tags: data.tags?.join(', ') || '',
    nsfw: data.nsfw ? 'yes' : 'no',
    level: data.level?.toString() || '',
    availability: data.availability || 'unknown',
    downloads: data.downloads?.toString() || '',
    creator: data.creator || 'unknown',
    versions: JSON.stringify(data.versions) || '[]',
  };
  log(obj);
  el.innerHTML = modelDetailsHTML.format({
    name: data.name || 'unknown',
    type: data.type || 'unknown',
    tags: data.tags?.join(', ') || '',
    nsfw: data.nsfw ? 'yes' : 'no',
    level: data.level?.toString() || '',
    availability: data.availability || 'unknown',
    downloads: data.downloads?.toString() || '',
    creator: data.creator || 'unknown',
    versions: JSON.stringify(data.versions) || '[]',
  });
}

const example = {
  id: 1157409,
  url: 'https://civitai.com/models/1157409',
  type: 'Checkpoint',
  name: 'Tempest-by-Vlad',
  html: '',
  desc: 'Base versionFlexible SDXL model with custom encoder and finetuned for larger landscape resolutions with high details and high contrast.Recommended to use medium-low...',
  tags: [
    'base model',
  ],
  nsfw: false,
  level: 15,
  availability: 'Public',
  downloads: 407,
  creator: 'vmandic',
  versions: [
    {
      id: 1301775,
      name: 'Base v0.1',
      base: 'SDXL 1.0',
      mtime: '2025-01-19T02:53:53.903Z',
      downloads: 346,
      availability: 'Public',
      html: '',
      desc: 'Initial release',
      files: [
        {
          id: 1206102,
          size: 6938089790,
          name: 'tempestByVlad_baseV01.safetensors',
          type: 'Model',
          hashes: [
            '79CB1E32',
            '8BFAD17222',
            '8BFAD1722243955B3F94103C69079C280D348B14729251E86824972C1063B616',
            '43E5E3BB',
            'DE83D56256411853AB6595CC3D8E865D5310D4A58D49A839DDC104C7F3429D4A',
            '4E933E1EBE61',
          ],
          url: 'https://civitai.com/api/download/models/1301775',
          dct: {},
        },
      ],
      images: [
        {
          id: 52503951,
          url: 'https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/18c749f2-42ec-4024-9d20-0b1202b6bacc/width=1024/52503951.jpeg',
          width: 1024,
          height: 1024,
          type: 'image',
          dct: {},
        },
        {
          id: 52508539,
          url: 'https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/634d0ea8-ecdb-4ca6-a4ff-145319bc3fd3/width=1024/52508539.jpeg',
          width: 1024,
          height: 1024,
          type: 'image',
          dct: {},
        },
        {
          id: 52508563,
          url: 'https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/be529820-a89e-458f-8a3d-86cb43b154ac/width=1024/52508563.jpeg',
          width: 1024,
          height: 1024,
          type: 'image',
          dct: {},
        },
        {
          id: 52508588,
          url: 'https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/b2eb456a-a664-4de8-8c3e-6ecd1c4acb38/width=1024/52508588.jpeg',
          width: 1024,
          height: 1024,
          type: 'image',
          dct: {},
        },
        {
          id: 52508654,
          url: 'https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/f7b09e3f-4a48-459b-9b32-fa207904f74c/width=1024/52508654.jpeg',
          width: 1024,
          height: 1024,
          type: 'image',
          dct: {},
        },
        {
          id: 52508659,
          url: 'https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/fff89f18-628a-43b3-b9f5-44951dc078f7/width=1024/52508659.jpeg',
          width: 1024,
          height: 1024,
          type: 'image',
          dct: {},
        },
        {
          id: 52508671,
          url: 'https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/fdd3f774-ce2e-4ea7-82a0-bdb523fb86f6/width=1024/52508671.jpeg',
          width: 1024,
          height: 1024,
          type: 'image',
          dct: {},
        },
        {
          id: 52512251,
          url: 'https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/f26971c8-1123-45e3-a85b-2b97c6334b85/width=1024/52512251.jpeg',
          width: 1024,
          height: 1024,
          type: 'image',
          dct: {},
        },
      ],
      dct: {},
    },
    {
      id: 1343512,
      name: 'Hyper v0.1',
      base: 'SDXL 1.0',
      mtime: '2025-01-28T22:54:12.734Z',
      downloads: 61,
      availability: 'Public',
      html: '',
      desc: 'Time-distilled version',
      files: [
        {
          id: 1246991,
          size: 6938085702,
          name: 'tempestByVlad_hyperV01.safetensors',
          type: 'Model',
          hashes: [
            '15943FD9',
            '4104FC6601',
            '4104FC6601F71C4C7A770AD422483FD700C8ECF72D06FCD8C4E8CD4B2D1C7DBB',
            '9F87BCEA',
            'CB52894625E9C13331285E4435799D707C4EAEF464974159C8B4B217EA32298E',
            'A0EE15E503DD',
          ],
          url: 'https://civitai.com/api/download/models/1343512',
          dct: {},
        },
      ],
      images: [
        {
          id: 54462987,
          url: 'https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/1dd020c8-8a9a-4eb3-afec-fe83613217c5/width=1024/54462987.jpeg',
          width: 1024,
          height: 768,
          type: 'image',
          dct: {},
        },
        {
          id: 54462992,
          url: 'https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/89edf233-939f-4b2e-97c8-175498704362/width=1536/54462992.jpeg',
          width: 1536,
          height: 640,
          type: 'image',
          dct: {},
        },
        {
          id: 54463002,
          url: 'https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/87aea8fb-7687-48d0-98e6-27555b2ff87f/width=768/54463002.jpeg',
          width: 768,
          height: 1024,
          type: 'image',
          dct: {},
        },
        {
          id: 54463010,
          url: 'https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/f1ee57cc-8920-4b8d-853a-ad1cfc7d9a5a/width=1024/54463010.jpeg',
          width: 1024,
          height: 1024,
          type: 'image',
          dct: {},
        },
        {
          id: 54463011,
          url: 'https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/dda6411b-fd36-484f-a9f0-0db847463128/width=1024/54463011.jpeg',
          width: 1024,
          height: 1024,
          type: 'image',
          dct: {},
        },
        {
          id: 54463016,
          url: 'https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/5f266a8c-f215-4e0f-8a64-aacc97f81d70/width=1024/54463016.jpeg',
          width: 1024,
          height: 1024,
          type: 'image',
          dct: {},
        },
        {
          id: 54463019,
          url: 'https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/0ad04bdc-41bd-433c-9c25-f5365fbe082c/width=1024/54463019.jpeg',
          width: 1024,
          height: 1024,
          type: 'image',
          dct: {},
        },
      ],
      dct: {},
    },
  ],
  dct: {},
};
