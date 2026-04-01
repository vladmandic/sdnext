#!/usr/bin/env node

// simple nodejs script to test sdnext api

const fs = require('node:fs');

const sd_url = process.env.SDAPI_URL || 'http://127.0.0.1:7860';
const sd_username = process.env.SDAPI_USR;
const sd_password = process.env.SDAPI_PWD;
const sd_options = {
  // first pass
  prompt: 'city at night',
  negative_prompt: 'foggy, blurry',
  sampler_name: 'UniPC',
  seed: -1,
  steps: 20,
  batch_size: 1,
  n_iter: 1,
  cfg_scale: 6,
  width: 512,
  height: 512,
  // api return options
  save_images: false,
  send_images: true,
};

async function main() {
  const method = 'POST';
  const headers = new Headers();
  const body = JSON.stringify(sd_options);
  headers.set('Content-Type', 'application/json');
  if (sd_username && sd_password) {
    // const credentials = btoa(`${sd_username}:${sd_password}`);
    const credentials = Buffer.from(`${sd_username}:${sd_password}`).toString('base64');
    headers.set('Authorization', `Basic ${credentials}`);
  }
  const res = await fetch(`${sd_url}/sdapi/v1/txt2img`, { method, headers, body });
  if (res.status !== 200) {
    const err = await res.text();
    console.log('Error', res.status, res.statusText, err);
  } else {
    const json = await res.json();
    console.log('result:', json.info);
    for (const i in json.images) {
      const f = `/tmp/test-${i}.jpg`;
      fs.writeFileSync(f, atob(json.images[i]), 'binary');
      console.log('image saved:', f);
    }
  }
}

main();
