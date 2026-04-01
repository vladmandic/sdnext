#!/usr/bin/env node

// simple nodejs script to test sdnext api

const fs = require('node:fs');
const path = require('node:path');

const argparse = require('argparse');

const sd_url = process.env.SDAPI_URL || 'http://127.0.0.1:7860';
const sd_username = process.env.SDAPI_USR;
const sd_password = process.env.SDAPI_PWD;
let args = {};

function b64(file) {
  const data = fs.readFileSync(file);
  const b64str = Buffer.from(data).toString('base64');
  const ext = path.extname(file).replace('.', '');
  const str = `data:image/${ext};base64,${b64str}`;
  // console.log('b64:', ext, b64.length);
  return str;
}

function options() {
  const opt = {
    // first pass
    prompt: args.prompt || 'beautiful lady, in the steampunk style',
    negative_prompt: args.negative || 'foggy, blurry',
    seed: -1,
    steps: 20,
    batch_size: 1,
    n_iter: 1,
    cfg_scale: 6,
    width: args.width || 1024,
    height: args.height || 1024,
    // api return options
    save_images: false,
    send_images: true,
  };
  if (args.pulid) {
    const b64image = b64(args.pulid);
    opt.script_name = 'pulid';
    opt.script_args = [
      b64image, // b64 encoded image, required param
      0.9, // strength, optional
      20, // zero, optional
      'dpmpp_sde', // sampler, optional
      'v2', // ortho, optional
      true, // restore (disable pulid after run), optional
      true, // offload, optional
      'v1.1', // version, optional
    ];
  }
  // console.log('options:', opt);
  return opt;
}

function init() {
  const parser = new argparse.ArgumentParser({ description: 'SD.Next API' });
  parser.add_argument('--prompt', { type: 'str', help: 'prompt' });
  parser.add_argument('--negative', { type: 'str', help: 'negative' });
  parser.add_argument('--width', { type: 'int', help: 'width' });
  parser.add_argument('--height', { type: 'int', help: 'height' });
  parser.add_argument('--pulid', { type: 'str', help: 'pulid init image' });
  parser.add_argument('--output', { type: 'str', help: 'output path' });
  const parsed = parser.parse_args();
  return parsed;
}

async function main() {
  const method = 'POST';
  const headers = new Headers();
  const opt = options();
  const body = JSON.stringify(opt);
  headers.set('Content-Type', 'application/json');
  if (sd_username && sd_password) headers.set({ Authorization: `Basic ${btoa('sd_username:sd_password')}` });
  const res = await fetch(`${sd_url}/sdapi/v1/txt2img`, { method, headers, body });

  if (res.status !== 200) {
    console.log('Error', res.status);
  } else {
    const json = await res.json();
    console.log('result:', json.info);
    for (const i in json.images) {
      const file = args.output || `/tmp/test-${i}.jpg`;
      const data = atob(json.images[i]);
      fs.writeFileSync(file, data, 'binary');
      console.log('image saved:', file);
    }
  }
}

args = init();
main();
