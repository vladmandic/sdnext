import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';
import { transformSync } from 'esbuild';

const source = readFileSync(new URL('../ui/progressBar.ts', import.meta.url), 'utf8');
const { code } = transformSync(source, { loader: 'ts', format: 'cjs' });

function setup() {
  const requests = [];
  const timers = [];
  const storage = new Map();
  const button = { style: {} };
  let notifications = 0;
  let completions = 0;
  const root = { querySelectorAll: () => [] };
  const context = {
    module: { exports: {} },
    window: { opts: {}, waitForUiReady: false },
    document: {
      hidden: false,
      getElementById: (id) => (id === 'control_generate' ? button : null),
    },
    localStorage: {
      setItem: (key, value) => storage.set(key, value),
      getItem: (key) => storage.get(key),
      removeItem: (key) => storage.delete(key),
    },
    setTimeout: (callback) => timers.push(callback),
    require: (name) => {
      if (name === './logger') return {
        log() {},
        debug() {},
        error() {},
        xhrPost: (url, data, success, error) => requests.push({ url, data, success, error }),
      };
      if (name === './script') return { gradioApp: () => root };
      if (name === './notification') return { sendNotification: () => { notifications += 1; } };
      throw new Error(`Unexpected import: ${name}`);
    },
  };
  vm.runInNewContext(code, context);
  const start = (id) => context.module.exports.requestProgress(id, null, null, () => { completions += 1; });
  const reply = (id, preview, state) => {
    const request = requests.find((item) => item.data.id_task === id && (item.data.id_live_preview !== -1) === preview);
    assert.ok(request);
    request.success({ progress: 0, eta: 0, ...state });
  };
  return { start, reply, button, storage, requests, timers, counts: () => ({ notifications, completions }) };
}

test('queued monitors cannot overwrite the running task', () => {
  const monitor = setup();
  monitor.start('running');
  monitor.reply('running', false, { active: true, job: 'Sampling', progress: 0.5, eta: 10 });
  const runningLabel = monitor.button.innerText;
  monitor.start('queued');
  monitor.reply('queued', false, { queued: true, textinfo: 'Queued...' });
  assert.equal(monitor.button.innerText, runningLabel);
  monitor.reply('running', true, { active: true, job: 'Sampling', progress: 0.6, eta: 8 });
  assert.match(monitor.button.innerText, /60%/);
  monitor.reply('queued', true, { queued: true, textinfo: 'Queued...' });
  assert.match(monitor.button.innerText, /60%/);
});

test('finishing an older task preserves newer progress and recovery id', () => {
  const monitor = setup();
  monitor.start('old');
  monitor.start('new');
  monitor.reply('new', false, { active: true, job: 'Story' });
  monitor.reply('old', false, { completed: true });
  assert.equal(monitor.button.innerText, 'Story Start');
  assert.equal(monitor.storage.get('task'), 'new');
});

test('both polling channels finish once and ignore stale responses and timers', () => {
  const monitor = setup();
  monitor.start('task');
  monitor.reply('task', false, { active: true, job: 'Story' });
  monitor.reply('task', true, { completed: true });
  monitor.reply('task', false, { completed: true });
  monitor.reply('task', true, { active: true, job: 'Story' });
  const count = monitor.requests.length;
  for (const timer of monitor.timers) timer();
  assert.equal(monitor.requests.length, count);
  assert.equal(monitor.button.innerText, 'Generate');
  assert.equal(monitor.storage.has('task'), false);
  assert.deepEqual(monitor.counts(), { notifications: 1, completions: 1 });
});

test('remaining queued work is displayed after the active task completes', () => {
  const monitor = setup();
  monitor.start('active');
  monitor.start('queued');
  monitor.reply('active', false, { active: true, job: 'Story' });
  monitor.reply('queued', false, { queued: true, textinfo: 'Queued...' });
  monitor.reply('active', true, { completed: true });
  assert.match(monitor.button.innerText, /Queued/);
});
