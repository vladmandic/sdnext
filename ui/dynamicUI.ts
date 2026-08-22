import { log } from './logger';
import { authFetch } from './authWrap';

interface Model {
  type: string;
  class: string;
  checkpoint: string;
  title: string;
  name: string;
}
let lastCheckpoint = '';

async function updateUI(model: Model) {
  if (model.checkpoint === lastCheckpoint) return;
  lastCheckpoint = model.checkpoint;
  log('modelUpdate', model);
}

export async function updateModel() {
  const req = await authFetch(`${window.api}/checkpoint`);
  if (req.ok) {
    const model = await req.json() as Model;
    if (model?.type?.length > 0) updateUI(model);
  }
}
