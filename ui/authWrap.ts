import { log, error } from './logger';

interface TokenResponse {
  user?: string;
  token?: string;
}

let user: string | undefined;
let token: string | undefined;

export async function getToken(): Promise<{ user: string | undefined; token: string | undefined }> {
  if (token === undefined || user === undefined) {
    const res = await fetch(`${window.subpath}/token`);
    if (res.ok) {
      const data = (await res.json()) as TokenResponse;
      user = data.user;
      token = data.token;
      log('getToken', user);
    }
  }
  return { user, token };
}

export async function authFetch(url: RequestInfo | URL, options: RequestInit = {}): Promise<Response | undefined> {
  await getToken();
  if (user && token) {
    const encoded = btoa(`${user}:${token}`);
    const headers = new Headers(options.headers);
    headers.set('Authorization', `Basic ${encoded}`);
    options.headers = headers;
  }
  let res: Response | undefined;
  try {
    res = await fetch(url, options);
    if (!res.ok) error('fetch', { status: res?.status || 503, url, user, token });
  } catch (err) {
    if (navigator.onLine) {
      error('fetch', { status: res?.status || 503, url, user, token, error: err });
    }
  }
  return res;
}
window.authFetch = authFetch;
