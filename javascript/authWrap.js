let user = null;
let token = null;

async function getToken() {
  if (!token || !user) {
    const res = await fetch(`${window.subpath}/token`);
    if (res.ok) {
      const data = await res.json();
      user = data.user;
      token = data.token;
      log('getToken', user);
    }
  }
  return { user, token };
}

async function authFetch(url, options = {}) {
  await getToken();
  if (user && token) {
    if (!options.headers) options.headers = {};
    const encoded = btoa(`${user}:${token}`);
    options.headers.Authorization = `Basic ${encoded}`;
  }
  let res;
  try {
    res = await fetch(url, options);
    if (!res.ok) error('fetch', { status: res.status, url, user, token });
  } catch (err) {
    error('fetch', { status: res.status, url, user, token, error: err });
  }
  return res;
}
