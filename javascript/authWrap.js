let user = null;
let token = null;

async function authFetch(url, options = {}) {
  if (!token) {
    const res = await fetch(`${window.subpath}/token`);
    if (res.ok) {
      const data = await res.json();
      user = data.user;
      token = data.token;
    }
  }
  if (user && token) {
    if (!options.headers) options.headers = {};
    const encoded = btoa(`${user}:${token}`);
    options.headers.Authorization = `Basic ${encoded}`;
  }
  const res = await fetch(url, options);
  return res;
}
