let user = null;
let token = null;

async function getToken() {
  if (!token || !user) {
    const res = await fetch(`${window.subpath}/token`);
    if (res.ok) {
      const data = await res.json();
      user = data.user;
      token = data.token;
    }
  }
  return { user, token };
}

async function authFetch(url, options = {}) {
  const { localUser, localToken } = await getToken();
  if (localUser && localToken) {
    if (!options.headers) options.headers = {};
    const encoded = btoa(`${localUser}:${localToken}`);
    options.headers.Authorization = `Basic ${encoded}`;
  }
  const res = await fetch(url, options);
  return res;
}
