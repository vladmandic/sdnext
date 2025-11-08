const loginCSS = `
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: #222;
  color: #ddd;
  font-family: monospace;
  z-index: 100;
`;

const loginHTML = `
  <div id="loginDiv" style="margin: 15% auto; max-width: 200px; padding: 2em; background: #444; border-radius: 4px; filter: drop-shadow(2px 4px 6px black);">
    <h2>Login</h2>
    <label for="username" style="margin-top: 0.5em">Username</label>
    <input type="text" id="loginUsername" name="username" style="width: 92%; padding: 0.5em; margin-top: 0.5em; border-radius: 4px;">
    <label for="password" style="margin-top: 0.5em">Password</label>
    <input type="text" id="loginPassword" name="password" style="width: 92%; padding: 0.5em; margin-top: 0.5em; border-radius: 4px;">
    <div id="loginStatus" style="margin-top: 0.5em"></div>
    <button type="submit" style="width: 100%; padding: 0.5em; margin-top: 0.5em; background: #366; color: #ddd; border: none; border-radius: 4px; filter: drop-shadow(2px 4px 6px black);">Login</button>
  </div>
`;

function forceLogin() {
  const form = document.createElement('form');
  form.method = 'POST';
  form.action = `${location.href}login`;
  form.id = 'loginForm';
  form.style.cssText = loginCSS;
  form.innerHTML = loginHTML;
  document.body.appendChild(form);
  const username = form.querySelector('#loginUsername');
  const password = form.querySelector('#loginPassword');
  const status = form.querySelector('#loginStatus');

  form.addEventListener('submit', (event) => {
    event.preventDefault();
    const formData = new FormData(form);
    formData.append('username', username.value);
    formData.append('password', password.value);
    console.warn('login', location.href, formData);
    fetch(`${location.href}login`, {
      method: 'POST',
      body: formData,
    })
      .then(async (res) => {
        const json = await res.json();
        const txt = `${res.status}: ${res.statusText} - ${json.detail}`;
        status.textContent = txt;
        console.log('login', txt);
        if (res.status === 200) location.reload();
      })
      .catch((err) => {
        status.textContent = err;
        console.error('login', err);
      });
  });
}

function loginCheck() {
  fetch(`${location.href}login_check`, {})
    .then((res) => {
      if (res.status === 200) console.log('login ok');
      else forceLogin();
    })
    .catch((err) => {
      console.error('login', err);
    });
}

window.onload = loginCheck;
