interface LoggerEntry {
  ts: string;
  type: 'log' | 'debug' | 'error';
  msg: unknown[];
}

window.logRingBuffer = [];
window.logBufferDirty = false;

const logBuffer = (ts: string, type: 'log' | 'debug' | 'error', msg: unknown[]): void => {
  const maxLogLength = 8;
  window.logRingBuffer.push({ ts, type, msg });
  if (window.logRingBuffer.length > maxLogLength) window.logRingBuffer.shift();
  window.logBufferDirty = true;
};

const scrollBottom = async (el: HTMLElement): Promise<void> => {
  const lastChild = el.lastElementChild;
  if (lastChild) lastChild.scrollIntoView({ behavior: 'smooth' });
};

export const log = async (...msg: unknown[]): Promise<void> => {
  const dt = new Date();
  const ts = `${dt.getHours().toString().padStart(2, '0')}:${dt.getMinutes().toString().padStart(2, '0')}:${dt.getSeconds().toString().padStart(2, '0')}.${dt.getMilliseconds().toString().padStart(3, '0')}`;
  if (window.logger) {
    if (window.logPrettyPrint) window.logger.innerHTML += window.logPrettyPrint(...msg);
    scrollBottom(window.logger);
  }
  console.log(ts, ...msg);
  logBuffer(ts, 'log', msg);
};

export const debug = async (...msg: unknown[]): Promise<void> => {
  const dt = new Date();
  const ts = `${dt.getHours().toString().padStart(2, '0')}:${dt.getMinutes().toString().padStart(2, '0')}:${dt.getSeconds().toString().padStart(2, '0')}.${dt.getMilliseconds().toString().padStart(3, '0')}`;
  if (window.logger) {
    if (window.logPrettyPrint) window.logger.innerHTML += window.logPrettyPrint(...msg);
    scrollBottom(window.logger);
  }
  console.debug(ts, ...msg);
  logBuffer(ts, 'debug', msg);
};

export const error = async (...msg: unknown[]): Promise<void> => {
  const dt = new Date();
  const ts = `${dt.getHours().toString().padStart(2, '0')}:${dt.getMinutes().toString().padStart(2, '0')}:${dt.getSeconds().toString().padStart(2, '0')}.${dt.getMilliseconds().toString().padStart(3, '0')}`;
  if (window.logger) {
    if (window.logPrettyPrint) window.logger.innerHTML += window.logPrettyPrint(...msg);
    scrollBottom(window.logger);
  }
  console.error(ts, ...msg);
  logBuffer(ts, 'error', msg);
  // const txt = msg.join(' ');
  // if (!txt.includes('asctime') && !txt.includes('xhr.')) xhrPost('/sdapi/v1/log', { error: txt }); // eslint-disable-line no-use-before-define
};

const xhrInternal = async (
  xhrObj: XMLHttpRequest,
  data: unknown,
  handler?: (json: unknown) => void,
  errorHandler?: (xhrObj: XMLHttpRequest) => void,
  ignore = false,
  serverTimeout = window.opts.ui_request_timeout || 30000,
): Promise<void> => {
  const err = (msg: string): void => {
    if (!ignore) {
      error(`${msg}: state=${xhrObj.readyState} status=${xhrObj.status} response=${xhrObj.responseText}`);
      if (errorHandler) errorHandler(xhrObj);
    }
  };

  // Authorization is not required for logger xhr calls and would create a dependency cycle.

  xhrObj.setRequestHeader('Content-Type', 'application/json');
  xhrObj.timeout = serverTimeout;
  xhrObj.ontimeout = () => err('xhr.ontimeout');
  xhrObj.onerror = () => err('xhr.onerror');
  xhrObj.onabort = () => err('xhr.onabort');
  xhrObj.onreadystatechange = () => {
    if (xhrObj.readyState === 4) {
      if (xhrObj.status === 200) {
        try {
          const json = JSON.parse(xhrObj.responseText);
          if (handler) handler(json);
        } catch {
          // error(`xhr.onreadystatechange: ${e}`);
        }
      } else {
        // err(`xhr.onreadystatechange: state=${xhrObj.readyState} status=${xhrObj.status} response=${xhrObj.responseText}`);
      }
    }
  };
  const req = JSON.stringify(data);
  xhrObj.send(req);
};

export const xhrGet = (
  url: string,
  data: Record<string, string | number | boolean>,
  handler?: (json: unknown) => void,
  errorHandler?: (xhrObj: XMLHttpRequest) => void,
  ignore = false,
  serverTimeout = window.opts.ui_request_timeout || 30000,
): void => {
  const xhr = new XMLHttpRequest();
  const args = Object.keys(data).map((k) => `${encodeURIComponent(k)}=${encodeURIComponent(data[k])}`).join('&');
  xhr.open('GET', `${url}?${args}`, true);
  xhrInternal(xhr, data, handler, errorHandler, ignore, serverTimeout);
};

export function xhrPost(
  url: string,
  data: unknown,
  handler?: (json: unknown) => void,
  errorHandler?: (xhrObj: XMLHttpRequest) => void,
  ignore = false,
  serverTimeout = window.opts.ui_request_timeout || 30000,
): void {
  const xhr = new XMLHttpRequest();
  xhr.open('POST', url, true);
  xhrInternal(xhr, data, handler, errorHandler, ignore, serverTimeout);
}

window.log = log;
window.debug = debug;
window.error = error;
window.xhrGet = xhrGet;
window.xhrPost = xhrPost;
