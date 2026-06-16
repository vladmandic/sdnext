import $ from 'jquery';

const globalAny = globalThis as any;
globalAny.$ = $;
globalAny.jQuery = $;
($ as any).isArray = Array.isArray; // polyfill for old jquery method

import './js/iframeResizer'; // eslint-disable-line import-x/first
