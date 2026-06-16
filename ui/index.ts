// ui/startup.ts
// Entrypoint for the SD.Next core UI bundle source tree.
// This file imports the legacy UI modules and vendor packages without changing runtime loader behavior.

import './vendor';
import './startup';
import './logger';
import './script';
import './ui';
import './uiConfig';
import './extensions';
import './contextMenus';
import './dragDrop';
import './inputAccordion';
import './settings';
import './imageViewer';
import './gallery';
import './generationParams';
import './civitai';
import './guidance';
import './hires';
import './changelog';
import './control';
import './logMonitor';
import './notification';
import './promptChecker';
import './setHints';
import './monitor';
import './history';
import './aspectRatioOverlay';
import './authWrap';
import './autocomplete';
import './autocomplete_xn';
import './editAttention';
import './extraNetworks';
import './gpu';
import './imageParams';
import './progressBar';
import './timers';
import './timesheet';
import './trainMonitor';
import './indexdb';

// Side-effect-only entrypoint; the generated bundle is separate from hand-authored source.
