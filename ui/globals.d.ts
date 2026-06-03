declare global {
  interface String {
    format(args: Record<string, string | number>): string;
  }

  // eslint-disable-next-line @typescript-eslint/consistent-indexed-object-style
  interface Node {
    [key: string]: any;
  }

  const $: any;

  interface JQueryStatic {
    range_map?: (options: Record<string, string>) => Record<string, string>;
    isArray?: any;
  }

  interface JQuery {
    sparkline?: (data: any, options?: any) => JQuery;
  }

  interface Window {
    // state objects
    api: string; // ui/startup.ts
    subpath: string; // ui/startup.ts
    opts: Record<string, any>; // ui/ui.ts
    localization?: Record<string, any>; // ui/ui.ts
    titles?: Record<string, any>; // ui/ui.ts
    submit_state?: string; // ui/ui.ts
    logRingBuffer: { ts: string; type: string; msg: unknown[] }[]; // ui/logger.ts
    logBufferDirty: boolean; // ui/logger.ts

    // global functions
    args_to_array?: typeof Array.from; // ui/ui.ts
    updateInput?: (target: EventTarget) => void; // ui/ui.ts
    cycleImageFit?: () => void; // ui/imageViewer.ts
    clip_gallery_urls?: (gallery: { data: string }[]) => void; // ui/ui.ts
    extract_image_from_gallery?: (gallery: { data?: string }[]) => ({ data?: string } | null)[]; // ui/ui.ts
    extensions_apply?: (_extensionsDisabledList: unknown, _extensionsUpdateList: unknown, disableAll: unknown) => [string, string, unknown]; // ui/extensions.ts
    extensions_check?: (_info: unknown, _extensionsDisabledList: unknown, searchText: unknown, sortColumn: unknown) => [string, string, unknown, unknown]; // ui/extensions.ts
    install_extension?: (button: HTMLButtonElement | HTMLInputElement, url: string) => void; // ui/extensions.ts
    uninstall_extension?: (button: HTMLButtonElement | HTMLInputElement, url: string) => void; // ui/extensions.ts
    update_extension?: (button: HTMLButtonElement | HTMLInputElement, url: string) => void; // ui/extensions.ts
    uiOpenSubmenus?: () => Record<string, boolean>; // ui/uiConfig.ts
    getCaptionActiveTab?: (...args: unknown[]) => unknown[]; // ui/ui.ts
    get_img2img_tab_index?: (...args: unknown[]) => unknown[]; // ui/ui.ts
    modelmerger?: (...args: unknown[]) => unknown[]; // ui/ui.ts
    restartReload?: (initial?: boolean) => void; // ui/ui.ts
    selected_gallery_index?: () => number; // ui/ui.ts
    selected_gallery_files?: (...args: unknown[]) => unknown[]; // ui/ui.ts
    send_to_kanvas?: (gallery: { data?: string }[]) => void; // ui/ui.ts
    submit?: (...args: unknown[]) => unknown[]; // ui/ui.ts
    submit_control?: (...args: unknown[]) => unknown[]; // ui/ui.ts
    submit_framepack?: (...args: unknown[]) => unknown[]; // ui/ui.ts
    submit_img2img?: (...args: unknown[]) => unknown[]; // ui/ui.ts
    submit_ltx?: (...args: unknown[]) => unknown[]; // ui/ui.ts
    submit_postprocessing?: (...args: unknown[]) => unknown[]; // ui/ui.ts
    submit_txt2img?: (...args: unknown[]) => unknown[]; // ui/ui.ts
    submit_video?: (...args: unknown[]) => unknown[]; // ui/ui.ts
    submit_video_wrapper?: (...args: unknown[]) => void; // ui/ui.ts
    currentImageResolutionimg2img?: (...args: unknown[]) => unknown[]; // ui/ui.ts
    currentImageResolutioncontrol?: (...args: unknown[]) => unknown[]; // ui/ui.ts
    switch_to_txt2img?: (...args: unknown[]) => unknown[]; // ui/ui.ts
    switch_to_img2img_tab?: (...args: unknown[]) => void; // ui/ui.ts
    switch_to_img2img?: (...args: unknown[]) => unknown[]; // ui/ui.ts
    switch_to_inpaint?: (...args: unknown[]) => unknown[]; // ui/ui.ts
    switch_to_sketch?: (...args: unknown[]) => unknown[]; // ui/ui.ts
    switch_to_composite?: (...args: unknown[]) => unknown[]; // ui/ui.ts
    switch_to_extras?: (...args: unknown[]) => unknown[]; // ui/ui.ts
    switch_to_control?: (...args: unknown[]) => unknown[]; // ui/ui.ts
    switch_to_video?: (...args: unknown[]) => unknown[]; // ui/ui.ts
    switch_to_caption?: (...args: unknown[]) => unknown[]; // ui/ui.ts
    recalculate_prompts_txt2img?: (...args: unknown[]) => unknown[]; // ui/ui.ts
    recalculate_prompts_img2img?: (...args: unknown[]) => unknown[]; // ui/ui.ts
    recalculate_prompts_inpaint?: (...args: unknown[]) => unknown[]; // ui/ui.ts
    recalculate_prompts_control?: (...args: unknown[]) => unknown[]; // ui/ui.ts
    consumeDesiredCheckpointName?: (...args: unknown[]) => unknown[]; // ui/ui.ts
    create_submit_args?: (args: unknown[]) => unknown[]; // ui/ui.ts
    selectCheckpoint?: (name: string) => void; // ui/ui.ts
    selectVAE?: (name: string) => void; // ui/ui.ts
    selectUNet?: (name: string) => void; // ui/ui.ts
    selectReference?: (name: string) => void; // ui/ui.ts
    consumeDesiredVAEName?: (...args: unknown[]) => unknown[]; // ui/ui.ts
    consumeDesiredUNetName?: (...args: unknown[]) => unknown[]; // ui/ui.ts
    getDesiredCheckpointName?: () => string | null; // ui/ui.ts
    updateImg2imgResizeToTextAfterChangingImage?: () => unknown[]; // ui/ui.ts
    authFetch: (url: RequestInfo | URL, options?: RequestInit) => Promise<Response | undefined>; // ui/authWrap.ts
    controlInputMode?: (inputMode: string, ...args: unknown[]) => unknown[]; // ui/control.ts
    clearModelDetails?: () => void; // ui/civitai.ts
    downloadCivitModel?: (...args: unknown[]) => unknown[]; // ui/civitai.ts
    modelCardClick?: (id: string | number) => Promise<void>; // ui/civitai.ts
    startCivitAllDownload?: (evt: Event) => void; // ui/civitai.ts
    startCivitDownload?: (url: string, name: string, type: string, base: string, modelId: number, versionId: number) => void; // ui/civitai.ts
    clickDocsPage?: (page: string) => Promise<void>; // ui/docs.ts
    getDocsPage?: () => string; // ui/docs.ts
    getGitHubWikiPage?: () => string; // ui/docs.ts
    getGuidanceDocs?: (guider: string | { label?: string }) => void; // ui/guidance.ts
    applyStyles?: (styles?: string | string[] | ArrayLike<{ textContent: string | null }>) => string; // ui/extraNetworks.ts
    cardClicked?: (textToAdd: string) => void; // ui/extraNetworks.ts
    closeDetailsEN?: (...args: unknown[]) => unknown[]; // ui/extraNetworks.ts
    extraNetworksFilterVersion?: (event: Event) => void; // ui/extraNetworks.ts
    extraNetworksSearchButton?: (event: Event) => void; // ui/extraNetworks.ts
    getCardDetails?: (...args: unknown[]) => unknown[]; // ui/extraNetworks.ts
    quickSaveStyle?: () => void; // ui/extraNetworks.ts
    getENActivePage?: () => string; // ui/extraNetworks.ts
    refeshDetailsEN?: (args?: unknown) => void; // ui/extraNetworks.ts
    showCardDetails?: (event: Event) => void; // ui/extraNetworks.ts
    sortExtraNetworks?: (fixed?: string) => string; // ui/extraNetworks.ts
    selectStyle?: (name: string) => void; // ui/extraNetworks.ts
    getGallerySelectedUrl?: () => string | null; // ui/gallery.ts
    getGallerySelection?: () => { index: number; files: { name?: string; title?: string; src?: string }[] }; // ui/gallery.ts
    gallerySendImage?: (images: unknown) => (string | null)[]; // ui/gallery.ts
    setGallerySelection?: (index: number, options?: { send?: boolean }) => void; // ui/gallery.ts
    gallerySort?: (key: string) => void; // ui/gallery.ts
    clearCache?: () => void; // ui/gallery.ts
    disableGPU?: () => Promise<void>; // ui/gpu.ts
    startGPU?: () => Promise<void>; // ui/gpu.ts
    refreshHistory?: () => void; // ui/history.ts
    inputAccordionChecked?: (id: string, checked: boolean) => void; // ui/inputAccordion.ts
    debug?: (...args: unknown[]) => Promise<void>; // ui/logger.ts
    error?: (...args: unknown[]) => Promise<void>; // ui/logger.ts
    log?: (...args: unknown[]) => Promise<void>; // ui/logger.ts
    xhrGet?: (url: string, data: Record<string, string | number | boolean>, handler?: (json: unknown) => void, errorHandler?: (xhrObj: XMLHttpRequest) => void, ignore?: boolean, serverTimeout?: number) => void; // ui/logger.ts
    xhrPost?: (url: string, data: unknown, handler?: (json: unknown) => void, errorHandler?: (xhrObj: XMLHttpRequest) => void, ignore?: boolean, serverTimeout?: number) => void; // ui/logger.ts
    checkPaused?: (state?: boolean) => void; // ui/progressBar.ts
    requestInterrupt?: () => void; // ui/progressBar.ts
    randomId?: () => string; // ui/progressBar.ts
    requestProgress?: (id_task?: string, progressEl?: HTMLElement | null, galleryEl?: HTMLElement | null, atEnd?: (() => void) | null, onProgress?: ((progress: unknown) => void) | null, once?: boolean) => void; // ui/progressBar.ts
    deleteFile?: (filename: string) => Promise<void>; // ui/script.ts
    gradioApp: () => Document | Element | ShadowRoot; // ui/script.ts
    onAfterUiUpdate?: (callback: () => void) => void; // ui/script.ts
    onOptionsChanged?: (callback: () => void) => void; // ui/script.ts
    onUiLoaded?: (callback: () => void) => void; // ui/script.ts
    onUiReady?: (callback: () => void) => void; // ui/script.ts
    onUiTabChange?: (callback: () => void) => void; // ui/script.ts
    onUiUpdate?: (callback: () => void) => void; // ui/script.ts
    timer?: (name: string, elapsed: number) => Promise<void>; // ui/timers.ts
    markIfModified?: (setting_name: string, value: unknown) => void; // ui/settings.ts
    appendContextMenuOption?: (targetElementSelector: string, entryName: string, entryFunction: () => void, primary?: boolean) => string; // ui/contextMenus.ts
    generateForever?: (genbuttonid: string) => void; // ui/contextMenus.ts
    removeContextMenuOption?: (id: string) => void; // ui/contextMenus.ts

    // legacy module
    Hash?: any; // ui/js/sha256.ts
    HMAC?: any; // ui/js/sha256.ts
    hmac?: (key: string | Uint8Array, data: string | Uint8Array) => Uint8Array; // ui/js/sha256.ts
    hash?: (data: string | Uint8Array) => Uint8Array; // ui/js/sha256.ts

    // modernui
    logger?: HTMLElement; // extensions-builtin/sdnext-modernui/src/logger.ts
    setupLogger?: () => Promise<void>; // extensions-builtin/sdnext-modernui/src/logger.ts
    logPrettyPrint?: (...args: unknown[]) => string; // extensions-builtin/sdnext-modernui/src/logger.ts
    waitForUiReady?: () => Promise<void>; // extensions-builtin/sdnext-modernui/src/index.ts

    // kanvas
    Kanvas?: new (containerId: string, opts?: { width?: number; height?: number }) => {
      stages: { maxStages: number };
      getImage: (index: number, includeMask: boolean, includeAlpha: boolean) => { kanvas: true; image: string | null; mask: string | null } | null;
      destroy: () => void;
      initialize: (defaultWidth?: number, defaultHeight?: number) => void;
    }; // extensions-builtin/sdnext-kanvas/src/Kanvas.ts
    kanvas?: {
      stages: { maxStages: number };
      getImage: (index: number, includeMask: boolean, includeAlpha: boolean) => { kanvas: true; image: string | null; mask: string | null } | null;
    }; // extensions-builtin/sdnext-kanvas/src/Kanvas.ts
    loadFromURL?: (url: string) => unknown; // external
    getKanvasData?: () => { kanvas: true; image: string | null; mask: string | null } | null; // extensions-builtin/sdnext-kanvas/javascript/kanvas.mjs

    // browser api
    showDirectoryPicker: () => Promise<FileSystemDirectoryHandle>;
  }

  const opts: Window['opts'];
  function updateInput(target: EventTarget): void;
  function panzoom(element: HTMLElement, options?: Record<string, unknown>): { dispose: () => void };
  function hash(data: string | Uint8Array): any;

  interface GlobalThis {
    $: any;
    jQuery: any;
  }
}

export {};
