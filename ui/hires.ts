import { gradioApp } from './script';

export function onCalcResolutionHires(
  width: number,
  height: number,
  hrScale: number,
  hrResizeX: number,
  hrResizeY: number,
  hrUpscaler: string,
): [number, number, number, number, number, string] {
  const setInactive = (elem: HTMLElement | null, inactive: boolean): void => {
    if (elem) elem.classList.toggle('inactive', !!inactive);
  };
  const hrUpscaleBy = gradioApp().getElementById('txt2img_hr_scale');
  const hrResizeXElem = gradioApp().getElementById('txt2img_hr_resize_x');
  const hrResizeYElem = gradioApp().getElementById('txt2img_hr_resize_y');
  setInactive(hrUpscaleBy, hrResizeX > 0 || hrResizeY > 0);
  setInactive(hrResizeXElem, hrResizeX === 0);
  setInactive(hrResizeYElem, hrResizeY === 0);
  return [width, height, hrScale, hrResizeX, hrResizeY, hrUpscaler];
}
