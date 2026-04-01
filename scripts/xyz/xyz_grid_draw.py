import time
from copy import copy
from PIL import Image
from modules.image.grid import GridAnnotation
from modules import shared, images, processing
from modules.logger import log
from modules.image.util import draw_text


def draw_xyz_grid(p, xs, ys, zs, x_labels, y_labels, z_labels, cell, draw_legend, include_lone_images, include_sub_grids, first_axes_processed, second_axes_processed, margin_size, no_grid: False, include_time: False, include_text: False): # pylint: disable=unused-argument
    x_texts = [[GridAnnotation(x)] for x in x_labels]
    y_texts = [[GridAnnotation(y)] for y in y_labels]
    z_texts = [[GridAnnotation(z)] for z in z_labels]
    list_size = (len(xs) * len(ys) * len(zs))
    processed_result = None

    t0 = time.time()
    i = 0

    def process_cell(x, y, z, ix, iy, iz):
        nonlocal processed_result, i
        i += 1
        log.debug(f'XYZ grid process: x={ix+1}/{len(xs)} y={iy+1}/{len(ys)} z={iz+1}/{len(zs)} total={i/list_size:.2f}')

        def index(ix, iy, iz):
            return ix + iy * len(xs) + iz * len(xs) * len(ys)

        res = cell(x, y, z, ix, iy, iz)
        processed: processing.Processed = res[0] if isinstance(res, tuple) else res
        elapsed = res[1] if isinstance(res, tuple) else 0
        if processed_result is None:
            processed_result = copy(processed)
            if processed_result is None:
                log.error('XYZ grid: no processing results')
                return processing.Processed(p, [])
            processed_result.images = [None] * list_size
            processed_result.all_prompts = [None] * list_size
            processed_result.all_seeds = [None] * list_size
            processed_result.infotexts = [None] * list_size
            processed_result.time = [0] * list_size
            processed_result.index_of_first_image = 1
        idx = index(ix, iy, iz)
        if processed is not None and processed.images:
            processed_result.images[idx] = processed.images[0]
            overlay_text = ''
            if include_text:
                if len(x_labels[ix]) > 0:
                    overlay_text += f'{x_labels[ix]}\n'
                if len(y_labels[iy]) > 0:
                    overlay_text += f'{y_labels[iy]}\n'
                if len(z_labels[iz]) > 0:
                    overlay_text += f'{z_labels[iz]}\n'
            if include_time:
                overlay_text += f'Time: {elapsed:.2f}'
            if len(overlay_text) > 0:
                processed_result.images[idx] = draw_text(processed_result.images[idx], overlay_text)
            processed_result.all_prompts[idx] = processed.prompt
            processed_result.all_seeds[idx] = processed.seed
            processed_result.infotexts[idx] = processed.infotexts[0]
            processed_result.time[idx] = round(elapsed, 2)
        else:
            cell_mode = "P"
            cell_size = (processed_result.width, processed_result.height)
            if processed_result.images[0] is not None:
                cell_mode = processed_result.images[0].mode
                cell_size = processed_result.images[0].size
            processed_result.images[idx] = Image.new(cell_mode, cell_size)
        shared.state.nextjob()

    if first_axes_processed == 'x':
        for ix, x in enumerate(xs):
            if second_axes_processed == 'y':
                for iy, y in enumerate(ys):
                    for iz, z in enumerate(zs):
                        process_cell(x, y, z, ix, iy, iz)
            else:
                for iz, z in enumerate(zs):
                    for iy, y in enumerate(ys):
                        process_cell(x, y, z, ix, iy, iz)
    elif first_axes_processed == 'y':
        for iy, y in enumerate(ys):
            if second_axes_processed == 'x':
                for ix, x in enumerate(xs):
                    for iz, z in enumerate(zs):
                        process_cell(x, y, z, ix, iy, iz)
            else:
                for iz, z in enumerate(zs):
                    for ix, x in enumerate(xs):
                        process_cell(x, y, z, ix, iy, iz)
    elif first_axes_processed == 'z':
        for iz, z in enumerate(zs):
            if second_axes_processed == 'x':
                for ix, x in enumerate(xs):
                    for iy, y in enumerate(ys):
                        process_cell(x, y, z, ix, iy, iz)
            else:
                for iy, y in enumerate(ys):
                    for ix, x in enumerate(xs):
                        process_cell(x, y, z, ix, iy, iz)

    if not processed_result:
        log.error("XYZ grid: failed to initialize processing")
        return processing.Processed(p, [])
    elif not any(processed_result.images):
        log.error("XYZ grid: failed to return processed image")
        return processing.Processed(p, [])

    t1 = time.time()
    grid = None
    for i in range(len(zs)): # create grid
        idx0 = (i * len(xs) * len(ys)) + i # starting index of images in subgrid
        idx1 = (len(xs) * len(ys)) + idx0  # ending index of images in subgrid
        to_process = processed_result.images[idx0:idx1]
        w, h = max(i.width for i in to_process if i is not None), max(i.height for i in to_process if i is not None)
        if w is None or h is None or w == 0 or h == 0:
            log.error("XYZ grid: failed get valid image")
            continue
        if (not no_grid or include_sub_grids) and images.check_grid_size(to_process):
            grid = images.image_grid(to_process, rows=len(ys))
            if draw_legend:
                grid = images.draw_grid_annotations(grid, w, h, x_texts, y_texts, margin_size, title=z_texts[i])
            processed_result.images.insert(i, grid)
            processed_result.all_prompts.insert(i, processed_result.all_prompts[idx0])
            processed_result.all_seeds.insert(i, processed_result.all_seeds[idx0])
            processed_result.infotexts.insert(i, processed_result.infotexts[idx0])
    if len(zs) > 1 and not no_grid and images.check_grid_size(processed_result.images[:len(zs)]): # create grid-of-grids
        grid = images.image_grid(processed_result.images[:len(zs)], rows=1)
        processed_result.images.insert(0, grid)
        processed_result.all_prompts.insert(0, processed_result.all_prompts[0])
        processed_result.all_seeds.insert(0, processed_result.all_seeds[0])
        processed_result.infotexts.insert(0, processed_result.infotexts[0])

    t2 = time.time()
    log.info(f'XYZ grid complete: images={list_size} results={len(processed_result.images)} size={grid.size if grid is not None else None} time={t1-t0:.2f} save={t2-t1:.2f}')
    p.skip_processing = True
    return processed_result
