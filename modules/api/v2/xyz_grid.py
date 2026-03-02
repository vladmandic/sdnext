import re
from typing import Optional
from itertools import permutations
from pydantic import BaseModel, Field
from fastapi import APIRouter, Query
import numpy as np


router = APIRouter(prefix="/sdapi/v2/xyz-grid", tags=["XYZ Grid"])


# --- Models ---

class XyzAxisInfo(BaseModel):
    label: str
    type: str = Field(description="int, float, str, str_permutations, or bool")
    cost: float
    category: str = Field(description="Extracted from [Category] prefix in label")
    has_choices: bool
    choices: Optional[list[str]] = None


class ResXyzAxes(BaseModel):
    items: list[XyzAxisInfo]
    categories: list[str] = Field(description="Unique sorted category names")


class ReqXyzValidate(BaseModel):
    axis_type: str
    values: str


class ResXyzValidate(BaseModel):
    ok: bool
    resolved: list
    count: int
    errors: list[str]


class XyzAxisInput(BaseModel):
    type: str = Field(description="Axis label, e.g. '[Param] Steps'")
    values: str = Field(description="Comma-separated values or range syntax")


class ReqXyzPreview(BaseModel):
    x_axis: Optional[XyzAxisInput] = None
    y_axis: Optional[XyzAxisInput] = None
    z_axis: Optional[XyzAxisInput] = None
    steps: int = Field(default=20, description="Base steps for estimation")


class ResXyzPreview(BaseModel):
    ok: bool
    dimensions: dict
    total_cells: int
    total_steps: int
    execution_order: list[str]
    x_values: list
    y_values: list
    z_values: list
    errors: list[str]


# --- Helpers ---

_re_category = re.compile(r'^\[([^\]]+)\]\s*')


def _get_axis_options():
    from scripts.xyz.xyz_grid_classes import axis_options
    return axis_options


def _find_axis(label: str):
    for opt in _get_axis_options():
        if opt.label == label:
            return opt
    return None


def _resolve_axis_values(axis_label: str, values_str: str) -> tuple[list, list[str]]:
    from scripts.xyz.xyz_grid_shared import re_range, re_plain_comma, restore_comma, str_permutations

    opt = _find_axis(axis_label)
    if opt is None:
        return [], [f"Unknown axis: {axis_label}"]

    if opt.label == 'Nothing':
        return [0], []

    valslist = [restore_comma(x.strip()) for x in re_plain_comma.split(values_str) if x]
    errors = []

    if opt.type == int:
        resolved = []
        for val in valslist:
            try:
                m = re_range.fullmatch(val)
                if m is not None:
                    start_val = int(m.group(1))
                    end_val = int(m.group(2))
                    num = int(m.group(3)) if m.group(3) is not None else int(end_val - start_val)
                    resolved += [int(x) for x in np.linspace(start=start_val, stop=end_val, num=max(2, num)).tolist()]
                else:
                    resolved.append(int(val))
            except Exception as e:
                errors.append(f"Invalid int value '{val}': {e}")
        return resolved, errors

    if opt.type == float:
        resolved = []
        for val in valslist:
            try:
                m = re_range.fullmatch(val)
                if m is not None:
                    start_val = float(m.group(1))
                    end_val = float(m.group(2))
                    num = int(m.group(3)) if m.group(3) is not None else int(end_val - start_val)
                    resolved += [round(float(x), 2) for x in np.linspace(start=start_val, stop=end_val, num=max(2, num)).tolist()]
                else:
                    resolved.append(float(val))
            except Exception as e:
                errors.append(f"Invalid float value '{val}': {e}")
        return resolved, errors

    if opt.type == str_permutations:
        return [list(p) for p in permutations(valslist)], errors

    # str or bool — return as-is after type conversion
    try:
        resolved = [opt.type(x) for x in valslist]
    except Exception as e:
        errors.append(f"Type conversion failed: {e}")
        resolved = valslist
    return resolved, errors


# --- Routes ---

@router.get("/axes", response_model=ResXyzAxes)
async def get_axes(expand: Optional[str] = Query(default=None, description="Axis label to expand choices for")):
    axis_options = _get_axis_options()
    items = []
    categories = set()
    for opt in axis_options:
        if opt.label == 'Nothing':
            continue
        m = _re_category.match(opt.label)
        category = m.group(1) if m else "Other"
        categories.add(category)
        info = XyzAxisInfo(
            label=opt.label,
            type=opt.type.__name__,
            cost=opt.cost,
            category=category,
            has_choices=opt.choices is not None,
        )
        if expand and opt.label == expand and opt.choices is not None:
            try:
                raw = opt.choices()
                info.choices = [str(c) for c in raw]
            except Exception:
                info.choices = []
        items.append(info)
    return ResXyzAxes(items=items, categories=sorted(categories))


@router.post("/validate", response_model=ResXyzValidate)
async def validate_axis(req: ReqXyzValidate):
    resolved, errors = _resolve_axis_values(req.axis_type, req.values)
    return ResXyzValidate(ok=len(errors) == 0, resolved=resolved, count=len(resolved), errors=errors)


@router.post("/preview", response_model=ResXyzPreview)
async def preview_grid(req: ReqXyzPreview):
    all_errors: list[str] = []
    axis_data: dict[str, tuple[list, float]] = {}

    for axis_name, axis_input in [('x', req.x_axis), ('y', req.y_axis), ('z', req.z_axis)]:
        if axis_input and axis_input.type and axis_input.values.strip():
            resolved, errs = _resolve_axis_values(axis_input.type, axis_input.values)
            all_errors.extend(errs)
            opt = _find_axis(axis_input.type)
            cost = opt.cost if opt else 0.0
            axis_data[axis_name] = (resolved, cost)
        else:
            axis_data[axis_name] = ([0], 0.0)

    x_vals, x_cost = axis_data['x']
    y_vals, y_cost = axis_data['y']
    z_vals, z_cost = axis_data['z']

    total_cells = len(x_vals) * len(y_vals) * len(z_vals)

    # Compute total steps: if an axis is Steps, sum its values instead of multiplying
    total_steps = 0
    steps_axis = None
    for axis_name, axis_input in [('x', req.x_axis), ('y', req.y_axis), ('z', req.z_axis)]:
        if axis_input and '[Param] Steps' in axis_input.type:
            steps_axis = axis_name
    if steps_axis == 'x':
        total_steps = sum(v for v in x_vals if isinstance(v, (int, float))) * len(y_vals) * len(z_vals)
    elif steps_axis == 'y':
        total_steps = sum(v for v in y_vals if isinstance(v, (int, float))) * len(x_vals) * len(z_vals)
    elif steps_axis == 'z':
        total_steps = sum(v for v in z_vals if isinstance(v, (int, float))) * len(x_vals) * len(y_vals)
    else:
        total_steps = req.steps * total_cells

    # Hires steps handling
    for axis_name, axis_input in [('x', req.x_axis), ('y', req.y_axis), ('z', req.z_axis)]:
        if axis_input and '[Refine] Hires steps' in axis_input.type:
            vals = axis_data[axis_name][0]
            other_counts = total_cells // max(1, len(vals))
            total_steps += sum(v for v in vals if isinstance(v, (int, float))) * other_counts

    # Execution order: sort by cost descending (highest cost = outermost loop)
    axes_with_cost = [('x', x_cost), ('y', y_cost), ('z', z_cost)]
    axes_with_cost.sort(key=lambda a: a[1], reverse=True)
    execution_order = [a[0] for a in axes_with_cost]

    return ResXyzPreview(
        ok=len(all_errors) == 0,
        dimensions={'x': len(x_vals), 'y': len(y_vals), 'z': len(z_vals)},
        total_cells=total_cells,
        total_steps=int(total_steps),
        execution_order=execution_order,
        x_values=x_vals,
        y_values=y_vals,
        z_values=z_vals,
        errors=all_errors,
    )


# --- Executor ---

def execute_xyz_grid(params: dict, job_id: str) -> dict:
    from modules.api.v2.executors import execute_generate

    # Build the positional script_args array that the xyz_grid script's run() expects
    axis_options = _get_axis_options()
    label_to_index = {opt.label: i for i, opt in enumerate(axis_options)}

    def axis_index(axis_input: Optional[dict]) -> int:
        if not axis_input or not axis_input.get('type'):
            return 0  # "Nothing"
        label = axis_input['type']
        return label_to_index.get(label, 0)

    x_axis = params.get('x_axis') or {}
    y_axis = params.get('y_axis') or {}
    z_axis = params.get('z_axis') or {}

    script_args = [
        axis_index(x_axis), x_axis.get('values', ''), '',  # x_type, x_values, x_values_dropdown
        axis_index(y_axis), y_axis.get('values', ''), '',  # y_type, y_values, y_values_dropdown
        axis_index(z_axis), z_axis.get('values', ''), '',  # z_type, z_values, z_values_dropdown
        False,                                               # csv_mode
        params.get('draw_legend', True),                     # draw_legend
        params.get('random_seeds', False),                   # no_fixed_seeds
        params.get('include_grid', True),                    # include_grid
        params.get('include_subgrids', False),               # include_subgrids
        params.get('include_images', True),                  # include_images
        params.get('include_time', False),                   # include_time
        params.get('include_text', False),                   # include_text
        params.get('margin_size', 0),                        # margin_size
        False,                                               # create_video
        'None',                                              # video_type
        2.0,                                                 # video_duration
        False,                                               # video_loop
        0,                                                   # video_pad
        0,                                                   # video_interpolate
    ]

    # Remove xyz-specific keys and set up the generate request
    gen_params = {k: v for k, v in params.items() if k not in (
        'x_axis', 'y_axis', 'z_axis', 'draw_legend', 'include_grid',
        'include_subgrids', 'include_images', 'include_time', 'include_text',
        'margin_size', 'random_seeds',
    )}
    gen_params['type'] = 'generate'
    gen_params['script_name'] = 'XYZ Grid Script'
    gen_params['script_args'] = script_args

    return execute_generate(gen_params, job_id)
