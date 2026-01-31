from typing import List


def xyz_grid_enum(option: str = "") -> List[dict]:
    from scripts.xyz import xyz_grid_classes
    options = []
    for x in xyz_grid_classes.axis_options:
        _option = {
            'label': x.label,
            'type': x.type.__name__,
            'cost': x.cost,
            'choices': x.choices is not None,
        }
        if len(option) == 0:
            options.append(_option)
        else:
            if x.label.lower().startswith(option.lower()) or x.label.lower().endswith(option.lower()):
                if callable(x.choices):
                    _option['choices'] = x.choices()
                options.append(_option)
    return options


def register_api():
    from modules.shared import api as api_instance
    api_instance.add_api_route("/sdapi/v1/xyz-grid", xyz_grid_enum, methods=["GET"], response_model=List[dict])
