

def xyz_grid_enum(option: str = "") -> list[dict]:
    """List XYZ grid axis options. Optionally filter by label prefix/suffix and expand choices."""
    from scripts.xyz import xyz_grid_classes # pylint: disable=no-name-in-module
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


def register_api(app):
    app.add_api_route("/sdapi/v1/xyz-grid", xyz_grid_enum, methods=["GET"], response_model=list[dict], tags=["Scripts"])
