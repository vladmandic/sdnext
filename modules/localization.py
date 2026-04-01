import json
import modules.errors as errors
from modules.logger import log


localizations = {}


def list_localizations(dirname): # pylint: disable=unused-argument
    localizations.clear()
    """
    for file in os.listdir(dirname):
        fn, ext = os.path.splitext(file)
        if ext.lower() != ".json":
            continue

        localizations[fn] = os.path.join(dirname, file)

    from modules import scripts
    for file in scripts.list_scripts("localizations", ".json"):
        fn, ext = os.path.splitext(file.filename)
        localizations[fn] = file.path
    """
    return localizations


def localization_js(current_localization_name):
    fn = localizations.get(current_localization_name, None)
    data = {}
    if fn is not None:
        try:
            with open(fn, encoding="utf8") as file:
                data = json.load(file)
        except Exception as e:
            log.error(f"Error loading localization from {fn}:")
            errors.display(e, 'localization')

    return f"var localization = {json.dumps(data)}\n"
