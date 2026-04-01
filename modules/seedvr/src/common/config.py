import importlib
from typing import Any, Callable, List, Union

try:
    from installer import install
    install('omegaconf')
    from omegaconf import DictConfig, ListConfig, OmegaConf
except Exception as e:
    raise ImportError(f"Failed to import omegaconf. Error: {e}") from e

try:
    OmegaConf.register_new_resolver("eval", eval)
except Exception as e:
    if "already registered" not in str(e):
        raise


def load_config(path: str, argv: List[str] = None):
    """
    Load a configuration. Will resolve inheritance.
    """

    config = OmegaConf.load(path)
    if argv is not None:
        config_argv = OmegaConf.from_dotlist(argv)
        config = OmegaConf.merge(config, config_argv)
    config = resolve_recursive(config, resolve_inheritance)
    return config


def resolve_recursive(
    config: Any,
    resolver: Callable[[Any], Any],
) -> Any:
    config = resolver(config)
    if isinstance(config, DictConfig):
        for k in config.keys():
            v = config.get(k)
            if isinstance(v, (DictConfig, ListConfig)):
                config[k] = resolve_recursive(v, resolver)
    if isinstance(config, ListConfig):
        for i in range(len(config)):
            v = config.get(i)
            if isinstance(v, (DictConfig, ListConfig)):
                config[i] = resolve_recursive(v, resolver)
    return config


def resolve_inheritance(config: Any) -> Any:
    """
    Recursively resolve inheritance if the config contains:
    __inherit__: path/to/parent.yaml or a ListConfig of such paths.
    """
    if isinstance(config, DictConfig):
        inherit = config.pop("__inherit__", None)

        if inherit:
            inherit_list = inherit if isinstance(inherit, ListConfig) else [inherit]

            parent_config = None
            for parent_path in inherit_list:
                assert isinstance(parent_path, str)
                parent_config = (
                    load_config(parent_path)
                    if parent_config is None
                    else OmegaConf.merge(parent_config, load_config(parent_path))
                )

            if len(config.keys()) > 0:
                config = OmegaConf.merge(parent_config, config)
            else:
                config = parent_config
    return config


def import_item(path: Union[str, List[str]], name: str) -> Any:
    """
    Import a python item with fallback support.

    Args:
        path: Single path string or list of paths to try (fallback order)
        name: Class/function name to import

    Returns:
        Imported object

    Example:
        import_item("path.to.file", "MyClass") -> MyClass
        import_item(["path1.to.file", "path2.to.file"], "MyClass") -> MyClass (first working path)
    """
    if isinstance(path, str):
        # Single path - original behavior
        return getattr(importlib.import_module(path), name)

    elif isinstance(path, (list, ListConfig)):
        # Multiple paths - try each until one works
        last_error = None
        for single_path in path:
            try:
                return getattr(importlib.import_module(single_path), name)
            except ImportError as e:
                last_error = e
                continue

        # If we get here, none of the paths worked
        raise ImportError(f"Could not import '{name}' from any of the paths: {path}. Last error: {last_error}")

    else:
        raise ValueError(f"Path must be string or list of strings, got: {type(path)}")


def create_object(config: Any) -> Any:
    """
    Create an object from config.
    The config is expected to contains the following:
    __object__:
      path: path.to.module
      name: MyClass
      args: as_config | as_params (default to as_config)
    """

    item = import_item(
        path=config.__object__.path,
        name=config.__object__.name,
    )
    args = config.__object__.get("args", "as_config")
    if args == "as_config":
        return item(config)
    if args == "as_params":
        config = OmegaConf.to_object(config)
        config.pop("__object__")
        return item(**config)
    raise NotImplementedError(f"Unknown args type: {args}")
