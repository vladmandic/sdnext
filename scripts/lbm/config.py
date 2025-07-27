import json
import os
import warnings
from dataclasses import asdict, field
from typing import Any, Dict, Union
import yaml
from pydantic import ValidationError
from pydantic.dataclasses import dataclass
from yaml import safe_load


@dataclass
class BaseConfig:
    """This is the BaseConfig class which defines all the useful loading and saving methods
    of the configs"""

    name: str = field(init=False)

    def __post_init__(self):
        self.name = self.__class__.__name__

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BaseConfig":
        """Creates a BaseConfig instance from a dictionnary

        Args:
            config_dict (dict): The Python dictionnary containing all the parameters

        Returns:
            :class:`BaseConfig`: The created instance
        """
        try:
            config = cls(**config_dict)
        except (ValidationError, TypeError) as e:
            raise e
        return config

    @classmethod
    def _dict_from_json(cls, json_path: Union[str, os.PathLike]) -> Dict[str, Any]:
        try:
            with open(json_path) as f:
                try:
                    config_dict = json.load(f)
                    return config_dict

                except (TypeError, json.JSONDecodeError) as e:
                    raise TypeError(
                        f"File {json_path} not loadable. Maybe not json ? \n"
                        f"Catch Exception {type(e)} with message: " + str(e)
                    ) from e

        except FileNotFoundError:
            raise FileNotFoundError(
                f"Config file not found. Please check path '{json_path}'"
            )

    @classmethod
    def from_json(cls, json_path: str) -> "BaseConfig":
        """Creates a BaseConfig instance from a JSON config file

        Args:
            json_path (str): The path to the json file containing all the parameters

        Returns:
            :class:`BaseConfig`: The created instance
        """
        config_dict = cls._dict_from_json(json_path)

        config_name = config_dict.pop("name")

        if cls.__name__ != config_name:
            warnings.warn(
                f"You are trying to load a "
                f"`{ cls.__name__}` while a "
                f"`{config_name}` is given."
            )

        return cls.from_dict(config_dict)

    def to_dict(self) -> dict:
        """Transforms object into a Python dictionnary

        Returns:
            (dict): The dictionnary containing all the parameters"""
        return asdict(self)

    def to_json_string(self):
        """Transforms object into a JSON string

        Returns:
            (str): The JSON str containing all the parameters"""
        return json.dumps(self.to_dict())

    def save_json(self, file_path: str):
        """Saves a ``.json`` file from the dataclass

        Args:
            file_path (str): path to the file
        """
        with open(os.path.join(file_path), "w", encoding="utf-8") as fp:
            fp.write(self.to_json_string())

    def save_yaml(self, file_path: str):
        """Saves a ``.yaml`` file from the dataclass

        Args:
            file_path (str): path to the file
        """
        with open(os.path.join(file_path), "w", encoding="utf-8") as fp:
            yaml.dump(self.to_dict(), fp)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "BaseConfig":
        """Creates a BaseConfig instance from a YAML config file

        Args:
            yaml_path (str): The path to the yaml file containing all the parameters

        Returns:
            :class:`BaseConfig`: The created instance
        """
        with open(yaml_path, "r") as f:
            try:
                config_dict = safe_load(f)
            except yaml.YAMLError as e:
                raise yaml.YAMLError(
                    f"File {yaml_path} not loadable. Maybe not yaml ? \n"
                    f"Catch Exception {type(e)} with message: " + str(e)
                ) from e

        config_name = config_dict.pop("name")

        if cls.__name__ != config_name:
            warnings.warn(
                f"You are trying to load a "
                f"`{ cls.__name__}` while a "
                f"`{config_name}` is given."
            )

        return cls.from_dict(config_dict)
