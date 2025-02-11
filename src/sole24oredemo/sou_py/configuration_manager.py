import os
from os.path import split

import yaml
import pprint
from pathlib import Path
from collections import OrderedDict
from sou_py.paths import SCHEDULE_DEFAULTS_PATH

from IPython.display import Markdown, display


# Custom presenter for single-quoted strings
def single_quoted_presenter(dumper, data):
    if isinstance(data, str):
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="'")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


yaml.add_representer(str, single_quoted_presenter)


class ConfigurationManager:
    def __init__(self, schedule_path, yaml_config_path):  #: Path | str,  #: Path | str
        self.operative_chain = os.path.basename(schedule_path)  # RADAR

        self.default_configs = {}
        self.config_dictionaries = {}

        # questo metodo va riadattato
        # legge dai .txt e scrive nel .yaml
        self._load_default_configs(SCHEDULE_DEFAULTS_PATH)

        self.schedules_configs = {}
        self._find_schedules_configs(schedule_path)

        # questo metodo va riadattato
        # legge dal .yaml e scrive nei .txt
        self.restore_original_config()

        self.yaml_config_path = yaml_config_path
        self.yaml_config = self._load_yaml_configuration(yaml_config_path)

    def _print_bold(self, string):
        print(f"\033[1m{string}\033[0m")

    def print_schedules_configurations(self):
        """
        Print the original configurations of schedules.

        This method displays the default values and descriptions of configuration parameters
        for all schedules. Each schedule's configuration is printed in bold for clarity.

        Args:
            None

        Returns:
            None

        """

        header = "The original configurations of the schedules are:\n"
        self._print_bold(header)
        for schedule_name in self.config_dictionaries.keys():
            self._print_bold(f"Schedule: {schedule_name}\n")
            for key, value in self.config_dictionaries[schedule_name].items():
                if "default" in self.config_dictionaries[schedule_name][key].keys():
                    print(f" - {key} = {value['default']} => {value['description']}")
            print("\n")

    def get_yaml_config(self):
        """
        Retrieve the loaded YAML configuration.

        This method provides access to the current in-memory YAML configuration
        stored in `self.yaml_config`.

        Args:
            None

        Returns:
            dict: The current YAML configuration as a Python dictionary.

        """

        return self.yaml_config

    def update_schedule_parameters(
        self, schedule_name: str, key_to_modify: str, new_value
    ):
        """
        Update a specific parameter in a schedule configuration.

        This method modifies a given key in the configuration of a specified schedule,
        updates the corresponding file, and logs the changes.

        Args:
            schedule_name (str): The name of the schedule to update.
            key_to_modify (str): The key in the schedule configuration to modify.
            new_value: The new value to set for the specified key.

        Returns:
            None
        """

        if (
            Path(self.operative_chain) / schedule_name
            not in self.schedules_configs.keys()
        ):
            print(f"{schedule_name} not found in the loaded schedules.")
            return

        config_path = self.schedules_configs[Path(self.operative_chain) / schedule_name]

        config = self._read_config(config_path)

        if key_to_modify not in config.keys():
            print(f"{key_to_modify} not found in the configuration of {schedule_name}")
            return

        config[key_to_modify] = new_value

        self._write_config(config_path, config)

        print(
            f"[{schedule_name} -> parameters.txt] {key_to_modify} updated to {new_value}."
        )

    def find_deepest_key_path(self, data, target_key, path=None):
        """
        Find the deepest path to a specific key in a nested dictionary.

        This method recursively searches a nested dictionary for the target key
        and returns the deepest path (list of keys) where the key is located.

        Args:
            data (dict): The dictionary to search.
            target_key (str): The key to find.
            path (list, optional): The current path during recursion. Defaults to None.

        Returns:
            list: The deepest path to the target key as a list of keys, or None if the key is not found.

        """

        if path is None:
            path = []
        if not isinstance(data, dict):
            return None
        deepest_path = None

        for key, value in data.items():
            current_path = path + [key]

            if key == target_key:
                deepest_path = current_path

            if isinstance(value, dict):
                result = self.find_deepest_key_path(value, target_key, current_path)
                if result and (deepest_path is None or len(result) > len(deepest_path)):
                    deepest_path = result

        return deepest_path

    def key_exists(self, data, target_key):
        """
        Check if a key exists in a nested dictionary.

        This method recursively searches a nested dictionary to determine if
        the target key exists anywhere within it.

        Args:
            data (dict): The dictionary to search.
            target_key (str): The key to check for existence.

        Returns:
            bool: True if the key exists, False otherwise.
        """

        if not isinstance(data, dict):
            return False

        if target_key in data:
            return True

        for key, value in data.items():
            if isinstance(value, dict) and self.key_exists(value, target_key):
                return True

        return False

    def count_key_instances(self, data, target_key):
        """
        Count the occurrences of a key in a nested dictionary.

        This method recursively searches a nested dictionary and counts
        how many times the target key appears.

        Args:
            data (dict): The dictionary to search.
            target_key (str): The key to count.

        Returns:
            int: The total number of occurrences of the target key.
        """

        count = 0
        if not isinstance(data, dict):
            return 0
        if target_key in data:
            count += 1
        for key, value in data.items():
            if isinstance(value, dict):
                count += self.count_key_instances(value, target_key)
        return count

    def restore_original_config(self):
        """
        Restore the schedule configurations to their original state.

        This method resets the current schedule configurations (`self.schedules_configs`) to their
        original values, based on the default configuration (`self.default_configs`). For each
        configuration key in `self.schedules_configs`, it traverses the corresponding structure
        in `self.default_configs` to find and apply the default values. Updated configurations
        are written back using the `_write_config` method.

        Args:
            None
        Returns:
            None
        """

        configs_items = list(self.schedules_configs.keys())

        for config in configs_items:
            default_dict = self.default_configs.copy()
            splitted_config = config.parts
            for elem in splitted_config:
                if elem in default_dict.keys():
                    default_dict = default_dict[elem]
                else:
                    break
            else:

                self._write_config(self.schedules_configs[config], default_dict)
            continue

    print("Schedule configurations restored to original.")

    def update_yaml_configuration(self, **kwargs):
        """
        Update and save the YAML configuration.

        This method updates specific keys in the existing YAML configuration, converts any
        `Path` objects to strings (for YAML compatibility), and saves the updated configuration
        back to the file. After saving, it prints the updated YAML configuration.

        Args:
            **kwargs: Key-value pairs to update in the YAML configuration. Keys represent
                      the configuration keys to update, and values are the new data to set.
                      `Path` objects in the values are automatically converted to strings.

        Returns:
            None

        Raises:
            OSError: If there are issues writing to the YAML file.
        """

        def _convert_paths(data):
            if isinstance(data, dict):
                return {k: _convert_paths(v) for k, v in data.items()}
            elif isinstance(data, Path):
                return str(data)
            return data

        for key, value in kwargs.items():
            self.yaml_config[key] = _convert_paths(value)

        with open(self.yaml_config_path, "w", encoding="UTF-8") as file:
            yaml.safe_dump(
                self.yaml_config, file, default_flow_style=False, sort_keys=False
            )

        self._print_yaml_config()

    def _read_config(self, file_path):
        config = {}
        with open(file_path, "r") as file:
            lines = file.readlines()
            for idx, line in enumerate(lines):
                # store the break lines
                if line.strip() == "":
                    config[f"break-{idx}"] = ""
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().replace("\n", "")
                config[key] = value
        return config

    def _get_original_config(self, default_config: dict):
        config = {}
        for key, value in default_config.items():
            config[key] = value.get("default", None)
            if config[key] is None:
                print(f"key {key}, value {value}")
        return config

    # def _printmd(self, string):
    #     display(Markdown(string))

    def _print_yaml_config(self):
        header = "The updated yaml configuration of the operative chain is:\n"
        self._print_bold(header)
        for key, value in self.yaml_config.items():
            print(f"{key} = {value}")
        print("\n")

    def _load_default_configs(self, yaml_config_path):
        with open(yaml_config_path, "r", encoding="UTF-8") as file:
            self.default_configs = yaml.safe_load(file)["defaults"]

    def _find_schedules_configs(self, schedule_path):  # schedule_path: Path | str):
        """
        Given the following schedule structure:
        TEST_CLUTT/
            Declutter/
                parameters.txt
                ClutterMap/
                    parameters.txt
                PHIDP/
                    parameters.txt
                ...
        This function will find all the parameters.txt files and stores them into the self.schedules dict.
        """
        schedule_path = Path(schedule_path)
        for root, _, files in os.walk(schedule_path):
            for file in files:
                if file == "parameters.txt":
                    schedule_rel_path = schedule_path.stem / Path(root).relative_to(
                        schedule_path
                    )
                    self.schedules_configs[schedule_rel_path] = os.path.join(
                        root, "parameters.txt"
                    )

    def _write_config(self, file_path, config):
        with open(file_path, "w", encoding="UTF-8") as file:
            for key, value in config.items():
                if not isinstance(value, dict):
                    file.write(f"{key} = {value}\n")
                elif "default" in value.keys():
                    file.write(f"{key} = {value['default']}\n")
                else:
                    continue

    def _load_yaml_configuration(
        self, yaml_config_path
    ):  # yaml_config_path: Path | str):
        with open(yaml_config_path, "r", encoding="UTF-8") as file:
            data = yaml.safe_load(file)
        return data

    def __del__(self):
        """
        Restore all the original parameters.txt files

        Returns:
             None
        """
        print("DEL CALLED!")
        self.restore_original_config()


# if __name__=="__main__":
#     cm = ConfigurationManager(
#         "/home/leobaro/workspace/labs/datamet/datamet_data/data/schedules/RADAR_0/TEST_CLUTT",
#         "/home/leobaro/workspace/labs/datamet/nbs/DECLUTTER/config.yaml"
#     )
#     print(cm.schedules_configs)
#     print(cm.default_configs.keys())
#     print(cm.default_configs["RHOHV"])
#     cm.print_schedules_default_configurations()
#     cm.restore_original_config()
