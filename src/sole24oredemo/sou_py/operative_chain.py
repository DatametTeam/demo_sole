import logging
import os
import re
import time
from pathlib import Path

from sou_py import dpg
from sou_py.dpg.log import initialize_general_log, log_message

# from sou_py.dpg.log import logger
from sou_py.paths import (
    DATAMET_DATA_PATH,
    DATAMET_RADVIEW_PATH,
    DATAMET_ROOT_PATH,
)
from sou_py.data_utils import empty_products_files, fill_products_files, fill_assessment_products
from sou_py.sites_utils import get_sites
from sou_py.dpg.scheduler__define import Scheduler
from sou_py.dpg.utility import delete_folders


class OperativeChain:

    @staticmethod
    def run(
            rv_name: str,
            rv_date: str,
            rv_time: str,
            rv_interactive: bool,
            rv_remove: bool,
            rv_schedule: str | None,
            rv_center: str | list | None,
            schedule_config: dict,
            data_folders_path: dict = None,
            rv_raw_path_prefix: str | Path = None,
            rv_raw_path_suffix: str = None,
            extra_prd_path: str = None,
            radar_prd_path: str = None,
            rv_sampled_path_prefix: str = None,
            rv_sampled_path_suffix: str = None,
            print_results_folder=False):
        """
        Execute the operative chain with the given parameters.

        Args:
            rv_name (str): The name of the operative chain (RADAR, RADAR_0, ..).
            rv_raw_path_prefix (str): The prefix of the path where the input files are located. The path is built as:
            DATAMET_DATA_PATH / rv_raw_path_prefix / Y / M / D / h+m / rv_raw_path_suffix.
            rv_raw_path_suffix (str): The suffix of the path where the input files are located. The path is built as:
            DATAMET_DATA_PATH / rv_raw_path_prefix / Y / M / D / h+m / rv_raw_path_suffix.
            rv_date (str): The date in the format DD-MM-YYYY.
            rv_time (str): The time in the format HH:MM.
            rv_interactive (bool): this is a flag that decide if run sites in parallel (false) or not
            rv_remove (bool): Removes execution files previously created
            rv_schedule (str): To execute a single specific schedule (and its subschedules), regardless it's active
            or not, ignoring all the others.
            rv_center (str): Sites that will bexecuted by the schedule
            schedule_config (dict): It's used to update the schedule configuration file. The dictionary must contain
            the key-value pairs to update in the schedule configuration file.

        Returns:
            results (dict): A dictionary containing the results of the operative chain execution, in terms of nodes.
        """
        # TODO: rv_center should be a list of string

        # from sou_py.dpg.log import logger
        logger = initialize_general_log()

        raw_path, sampled_path = OperativeChain._configure(
            schedule_config,
            rv_name,
            rv_raw_path_prefix,
            rv_date,
            rv_time,
            rv_raw_path_suffix,
            extra_prd_path,
            radar_prd_path,
            rv_sampled_path_prefix,
            rv_sampled_path_suffix,
            data_folders_path=data_folders_path,
        )

        # OperativeChain._set_environment(raw_path)
        OperativeChain._set_cwd(DATAMET_ROOT_PATH)

        log_message("Current configuration:")
        log_message("---------------------------------------------")
        log_message(f"rv_name: {rv_name}")
        log_message(f"rv_date: {rv_date}")
        log_message(f"rv_time: {rv_time}")
        log_message(f"rv_remove: {rv_remove}")
        log_message(f"rv_interactive: {rv_interactive}")
        log_message(f"rawPath = {raw_path}")
        log_message(f"sampledPath = {sampled_path}")
        log_message(f"rv_home = {DATAMET_RADVIEW_PATH}")
        log_message(f"rv_data_path = {DATAMET_DATA_PATH}")
        log_message(f"rv_schedule = {rv_schedule}")
        log_message("---------------------------------------------")

        st = time.time()
        log_message(f"Starting operative chain...")

        sites = get_sites(rv_center)

        radar = Scheduler(
            rv_name,
            sites,
            date=rv_date,
            time=rv_time,
            remove=rv_remove,
            # rawPath=str(raw_path),  # .. DATAMET_DATA_PATH / rv_name VOL/LR  -- input!
            interactive=rv_interactive,
            rv_home=str(DATAMET_RADVIEW_PATH),
            rv_data_path=str(DATAMET_DATA_PATH),
            rv_schedule=rv_schedule,
            # sampledPath=str(sampled_path)
        )

        phase_1_out_path = radar.phase_1(logger=logger)

        radar.phase_2()

        et = time.time()
        elapsed_time = et - st
        dpg.log.log_message("==" * 50)
        dpg.log.log_message(f"TOTAL Execution Time: {elapsed_time} seconds")

        if print_results_folder:
            if raw_path is not None:
                h_nodes = OperativeChain._get_H_results_nodes(
                    rv_raw_path_prefix, rv_date, rv_time, rv_raw_path_suffix, sites
                )
            else:
                h_nodes = []
            s_nodes = OperativeChain._get_S_results_nodes(
                phase_1_out_path, [s.schedule_path for s in radar.schedules], sites
            )

            print(f"The Operative Chain {rv_name} produced the following nodes:")

            for n in OperativeChain._remove_elevation_folder(h_nodes):
                if OperativeChain._get_elevation_folders(n):
                    print(
                        f"- {n} \n\tElevation levels: {sorted(OperativeChain._get_elevation_folders(n))}"
                    )
                else:
                    print(f"- {n}")

            for n in OperativeChain._remove_elevation_folder(s_nodes):
                if OperativeChain._get_elevation_folders(n):
                    print(
                        f"- {n} \n\tElevation levels: {sorted(OperativeChain._get_elevation_folders(n))}"
                    )
                else:
                    print(f"- {n}")
        else:
            h_nodes, s_nodes = [], []

        # Reset dei file modificati durante l'esecuzione
        # - products files
        # - RADAR.txt per LR e RADAR_0.txt per HR
        empty_products_files()
        OperativeChain._update_schedule_configuration(
            DATAMET_RADVIEW_PATH / "cfg" / "schedules" / f"{rv_name}.txt",
            {"out_path": "[[OUT_PATH]]"},
        )

        return h_nodes + s_nodes

    @staticmethod
    def delete_products(products_to_delete: list, **args):
        nodes = OperativeChain._get_H_results_nodes(
            args["rv_raw_path_prefix"],
            args["rv_date"],
            args["rv_time"],
            args["rv_raw_path_suffix"],
            get_sites(args["rv_center"]),
        )

        nodes = OperativeChain._remove_elevation_folder(nodes)

        nodes_to_delete = [str(n) for n in nodes if n.name in products_to_delete]

        print("The following directories will be deleted:", nodes_to_delete)
        delete_folders(nodes_to_delete)

    @staticmethod
    def _set_environment(raw_path: Path):
        """
         Sets environment variables for the application based on the provided raw path and predefined constants.

        Args:
            raw_path: Path to be set for the environment variable `RV_RAW_PATH`.

        Returns:
            - **None**
        """
        os.environ["DMT_DATA_PATH"] = str(DATAMET_DATA_PATH)
        os.environ["DMT_HOME"] = str(
            DATAMET_RADVIEW_PATH
        )  # TODO: why it's equal to RV_HOME?
        os.environ["DMT_TARGET_PATH"] = str(DATAMET_RADVIEW_PATH / "target")
        os.environ["RV_RAW_PATH"] = str(raw_path)
        os.environ["RV_HOME"] = str(DATAMET_RADVIEW_PATH)
        os.environ["RV_DATA_PATH"] = str(DATAMET_DATA_PATH)
        os.environ["RV_SAV_PATH"] = str(DATAMET_RADVIEW_PATH / "sav")
        os.environ["RV_SAV_FILE"] = str(DATAMET_RADVIEW_PATH / "sav" / "datamet.sav")
        os.environ["RV_TARGET_PATH"] = str(DATAMET_RADVIEW_PATH / "target")

    @staticmethod
    def _set_cwd(path: Path):
        """
         Sets the current working directory for the application.

        Args:
            path: Path to be set as cwd.

        Returns:
            - **None**
        """
        os.chdir(path)

    @staticmethod
    def _configure(
            schedule_config: dict,
            rv_name,
            rv_raw_path_prefix: str,
            rv_date: str,
            rv_time: str,
            rv_raw_path_suffix: str,
            extra_prd_path: str = None,
            radar_prd_path: str = None,
            rv_sampled_path_prefix: str = None,
            rv_sampled_path_suffix: str = None,
            data_folders_path: dict = None,
    ):
        """
        Configures the schedule and updates the necessary files based on the provided parameters.

        Args:
            schedule_config: Dictionary containing configuration settings for the schedule.
            rv_name: Name associated with the configuration.
            rv_raw_path_prefix: Prefix for the raw path.
            rv_date: Date in the format 'DD-MM-YYYY'.
            rv_time: Time in the format 'HH:MM'.
            rv_raw_path_suffix: Optional suffix to append to the base path.

        Returns:
            - **Path**: A Path object representing the constructed raw path.
        """
        empty_products_files()

        if rv_name == "EXTRA":
            fill_products_files(
                rv_name,
                rv_date,
                rv_time,
                rv_raw_path_prefix,
                extra_prd_path,
                radar_prd_path,
            )

        if rv_name == 'RADAR':
            fill_assessment_products(rv_date, rv_time)

        # TODO: what is this?
        dpg.globalVar.GlobalState.remove_global_var_log()  # Remove old globalVar.GlobalState log files

        # if rv_raw_path_prefix is not None:
        #     raw_path = OperativeChain._set_raw_path(
        #         rv_raw_path_prefix, rv_date, rv_time, rv_raw_path_suffix
        #     )
        # else:
        #     raw_path = None
        #
        # if rv_sampled_path_prefix is not None:
        #     sampled_path = OperativeChain._set_raw_path(
        #         rv_sampled_path_prefix, rv_date, rv_time, rv_sampled_path_suffix
        #     )
        # else:
        #     sampled_path = None

        if rv_raw_path_prefix is not None:
            rv_raw_path_prefix = DATAMET_ROOT_PATH / rv_raw_path_prefix
        if rv_sampled_path_prefix is not None:
            rv_sampled_path_prefix = DATAMET_ROOT_PATH / rv_sampled_path_prefix

        if schedule_config:
            OperativeChain._update_schedule_configuration(
                DATAMET_RADVIEW_PATH / "cfg" / "schedules" / f"{rv_name}.txt",
                schedule_config,
            )
        else:
            OperativeChain._update_schedule_configuration(
                DATAMET_RADVIEW_PATH / "cfg" / "schedules" / f"{rv_name}.txt",
                {"out_path": str(DATAMET_ROOT_PATH / "datamet_data" / rv_name / "PRD")},
            )
        OperativeChain._update_schedule_configuration(
            DATAMET_RADVIEW_PATH / "cfg" / "schedules" / f"{rv_name}.txt",
            {
                "phase_1.rawPath": rv_raw_path_prefix,
                "phase_1.sampledPath": rv_sampled_path_prefix,
                "phase_1.sampledSub": rv_sampled_path_suffix},
        )

        if data_folders_path:
            for key, val in data_folders_path.items():
                dpg.globalVar.GlobalState.update(key, val)

        return rv_raw_path_prefix, rv_sampled_path_prefix

    @staticmethod
    def _set_raw_path(
            rv_raw_path_prefix: str, rv_date: str, rv_time: str, rv_raw_path_suffix: str
    ):
        """
        Constructs a raw path using the provided parameters.

        Args:
            rv_raw_path_prefix: Prefix for the raw path.
            rv_date: Date in the format 'DD-MM-YYYY'.
            rv_time: Time in the format 'HH:MM'.
            rv_raw_path_suffix: Optional suffix to append to the base path.

        Returns:
            - **Path**: A Path object representing the constructed raw path.
        """
        raw_path = OperativeChain._get_raw_path(
            rv_raw_path_prefix=rv_raw_path_prefix,
            rv_date=rv_date,
            rv_time=rv_time,
            rv_raw_path_suffix=rv_raw_path_suffix,
        )
        return raw_path

    @staticmethod
    def _get_raw_path(**kwargs) -> Path:
        """
        Constructs a raw path based on the provided keyword arguments.

        Args:
            **kwargs: Keyword arguments including:
                - rv_date: Date in the format 'DD-MM-YYYY'.
                - rv_time: Time in the format 'HH:MM'.
                - rv_raw_path_prefix: Prefix for the raw path.
                - rv_raw_path_suffix: Optional suffix to append to the base path.

        Returns:
            - **Path**: A Path object representing the constructed raw path.

        Raises:
            - **ValueError**: If the date or time format is invalid.
        """
        OperativeChain._check_datetime_format(kwargs["rv_date"], kwargs["rv_time"])
        path = OperativeChain._build_base_path(
            kwargs["rv_raw_path_prefix"], kwargs["rv_date"], kwargs["rv_time"]
        )
        if kwargs["rv_raw_path_suffix"]:
            path = path / kwargs["rv_raw_path_suffix"]
        return path

    @staticmethod
    def _get_H_results_nodes(
            rv_raw_path_prefix: str,
            rv_date: str,
            rv_time: str,
            rv_raw_path_suffix: str,
            sites: list,
    ):

        products_nodes = []
        raw_path = OperativeChain._get_raw_path(
            rv_raw_path_prefix=rv_raw_path_prefix,
            rv_date=rv_date,
            rv_time=rv_time,
            rv_raw_path_suffix=rv_raw_path_suffix,
        )
        for s in sites:
            raw_path = raw_path / s / "H"
            if raw_path.exists():
                products_nodes += OperativeChain._get_all_sub_dirs(raw_path)

        return OperativeChain._get_data_nodes(products_nodes)

    @staticmethod
    def _get_S_results_nodes(phase_1_out_path, schedules: list, sites: list):
        products_nodes = []
        for sched in schedules:
            output_base_path = (
                    Path(phase_1_out_path) / Path(sched).stem
            )  # datamet_data/RADAR_0/PRD/2024/01/17/1100/TEST_CLUTT
            subschedules = [
                f for f in output_base_path.iterdir() if f.is_dir()
            ]  # datamet_data/RADAR_0/PRD/2024/01/17/1100/TEST_CLUTT/Declutter
            if len(schedules) == 0:
                print("[WARNING] No sub-schedules found in the output path.")

            sites += ["MOSAIC"]
            for subsched in subschedules:
                for si in sites:
                    schedule_site = (
                            subsched / si
                    )  # datamet_data/RADAR_0/PRD/2024/01/17/1100/TEST_CLUTT/Declutter/SERANO
                    if schedule_site.exists():
                        if si == "MOSAIC":
                            products_nodes += OperativeChain._get_all_sub_dirs(
                                schedule_site
                            )
                            products_nodes += [schedule_site]
                        else:
                            products_nodes += OperativeChain._get_all_sub_dirs(
                                schedule_site
                            )

        if len(products_nodes) == 0:
            print(
                f"[WARNING] No products found in the output path: {phase_1_out_path}."
            )

        return OperativeChain._get_data_nodes(products_nodes)

    @staticmethod
    def _get_data_nodes(products_nodes):
        data_nodes = []

        for p in products_nodes:

            if OperativeChain._contains_dat_file(p):
                data_nodes.append(p)
            else:
                elevation_dirs = OperativeChain._get_all_sub_dirs(p)
                data_nodes += elevation_dirs

        return data_nodes

    @staticmethod
    def _contains_dat_file(product_node: Path):
        for f in product_node.iterdir():
            if f.is_file() and f.suffix == ".dat":
                return True
        return False

    @staticmethod
    def _remove_elevation_folder(nodes):
        return list(set([n.parent if n.name.isnumeric() else n for n in nodes]))

    @staticmethod
    def _get_elevation_folders(node):
        if OperativeChain._contains_dat_file(node):
            return []
        else:
            return [f.name for f in node.iterdir() if f.is_dir()]

    @staticmethod
    def _get_all_sub_dirs(base_path: Path):
        return [s for s in base_path.iterdir() if s.is_dir()]

    @staticmethod
    def _update_schedule_configuration(file_path: Path, update_dict: dict):
        """
        Updates the schedule configuration in the specified file based on the provided update dictionary.

        Args:
            file_path: Path to the file that needs to be updated.
            update_dict: Dictionary containing keys to update in the file along with their new values.

        Returns:
            - **None**
        """
        if update_dict:
            with open(file_path, "r", encoding="UTF-8") as file:
                lines = file.readlines()
            with open(file_path, "w", encoding="UTF-8") as file:
                for line in lines:
                    OperativeChain._update_line_if_contains_key(file, line, update_dict)
            print(f"File {file_path} updated successfully.")

    @staticmethod
    def _update_line_if_contains_key(file, line, update_dict):
        """
        Updates a line in a file if it contains a key from the update dictionary.

        Args:
            file: File object where the line will be written.
            line: The line from the file to check and potentially update.
            update_dict: Dictionary containing keys to check against the line and their corresponding update values.

        Returns:
            - **None**
        """

        if any(line.startswith(key) for key in update_dict.keys()):
            key = line.split("=")[0].strip()
            value = update_dict[key]
            file.write(f"{key} = {value}\n")
            return
        file.write(line)

    @staticmethod
    def _check_datetime_format(date: str, time: str):
        """
        Validates the format of the provided date and time strings.

        Args:
            date: Date in the format 'DD-MM-YYYY'.
            time: Time in the format 'HH:MM'.

        Raises:
            - **ValueError**: If the date does not match the expected format 'DD-MM-YYYY'.
            - **ValueError**: If the time does not match the expected format 'HH:MM'.
        """

        if not re.match(r"\d{2}-\d{2}-\d{4}", date):
            raise ValueError(
                "Invalid date format for 'date' arg. Expected format: DD-MM-YYYY"
            )
        if not re.match(r"\d{2}:\d{2}", time):
            raise ValueError(
                "Invalid time format  for 'time' arg. Expected format: HH:MM"
            )

    @staticmethod
    def _build_base_path(rv_raw_path_prefix: str, date: str, time: str):
        """
         Constructs a base path using the provided date and time.

        Args:
            rv_raw_path_prefix: Prefix for the raw relative path.
            date: Date in the format 'DD-MM-YYYY'.
            time: Time in the format 'HH:MM'.

        Returns:
            - **Path**: A Path object representing the constructed path, following the structure:
              DATAMET_ROOT_PATH / rv_raw_path_prefix / YYYY / MM / DD / HHMM.
        """

        D, M, Y = date.split("-")
        h, m = time.split(":")
        return DATAMET_ROOT_PATH / rv_raw_path_prefix / Y / M / D / (h + m)
