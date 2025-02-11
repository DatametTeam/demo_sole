import os
import shutil
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile
from sou_py.dpg.utility import delete_folders
import yaml

from sou_py.paths import DATAMET_DATA_PATH, SHAPEFILE_FOLDER, DATAMET_ROOT_PATH


def empty_products_files():
    """
    Empties the contents of all product files.

    This method retrieves a list of file paths using the `_get_product_files()`
    function and then iterates through each file path. For each file, it opens the file
    in write mode, which clears its contents. The files are then immediately
    closed, effectively emptying them.
    """
    for file_path in _get_product_files(
            DATAMET_DATA_PATH / "schedules" / "RADAR"
    ):
        open(
            DATAMET_DATA_PATH / "schedules" / "RADAR" / file_path,
            "w",
            encoding="utf-8",
        ).close()
    for file_path in _get_product_files(
            DATAMET_DATA_PATH / "schedules" / "EXTRA"
    ):
        open(
            DATAMET_DATA_PATH / "schedules" / "EXTRA" / file_path,
            "w",
            encoding="utf-8",
        ).close()
    print("All products files emptied")


def fill_assessment_products(date, time, data_path_prefix=None):
    assessment_product_file = _get_product_files(
        DATAMET_DATA_PATH / "schedules" / "SENSORS" / "VALID" / "Assessment")[0]
    datetime_str = f"{date} {time}"
    datetime_obj = datetime.strptime(datetime_str, "%d-%m-%Y %H:%M")

    if datetime_obj.minute < 30:
        adjusted_dt = datetime_obj.replace(minute=0, second=0, microsecond=0)
    else:
        adjusted_dt = datetime_obj.replace(minute=30, second=0, microsecond=0)

    time_str_no_colon = adjusted_dt.strftime("%H%M")

    if data_path_prefix is None:
        data_path_prefix = DATAMET_ROOT_PATH / "datamet_data"

    with open(assessment_product_file, "w", encoding="utf-8", ) as f:
        f.writelines(
            f"path = "
            f"{data_path_prefix / 'SENSORS' / 'PRD' / str(datetime_obj.year) / str(datetime_obj.month).zfill(2) / str(datetime_obj.day).zfill(2) / time_str_no_colon / 'VALID' / 'Assessment'}"
        )
        f.close()


def fill_products_files(name, date, time, rv_raw_path_prefix, extra_prd_path=None, radar_prd_path=None):
    if not isinstance(rv_raw_path_prefix, Path):
        rv_raw_path_prefix = Path(rv_raw_path_prefix)

    radar_product_list = _get_product_files(DATAMET_DATA_PATH / "schedules" / "RADAR" / "RRN")
    radar_product_list += _get_product_files(DATAMET_DATA_PATH / "schedules" / "RADAR" / "DUAL")
    radar_product_list += _get_product_files(DATAMET_DATA_PATH / "schedules" / "RADAR" / "SRT")
    radar_product_list += _get_product_files(DATAMET_DATA_PATH / "schedules" / "EXTRA" / "HRW")

    datetime_str = f"{date} {time}"
    datetime_obj = datetime.strptime(datetime_str, "%d-%m-%Y %H:%M")
    time_str_no_colon = datetime_obj.strftime("%H%M")

    for file_path in radar_product_list:
        operative_chain, schedule, product = _extract_elements_after_schedules(
            file_path
        )

        if operative_chain == 'EXTRA' and extra_prd_path:
            raw_path_prefix = Path(extra_prd_path)
        elif radar_prd_path:
            raw_path_prefix = Path(radar_prd_path)
        else:
            raw_path_prefix = Path(rv_raw_path_prefix.parts[0]) / operative_chain / rv_raw_path_prefix.parts[2]

        with open(
                file_path,
                "w",
                encoding="utf-8",
        ) as f:
            f.writelines(
                f"path = "
                f"{raw_path_prefix / str(datetime_obj.year) / str(datetime_obj.month).zfill(2) / str(datetime_obj.day).zfill(2) / time_str_no_colon / schedule / product}"
            )
            f.close()

        print(f"File {file_path} filled")


def _extract_elements_after_schedules(file_path):
    # Ensure file_path is a Path object
    if not isinstance(file_path, Path):
        file_path = Path(file_path)

    # Split the path into parts
    path_parts = file_path.parts

    # Find the index of "schedules"
    if "schedules" in path_parts:
        schedules_index = path_parts.index("schedules")
        elements_after_schedules = path_parts[schedules_index + 1: schedules_index + 4]

        if len(elements_after_schedules) == 3:
            operative_chain, schedule, product = elements_after_schedules
            return operative_chain, schedule, product
        else:
            raise ValueError("Not enough elements after 'schedules'")
    else:
        raise ValueError("'schedules' not found in the path")


def _get_data_folders():
    """
    Returns a list of paths to various data directories used.

    This function constructs and returns a list of directory paths, this includes locations for radar data,
    RPG data, VPR data, clutter data and several configuration directories. These paths are based on the
    `DATAMET_DATA_PATH` base directory.

    Returns:
        List[str]: A list of directory paths as strings, representing different data folders.
    """
    return [
        os.path.join(DATAMET_DATA_PATH, "RADAR"),
        os.path.join(DATAMET_DATA_PATH, "RPG"),
        os.path.join(DATAMET_DATA_PATH, "data", "vpr"),
        os.path.join(DATAMET_DATA_PATH, "data", "clutter"),
        os.path.join(DATAMET_DATA_PATH, "data", "underlays"),
        DATAMET_DATA_PATH / "RadView" / "sav",
        DATAMET_DATA_PATH / "RadView" / "target",
        DATAMET_DATA_PATH / "RadView" / "cfg" / "hdf",
        DATAMET_DATA_PATH / "RadView" / "cfg" / "init",
        DATAMET_DATA_PATH / "RadView" / "cfg" / "properties",
        DATAMET_DATA_PATH / "RadView" / "cfg" / "resources",
        DATAMET_DATA_PATH / "RadView" / "cfg" / "sensors",
        os.path.join(DATAMET_DATA_PATH, "RadView", "cfg", "templates"),
        os.path.join(DATAMET_DATA_PATH, "RadView", "cfg", "users"),
    ]


def _get_data_files():
    return [DATAMET_DATA_PATH / "data" / "IDL.csh"]


def _get_product_files(start_folder):
    """
    Recursively searches for "products.txt" files within a directory and its subdirectories.

    This function traverses the directory tree starting from `start_folder` and looks for
    files named "products.txt". When found, the full path to each "products.txt" file is
    added to a list.

    Args:
        start_folder (Path): The root directory from which the search begins.

    Returns:
        List[str]: A list of full paths to each "products.txt" file found within the
                   directory tree.
    """
    products_files = []

    for root, dirs, files in os.walk(start_folder):
        if "products.txt" in files:
            products_files.append(os.path.join(root, "products.txt"))

    return products_files


def get_italian_region_shapefile() -> Path:
    italian_regions_folder_path = SHAPEFILE_FOLDER / "italian_regions"
    files_in_folder = list(italian_regions_folder_path.glob("*"))
    filename = files_in_folder[0].stem
    return italian_regions_folder_path / filename


def clean_input_folder_for_HR_execution(yaml_path: Path) -> bool:
    # Open and read the YAML file
    with open(yaml_path, 'r') as file:
        yaml_content = yaml.safe_load(file)

    # Accessing individual fields in the YAML
    rv_raw_path_prefix = yaml_content.get('rv_raw_path_prefix')
    rv_date = yaml_content.get('rv_date')
    day, month, year = rv_date.split('-')
    rv_time = yaml_content.get('rv_time').replace(":", "")
    rv_center = yaml_content.get('rv_center')

    data_path = os.path.join(rv_raw_path_prefix, year, month, day, rv_time, rv_center)
    if os.path.exists(os.path.join(data_path, "H")):
        data_path = os.path.join(data_path, "H")
    if not os.path.exists(data_path):
        print(f"La cartella {data_path} non esiste")
        return False

    data_path = Path(data_path)
    try:
        # ATTENUATION
        delete_folders(
            [data_path / _dir for _dir in ["UZ_ATT_CORR", "Quality_att_corr"]]
        )
        print("Pulita ATTENUATION")
        # DECLUTTER
        delete_folders(
            [data_path / _dir for _dir in ["ClutterMask", "Quality", "UZ_CLUTTER_CORR"]]
        )
        print("Pulita DECLUTTER")
        # FLH
        delete_folders(
            [data_path / _dir for _dir in ["Quality", "Quality_after_flh"]]
        )
        print("Pulita FLH")
        # KDP
        delete_folders([data_path / _dir for _dir in ["CKDP", "CPHIDP"]])
        print("Pulita KDP")
        # OCCLUSION
        delete_folders(
            [data_path / _dir for _dir in ["Quality", "Quality_after_occlusion"]]
        )
        print("Pulita OCCLUSION")
        # PBB
        delete_folders([data_path / _dir for _dir in ["UZ_PBB_CORR"]])
        print("Pulita PBB")
    except Exception as e:
        print(e)
        return False
    return True
