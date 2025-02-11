"""

THIS IS A DUPLICATION OF THE FILE IN THE TEST_PY DIRECTORY

"""

from typing import Tuple
from pathlib import Path
import numpy as np

# from sou_py.dpb.dpb import get_data as datamet_get_data
from sou_py.dpg.cfg import getGeoDescName
from sou_py.dpg.navigation import get_radar_par
from sou_py.dpg.node__define import Node
from sou_py.dpg.tree import createTree
from sou_py.paths import SHAPEFILE_FOLDER
from sou_py.dpg.attr__define import Attr


# def get_data(path: Path):
#     return datamet_get_data(createTree(str(path)))


# def get_data_with_nodes(path: Path) -> Tuple[object, np.array]:
#     node = createTree(str(path))
#     return (
#         node, datamet_get_data(node)
#     )


def get_azoff(prd_node) -> float:
    """
    Retrieves the azimuth offset value from a product node's radar parameters.

    Args:
        - prd_node: Node object representing the product.

    Returns:
        - float: The azimuth offset value. Defaults to 0 if not found.
    """

    navigation_dict = get_radar_par(prd_node, get_az_coords_flag=True)
    if "azimut_off" in navigation_dict.keys():
        return navigation_dict["azimut_off"]
    else:
        return 0


def get_calibration_data(node) -> list:
    """
    Fetches calibration data associated with a given node.

    Parameters:
        - node: Node object from which calibration data is retrieved.

    Returns:
        - list: Calibration data as a list. Ensures data is wrapped in a list if it's not already.
    """

    calibrationData, _ = node.getValues()
    if not isinstance(calibrationData, list):
        calibrationData = [calibrationData]
    return calibrationData


def get_product_unit_max_min(prd_node, prd_image, gt_image, orig_path):
    """
    Fetches calibration data associated with a given node.

    Args:
        - node: Node object from which calibration data is retrieved.

    Returns:
        - list: Calibration data as a list. Ensures data is wrapped in a list if it's not already.
    """
    calibrationData = get_calibration_data(prd_node)
    if None in calibrationData and prd_node.getSon(str(orig_path)) is not None:
        calibrationData = get_calibration_data(prd_node.getSon(str(orig_path)))

    attr: Attr
    for attr in calibrationData:
        if "unit" in attr.pointer.keys():
            product_unit = attr.pointer["unit"]
            product_min = attr.pointer["offset"]
            product_max = max(np.nanmax(gt_image), np.nanmax(prd_image))
            return product_unit, product_min, product_max
        else:
            product_unit = "General"
            product_min = np.nanmin(prd_image)
            product_max = max(np.nanmax(gt_image), np.nanmax(prd_image))

    return product_unit, product_min, product_max


def get_italian_region_shapefile() -> Path:
    """
    Retrieves the path to the shapefile for Italian regions.

    Returns:
        - Path: File path of the Italian regions shapefile.
    """
    italian_regions_folder_path = SHAPEFILE_FOLDER / "italian_regions"
    files_in_folder = list(italian_regions_folder_path.glob("*"))
    filename = files_in_folder[0].stem
    return italian_regions_folder_path / filename


def check_quality_path(path: Path) -> bool:
    """
    Checks if a given file path corresponds to a "Quality" file.

    Args:
        - path: Path object representing the file path.

    Returns:
        - bool: True if the path's stem is "Quality", otherwise False.
    """
    if path.stem == "Quality":
        return True
    return False


def check_navigation(node: Node) -> bool:
    """
    Verifies the presence of navigation attributes for a given node.

    Args:
        - node: Node object to check.
    Returns:
        - bool: True if navigation attributes are present, otherwise False.
    """
    nav = node.getAttr(getGeoDescName())
    if nav is None:
        return False
    else:
        return True
