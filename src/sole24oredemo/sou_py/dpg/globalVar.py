# File create per gestire attualmente le variabili globali utilizzate all'interno del programma
import multiprocessing
from datetime import datetime
import inspect
from typing import Any, Optional
from sou_py.paths import (
    DATAMET_ROOT_PATH,
    DATAMET_GLOBVAR_DIR,
)
import os

lock = multiprocessing.Lock()


class GlobalState:
    DATAMET_GLOBVAR_DIR.mkdir(exist_ok=True)
    GLOBAL_STATE_LOG = (
            DATAMET_GLOBVAR_DIR
            / f"global_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    # All these vars are updated in access.py

    SAMPLED_PATH = {}  # this is updated also in scheduler__define.py
    SAMPLED_SITE = {}  # this is updated also in scheduler__define.py
    SAMPLED_DATE = {}  # this is updated also in scheduler__define.py
    SAMPLED_TIME = {}  # this is updated also in scheduler__define.py
    SAMPLED_VOL_PATH = {}  # not used anywhere

    CHECK_PATH = {}

    RAW_VOL_PATH = {}  # not used anywhere
    RAW_PATH = {}
    RAW_SITE = None
    RAW_DATE = {}
    RAW_TIME = {}

    LAST_PATH = None  # not used anywhere
    LAST_RAW = {}  # this is updated also in scheduler__define.py
    LAST_SAMPLED = {}  # this is updated also in scheduler__define.py

    RRPC_HOME = None  # not used anywhere
    RV_HOME = None
    RV_HOST = None
    RV_SEP = None
    RV_PATHS = None  # not used anywhere

    # the following vars are updated in getDir() (path.py)
    dataPath = None
    rpgPath = None
    rpvPath = None
    souPath = None
    externPath = None  # this is also update in mosaic.py
    miscPath = None
    rdpPath = None
    modelsPath = None
    rawPath = None
    clutterRoot = None
    vprRoot = None

    PREFIX = None  # not used anywhere
    pref_in = None
    pref_out = None

    LOW = None
    HIGH = None

    isInteractive = None

    SHARED_MAPS = []  # not tracked
    SHARED_TREES = []  # not tracked

    DEM_HROOT = None
    DEM_ROOT = None

    DATA_SEC = None

    ANAG_FILE = None

    @staticmethod
    def remove_global_var_log():
        """
        Deletes log files related to global variables in the application.

        This static method searches for files in the `DATAMET_ROOT_PATH` directory that
        follow a specific naming pattern (files starting with "global_state_" and ending
        with ".log"). Any matching files are deleted from the directory.

        Returns:
            None
        """
        files = os.listdir(DATAMET_ROOT_PATH)

        # Loop through the files and delete the ones that match the criteria
        for file in files:
            if file.startswith("global_state_") and file.endswith(".log"):
                file_path = os.path.join(DATAMET_ROOT_PATH, file)
                os.remove(file_path)
                print(f"Deleted: {file_path}")

    @staticmethod
    def update(
            global_var_name: str,
            value: Any,
            key: Optional[str] = None,
    ):
        """
        Assign value to class variable name and update related log

        Args:
            global_var_name (str): variable to update
            value (Any): value to assign.
            key (Optional[str], optional): the target keys
                in case the variable to update is a dict. Defaults to None.

        """

        selected_var = getattr(GlobalState, global_var_name)
        if isinstance(selected_var, dict) and key is not None:
            selected_var[key] = value
        else:
            setattr(
                GlobalState,
                global_var_name,
                value,
            )
        update_global_state_log(global_var_name, value, key)


def update_global_state_log(
        variable_name: str,
        value: Any,
        key: Optional[str] = None,
):
    """
    Update the global state log for the modified variable name in GlobalState

    Args:
        variable_name (str): variable to update
        value (Any): value to assign.
        key (Optional[str], optional): the target keys
            in case the variable to update is a dict. Defaults to None.
    """
    curframe = inspect.currentframe()
    calframe = inspect.getouterframes(curframe, 2)
    with open(GlobalState.GLOBAL_STATE_LOG, "a") as f:
        log_str = f"{datetime.now().strftime('[%Y/%m/%d %H:%M:%S]')} {variable_name}"
        if key:
            log_str += f"[{key}]"
        log_str += f" = {value}, Called by: {calframe[1][3]}(..)\n"
        f.write(log_str)
