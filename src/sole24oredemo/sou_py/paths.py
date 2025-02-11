import os
from pathlib import Path

DATAMET_ROOT_PATH: Path = Path(__file__).parent.parent.absolute()

if os.getenv("RV_DATA_PATH"):
    DATAMET_DATA_PATH = Path(os.getenv("RV_DATA_PATH"))
else:
    print("Cannot find $RV_DATA_PATH")
    DATAMET_DATA_PATH: Path = DATAMET_ROOT_PATH / "datamet_data" / "data"

if os.getenv("RV_HOME"):
    DATAMET_RADVIEW_PATH = Path(os.getenv("RV_HOME"))
else:
    print("Cannot find $RV_HOME")
    DATAMET_RADVIEW_PATH: Path = DATAMET_ROOT_PATH / "datamet_data" / "RadView"

DATAMET_GLOBVAR_DIR: Path = DATAMET_ROOT_PATH / "tmp"
SHAPEFILE_FOLDER: Path = DATAMET_DATA_PATH / "shapefiles"
SCHEDULE_DEFAULTS_PATH: Path = (
        DATAMET_DATA_PATH / "schedules" / "schedules_default.yaml"
)
