import multiprocessing
from constants import OUTPUT_DATA_DIR
from time import sleep
import os
from pathlib import Path
import numpy as np

IN_PROCESS = None

def get_job_status(job_id):
    if IN_PROCESS is not None and IN_PROCESS.is_alive():
        return "R"
    else:
        return "ended"

def inference_mock(inf_args) -> None:
    mj = MockJob(job_id="123456")
    print("inf_args", inf_args)
    from datetime import datetime
    date_now = datetime.now().strftime("%Y%m%d%H%M%S")      
    p = multiprocessing.Process(target=mock_inference_job_worker, args=(mj, date_now, *inf_args))
    p.start()
    global IN_PROCESS
    IN_PROCESS = p

    start_date, end_date, model_name, submitted = inf_args
    output_dir = OUTPUT_DATA_DIR / str(start_date) / str(end_date) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return mj.job_id, output_dir

def mock_inference_job_worker(mj, date_now, *inf_args):
    mj.mock_inference_job(date_now, *inf_args)

class MockJob:
    def __init__(self, job_id):
        self.job_id = job_id

    def mock_inference_job(self, date_now, start_date, end_date, model_name, submitted) -> None:

        output_dir = OUTPUT_DATA_DIR / str(start_date) / str(end_date) / model_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Define dimensions
        starttimestamps = np.arange(0, 24)  # 24 timestamps (0 to 23)
        hourstep = np.arange(1, 13)  # 12 steps (1 to 12)
        width, height = 512, 512
        
        # Generate random noise data
        data_shape = (len(starttimestamps), len(hourstep), width, height)
        array1 = np.random.rand(*data_shape).astype(np.float32)
        array2 = np.random.rand(*data_shape).astype(np.float32)
        
        # Save arrays as .npy
        np.save(output_dir / "gt_mock_data.npy", array1)
        np.save(output_dir / "pred_mock_data.npy", array2)
        
        # Log file
        log_file = Path(os.environ["HOME"]) / f"sole24ore_demo.o{self.job_id}"
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"Mock inference job {self.job_id} completed. Data saved to {output_dir}\n")