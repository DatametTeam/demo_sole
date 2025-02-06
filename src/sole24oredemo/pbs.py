import subprocess
from pathlib import Path

from constants import OUTPUT_DATA_DIR, TARGET_GPU
from datetime import datetime


def is_pbs_available() -> bool:
    import subprocess
    return subprocess.call(["qstat"], shell=True) == 0


def get_job_status(job_id):
    try:
        # Run the command and capture the output
        result = subprocess.run(
            ["qstat", "-f", job_id],
            check=True,
            text=True,
            capture_output=True
        )
        # Filter for the "job_state" line
        for line in result.stdout.splitlines():
            if "job_state" in line:
                # Extract the status after the "=" sign
                _, status = line.split("=", 1)
                return status.strip()  # Remove any surrounding whitespace
        return "ended"  # job_state line not found, assume the job ended
    except subprocess.CalledProcessError:
        # If qstat fails, the job likely doesn't exist
        return "ended"


def get_pbs_header(job_name, q_name, pbs_log_path, target_gpu=None):
    if target_gpu is None:
        return f"""
#PBS -N {job_name}
#PBS -q {q_name}
#PBS -l select=1:ncpus=1:ngpus=4,walltime=12:00:00
#PBS -k oe
#PBS -j oe
#PBS -o {pbs_log_path} 
"""
    else:
        return f"""
#PBS -N {job_name}
#PBS -q {q_name}
#PBS -l host={target_gpu},walltime=12:00:00
#PBS -k oe
#PBS -j oe
#PBS -o {pbs_log_path} 
"""


# TO UPDATE!
def get_pbs_env():
    return f"""
module load proxy/proxy_20 python39 cuda12.1 gcc/11.2.0
WORKDIR="$HOME/workspace/faradai"
source "$WORKDIR/faradai/dreambooth_scripts/.venv/bin/activate"
"""


def submit_inference(args) -> tuple[str, str]:

    return 0, []

    start_date, end_date, model_name, submitted = args

    # TO UPDATE!
    out_dir = OUTPUT_DATA_DIR / model_name / start_date.strftime("%Y%m%d") / end_date.strftime("%Y%m%d")
    
    # DEFINE THE OUTPUT DIRECTORY ! TO UPDATE!
    date_now = datetime.now().strftime("%Y%m%d%H%M%S")
    out_images_dir = out_dir / "generations" / date_now
    out_images_dir.mkdir(parents=True, exist_ok=True)

    fine_tuned_model_dir = out_dir / "finetuned_model"

    cmd_string = f"""
python3 "$WORKDIR/faradai/dreambooth_scripts/run_inference.py" \
--fine-tuned-model-dir={str(fine_tuned_model_dir)}
"""

    print(f"cmd_string: \n > {cmd_string}")
    pbs_logs = out_dir / "pbs_logs"
    pbs_logs.mkdir(parents=True, exist_ok=True)

    pbs_script = "#!/bin/bash"
    pbs_script += get_pbs_header("sole24ore_demo", TARGET_GPU, str(pbs_logs / "pbs.log"))
    pbs_script += get_pbs_env()
    pbs_script += f"\n{cmd_string}"

    pbs_scripts = out_dir / "pbs_script"
    pbs_scripts.mkdir(parents=True, exist_ok=True)
    pbs_script_path = pbs_scripts / "run_inference.sh"
    with open(pbs_script_path, "w", encoding="utf-8") as f:
        f.write(pbs_script)

    # Command to execute the script with qsub
    command = ["qsub", pbs_script_path]

    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        print("Inference job submitted successfully!")
        job_id = result.stdout.strip().split(".davinci-mgt01")[0]
        print("Job ID:", job_id)
        return job_id, out_images_dir

    except subprocess.CalledProcessError as e:
        print("Error occurred while submitting the job!")
        print("Error message:", e.stderr.strip())
        return None, None
