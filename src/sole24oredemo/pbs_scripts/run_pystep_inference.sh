#!/bin/bash
#PBS -N sole24ore_demo
#PBS -q fast
#PBS -l select=1:ncpus=0:ngpus=0
#PBS -k oe
#PBS -j oe
#PBS -o /davinci-1/home/guidim/pbs_logs/pbs.log 

            module load proxy
            module load anaconda3
            source activate nowcasting
            

    python "/archive/SSD/home/guidim/demo_sole/src/sole24oredemo/inference_scripts/run_pystep_inference.py"         --start_date=13-02-2025-21-50
        