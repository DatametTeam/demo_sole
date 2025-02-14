#!/bin/bash
#PBS -N sole24ore_demo
#PBS -q fast
#PBS -l select=1:ncpus=0:ngpus=0
#PBS -k oe
#PBS -j oe
#PBS -o /davinci-1/home/guidim/pbs_logs/pbs.log 

            module load proxy
            module load anaconda3
            source activate protezionecivile
            

    python "/archive/SSD/home/guidim/protezione_civile/nowcasting/nwc_test_webapp.py"         start_date=14-02-2025-01-30
        