import os

import numpy as np

import sou_py.dpg as dpg
from sou_py.dpg.log import log_message


def check_RPG_options(options):
    nArgs = np.size(options)
    if nArgs == 0 or options is None:
        return None

    log_message("CHECK RPG PASSED WITH OPTIONS: TO DO PORTING", level='ERROR')


def replaceSiteFiles(path):
    """
    Updates the site files within the given directory. The function traverses the directory
    structure to locate schedule, product, and site directories. For each product, if multiple
    site directories are found, it generates a file named 'sites.txt' containing the list
    of site tags and their names.

    Args:
        path (str): The root directory containing schedule, product, and site subdirectories.

    Output:
        Updates the directory by saving a 'sites.txt' file in the product folder for each schedule.
    """
    list_path = os.listdir(path)
    sched = [i for i in list_path if os.path.isdir(os.path.join(path, i))]
    for sss in range(len(sched)):
        list_prod = os.listdir(os.path.join(path, sched[sss]))
        prod = [i for i in list_prod if os.path.isdir(os.path.join(path, sched[sss], i))]
        for ppp in range(len(prod)):
            list_sites = os.listdir(os.path.join(path, sched[sss], prod[ppp]))
            sites = [i for i in list_sites if os.path.isdir(os.path.join(path, sched[sss], prod[ppp], i))]
            if len(sites) > 1:
                tags = ['site'] * len(sites)
                dpg.attr.saveAttr(os.path.join(path, sched[sss], prod[ppp]), 'sites.txt', tags, sites, replace=True)
