import os
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import sou_py.dpg as dpg
import sou_py.dpb as dpb
from sou_py.dpg.log import log_message


def readAnagFile(old=False):
    initDir = dpg.path.getDir('SENSORS')

    if old:
        records = pd.read_csv(os.path.join(initDir, 'raingauges.old'), sep=';', header=None)
    else:
        records = pd.read_csv(os.path.join(initDir, 'raingauges.txt'), sep=';', header=None)

    records.columns = ['codes', 'names', 'regions', 'alts', 'lons', 'lats', 'alerts', 'com', 'pv']

    return records


def updateCurrCount(path, date, time, count):
    log_message(f'Time {time} ... Found {str(count).strip()} valid raingauges')

    ddd = date
    ttt = time
    ddd, hh = dpg.times.addMinutesToDate(ddd, ttt, -60)

    ttt, hh, _ = dpg.times.checkTime(time=ttt)  # TODO. controllare

    name = ddd + '.txt'

    strings, _ = dpg.io.read_strings(path, name)
    sCount = len(strings)

    if sCount != 25:
        sCount = np.array([""] * 25, dtype='str')
        sCount[0] = ddd

    ttt = time
    hh += 1
    if hh == 24:
        ttt = '24:00'

    chan = sCount[hh].split()
    if len(chan) == 2:
        if count == 0 or count > chan[1]:
            sCount[hh] = chan[0] + str(count)
    else:
        sCount[hh] = ttt + str(count)

    dpg.io.save_values(path, name, sCount)

    if hh <= 3:
        return

    ddd = dpg.times.getPrevDay(ddd)
    prevFile = dpg.path.getFullPathName(path, ddd + '.txt')
    if not os.path.isfile(prevFile):
        return

    currFile = dpg.path.getFullPathName(path, 'count.txt')
    shutil.copyfile(prevFile, currFile)

    return


def loadSRT5(path, date):
    if path is None or path == '':
        return

    if not os.path.isfile(os.path.join(path, date + '.txt')):
        return None
    srt5 = pd.read_csv(os.path.join(path, date + '.txt'), sep='', header=0)
    if len(srt5) <= 1:
        return None

    return srt5


def raingauge_srt(currDate, currTime, current_data, srt5Path):
    countLast = 0
    countPrev = 0
    nG = len(current_data)
    if nG <= 0:
        return

    currSec = dpg.times.convertDate(currDate, currTime)
    seconds = dpg.times.convertDate(current_data['dates'].to_list(), time=current_data['times'].to_list())

    last = np.zeros(nG) * np.nan
    prev = np.zeros(nG) * np.nan
    offset = 0
    nValues = len(current_data)
    firstSec = currSec[0] - 3600

    srt5Date, srt5Time = dpg.times.seconds2Date(firstSec - 3600)
    srt_data = loadSRT5(srt5Path, srt5Date)

    changed = 0
    while offset < nValues:
        ind = getNextSensorIndex(current_data)



def raingauge_import(pathname, date, time, summaryPath, srt5Path):
    time = time[:3] + "00" + time[5:]

    current_data = pd.read_csv(pathname, sep=' ', header=None)
    current_data.columns = ['pre', 'codes', 'dates', 'times', 'values']

    path = str(Path(pathname).parent)  # TODO: da controllare

    if len(current_data) < 1:
        updateCurrCount(path, date, time, 0)
        log_message(f"Cannot read {pathname}", level='WARNING+')
        return

    raingauge_srt(date, time, current_data, srt5Path=srt5Path)


def raingauge(pathname, summaryPath=None, srt5Path=None, tag=None, value=None, init=False, date=None, time_=None,
              tmp_file=None, cum=None, SRT=None):
    # TODO: check_RPG_options, options, INIT=init

    if init:
        raingauge_init_codes(pathname)
        return

    if dpg.globalVar.GlobalState.ANAG_FILE is None:
        anag_df = readAnagFile()
        dpg.globalVar.GlobalState.update("ANAG_FILE", anag_df)
        if len(anag_df) <= 1:
            log_message("AnagFile read returned 0 elements. ", level='ERROR')
            return

    if date is not None and cum:
        summary('RAIN', date, None, SRT, cum=True)
        return

    # TODO. check RPG
    # if date is not None or time is not None:
    #     check_RPG_options()

    if date is None or time_ is None:
        log_message("raingauge(): date or time is None", level='WARNING+')
        return

    if summaryPath is None:
        summaryPath = 'RAIN'

    if pathname is not None:
        if pathname != '':
            if tmp_file is not None:
                name = dpg.times.checkDate(date, sep='', year_first=True)
                new_file = dpg.path.getFullPathName(pathname, name + '.asc')
                if not os.path.isfile(new_file):
                    log_message(f"Cannot find {new_file}", level='WARNING')
                    return
                log_message(f"Importing {new_file}")
                if tmp_file != new_file:
                    shutil.copy2(new_file, tmp_file)
            else:
                tmp_file = pathname
            raingauge_import(tmp_file, date, time_, summaryPath, srt5Path)
            return

    summary(summaryPath, date, time_, SRT)
