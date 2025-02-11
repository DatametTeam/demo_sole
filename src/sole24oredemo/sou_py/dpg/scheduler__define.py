import copy
import glob
import multiprocessing
import re
import sys
import time
import traceback
from datetime import datetime, timedelta
from functools import partial
from multiprocessing.dummy import Pool
import time as pytime

import numpy as np
import os
import shutil
import sou_py.dpg as dpg
from sou_py.dpg.log import log_message

from deprecated.sphinx import deprecated

from sou_py.paths import DATAMET_RADVIEW_PATH


class Scheduler(object):
    def __init__(
            self,
            name,
            sites,
            out_path=None,
            schedules=None,
            rawPath="",
            center="",
            # phase=,
            date="",
            time="",
            remove=False,
            force=False,
            export=True,
            endCommand="",
            series=False,
            delay=0,
            max_delay=0,
            interactive=False,
            rv_home="",
            rv_data_path="",
            rv_schedule=None,
            sampledPath="",
    ):

        self.name = name
        self.out_path = out_path
        self.schedules = schedules

        # self.rawPath = {}
        # for site in sites:
        #     self.rawPath[site] = rawPath

        # self.sampledPath = {}
        # for site in sites:
        #     self.sampledPath[site] = sampledPath

        self.center = center
        self.date = date
        self.time = time
        self.remove = remove
        self.force = force
        self.export = export
        self.endCommand = endCommand
        self.series = series
        self.delay = delay
        self.max_delay = max_delay
        self.interactive = interactive
        self.sites = sites
        self.rv_home = rv_home
        self.rv_data_path = rv_data_path

        if rv_schedule is None:
            self.schedules = []
        else:
            self.schedules = [rv_schedule]

    def check_out_path(self, export: int = 1) -> str:
        """
        Checks and retrieves the output path, updating it if necessary.

        Args:
            export (int, optional): Default export value if not found in attributes. Defaults to 1.

        Returns:
            str: The verified or updated output path.
        """
        if isinstance(self.out_path, str):
            return self.out_path
        path = dpg.path.getDir("CFG_SCHED")
        attr_dict = dpg.attr.loadAttr(path, self.name + ".txt")
        out_path, _, _ = dpg.attr.dpg.attr.getAttrValue(attr_dict, "out_path", "")
        export, _, _ = dpg.attr.dpg.attr.getAttrValue(attr_dict, "export", export)
        out_path = dpg.path.checkPathname(out_path)
        if out_path.find("20") < 0:
            out_path = os.path.join(out_path, "2000")
        self.out_path = out_path
        return out_path

    def getSchedulePath(self, sep: str = "") -> str:
        """
         Constructs and returns the full path to the schedule directory for the current object.

        Args:
            sep: Optional; A string separator to be used for constructing the path. Defaults to an empty string.

        Returns:
            str: The full path to the schedule directory, combining the base data directory, "schedules",
            and the object's name.
        """
        path = dpg.path.getDir("DATA", sep=sep)
        return os.path.join(path, "schedules", self.name)

    def getAllSchedules(
            self,
            schedules,
            onlyIfActive: bool = False,
            sites: list = [],
            nominalTime: str = "",
    ):
        """
        Retrieves all schedules from the schedule path, optionally filtering by active status, site list, and nominal
        time

        Args:
            schedules (list): A list to store the schedule paths
            onlyIfActive (bool, optional): If True, only retrieves schedules that are active. Defaults to False
            sites (list, optional): A list of sites to check for each schedule. Defaults to an empty list
            nominalTime (str, optional): The nominal time to check for each schedule. Defaults to an empty string

        Returns:
            tuple: A tuple containing:
                   - A list of active schedule paths
                   - A list of corresponding parallel execution flags
        """
        path = self.getSchedulePath()
        schedules = dpg.cfg.getSubDir(path)
        nSched = len(schedules)

        if nSched <= 0:
            return

        parallel = []
        if not onlyIfActive:
            return schedules, parallel

        priority = []
        name = dpg.cfg.getScheduleDescName()
        activeSchedules = []
        for schedPath in schedules:
            attr = dpg.attr.loadAttr(schedPath, name)
            active, _, _ = dpg.attr.dpg.attr.getAttrValue(attr, "active", 0)
            if active:
                active = dpg.rpk.checkSiteList(schedPath, sites, sc_attr=attr)
            if active:
                active = dpg.rpk.checkTimeList(schedPath, nominalTime, sc_attr=attr)
                if active == 0:
                    log_message(
                        f"Skip schedule {schedPath} for time {nominalTime}. Time not set in times.txt",
                        level="WARNING",
                    )

            if active:
                activeSchedules.append(schedPath)
                parallel.append(dpg.attr.dpg.attr.getAttrValue(attr, "parallel", 0)[0])
                priority.append(dpg.attr.dpg.attr.getAttrValue(attr, "priority", 10)[0])
            # endif
        # endfor

        if len(activeSchedules) <= 0:
            return activeSchedules, parallel

        ind = np.argsort(priority)
        self.schedules = [activeSchedules[i] for i in ind]
        self.parallel = [parallel[i] for i in ind]
        log_message(
            f"Found {len(self.schedules)} active schedules: {['/'.join(i.split('/')[-2:]) for i in self.schedules]}"
        )
        return

    def checkSchedules(
            self, options: str = "", time: str = "", sites: list = [], schedules: list = []
    ):
        if options:
            log_message(
                "Options setted in CheckSchedules(): NOT IMPLEMENTED", level="ERROR"
            )
            sys.exit()
            # pos = strpos(strupcase(options), '-SCHEDULE')
            # ind = where(pos eq 0, count)
            # if count eq 1 then $
            # schedules = strmid(options[ind[0]], 10)
        # endif
        if len(self.schedules) > 0:
            self.schedules = [
                os.path.join(self.getSchedulePath(), i) for i in self.schedules
            ]
            log_message(
                f"Found {len(self.schedules)} selected schedules: {self.schedules}"
            )
            return

        return self.getAllSchedules(
            schedules, onlyIfActive=True, nominalTime=time, sites=sites
        )

    def phase_1_init(self, date="", time=""):
        """
        Initializes the scheduling phase by setting up the schedule objects.

        Args:
            date (str, optional): The date for initialization (format: "YYYY-MM-DD"). Default is an empty string.
            time (str, optional): The time for initialization (format: "HH:MM:SS"). Default is an empty string.

        Returns:
            None
        """

        outPath = dpg.path.checkPathname(self.out_path)

        schedules = []
        for i, sched in enumerate(self.schedules):
            schName = os.path.basename(sched)
            outSchedule = dpg.schedule__define.Schedule(
                outPath, name=schName, schedule_path=sched, scheduler=self
            )
            schedules.append(outSchedule)

        self.schedules = schedules
        return

    def run_site(
            self,
            site: str,
            date: str,
            time: str,
            general_logger=None,
            series=False,
            last=False,
            endCommand=False,
            interactive=True
    ):
        # da controllare che tutto questo metodo venga chiamato su tutti i siti in fase 1
        time1 = pytime.time()
        formatted_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        site_logger = dpg.log.DynamicLogger(name=f"{site}")
        site_logger.initialize()
        site_logger.set_log_path(self.out_path)
        log_message(f"==" * 50, all_logs=True)
        log_message(f"Site {site} started at time {formatted_time}", all_logs=True)

        out_path = self.out_path
        out_path = dpg.path.checkPathname(out_path)
        if not os.path.isdir(out_path):
            log_message("Output path does not exists", level="ERROR", all_logs=True)
            return
        if len(self.schedules) <= 0:
            log_message("There are 0 schedules", level="ERROR", all_logs=True)
            return

        phase = 1
        ignored = []

        # Da implementare: Attendere N minuti l'arrivo dei dati,
        # altrimenti sostiuire il raw_path

        # questa chiamata restituisce False qual'ora non vengano trovati i dati RAW del sito corrente
        # site_found = self.phase_wait_start(
        #     site, date=date, time=time, max_wait_minutes=15
        # )
        rawPath = self.new_phase_wait_start(site, phase_num=1, out_path=out_path,
                                            date_=date, time_=time, )  # interactive=interactive)
        # rawPath=self.rawPath[site])

        log_message(f" " * 50, all_logs=True, newlines=True)
        for sss in self.schedules:
            schedule_logger = dpg.log.DynamicLogger(name=os.path.join(sss.name, site))
            schedule_logger.initialize()
            schedule_logger.set_log_path(self.out_path)

            schedule_start_time = pytime.time()
            formatted_schedule_time = datetime.utcnow().strftime(
                "%Y-%m-%d %H:%M:%S.%f"
            )[:-3]

            log_message(
                f"Schedule {sss.name} for site {site} started at time {formatted_schedule_time}",
                all_logs=True,
            )
            # If sites.txt is present filter sites
            # if os.getenv('RV_CENTER') is None:

            attr = dpg.attr.loadAttr(sss.schedule_path, dpg.cfg.getScheduleDescName())
            active = dpg.rpk.checkSiteList(sss.schedule_path, [site], sc_attr=attr)
            if not active:
                log_message(
                    f"Skip schedule {sss.schedule_path} for site {site}. Site not present in sites.txt",
                    level="WARNING",
                    all_logs=True,
                )
                continue

            schedule_mp = copy.deepcopy(sss)
            # print(f"PROCESSING SITE: {site} | ADDRESS SCHEDULE MP = {id(schedule_mp)}")
            check = schedule_mp.schedule_check(site, rawPath)
            if check > 0:
                schedule_mp.schedule_init(site=site, date=date, time=time, rawPath=rawPath, no_log=True)
                try:
                    schedule_mp.schedule_exec(site=site, phase=phase)
                except Exception as e:
                    stack_trace = traceback.format_exc()
                    log_message(
                        f"An exception has occurred at schedule {sss.name}:\n{stack_trace}",
                        level="ERROR",
                        newlines=True,
                        all_logs=True,
                    )
                schedule_mp.schedule_end(
                    site=site, phase=phase, interactive=self.interactive
                )
                log_message(
                    f"Schedule {sss.name} executed in {pytime.time() - schedule_start_time}",
                    all_logs=True,
                )
            else:
                ignored.append(schedule_mp.name)
                if rawPath == '':
                    msg = (f"Data not found at: {datetime.strptime(self.date, '%d-%m-%Y').strftime('%Y-%m-%d')} "
                           f"{self.time}")
                else:
                    msg = ''
                log_message(
                    f"Schedule {sss.name} on site {site} skipped! {msg}",
                    all_logs=True,
                )

            log_message(f"--" * 50, all_logs=True)

            schedule_logger.unregister()
        del schedule_logger
        log_message(f"**" * 50, all_logs=True)
        log_message(
            f"Site {site} executed in {round(pytime.time() - time1, 4)} seconds",
            level="INFO",
        )

        site_logger.unregister()
        return

    def add_n_minutes_to_raw_path(self, minutes: int, site: str, date: str):
        """
        Adjusts the time component in the raw path of the specified site by subtracting a given number of minutes

        Args:
            minutes (int):  The number of minutes to subtract from the time component in the raw path
            site (str):     The site for which the raw path needs to be adjusted
        """
        self.rawPath[site] = self.rawPath[site].replace("\\", "/")
        list_raw_path = self.rawPath[site].split("/")

        D, M, Y = date.split("-")
        year_index = None
        for i, part in enumerate(list_raw_path):
            if part == str(Y):
                year_index = i
                break

        time_raw = list_raw_path[year_index + 3]
        time_raw = datetime.strptime(self.rawPath[site].split("/")[year_index + 3], "%H%M")
        time_raw = time_raw - timedelta(minutes=minutes)
        time_raw = time_raw.strftime("%H%M")
        list_raw_path[year_index + 3] = time_raw
        self.rawPath[site] = "/".join(list_raw_path)

    def phase_wait_start(
            self,
            site: str,
            date: str,
            time: str,
            max_wait_minutes: int = 15,
            minutes: int = 5,
    ):
        """
        Monitors and waits for the raw data corresponding to a specific site to become available.
        If the raw data is not found for the given site, the function attempts to search data generated
        earlier in 5-minute intervals until the maximum wait time is reached.

        Args:
            site (str): The site for which raw data is being searched.
            date (str): The date associated with the raw data.
            time (str): The time associated with the raw data.
            max_wait_minutes (int, optional): Maximum time (in minutes) to wait for the raw data to appear. Defaults
            to 15 minutes.
            minutes (int, optional): The interval (in minutes) to add to the raw data search path if the site is not
            found. Defaults to 5 minutes.
        Returns:
            bool: True if the raw data is found within the specified timeframe; False if it is not.
        """

        counter = 0
        while True:
            if self.sampledPath is not None and self.sampledPath != "":
                dpg.access.setSampledVolumePath(self.sampledPath[site], site, date, time)

            if os.path.isdir(self.rawPath[site]):
                # qua non vengono aggiunti siti che non hanno dati RAW..
                if site not in os.listdir(self.rawPath[site]):
                    log_message(
                        f"Site {site} RAW data NOT found at path {self.rawPath[site]}"
                    )
                    self.add_n_minutes_to_raw_path(minutes=minutes, site=site, date=date)
                    counter += minutes
                    if counter > max_wait_minutes:
                        log_message(
                            f"Site {site} RAW data NOT found in the previous {max_wait_minutes} minutes",
                            level="WARNING",
                        )
                        self.rawPath[site] = None
                        return False
                else:
                    log_message(
                        f"Site {site} RAW data found at path {self.rawPath[site]}"
                    )
                    if self.name == "RADAR_0":
                        # TODO: ACCROCCHIO FATTO PER FARLO FUNZIONARE. A regime deve essere letto dal radview/cfg
                        # new_raw_path = os.path.join(self.rawPath[site], site, "H")
                        if os.path.isdir(os.path.join(self.rawPath[site], site, "H")):
                            new_raw_path = os.path.join(self.rawPath[site], site, "H")
                        else:
                            new_raw_path = os.path.join(self.rawPath[site], site)
                        if not os.path.isdir(new_raw_path):
                            log_message(
                                f"RAW data not found at path {new_raw_path}",
                                level="ERROR",
                            )

                        dpg.access.setRawVolumePath(
                            new_raw_path, site, date=date, time=time
                        )
                        dpg.globalVar.GlobalState.update(
                            "LAST_RAW", new_raw_path, key=site
                        )
                    return True
            else:
                log_message(
                    f"Site {site} RAW data NOT found at path {self.rawPath[site]}"
                )
                self.add_n_minutes_to_raw_path(minutes=minutes, site=site, date=date)
                counter += minutes
                if counter > max_wait_minutes:
                    log_message(
                        f"Site {site} RAW data not found in the previous {max_wait_minutes} minutes",
                        level="WARNING",
                    )
                    self.rawPath[site] = None
                    return False

    def phase_1(self, center="", schedules=[], remove=False, logger=None):
        """
        This is the peripheral phase. It's executed on sites radar. This phase performs a serie of operation
        depending on the operative chain.

        Args:
            center (str, optional): Center name for local site retrieval (default is an empty string).
            schedules (list, optional): List of schedule file paths to be processed (default is an empty list).
            remove (bool, optional): Whether to remove the output directory if it exists (default is False).
            logger (logging.Logger, optional): Logger instance to record log messages (default is None).

        Returns:
            str: The path where the output is saved.
        """

        log_message("Starting phase1")

        # qua viene ottenuto correttamente il numero dei siti
        sites = self.sites

        if len(sites) <= 0:
            raise ValueError("No sites found")
            # sites = GetLocalSites(center_name=center)
        if len(sites) <= 0:
            return

        self.out_path = self.check_out_path()
        self.out_path = dpg.times.changePath(self.out_path, self.date, self.time)
        self.out_path = dpg.path.checkPathname(self.out_path)
        log_files = glob.glob(os.path.join(self.out_path, "*.log"))
        for log_file in log_files:
            os.remove(log_file)
        if self.remove:
            if os.path.isdir(os.path.join(self.out_path)):
                shutil.rmtree(self.out_path)

        for sched in self.schedules:
            if os.path.isdir(os.path.join(self.out_path, sched)):
                if self.remove:
                    shutil.rmtree(os.path.join(self.out_path, sched))

        self.checkSchedules(time=self.time, sites=sites)
        if len(self.schedules) <= 0:
            log_message(f"Cannot find any active schedule @ path {self.getSchedulePath()}")
            return

        # self.phase_1_init(time=self.time, date=self.date)

        for schedule_path in self.schedules:
            schedule_name = os.path.basename(schedule_path)
            if not os.path.exists(os.path.join(self.out_path, schedule_name)):
                self.copySchedule(schedule_path, self.out_path)

        logger.set_log_path(self.out_path)
        log_message(f"Output path: {self.out_path}", level="INFO")

        self.phase_1_init(time=self.time, date=self.date)

        if self.interactive:
            for site in sites:
                try:
                    site_execution_time = time.time()
                    self.run_site(site, date=self.date, time=self.time)
                except Exception as e:
                    stack_trace = traceback.format_exc()
                    log_message(
                        f"Exception occurred at {site}\n{stack_trace}",
                        level="EXCEPTION",
                        newlines=True,
                    )
                    raise e
        else:
            log_message("Run site running in parallel")

            loggers = dpg.log.get_loggers_from_stack()
            general_logger = loggers[0]
            partial_run_site = partial(
                self.run_site,
                date=self.date,
                time=self.time,
                general_logger=general_logger,
            )
            multiprocessing.set_start_method("spawn", force=True)
            print("Use of processes Pool:")
            with Pool(len(sites)) as pool:
                pool.map(partial_run_site, sites)

        return self.out_path

    @staticmethod
    def copySchedule(fromPath: str, toPath: str):
        """
        Copies the first level of directories and the second level of files from the source path to the destination
        path.

        Args:
            fromPath (str): The source path from which to copy the schedule.
            toPath (str):   The destination path to which the schedule should be copied.

        Example:
            src: RNN/ETM/calibration.txt
                 RNN/ETM/QUALITY/file.txt
            dst: RNN/ETM/calibration.txt
        """
        schName = os.path.basename(fromPath)
        out = os.path.join(toPath, schName)

        folders = os.listdir(fromPath)
        os.makedirs(out)
        for f in folders:
            file = os.path.join(fromPath, f)
            if os.path.isdir(file):
                # Create first level folders
                dst_folder = os.path.join(out, f)
                os.makedirs(dst_folder)
                # Copy second level files
                src_folder = os.path.join(fromPath, f)
                files = [
                    f
                    for f in os.listdir(src_folder)
                    if os.path.isfile(os.path.join(src_folder, f))
                ]
                for file in files:
                    shutil.copy2(
                        os.path.join(src_folder, file), os.path.join(dst_folder, file)
                    )
            else:
                # Copy first level files
                shutil.copy2(file, out)
        return

    def correct_time(self) -> int:
        """
        Updates the `date` and `time` attributes of the object if they are not set,
        and adjusts the output path based on these values and a potential delay.

        This function ensures that `date` and `time` are set before computing the
        `out_path`. If `date` or `time` is missing, it assigns the current date and time
        respectively. If a delay is specified, it adjusts the `date` and `time` accordingly.

        Returns:
            int: Always returns 0, indicating the function's completion status.
        """

        if self.out_path:
            return 0

        if not self.date:
            date = ""
        if not self.time:
            time = ""

        if self.date == "" or self.time == "":
            date = datetime.now()
            date = date.strftime("%d-%m-%Y")
            self.date = date

        if self.delay:
            if self.delay != 0:
                dpg.times.addMinutesToDate(
                    self.date, self.time, -self.delay
                )  # todo: controllare

        self.out_path = dpg.times.changePath(self.out_path, self.date, self.time)

        return 0

    def phase_2(self):
        """
        Phase completed at the end of phase 1. Compressed data from phase 1 is collected. In this phase,
        the aggregate of what is received through the mosaic process takes place.

        This method sets up the output path, initializes and executes schedules,
        and handles logging. If no schedules are found or if the schedules are not
        of the expected type, it rechecks and initializes the schedules.

        Returns:
            None: This method does not return a value.

        Raises:
            Exception: If an error occurs during schedule execution, it logs the exception details.
        """

        log_message("/==/" * 30)
        log_message("/==/" * 30)
        log_message("Starting phase2")

        sites = self.sites
        if sites is None:
            sites = []

        self.out_path = self.check_out_path()
        self.out_path = dpg.times.changePath(self.out_path, self.date, self.time)
        self.out_path = dpg.path.checkPathname(self.out_path)

        self.phase_2_wait_start()

        phase = 2

        if len(self.schedules) == 0 or all(
                not isinstance(elem, dpg.schedule__define.Schedule)
                for elem in self.schedules
        ):
            self.checkSchedules(time=self.time, sites=sites)
            self.phase_1_init(date=self.date, time=self.time)

        for sss in self.schedules:
            schedule_execution_time = pytime.time()
            formatted_schedule_time = datetime.utcnow().strftime(
                "%Y-%m-%d %H:%M:%S.%f"
            )[:-3]

            attr = dpg.attr.loadAttr(sss.schedule_path, dpg.cfg.getScheduleDescName())
            origin, _, _ = dpg.attr.getAttrValue(attr, "origin", "")

            # mosaic non va bene, sostituire con origin_logger? (Mimmo)
            mosaic_logger = dpg.log.DynamicLogger(name=os.path.join(sss.name, "MOSAIC"))
            mosaic_logger.initialize()
            mosaic_logger.set_log_path(self.out_path)

            log_message(f"==" * 50, all_logs=True)
            log_message(
                f"Schedule {sss.name} for site {origin} started at time {formatted_schedule_time}",
                all_logs=True,
            )

            sss.schedule_init(
                site=origin,
                date=self.date,
                time=self.time,
                # rawPath=self.rawPath,
                no_log=True,
            )
            try:
                sss.schedule_exec(site=origin, phase=phase)
            except Exception as e:
                stack_trace = traceback.format_exc()
                log_message(
                    f"An exception has occurred during phase2:\n{stack_trace}",
                    level="EXCEPTION",
                    newlines=True,
                    all_logs=True,
                    general_log=False,
                )
                log_message(
                    f"An exception has occurred during phase2:\n",
                    level="EXCEPTION",
                    newlines=True,
                    only_general=True,
                )

            sss.schedule_end(site=origin, phase=phase, export=self.export)
            log_message(
                f"{origin} for schedule {sss.name} executed in "
                f"{round(time.time() - schedule_execution_time, 4)} seconds",
                level="INFO",
                all_logs=True,
            )
            log_message(f"--" * 50, all_logs=True)

            with open(os.path.join(sss.outPath, sss.name, "phase2.end"), "w"):
                log_message("Created phase2.end", only_general=True)

    def phase_2_wait_start(self):
        """
        Updates the site files within the directory "out_path". Use method "replaceSiteFiles" contained in dpg.

        Returns:
            None
        """
        # if sys_wait:
        #     self.sys_wait()
        #
        # if prdWait:
        #     self.prd_wait()

        dpg.dpg.replaceSiteFiles(self.out_path)

    def new_phase_wait_start(self, site, phase_num, out_path, date_, time_, rawPath='', interactive=False):
        exclude = None
        status = None
        lastPath = None

        if self.name is None or self.name == '' or date_ is None or time_ is None:
            return rawPath

        if interactive:
            path = dpg.path.getDir('LOCAL_QUERY')
            attr = dpg.attr.loadAttr(path, 'run.txt')
            prefix = ''
        else:
            path = dpg.path.getDir('CFG_SCHED', with_separator=True)
            attr = dpg.attr.loadAttr(path, self.name + '.txt')
            prefix = 'phase_' + str(phase_num)

        prdWait, _, _ = dpg.attr.getAttrValue(attr, 'prdWait', 0, prefix=prefix)
        rawWait, _, _ = dpg.attr.getAttrValue(attr, 'rawWait', 0, prefix=prefix)
        sysWait, _, _ = dpg.attr.getAttrValue(attr, 'sysWait', '', prefix=prefix)

        sPath, _, _ = dpg.attr.getAttrValue(attr, 'sampledPath', '', prefix=prefix)
        if sPath == '<current>':
            sPath = out_path

        sampledSub, _, _ = dpg.attr.getAttrValue(attr, 'sampledSub', '', prefix=prefix)
        sampledCheck, _, _ = dpg.attr.getAttrValue(attr, 'sampledCheck', 0, prefix=prefix)
        rPath, _, _ = dpg.attr.getAttrValue(attr, 'rawPath', '', prefix=prefix)
        rawSub, _, _ = dpg.attr.getAttrValue(attr, 'rawSub', '$site/H', prefix=prefix)

        minFreq, _, _ = dpg.attr.getAttrValue(attr, 'minFreq', 0, prefix=prefix)
        altFreq, _, _ = dpg.attr.getAttrValue(attr, 'altFreq', 0, prefix=prefix)
        maxMin, _, _ = dpg.attr.getAttrValue(attr, 'maxMin', 15, prefix=prefix)
        lastCheck, _, _ = dpg.attr.getAttrValue(attr, 'lastCheck', 40, prefix=prefix)
        rawFile, _, _ = dpg.attr.getAttrValue(attr, 'rawFile', '', prefix=prefix)
        endFile, _, _ = dpg.attr.getAttrValue(attr, 'rawEnd', '', prefix=prefix)

        if not interactive:
            if site != '':
                rawWait, _, _ = dpg.attr.getAttrValue(attr, 'rawWait', rawWait, prefix=site)
                minFreq, _, _ = dpg.attr.getAttrValue(attr, 'minFreq', minFreq, prefix=site)
                altFreq, _, _ = dpg.attr.getAttrValue(attr, 'altFreq', altFreq, prefix=site)
                maxMin, _, _ = dpg.attr.getAttrValue(attr, 'maxMin', maxMin, prefix=site)
                lastCheck, _, _ = dpg.attr.getAttrValue(attr, 'lastCheck', lastCheck, prefix=site)
                tmp, _, _ = dpg.attr.getAttrValue(attr, 'rawExclude', '', prefix=site)

                if tmp != '':
                    exclude = tmp
                if rawWait > 0:
                    checkStatus, _, _ = dpg.attr.getAttrValue(attr, 'checkStatus', 0, prefix=site)
                    if checkStatus > 0:
                        stAttr = dpg.attr.loadAttr(pathname=dpg.cfg.getDefaultStatusFile(site))
                        status, _, _ = dpg.attr.getAttrValue(stAttr, 'status', -1)

        else:
            rawWait = 0
            prdWait = 0
            if rawPath is not None and rawPath != '':
                rPath = ''

        if rPath != '':
            rawPath, lastPath = self.rawWait(rPath, site, date_, time_, maxMin, rawWait, minFreq, altFreq, lastCheck,
                                             sub=rawSub, check_files=rawFile, exclude=exclude, status=status)
            dpg.access.setRawVolumePath(rawPath, site, date_, time_)

        if sysWait is not None and sysWait != '':
            log_message("DA IMPLEMENTARE sys_wait", level='ERROR', all_logs=True)
            self.sys_wait()

        if sPath != '':
            dpg.access.setSampledVolumePath(sPath, site, date_, time_, maxMin=maxMin, sub=sampledSub,
                                            lastPath=lastPath, sampledCheck=sampledCheck)

        if phase_num == 2:
            if prdWait > 0:
                log_message("TODO PRD_WAIT > 0", level='ERROR', all_logs=True)
                self.prd_wait()
            self.replaceSiteFiles(out_path)

        return rawPath

    @staticmethod
    def rawWait(searchPath, site, date_, time_, maxMin, seconds, minFreq, altFreq, lastCheck, sub, check_files,
                exclude, status):
        lastPath = ''

        if exclude is not None:
            if dpg.times.isTime(time, exclude) <= 0:
                rawPath = ''
                return rawPath, lastPath

        rawPath = dpg.path.checkPathname(searchPath, with_separator=False)
        if not re.search(r"20", rawPath) and rawPath is not None and rawPath != '':
            rawPath = os.path.join(rawPath, '2000')

        rawPath = dpg.times.changePath(rawPath, date_, time_)

        if sub is not None:
            pos = sub.find(r"$site")
            if pos >= 0:
                sss = sub[:pos] + site + sub[pos + 5:]
            else:
                sss = sub
            if rawPath is not None and rawPath != '':
                rawPath = dpg.path.checkPathname(os.path.join(rawPath, sss), with_separator=False)

        if lastCheck > 0:
            lastPath, current = dpg.utility.getLastPath(rawPath, date_, time_, lastCheck, check_files, get_current=True)
            if current > 0:
                return rawPath, lastPath
            if lastPath == '':
                log_message(f"Cannot find Raw Data @ {rawPath}", level='ERROR', all_logs=True)
                if rawPath is None or rawPath == '':
                    log_message(f"Raw path was not set correctly. Check value @ {DATAMET_RADVIEW_PATH}/cfg/schedules",
                                level='ERROR', all_logs=True)
                rawPath = ''
                return rawPath, lastPath

        if seconds:
            ttt, _, mm = dpg.times.checkTime(time_)
            if minFreq > 0:
                if mm / minFreq > mm / int(minFreq):
                    seconds = 0
            if status is not None and status != '':
                if status == 0 and mm / 10. > mm / 10:
                    seconds = 0

        if dpg.globalVar.GlobalState.DATA_SEC is None:
            dpg.globalVar.GlobalState.update("DATA_SEC", 0)

        while dpg.globalVar.GlobalState.DATA_SEC < seconds:
            lastPath, current = dpg.utility.getLastPath(rawPath, date_, time_, altFreq, check_files, get_current=True)

            if current > 0:
                return rawPath, lastPath
            if lastPath != '':
                rawPath = ''
                return rawPath, lastPath
            # time.sleep(5)
            dpg.globalVar.GlobalState.update("DATA_SEC", dpg.globalVar.GlobalState.DATA_SEC + 5)

        lastPath, current = dpg.utility.getLastPath(rawPath, date_, time_, maxMin, check_files, get_current=True)
        if current > 0:
            return rawPath, lastPath

        rawPath = ''

        return rawPath, lastPath

    @staticmethod
    def replaceSiteFiles(path):

        sched = dpg.utility.getAllDir(path)
        for sss in sched:
            prod = dpg.utility.getAllDir(sss)
            for ppp in prod:
                sites = dpg.utility.getAllDir(ppp, withoutPath=True)
                if len(sites) > 1:
                    tags = ['sites' * len(sites)]
                    dpg.attr.saveAttr(ppp, 'sites.txt', tags, sites, replace=True)
