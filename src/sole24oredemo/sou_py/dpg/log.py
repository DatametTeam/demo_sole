import inspect
import logging
import os
import sys
import time
from threading import Lock

import numpy as np

from sou_py import dpg


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    ORANGE = "\033[38;5;208m"  # Adding orange color using 256-color mode


class LoggerRegistry:
    def __init__(self):
        self._loggers = {}
        self._lock = Lock()

    def register(self, logger):
        with self._lock:
            if logger.name == 'general':
                self._loggers['general'] = logger

            elif logger.process_pid not in self._loggers.keys():
                self._loggers[logger.process_pid] = [logger]
            else:
                self._loggers[logger.process_pid].append(logger)

    def unregister(self, logger):
        with self._lock:
            if logger.process_pid in self._loggers:
                if isinstance(self._loggers[logger.process_pid], list):
                    self._loggers[logger.process_pid] = [i for i in self._loggers[logger.process_pid] if i.name != logger.name]
                else:
                    self._loggers.pop(logger.process_pid)

    def get_active_loggers(self):
        with self._lock:
            if len(self._loggers) == 1:
                active_loggers = [self._loggers['general']]
            else:
                active_loggers = [self._loggers['general']] + self._loggers[os.getpid()]
            return active_loggers

    def unregister_all_loggers(self):
        with self._lock:
            self._loggers = {}


logger_registry = LoggerRegistry()


class DynamicLogger:
    def __init__(self, name):
        self.log_path = None
        self.name = name
        self.creation_time = time.time()
        self.process_pid = os.getpid()

        self.logger = logging.getLogger("dynamic_logger")
        self.logger.setLevel(logging.DEBUG)

        self.logFormatter = logging.Formatter(
            "%(asctime)s [%(levelname)s]  %(message)s"
        )
        self.logFormatter.default_msec_format = "%s.%03d"

        self.first_message = True
        self.before_messages = []

        logger_registry.register(self)

    def append_previous_messages(self, message: str):
        """
        Appends a message to the list of previously logged messages.

        This function stores a message in the `before_messages` list, which holds messages
        that were generated before the logger was fully initialized.

        Args:
            message (str): The message to be appended to the `before_messages` list.

        Returns:
            None
        """
        self.before_messages.append(message)

    def initialize(self):
        """
        Initializes the logger for the current instance.

        This function first shuts down the logging system. It then retrieves or creates a logger based
        on the instance's `name` and removes any existing handlers attached to it. The logger is
        then set to the DEBUG level.

        Returns:
            None
        """
        logging.shutdown()

        log_ = logging.getLogger(self.name + "_logger")
        if log_:
            # rimozione degli handler
            for handler in log_.handlers:
                handler.close()
                log_.removeHandler(handler)
        self.logger = log_
        self.logger.setLevel(logging.DEBUG)

    def set_log_path(self, log_path):
        """
        Sets the log file path for the logger and handles any previous messages.

        This function configures the logger to write logs to a new file at the specified `log_path`.
        It creates a new file handler, sets its logging level to DEBUG, and applies the existing
        log formatter. If any messages were logged before the logger was fully initialized,
        these messages are logged to the new file.

        Args:
            log_path (str): The directory path where the log file should be created.

        Returns:
            None
        """
        if self.logger:
            # Create a new file handler with the updated log path
            handler = logging.FileHandler(os.path.join(log_path, self.name + ".log"))
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(self.logFormatter)

            # Add the new handler to the logger
            self.logger.addHandler(handler)

            self.log_path = log_path

            if self.first_message:
                for prev_msg in self.before_messages:
                    self.logger.info(prev_msg)
                    print(prev_msg)
                self.first_message = False
        else:
            print("Logger is not initialized.")

    def unregister(self):
        logger_registry.unregister(self)


def log_msg(message, logger=None, level="INFO"):
    """
    Logs a message with a specified logging level and optionally prints it to the console.

    This function logs a message using the provided `logger` and the specified
    logging `level`. If the `logger` is not provided, a default message indicating that
    the logger is not initialized will be printed. Additionally, if the `logger` name
    is "general", the message will also be printed to the console, potentially with color
    coding depending on the log level.

    Args:
        message (str): The message to be logged.
        logger (Optional): The logger object used to log the message. If `None`,
                           a default message is printed instead.
        level (str): The logging level for the message. Accepted values are "INFO",
                     "DEBUG", "ERROR", "WARNING", and "EXCEPTION". The default is "INFO".

    Returns:
        None
    """
    if logger:
        if level == "INFO":
            logger.logger.info(message)
            if logger.name == "general":
                print(message)
        if level == "DEBUG":
            logger.logger.debug(message)
            if logger.name == "general":
                print(bcolors.OKCYAN + "DEBUG" + bcolors.ENDC + message)
        if level == "ERROR":
            logger.logger.error(message)
            if logger.name == "general":
                print(bcolors.FAIL + message + bcolors.ENDC)
        if level == "WARNING":
            logger.logger.warning(message)
            if logger.name == "general":
                print(bcolors.WARNING + message + bcolors.ENDC)
        if level == "WARNING+":
            logger.logger.warning(message)
            if logger.name == "general":
                print(bcolors.ORANGE + message + bcolors.ENDC)
        if level == "EXCEPTION":
            logger.logger.exception(message)
            if logger.name == "general":
                print(bcolors.FAIL + message + bcolors.ENDC)

    else:
        print("Logger is not initialized.")


def log_message(
        message: str,
        level: str = "INFO",
        newlines: bool = False,
        all_logs: bool = False,
        general_log: bool = True,
        only_general: bool = False,
):
    """
    Logs a message across multiple loggers based on specified criteria.

    This function logs a message using a set of loggers obtained from the current call stack.
    It allows for flexible logging behavior, including logging to all available loggers,
    only the general logger, or the most recently created logger. The message can be logged
    at different levels (INFO, DEBUG, ERROR, WARNING, EXCEPTION).

    Args:
        message (str): The message to be logged.
        level (str): The logging level for the message. Defaults to "INFO".
        newlines (bool): If `False`, newlines in the message will be removed. Defaults to `False`.
        all_logs (bool): If `True`, the message is logged to all loggers in the stack. Defaults to `False`.
        general_log (bool): If `True`, the message is logged to the general logger (if it exists).
                            Defaults to `True`.
        only_general (bool): If `True`, the message is logged only to the general logger.
                             Overrides other logging options. Defaults to `False`.

    Returns:
        None
    """
    if not newlines:
        message = message.replace("\n", "")
    if not message:
        return

    # loggers = get_loggers_from_stack()
    loggers = logger_registry.get_active_loggers()

    ordered_logs = sorted(loggers, key=lambda obj: obj.creation_time)

    if not isinstance(ordered_logs, list) or len(ordered_logs) == 0:
        print(message)
        return
    else:
        if ordered_logs[0].name == "general":
            if ordered_logs[0].log_path is None:
                ordered_logs[0].append_previous_messages(message)
                return

    # qua fa un log doppia
    if all_logs:
        for logger in ordered_logs:
            if not general_log and logger.name == "general":
                pass
            else:
                log_msg(message, logger=logger, level=level)

    elif only_general:
        log_msg(message, logger=ordered_logs[0], level=level)

    else:
        if len(ordered_logs) == 1 and ordered_logs[0].name == "general":
            log_msg(message, logger=ordered_logs[0], level=level)
        else:
            log_msg(message, logger=ordered_logs[-1], level=level)
            if general_log:
                log_msg(message, logger=ordered_logs[0], level=level)


def get_loggers_from_stack():
    """
    Retrieves logger instances from the current call stack.

    This function inspects the current call stack and collects any logger instances
    found in the local variables of the stack frames. It looks for variables containing
    the word "logger" and filters the collected loggers to ensure they are instances of
    `dpg.log.DynamicLogger`. The function stops searching when it reaches the main module.

    Returns:
        List[dpg.log.DynamicLogger]: A list of unique logger instances found in the stack.
    """
    stack = inspect.stack()
    loggers = []
    for frame_info in stack[1:]:
        module = inspect.getmodule(frame_info.frame)
        if module:
            module_name = module.__name__
        else:
            continue
        function_name = frame_info.function
        frame = frame_info.frame
        local_vars = frame.f_locals
        for key, val in local_vars.items():
            if "logger" in key:
                loggers.append(val)

        if module_name == "run" and function_name == "<module>":
            break  # Stop the loop when reaching the main module

    loggers = [log for log in loggers if isinstance(log, dpg.log.DynamicLogger)]
    return list(set(loggers))


def initialize_general_log():
    """
    Initializes a general-purpose logging instance. The function creates a dynamic logger
    named "general," initializes it, and appends a message indicating that the general log
    has been initialized.


    Returns:
        logger (DynamicLogger): An initialized logger instance with the name "general" and
                                a message appended indicating initialization.
    """

    logger_registry.unregister_all_loggers()
    logger = DynamicLogger(name="general")
    logger.initialize()
    logger.append_previous_messages("General log initialized")
    return logger
