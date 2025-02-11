import glob
from pathlib import Path

import numpy as np
import shutil
import os
import sou_py.dpg as dpg

"""
Funzioni ancora da portare
FUNCTION CopyDirectory 			dpg/UtilityDir.pro
FUNCTION CopyTemplate 			dpg/UtilityDir.pro
FUNCTION CopyToClipBoard 		dpg/UtilityDir.pro
FUNCTION CreateDirectory 		dpg/UtilityDir.pro
FUNCTION CutToClipBoard 		dpg/UtilityDir.pro
FUNCTION DirDelete 				dpg/UtilityDir.pro
FUNCTION GetAllDir 				dpg/UtilityDir.pro
FUNCTION GetClipBoard 			dpg/UtilityDir.pro
FUNCTION GetOldDir 				dpg/UtilityDir.pro
FUNCTION GetTemporaryPath 		dpg/UtilityDir.pro
PRO TouchFile 				    dpg/UtilityFile.pro
FUNCTION PasteFromClipBoard 	dpg/UtilityDir.pro      // UNUSED
PRO RemoveSubDir 				dpg/UtilityDir.pro      // UNUSED
"""


def getLastPath(
        path: str,
        date: str,
        time: str,
        nMin: int,
        check_files: list = [],
        get_current: list = False,
) -> str | tuple[str, bool]:
    """
    Finds the last valid path within a specified time range, checking the existence of certain files.

    This function iterates backwards in time from a given date and time, checking for the existence of specified files
    in paths determined by the date and time. It searches for a valid path within 'nMin' minutes before the specified
    date and time. If 'get_current' is True, the function also returns a flag indicating whether the returned path
    is the current one.

    Args:
        path (str): The base path used for generating the paths to check.
        date (str): The starting date for the search.
        time (str): The starting time for the search.
        nMin (int): The number of minutes to go back in time from the specified date and time.
        check_files (list[str]): A list of filenames to check for in each generated path.
        get_current (bool, optional): If True, returns a flag indicating if the returned path is the current one.
        Defaults to False.

    Returns:
        str (and int if get_current is True): The last valid path found within the specified time range.
        If 'get_current' is True, also returns a flag (1 for current, 0 for not current).
        Returns an empty string ('') if no valid path is found.

    Raises:
        None: This function does not explicitly raise any exceptions but relies on 'changePath' and
        'addMinutesToDate' from 'dpg.times'.
    """
    ddd = date
    ttt = time
    altPath = dpg.times.changePath(path, ddd, ttt)
    current = 1

    for fff in range(nMin):
        if checkAllFiles(altPath, check_files):
            if get_current:
                return altPath, current
            return altPath
        ddd, ttt = dpg.times.addMinutesToDate(ddd, ttt, -1)
        altPath = dpg.times.changePath(path, ddd, ttt)
        current = 0

    if checkAllFiles(altPath, check_files):
        if get_current:
            return altPath, current
        return altPath

    current = 0
    if get_current:
        return "", current
    return ""


def checkAllFiles(path: str, check_files) -> bool:
    """
    Checks if all specified files exist in a given path.

    This function verifies the existence of specified files within a given directory ('path'). It supports
    checking multiple files if 'check_files' is a list or a single file if it is a string. The function returns
    True only if all specified files exist in the directory.

    Args:
        path (str): The path of the directory to check for file existence.
        check_files (list[str] or str): The file(s) to check for. Can be a list of file names or a single file name.

    Returns:
        bool: True if all specified files exist in the directory, False otherwise.

    Raises:
        None: This function does not explicitly raise any exceptions but relies on file existence checks using
        'os.path'.
    """
    if not isinstance(path, str) or path == "":
        return False
    if not (os.path.exists(path) and os.path.isdir(path)):
        return False

    if isinstance(check_files, list):
        for single_file in check_files:
            if single_file != "":
                file_path = dpg.path.getFullPathName(path, single_file)
                if not (os.path.exists(file_path) and os.path.isfile(file_path)):
                    return False
    if isinstance(check_files, str):
        if check_files != "":
            file_path = dpg.path.getFullPathName(path, check_files)
            if not (os.path.exists(file_path) and os.path.isfile(file_path)):
                return False

    return True


def delete_path(path: str):
    """
    Deletes a specified path, which can be a file or directory.

    This function attempts to delete the specified path ('path'). It handles different cases: if the path
    is a file, it is removed; if it's a directory, it is removed if empty, otherwise, it is deleted recursively.
    Any errors encountered during the deletion process are caught and printed.

    Args:
        path (str): The path to be deleted. It can be a file or a directory.

    Raises:
        Exception: Catches and prints exceptions that occur during the deletion process, without re-raising them.
    """
    """
    Funzione creata appositamente per replicare il comportamento della chiamata
    FILE_DELETE, path, /RECURSIVE, /ALLOW_NONEXISTENT
    """
    try:
        if os.path.exists(path):
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                os.rmdir(path)
            else:
                # delete recursively
                os.removedirs(path)
    except Exception as e:
        print(f"An error occurred while deleting {path}: {e}")


def deleteFile(filename: bool):
    """
    Deletes the specified file.

    This function attempts to delete the file specified by 'filename'. If the file does not exist,
    the function silently handles the 'FileNotFoundError' without raising an exception.

    Args:
        filename (str): The path of the file to be deleted.

    Raises:
        None: This function does not explicitly raise any exceptions. It silently handles 'FileNotFoundError'.
    """
    try:
        os.remove(filename)
    except FileNotFoundError:
        pass


def removeFile(path: str, name: str):
    """
    Deletes a file specified by its path and name.

    This function constructs the full path of a file from its path and name, and then attempts to delete the file.
    If the file does not exist (FileNotFoundError), the function silently ignores the error.

    Args:
        path (str): The directory path where the file is located.
        name (str): The name of the file to be deleted.

    Raises:
        None: This function does not explicitly raise any exceptions. It handles FileNotFoundError silently.
    """
    full_path = dpg.path.getFullPathName(path, name)
    try:
        os.remove(full_path)
    except FileNotFoundError:
        pass


def copyFile(oldname: str, newname: str, silent: bool = False):
    """
    Copies a file from one location to another.

    This function attempts to copy a file specified by 'oldname' to a new location specified by 'newname'.
    If 'silent' is False, the function checks for the existence of the source file and prints a message
    if the file does not exist or if an IOError occurs during copying.

    Args:
        oldname (str): The path of the source file.
        newname (str): The path for the destination file.
        silent (bool, optional): If True, suppresses printing of error messages. Defaults to False.

    Raises:
        None: This function does not explicitly raise any exceptions. It prints error messages unless 'silent' is True.
    """
    if not silent:
        if not os.path.isfile(oldname):
            print(f"The file {oldname} does not exist.")
            return

    try:
        shutil.copy2(oldname, newname)
    except IOError as e:
        if not silent:
            print(f"Error copying file {oldname} to {newname}: {e}")


def readStringFile(filename: str) -> list:
    """
    Reads a text file and returns its contents as a list of lines.

    This function opens a file specified by 'filename' and reads its contents line by line. Each line is
    stripped of leading and trailing whitespace. If the file does not exist, the function returns an empty list.

    Args:
        filename (str): The path of the file to be read.

    Returns:
        list[str]: A list of strings, where each string is a line from the file, or an empty list if the file does
        not exist.

    Raises:
        None: This function does not explicitly raise any exceptions. It handles FileNotFoundError by returning an
        empty list.
    """
    try:
        with open(filename, "r") as f:
            lines = f.readlines()
        return [line.strip() for line in lines]
    except FileNotFoundError:
        return []


def delete_folders(folders: list):
    """
    Deletes the specified folders and all their contents.

    This function iterates over a list of folder paths, and for each path, it deletes
    the folder and all its contents if the folder exists.

    Args:
        folders (list): A list of folder paths to delete. Each element should is a Path object.

    Returns:
        None
    """
    for folder in folders:
        folder = Path(folder)
        if folder.exists():
            shutil.rmtree(folder)


def delete_files_in_sub_folder(folder: Path):
    """
    Deletes all files in the specified folder and its subfolders, excluding '.gitkeep' files.

    This function searches through the specified folder and all its subfolders to find
    and delete all files, except those named '.gitkeep'. It then prints the number of
    files deleted.

    Args:
        folder (Path): The path to the folder in which files will be deleted.

    Returns:
        None
    """
    files = glob.glob(os.path.join(folder, "**", "*"))
    files_to_delete = [
        f for f in files if os.path.basename(f) != ".gitkeep" and os.path.isfile(f)
    ]
    for file_path in files_to_delete:
        os.remove(file_path)
    print(f"Removed {len(files_to_delete)} files in {folder}")


def getFilesDir(pathroot):
    """
    Retrieves the absolute paths of all files within a specified directory.

    Args:
        pathroot (str): The root directory path from which to list files.

    Returns:
        tuple: A tuple containing:
            - list of str: The absolute paths of files in the specified directory.
            - int: The count of files in the directory.
        If the directory does not exist, returns an empty string and zero count.

    Notes:
        - The function first checks if the provided directory path exists.
        - If the directory exists, it returns the absolute paths of each file within it and the total number of files.
        - If the directory does not exist, it returns an empty string and skips further operations.
    """
    if not os.path.isdir(pathroot):
        return ''
    files = [os.path.abspath(os.path.join(pathroot, f)) for f in os.listdir(pathroot)]
    count = len(files)
    return files, count


def getAllDir(dir, withoutPath=False):
    if not os.path.isdir(dir):
        return 0
    if withoutPath:
        full = 0
    else:
        full = 1

    subdir = [os.path.join(dir, d) + os.sep
              for d in os.listdir(dir)
              if os.path.isdir(os.path.join(dir, d))]

    if full:
        subdir = [os.path.abspath(d) for d in subdir]
    return subdir
