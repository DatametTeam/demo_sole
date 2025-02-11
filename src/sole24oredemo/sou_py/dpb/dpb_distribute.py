import os

"""
Funzioni ancora da portare
FUNCTION ReplaceProdTags 
FUNCTION GetDestName 
PRO DISPATCH_FILE 
PRO DPB_DISTRIBUTE 
PRO REEXPORT //UNUSED
"""


def searchAuxFile(source_file, dest_name):
    """
    Searches for auxiliary files related to a given source file and generates corresponding destination file names.

    This function locates files with specific extensions related to the source shapefile (.shp) and prepares their
    corresponding destination file names based on the given destination name. If the source or destination does not
    include a '.shp' or '.SHP' extension, the function returns indicating no files were found.

    Args:
        source_file (str): The name of the source file, typically a shapefile with a '.shp' extension.
        dest_name (str): The base name for the destination files, where auxiliary files will be renamed.

    Returns:
        int: The number of valid auxiliary files found.
        list of str: A list of full paths to the valid auxiliary files.
        list of str: A list of new names for the auxiliary files based on the destination name.

    Note:
        The function searches for auxiliary files with the following extensions: ['.dbf', '.shx', '.prj', '.sbn', '.sbx'].
        If the destination name contains '.SHP', the extensions are converted to uppercase.

    """

    pos = source_file.find(".shp")
    if pos <= 0:
        return 0, None, None

    in_file = source_file[:pos]
    inext = [".dbf", ".shx", ".prj", ".sbn", ".sbx"]
    outext = inext

    pos = dest_name.find(".shp")
    if pos <= 0:
        pos = dest_name.find(".SHP")
        if pos <= 0:
            return 0, None, None
        outext = [ext.upper() for ext in outext]

    aux_name = dest_name[:pos]
    n_files = len(outext)
    aux_files = []
    dest_files = []
    valids = 0

    for fff in range(n_files):
        new_file = in_file + inext[fff]
        if os.path.isfile(new_file):
            aux_files.append(new_file)
            dest_files.append(aux_name + outext[fff])
            valids += 1

    return valids, aux_files, dest_files
