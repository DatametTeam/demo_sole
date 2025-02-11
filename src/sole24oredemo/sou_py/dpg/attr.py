import numpy as np

import sou_py.dpg as dpg
import os
import copy

from sou_py.dpg.attr__define import Attr
from sou_py.dpg.log import log_message

"""
Funzioni ancora da portare
PRO RemoveComments 
PRO ReplaceExistingTags 
"""


def createAttrStruct(varnames: list, values: list) -> dict:
    """
    Creates a dictionary structure where variable names are the keys and their corresponding values are the values.

    Args:
        varnames (list): A list of variable names to be used as keys in the dictionary. If a single string is
        provided, it is converted to a list.
        values (list):   A list of values corresponding to each variable name. If a non-list value is provided,
        it is converted to a list.

    Returns:
        dict:            A dictionary where the keys are the variable names (converted to uppercase) and the values
        are their corresponding values.
                         If a key appears multiple times, the values are stored in a list. Empty string keys are
                         ignored.
    """
    mydict = {}
    if not isinstance(varnames, (str, list)):
        return mydict
    if isinstance(varnames, str):
        varnames = [varnames]
    if not isinstance(values, list):
        values = [values]
    for key, val in zip(varnames, values):
        if key == "" or not isinstance(key, str):
            continue
        if key not in mydict.keys():
            mydict[key] = val
        else:
            tmp = mydict[key]
            if not isinstance(tmp, list):
                mydict[key] = [tmp]
            mydict[key].append(val)
    return mydict


def compareDate(lastRead: float, path: str = "", name: str = "", pathname: str = ""):
    """
    Given a path and a name (or a pathname=path+name), it returns the maximum value batween the last modified date
    and lastRead.

    Args:
        last_read (float): The time against which to compare the file's modification time.
        path (str): The directory path where the file is located.
        name (str): The name of the file.
        pathname (str, optional): The full path to the file. If not provided, it's constructed from `path` and `name`.

    Returns:
        None or int:
    """
    file_date = None
    if pathname is None and path is None and name is None:
        # raise TypeError('Error: either pathname or path and name must be provided')
        print("Error: either pathname or path and name must be provided")
        return
    if (not isinstance(pathname, str)) and (
            (not isinstance(path, str)) and (not isinstance(name, str))
    ):
        print("Error: either pathname or path and name must be strings")
        return
    if (pathname == "") and ((path == "") and (name == "")):
        print("Error: either pathname or path and name must be non empty strings")
        return
    if pathname == "":
        pathname = dpg.path.getFullPathName(path, name)

    file_date = os.path.getmtime(pathname)
    if lastRead is None:
        return file_date
    if lastRead < file_date:
        return file_date
    return lastRead


def formatIsAscii(format: str) -> bool:
    """
    Determines if the provided format is considered ASCII-based.

    Args:
        format (str): The format string to check.

    Returns:
        bool: It returns True if format is in numerics, False otherwise
    """
    if not isinstance(format, str):
        return False
    numerics = ["TXT", "LUT", "VAL", "VAL_256", "TIM", "STR", "ASCII"]
    if format.upper() in numerics:
        return True
    return False


def loadAttr(path: str = None,
             name: str = None,
             pathname: str = None) -> dict:
    """
    Load attributes from a file, parsing them into a dictionary.

    Args:
        path (str): The directory path where the attributes file is located.
        name (str): The name of the attributes file.
        pathname (str, optional): The full path to the file. If not provided, it's constructed from `path` and `name`.

    Returns:
        dict: A dictionary containing attribute names and values.
    """
    data, err = dpg.io.read_strings(path, name, pathname=pathname)
    count = len(data)
    if (err > 0) or (count == 0):
        return None

    pos = [string.find("=") for string in data]
    countpos = len(pos)

    if countpos != count:
        return None

    tags = [data[i][: pos[i]].strip() for i in range(count)]
    values = [data[i][pos[i] + 1:].strip() for i in range(count)]
    tags = [tag for tag in tags]

    # Sostituito chiamando createAttrStruct(tags, values)
    # struct = {}
    # for i in range(count):
    #    if tags[i] in struct.keys():
    #        if isinstance(struct[tags[i]], list):
    #            struct[tags[i]].append(values[i])
    #        else:
    #            struct[tags[i]] = [struct[tags[i]], values[i]]
    #    else:
    #        struct[tags[i]] = values[i]
    # return struct

    return createAttrStruct(tags, values)


def saveAttr(
        path: str,
        name: str,
        varnames: dict,
        values: dict,
        append: bool = False,
        replace: bool = False,
        pathname: str = None,
) -> bool:
    """
    Saves attributes to a specified file, with options to append, replace, and handle the file path

    Args:
        path (str):               The directory path where the file is to be saved
        name (str):               The name of the file
        varnames (dict):          Dictionary of variable names to be saved as keys
        values (dict):            Dictionary of values corresponding to each variable name
        append (bool, optional):  If True, the attributes will be appended to the file. Defaults to False
        replace (bool, optional): If True, existing attributes in the file will be replaced. Defaults to False
        pathname (str, optional): Full path name for the file. If None, it is constructed from path and name.
        Defaults to None

    Returns:
        bool: True if the attributes are successfully saved, False otherwise
    """
    if (len(varnames) == 0) or (len(values) == 0) or (len(varnames) != len(values)):
        return False

    if (pathname == "") or pathname is None:
        pathname = dpg.path.getFullPathName(path, name)
    if len(pathname) < 4:
        return False
    dirname = os.path.dirname(pathname)
    if dirname == ".":
        return False

    key = varnames
    keyVal = [str(v) if not isinstance(v, list) else v for v in values]
    if replace:
        attr_dict = loadAttr(path, name, pathname=pathname)
        if attr_dict is not None:
            replaceTags(attr_dict, varnames, values)
            key = attr_dict.keys()
            keyVal = attr_dict.values()

    ind = [i for i in range(len(keyVal)) if keyVal != ""]
    if len(ind) == 0:
        if os.path.isfile(pathname):
            os.remove(pathname)
        return True
    # endif
    if not os.path.isdir(dirname):
        os.makedirs(dirname)

    if append:
        mode = "a"
    else:
        mode = "w"
    # endif
    try:
        f = open(pathname, mode)
    except:
        print(
            "LogMessage", "Cannot write datafile " + path + name
        )  # , PROCNAME='SaveAttr'
        return False
    for k, v in zip(key, keyVal):
        if isinstance(v, str):
            f.write(f"{k}    = {v}\n")
        elif isinstance(v, list):
            for elem in v:
                f.write(f"{k}    = {elem}\n")
        else:
            raise Exception("Attribute cannot be saved. Check data format!")
    f.close()
    return True


def writeAttr(
        attr,
        path: str = None,
        name: str = None,
        format: str = "",
        pathname: str = "",
        str_format: str = None,
) -> bool:
    """
    Writes attributes to a specified file in various formats

    Args:
        attr (dict or np.ndarray):  The attributes to be written. Must be either a dictionary or a NumPy array
        path (str):                 The directory path where the file is to be saved
        name (str):                 The name of the file
        format (str, optional):     The format in which to save the attributes. Defaults to an empty string
        pathname (str, optional):   Full path name for the file. If empty, it is constructed from path and name.
        Defaults to an empty string
        str_format (str, optional): A string format to be used when saving values in specific formats. Defaults to None

    Returns:
        bool:                       True if the attributes are successfully written, False otherwise
    """
    ret = False
    if (not isinstance(attr, dict)) and (not isinstance(attr, np.ndarray)):
        return ret

    if (format == "") or format is None:
        if isinstance(attr, dict):
            format = "txt"
    # endif

    if isinstance(format, str) and (format != ""):
        fff = format.upper()
    else:
        fff = "NONE"

    if fff == "TXT":
        if not isinstance(attr, dict):
            values = str(attr)
            key = "VALUE"
        else:
            values = attr.values()
            key = attr.keys()
        ret = saveAttr(path, name, key, values, pathname=pathname)
    elif fff == "LUT":
        ret = dpg.io.save_lut(path, name, attr, pathname=pathname)
    elif fff == "VAL":
        ret = dpg.io.save_values(
            path, name, attr, pathname=pathname, str_format=str_format
        )
    elif fff == "VAL_256":
        ret = dpg.io.save_values(
            path, name, attr, pathname=pathname, str_format=str_format
        )
    elif fff == "ASCII":
        ret = dpg.io.save_values(
            path, name, attr, pathname=pathname, str_format=str_format
        )
    elif fff == "STR":
        ret = dpg.io.save_values(
            path, name, attr, pathname=pathname, str_format=str_format
        )
    elif fff == "WIND":
        ret = dpg.io.save_values(
            path, name, attr, pathname=pathname, str_format=str_format
        )
    elif fff == "DAT":
        ret = save_var(path, name, attr)
    else:
        print("WARNING: unknown format in writeAttr")  ##    ret = 0
    return ret


def readAttr(path: str, name: str, format: str = "", file_date=None):
    """
    Reads an attribute file from the specified path and returns its contents.

    Args:
        path (str):             The directory path where the file is located.
        name (str):             The name of the file to read.
        format (str, optional): The format of the file. Defaults to an empty string, which implies 'TXT'.
        file_date (optional):   A variable to store the file date if provided. Defaults to None.

    Returns:
        tuple:                  A tuple containing the read data and the file date. If the file is not found or an
        error occurs, the data is None.
                                If file_date is provided and the file is read successfully, file_date is updated with
                                the comparison date.
    """
    shared = 0
    if not isinstance(format, str):
        format = "TXT"
    ret = None
    pathname = dpg.path.getFullPathName(path, name)

    if format.upper() == "LUT":  # TODO ret = read_lut(path, name, PATHNAME=pathname)
        print("TO DO: implement read_lut in attr.py")
        pass
    elif format.upper() == "VAL":
        ret, err = dpg.io.read_values(path, name, pathname=pathname)
    elif format.upper() == "VAL_256":
        ret, err = dpg.io.read_values(path, name, pathname=pathname, mode_256=True)
        pass
    elif format.upper() == "TIM":  # TODO ret = read_ulongs(path, name, PATHNAME=pathname)
        print("TO DO: implement read_ulong in attr.py")
        pass
    elif format.upper() == "STR":  # TODO ret = read_strings(path, name, /POINTER, PATHNAME=pathname)
        print("TO DO: implement read_strings in attr.py")
        pass
    elif format.upper() == "ASCII":
        ret = dpg.io.load_ascii(path, name, pathname=pathname)
    else:
        ret = loadAttr(path, name, pathname=pathname)

    if file_date is not None:
        if ret is not None:
            file_date = compareDate(0, path=path, name=name, pathname=pathname)
        else:
            file_date = 0
    # endif

    return ret, file_date


def getAttrValue(
        attrs: list[Attr] | Attr | dict,
        key,
        default="",
        prefix: str = "",
        only_with_prefix: bool = False,
        round_value=None,
):
    """
    Retrieves the value of a specified attribute from a list of attributes, with options for prefix matching,
    type casting, and value rounding

    Args:
        attrs (list[Attr] or Attr or dict):       The attributes to search within. Can be a dictionary,
        a list of Attr or a single Attr
        key (str):                          The key of the attribute to retrieve.
        default (any, optional):            The default value to return if the key is not found. Defaults to an empty
        string
        get_exists (bool, optional):        If True, also returns a boolean indicating if the key exists. Defaults to
        False
        prefix (str, optional):             A prefix to prepend to the key when searching. Defaults to an empty string
        only_with_prefix (bool, optional):  If True, only searches for the key with the prefix. Defaults to False
        get_attr_ind (bool, optional):      If True, also returns the index of the attribute where the key was found.
        Defaults to False
        round_value (int, optional):        If provided, rounds the retrieved value to the specified number of
        decimal places. Defaults to None

    Returns:
        Depending on the combination of get_exists and get_attr_ind:
        - The value of the attribute if found, or the default value if not
        - A boolean indicating if the key exists
        - The index of the attribute where the key was found
    """
    if attrs is None:
        return default, False, None

    exists = False

    if not isinstance(attrs, list):
        attrs = [attrs]

    ret_val = default
    type_cast = type(ret_val)
    if type_cast == float:
        ret_val = np.float32(ret_val)

    attr_ind = 0

    for ind, attr in enumerate(attrs):
        attr_ind = ind
        if isinstance(attr, dpg.attr__define.Attr):
            attr = attr.pointer

        exists, value = False, default

        if isinstance(attr, dict):
            if prefix != "" and prefix is not None:
                prefix_key = prefix + "." + key
                attr_upper_keys = {k.upper(): v for k, v in attr.items()}
                prefix_key_upper = prefix_key.upper()
                if prefix_key_upper in attr_upper_keys.keys():
                    exists = True
                    value = attr_upper_keys[prefix_key_upper]

            if not exists:
                if not only_with_prefix:
                    attr_upper_keys = {k.upper(): v for k, v in attr.items()}
                    key_upper = key.upper()
                    if key_upper in attr_upper_keys.keys():
                        exists = True
                        value = attr_upper_keys[key_upper]

            if exists:
                type_cast = type(default)
                if type_cast == float:
                    type_cast = np.float32
                if isinstance(value, list):
                    if round_value is not None:
                        ret_val = [type_cast(np.round(v, round_value)) for v in value]
                    else:
                        ret_val = [type_cast(v) for v in value]
                else:
                    ret_val = type_cast(value)
                    if round_value is not None:
                        ret_val = np.round(ret_val, round_value)
                break

    if ret_val == 'None':
        ret_val = None

    return ret_val, exists, attr_ind


def getInherited(attr: Attr):
    """
    Retrieves the inherited attributes from the given attribute object.

    Args:
        attr (list or dict): The attribute object to search within.

    Returns:
        list: A list of inherited attributes if the 'inherits' key exists, otherwise an empty list.
    """
    inherit, exists, _ = getAttrValue(attr, "inherits", "")
    if not exists:
        return []
    return inherit


def removeTags(attr, tags: list):
    """
    Remove specified tags from the attribute structure.

    Args:
        attr (dict | Attr): The attribute object (either a dictionary or an Attr object) from which tags will be
        removed.
        tags (list): A list of tags to remove from the attribute structure.

    Returns:
        list: A list of removed elements. If no tags are removed or if the attribute object is invalid, an empty list
        is returned.
        None: If the attribute structure is neither a dictionary nor an Attr object.
    """
    attr = attr[0] if isinstance(attr, list) and all(isinstance(elem, Attr) for elem in attr) and len(
        attr) > 0 else attr

    isAttr = True if isinstance(attr, dpg.attr__define.Attr) else False
    if isAttr:
        el = attr.removeTags(tags)
        return el

    isDict = True if isinstance(attr, dict) else False
    if isDict:
        # log_message("Remove tags sta cercando di rimuovere un tag da un dict.. DA CONTROLLARE", level="WARNING+")
        el = []
        if len(attr) == 0:
            return el

        # a volte attr key è camel case e altre no, lo stesso vale per tags
        attr_keys = [a for a in attr.keys()]
        tags_lower = [a.lower() for a in tags]
        for key in attr_keys:
            if key.lower() in tags_lower:
                # attr contiene le chiavi in camel case mentre tags tutto in lower
                el.append(attr.pop(key))
        return el

    return None


def replaceTags(
        attr,
        tags: list,
        values: list,
        to_add: bool = False,
        clean: bool = False,
        rem_inherits: bool = False,
        to_create: bool = False,
):
    """
    Replaces, adds, or removes tags and their corresponding values in an attribute structure.

    Args:
        attr: The attribute structure (dictionary or Attr object) to modify.
        tags (list): A list of tags to replace or add.
        values (list): A list of values corresponding to the tags.
        to_add (bool, optional): If True, adds new tags instead of replacing existing ones. Defaults to False.
        clean (bool, optional): If True, removes all existing tags before adding the new ones. Defaults to False.
        rem_inherits (bool, optional): If True, removes any 'inherits' tags from the attribute structure. Defaults to
        False.
        to_create (bool, optional): If True, creates a new attribute structure if `attr` is None. Defaults to False.

    Returns:
        None or dict: If a new attribute structure is created, it is returned as a dictionary. Otherwise, returns None.
    """
    # if not isinstance(tags, list):
    #     tags = [tags]
    # if not isinstance(values, list):
    #     values = [values]

    if (
            (np.size(tags) == 0)
            or (np.size(values) == 0)
            or (np.size(values) != np.size(tags))
    ):
        return None
    if attr is None and not to_create:
        return None

    if isinstance(attr, list):
        attr = attr[0]

    isAttr = True if isinstance(attr, dpg.attr__define.Attr) else False
    if isAttr:
        attr.replaceTags(
            tags, values, to_add=to_add, clean=clean, rem_inherits=rem_inherits
        )
    else:
        if attr is None:
            str = createAttrStruct(tags, values)
            return str
        # endif
        if not isinstance(attr, dict):
            return None
        if not to_add:
            removeTags(attr, tags)
        if rem_inherits:
            removeTags(attr, "inherits")
        if clean:
            removeTags(attr, list(attr.keys()))
        # endif
        for key, val in zip(tags, values):
            attr[key] = val
        # endfor
    # endif
    return attr


def save_var(path: str, name: str, var: np.ndarray, pathname: str = "") -> bool:
    """
    Saves a NumPy array to a binary file at the specified path.

    Args:
        path (str): The directory path where the file will be saved.
        name (str): The name of the file to be saved.
        var (np.ndarray): The NumPy array to save.
        pathname (str, optional): The full file path. If not provided, it will be constructed from `path` and `name`.

    Returns:
        bool: Returns True if the array is successfully saved, False otherwise.
    """
    if (pathname == "") or pathname is None:
        pathname = dpg.path.getFullPathName(path, name)
    if not isinstance(var, np.ndarray):
        print("LogMessage", "Invalid var for " + pathname)  # , PROCNAME='WriteAttr')
        return False
    # endif

    if not os.path.isdir(os.path.dirname(pathname)):
        os.makedirs(os.path.dirname(pathname))

    array = var.flatten("C")
    dim = var.shape

    try:
        f = open(pathname, "wb")
    except:
        log_message("Cannot write datafile " + pathname, level='ERROR')
        return False
    # content = struct.pack("B"*prod(dim), *array)
    # f.write(content)
    array.tofile(f)
    f.close()

    return True


def checkBoolVal(val: np.ndarray):
    """
    Converts an array of strings or boolean values to a boolean array based on specific conditions.

    Args:
        val (np.ndarray): Input array which can contain strings or boolean values.

    Returns:
        np.ndarray: A boolean array where each element corresponds to the condition:
                    - For string arrays: True if the string does not contain 'N' (case insensitive), False otherwise.
                    - For boolean arrays: The boolean value is directly used.

    Usage example:
        data = np.array(['Yes', 'No', 'yes', 'nO'], dtype='U')
        result = checkBoolVal(data)
    """
    # Check if the input is an empty array
    if len(val) == 0:
        return np.array([])

    # Determine the data type of the input
    data_type = val.dtype

    # Check if the data type is string (type 7 in IDL)
    if data_type == np.dtype("U") or data_type == np.dtype("S"):
        # Convert the strings to uppercase and check for 'N'
        # TODO: task check string case --> questo for non viene mai richiamato ne in bassa ne in alta risoluzione..
        #  NON è stato testato
        bool_array = np.array(
            [1 if "N" not in item.upper() else 0 for item in val], dtype=bool
        )
    else:
        # Assuming the input is a boolean byte array
        bool_array = np.array(val, dtype=bool)

    return bool_array


def filterTags(attr, tags, values, prefix=None) -> int:
    # (funzione usata in dps e dpv)
    """
    Filters attribute tags and values based on a given prefix and updates the provided lists

    Args:
        attr (tuple):           A tuple containing two lists:
                                - The first list contains attribute names (strings)
                                - The second list contains corresponding values
        tags (list):            A list to store the filtered tags
        values (list):          A list to store the filtered values
        prefix (str, optional): A prefix to filter the attribute names. Defaults to None

    Returns:
        int:                    The number of filtered tags.

    """
    # Check if the attribute pointer is valid
    if attr is None:
        return len(tags)

    # Get the attribute names
    # TODO: task 51 --> questo for non viene mai richiamato ne in bassa ne in alta risoluzione.. NON è stato testato
    names = [name.upper() for name in attr[0]]

    if prefix is not None:
        prefix = prefix.upper()
        # Find positions where names start with the given prefix
        ind = [i for i, name in enumerate(names) if name.startswith(prefix + ".")]
    else:
        # Find positions where names contain a period
        ind = [i for i, name in enumerate(names) if "." in name]

    if not ind:
        return len(tags)

    if tags:
        # Extend the tags and values lists
        tags.extend(attr[0][i] for i in ind)
        values.extend(attr[1][i] for i in ind)
        return len(tags)
    else:
        # Set the tags and values lists
        tags = [attr[0][i] for i in ind]
        values = [attr[1][i] for i in ind]
        return len(tags)


def getAllTags(attr) -> int:
    # TODO: cambiare head comment per riadattarlo alla nuova implementazione
    """
    Aggregates all tags from a list of attribute tuples and returns the total number of tags

    Args:
        attr (list): A list of tuples where each tuple contains:
                     - A list of tag names (strings)
                     - A list of corresponding values

    Returns:
        int:         The total number of aggregated tags
    """
    if attr is None:
        return None, None, None

    if isinstance(attr, dict):
        attr = [attr]
    if not isinstance(attr, list):
        return None, None, None

    tags, values = [], []
    for aaa in attr:
        if isinstance(aaa, dict):
            tags += list(aaa.keys())
            values += list(aaa.values())
    return tags, np.array(values), len(tags)


def getValidTags(attr):
    """
    Retrieves all valid (non-empty) tags from a list of attribute tuples

    Args:
        attr (list): A list of tuples where each tuple contains:
                     - A list of tag names (strings)
                     - A list of corresponding values

    Returns:
        tuple:       A tuple containing:
                     - A list of valid tags (non-empty strings)
                     - An integer representing the count of valid tags
    """
    # Questa funzione non viene mai chiamata al momento.
    # la chiamata a getAllTags potrebbe rompersi in quanto
    # getAllTags è stata reimplementata per gestire dizionari
    tags = []

    tags, _, n_tags = getAllTags(attr)
    if n_tags <= 0:
        return 0

    for aaa in range(n_tags):
        if attr[aaa][0]:
            tags.extend(attr[aaa][0])

    valid_tags = [tag for tag in tags if tag != ""]
    return valid_tags, len(valid_tags)


def mergeAttr(attr: Attr) -> Attr:
    """
    Merges a list of attribute tuples into a single attribute tuple.

    Args:
        attr (list): A list of attribute tuples. Each tuple contains:
                     - A list of tag names (strings).
                     - A list of corresponding values.

    Returns:
        Attr:        A single merged attribute tuple.
    """
    # Get the number of attributes in the input list
    if not isinstance(attr, list):
        attr = [attr]

    n_attr = len(attr)
    # if n_attr < 2:
    #     return attr[0]

    # Check if the input list is empty or the last element is not a valid pointer
    if n_attr == 0 or not attr[-1]:
        # Return a new empty pointer
        return []

    # Create a new pointer based on the last attribute
    # new = copy.deepcopy(attr[-1])
    new = dpg.attr.Attr(name=attr[-1].name,
                        owner=attr[-1].owner,
                        pointer=copy.copy(attr[-1].pointer),
                        file_date=attr[-1].file_date,
                        format=attr[-1].format,
                        str_format=attr[-1].str_format)

    # Loop through the remaining attributes in reverse order
    for aaa in range(n_attr - 2, -1, -1):
        # Extract tags and values from the current attribute
        tags = attr[aaa].pointer.keys()
        values = attr[aaa].pointer.values()

        # Replace tags and values in the new attribute
        ret = replaceTags(new, tags, values)

    # Return the merged attribute
    return new


def deleteAttr(path: str, name: str, pathname: str = None):
    """
    Deletes an attribute file from the specified path

    Args:
        path (str):               The directory path where the file is located
        name (str):               The name of the file to be deleted
        pathname (str, optional): Full path name for the file. If not provided, it is constructed from path and name.
        Defaults to None
    """
    # If pathname is not provided, construct it from path and name
    if pathname is None:
        pathname = os.path.join(path, name)

    # Check if the file exists before attempting to delete it
    if os.path.exists(pathname):
        os.remove(pathname)
        print(f"Deleted attribute file: {pathname}")
    else:
        print(f"Attribute file does not exist: {pathname}")


def removePrefix(tags: list, prefix: str):
    """
    Removes a specified prefix from each tag in a list of tags

    Args:
        tags (list):  A list of tag names (strings) from which the prefix should be removed
        prefix (str): The prefix to be removed from each tag
    """
    # Check if the input list is empty
    if len(tags) == 0:
        return

    # Determine the length of the prefix
    if len(prefix) == 1:
        prefix_len = 1
    else:
        prefix_len = len(prefix) + 1

    # Iterate through the list and remove the prefix from each tag
    for i in range(len(tags)):
        if tags[i].startswith(prefix + "."):
            tags[i] = tags[i][prefix_len:]
        elif tags[i].startswith(prefix):
            tags[i] = tags[i][len(prefix):]


def getTag(
        attr,
        tag,
        default,
        prefix=None,
        only_with_prefix: bool = False,
        str_format=None,
        contains: bool = False,
        start_with: bool = False,
        split=None,
):
    """
    Retrieves a tag value from an attribute dictionary based on various search conditions.

    Args:
        attr (dict): A dictionary of attributes where tags and values are stored.
        tag (str): The tag to search for within the attribute dictionary.
        default: The default value to return if the tag is not found or if the value is invalid.
        prefix (str, optional): A prefix to prepend to the tag when searching. Defaults to None.
        only_with_prefix (bool, optional): If True, only searches for tags with the specified prefix. Defaults to False.
        str_format (str, optional): A format string to apply to the retrieved value. Defaults to None.
        contains (bool, optional): If True, searches for tags containing the tag string. Defaults to False.
        start_with (bool, optional): If True, searches for tags starting with the tag string. Defaults to False.
        split (optional): If provided, splits the retrieved value using a specified delimiter. Defaults to None.

    Returns:
        The value associated with the tag if found and valid, otherwise the `default` value.
    """
    if not attr:
        return default

    exists = False

    if not isinstance(attr, dict):  # TODO Penso non sia ok
        return default

    count = 0
    # TODO: task 51 --> questo for non viene mai richiamato ne in bassa ne in alta risoluzione.. NON è stato testato
    curTag = tag.strip().upper()
    curTags = {k.upper(): v for k, v in attr.items()}

    indices = []
    if prefix:
        compName = (prefix + "." + curTag).upper()
        if contains:
            indices = [key for key in curTags.keys() if compName in key]
        else:
            if start_with:
                indices = [key for key in curTags.keys() if key.startswith(compName)]
            else:
                indices = [key for key in curTags.keys() if key == compName]

    if not indices and not only_with_prefix:
        if contains:
            indices = [key for key in curTags.keys() if curTag in key]
        else:
            if start_with:
                indices = [key for key in curTags.keys() if key.startswith(curTag)]
            else:
                indices = [key for key in curTags.keys() if key == curTag]

    if not indices:
        return default

    if len(indices) == 1:
        val = curTags[indices[0]]
        if not val or val == " ":
            return default

    exists = True
    val = curTags[indices[0]]

    if str_format:
        val = format(val, str_format)

    val = val.strip()

    if split:
        vals = splitValues(val, _type=type(default))
        return vals

    if isinstance(default, int):
        try:
            val = int(val)
        except ValueError:
            return default
    elif isinstance(default, float):
        try:
            val = float(val)
        except ValueError:
            return default

    return val


def splitValues(values, _type=int):
    """
    Split a list of values based on spaces and return the result in the specified type format

    Args:
        values (list): A list of string values to be split
        _type (type, optional): Data type for the result. Default is int

    Returns:
        list: A list or a list of lists based on the split results. If no values are provided, returns empty list []
    """

    if not values:
        return []

    # Split the first value based on spaces
    chan = values[0].split()

    if len(chan) == 1:
        if len(values) == 1:
            if not chan[0] or chan[0] == '':
                # Codice morto, non si riesce a raggiungere
                # capire se ha senso verificare se chan è lista vuota
                return []
            return [_type(chan[0])]

        # Convert the values to the specified type
        split = [_type(val.split()[0]) for val in values]
        return split

    # If there are multiple elements after splitting the first value
    split = []
    for val in values:
        chan = val.split()
        split.append([_type(c) for c in chan[: len(chan)]])
    return split
