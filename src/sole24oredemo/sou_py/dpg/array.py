import numpy as np
import os
import sou_py.dpg as dpg
from sou_py.dpg.log import log_message

"""
Funzioni ancora da portare
FUNCTION CheckCurrPointer 
FUNCTION IDL_rv_get_quality 
FUNCTION IDL_rv_get_shape_values 
FUNCTION IDL_rv_replace_array 
PRO IDL_rv_clean_nodes 
PRO IDL_rv_remove_array 
PRO IDL_rv_save_array_info 
PRO IDL_rv_set_quality 
PRO IDL_rv_set_wind_values 
PRO IDL_rv_unload_array 
"""


def set_array_info(
    node,
    name: str = "",
    filename: str = "",
    tag: str = "",
    dtype=None,
    dim: tuple = None,
    bitplanes=None,
    mode=None,
    format: str = "",
    str_format: str = None,
    corrected: bool = None,
    sampled: bool = None,
    lut=None,
    legend=None,
    aux: bool = False,
    date: str = None,
    time: str = None,
):
    """
    Sets various attributes and information for a given node based on provided parameters.

    This function updates a node with an array of information, including dimensions, file
    names, data type, mode, format, and other metadata. It constructs attribute names
    and values based on the input parameters and applies them to the specified node.

    Args:
        node: The node to be updated with array information.
        name (str, optional): The name of the array. Defaults to an empty string.
        filename (str, optional): The file name associated with the array. Defaults to
                                  an empty string.
        tag (str, optional): A tag for the data file. Defaults to an empty string.
        dtype: The data type of the array. Defaults to None.
        dim (tuple, optional): The dimensions of the array. Defaults to None.
        bitplanes: The bitplanes information. Defaults to None.
        mode: The mode of the array. Defaults to None.
        format (str, optional): The format of the array. Defaults to an empty string.
        str_format (str, optional): String format for the array. Defaults to None.
        corrected: Flag indicating whether the array is corrected. Defaults to None.
        sampled: Flag indicating whether the array is sampled. Defaults to None.
        lut: The lookup table file name. Defaults to None.
        legend: The legend file name. Defaults to None.
        aux (bool, optional): Auxiliary flag. Defaults to False.
        date (str, optional): Date associated with the array. Defaults to None.
        time (str, optional): Time associated with the array. Defaults to None.

    Returns:
        None: The function does not return any value.

    Note:
        The function dynamically builds a list of attribute names and values based on
        the provided parameters. It then updates or creates these attributes in the
        specified node using 'dpg.tree.replaceAttrValues'. Special handling is applied
        to dimensions, where the number of dimensions dictates the specific attributes
        to be set or removed.
    """

    varnames = []
    values = []

    if dim is not None:
        if isinstance(dim, tuple):
            dim = [val for val in dim]
        if not isinstance(dim, list) and not isinstance(dim, np.ndarray):
            dim = [dim]
        if len(dim) == 1:
            values = [str(dim[0])]
            varnames = ["ncols"]
        if len(dim) == 2:
            values = [str(dim[0]), str(dim[1])]
            varnames = ["nlines", "ncols"]
        if len(dim) == 3:
            values = [str(dim[0]), str(dim[1]), str(dim[2])]
            varnames = ["nplanes", "nlines", "ncols"]

    if (filename != "") and filename is not None:
        if (tag == "") or tag is None:
            tag = "datafile"
            if aux:
                tag += ".aux"
        # endif
        varnames.append(tag)
        values.append(filename)
    # endif
    if (name != "") and name is not None:
        varnames.append("name")
        values.append(name)
    # endif
    if lut is not None:
        varnames.append("lutfile")
        values.append(lut)
    # endif
    if legend is not None:
        varnames.append("legendfile")
        values.append(legend)
    # endif
    if dtype is not None:
        varnames.append("type")
        values.append(str(dtype))
    # endif
    if mode is not None:
        varnames.append("mode")
        values.append(mode)
    # endif
    if bitplanes is not None:
        varnames.append("bitplanes")
        values.append(str(bitplanes))
    # endif
    if corrected is not None:
        varnames.append("corrected")
        values.append(str(int(corrected)))
    # endif
    if sampled is not None:
        varnames.append("sampled")
        values.append(str(int(sampled)))
    # endif

    if date is not None:
        varnames.append("date")
        values.append(date)
    # endif

    if time is not None:
        varnames.append("time")
        values.append(time)
    # endif

    if (format != "") and format is not None:
        varnames.append("format")
        values.append(format)
    # endif
    if str_format is not None:
        varnames.append("str_format")
        values.append(str_format)
    # endif

    idname = dpg.cfg.getArrayDescName()
    attr = dpg.tree.replaceAttrValues(node, idname, varnames, values, only_current=True)
    if dim is not None and attr is not None:
        if len(dim) == 2:
            dpg.attr.removeTags(attr, ["nplanes"])
        if len(dim) == 1:
            dpg.attr.removeTags(attr, ["nlines", "nplanes"])
    return


# TODO da controllare perché leggermente diversa da IDL
def save_array(node, aux: bool = False, str_format: str = None):
    """
    Saves an array to a file based on the node's attributes, with optional format specification.

    This function saves the array data associated with a given node to a file. The file
    name and other properties are determined based on the node's attributes. Optionally,
    the function can handle auxiliary information and a specified string format for the
    output.

    Args:
        node (dpg.node__define.Node): The node associated with the array to be saved.
        aux (bool, optional): A flag to indicate whether to use auxiliary information
                              in the node. Defaults to False.
        str_format (str, optional): The string format to be applied when saving the array.
                                    Defaults to None.

    Returns:
        None: The function does not return any value.

    Note:
        The function first checks if the provided 'node' is a valid 'dpg.node__define.Node'.
        If a 'str_format' is provided, the function retrieves the array information from
        the node and fetches the filename from its attributes. If the filename is not empty,
        it then obtains the attribute associated with this filename and sets its property
        with the specified string format. Finally, it saves the node data to the file
        specified in its attributes. This process allows for dynamic handling of array
        saving based on node configuration and optional formatting requirements.
    """
    if not isinstance(node, dpg.node__define.Node):
        return

    if str_format:
        out = node.getArrayInfo(aux=aux)
        filename, _, _ = dpg.attr.getAttrValue(out, "filename", "")
        if filename == "":
            return
        oAttr = node.getMyAttr(filename)
        if isinstance(oAttr, dpg.attr__define.Attr):
            oAttr.setProperty(str_format=str_format)
    # endif

    node.save(only_current=True)


def set_array(
    node,
    pointer: np.ndarray = None,
    data: np.ndarray = None,
    no_copy: bool = False,
    filename: str = "",
    tag: str = "",
    mode=None,
    date: str = None,
    time: str = None,
    format: str = "",
    str_format: str = None,
    aux: bool = False,
    to_save: bool = False,
):
    """
    Sets array data to a node and updates its attributes, with options to save the node.

    This function is used to assign array data to a given node and update the node's
    attributes based on provided parameters. The function can handle different scenarios
    such as copying data, setting file names, and saving the node after setting the array.

    Args:
        node (dpg.node__define.Node): The node to which the array data is to be set.
        pointer (np.ndarray, optional): A pointer to an array, typically used for setting
                                        data without copying. Defaults to None.
        data (np.ndarray, optional): The array data to be set to the node. Defaults to None.
        no_copy (bool, optional): If True, avoids copying the data and uses the pointer directly.
                                  Defaults to False.
        filename (str, optional): The file name to be associated with the node. Defaults to
                                  an empty string.
        tag (str, optional): A tag to be set to the node. Defaults to an empty string.
        mode (optional): The mode to be set to the node. Defaults to None.
        date (str, optional): The date to be set to the node. Defaults to None.
        time (str, optional): The time to be set to the node. Defaults to None.
        format (str, optional): The format of the data to be set. Defaults to an empty string.
        str_format (str, optional): The string format for the data. Defaults to None.
        aux (bool, optional): If True, auxiliary data is handled. Defaults to False.
        to_save (bool, optional): If True, saves the node after setting the data. Defaults
                                  to False.

    Returns:
        None: The function does not return any value.

    Note:
        The function first checks if the provided 'node' is a valid 'dpg.node__define.Node'.
        It then assigns the 'data' to 'pointer' based on the 'no_copy' flag. The function
        updates the node's attributes like filename, tag, mode, etc., and adds the array
        data to the node using 'node.addAttr'. If 'to_save' is True, the node is saved
        after setting the array. This function is essential for managing array data in nodes,
        including updating node attributes and handling data storage efficiently.
    """

    if not isinstance(node, dpg.node__define.Node):
        return

    if isinstance(data, np.ndarray):
        pointer, _ = node.getArray(to_not_load=True, aux=aux, silent=True)
        if pointer is not None:
            if no_copy:
                pointer = data
            else:
                pointer = data.copy()
        else:
            if no_copy:
                pointer = data
            else:
                pointer = data.copy()
        # endelse
    # endif

    if not isinstance(pointer, np.ndarray) or np.size(pointer) == 0:
        return

    if not aux:
        dtype_attr = getattr(pointer, "dtype", None)
        dtype = dpg.io.type_py2idl(
            dtype_attr if dtype_attr is not None else type(pointer)
        )

        dim = pointer.shape
        if dim[0] == 0:  # TODO potrebbe essere sbagliato cablare l'indice, capire il caso in cui viene usato
            dim = (1, 1)
    else:
        dtype = None
        dim = None
    # endif
    if filename == "" or filename is None:
        _, _, _, _, filename, _, _, _ = node.getArrayInfo(aux=aux)
    # endif
    if filename == "" or filename is None:
        if aux:
            filename = node.getProperty("name") + ".aux"
        else:
            filename = node.getProperty("name") + ".dat"
    # endif
    if format == "" or format is None:
        format = "dat"
    if dtype is not None:
        if dtype == 7:
            format = "ascii"
    # endif

    set_array_info(
        node,
        tag=tag,
        filename=filename,
        dtype=dtype,
        dim=dim,
        date=date,
        time=time,
        mode=mode,
        format=format,
        str_format=str_format,
        aux=aux,
    )

    node.addAttr(filename, pointer=pointer, format=format, str_format=str_format)

    if to_save:
        save_array(node)


def get_array(
    node,
    replace_quant: bool = False,
    reload: bool = False,
    to_not_load: bool = False,
    check_date: bool = False,
    aux: bool = False,
    silent: bool = False,
):
    """
    Retrieves array data and its metadata from a node based on specified parameters.

    This function obtains array data from a given node and returns the data along with
    its associated metadata. It offers various options for how the data is retrieved,
    including whether to replace quantities, reload data, and check dates. The function
    also supports auxiliary data handling and silent operation without printing error messages.

    Args:
        node (dpg.node__define.Node): The node from which the array data is to be retrieved.
        replace_quant (bool, optional): If True, replaces quantities in the data. Defaults
                                        to False.
        reload (bool, optional): If True, forces the data to be reloaded. Defaults to False.
        to_not_load (bool, optional): If True, prevents the data from being loaded. Defaults
                                      to False.
        check_date (bool, optional): If True, checks the date associated with the data.
                                     Defaults to False.
        aux (bool, optional): If True, handles auxiliary data. Defaults to False.
        silent (bool, optional): If True, suppresses print statements for errors. Defaults
                                 to False.

    Returns:
        tuple: A tuple containing the following two elements:
               - pointer (np.ndarray or None): The array data retrieved from the node,
                                               or None if not found.
               - dict: A dictionary containing metadata about the array, including 'type',
                       'file_type', 'dim', 'bitplanes', 'filename', 'format', and
                       'file_changed'.

    Note:
        The function first checks if the provided 'node' is a valid 'dpg.node__define.Node'.
        It then retrieves the array and its metadata based on the specified parameters. If
        the data is not found and 'silent' is False, an error message is printed. This function
        is useful for efficiently accessing and handling array data and metadata within a node.
    """

    if isinstance(node, dpg.node__define.Node):
        pointer_data, out_dict = node.getArray(
            replace_quant=replace_quant,
            reload=reload,
            to_not_load=to_not_load,
            check_date=check_date,
            aux=aux,
            silent=silent,
        )
    else:
        # log_message(f"Error: {node} is not type Node")
        return None, None

    if pointer_data is None and not silent:
        log_message(f"Cannot find data in {node.path}", level="WARNING", all_logs=True)

    return pointer_data, out_dict


def get_idgeneric(
    node,
    reload: bool = False,
    load_if_changed: bool = False,
    check_date: bool = False,
    only_current: bool = False,
    standard: bool = False,
):
    """
    Retrieves the 'idgeneric' attribute from a given node, applying various data handling options.

    This function fetches the 'idgeneric' attribute, which typically contains metadata about
    an array, from a specified node. It offers options for reloading the attribute, loading
    it if it has changed, checking the date, and limiting the retrieval to only the current
    attribute.

    Args:
        node (dpg.node__define.Node): The node from which to retrieve the 'idgeneric' attribute.
        reload (bool, optional): If True, forces the attribute to be reloaded. Defaults to False.
        load_if_changed (bool, optional): If True, loads the attribute if it has been changed.
                                          Defaults to False.
        check_date (bool, optional): If True, checks the date associated with the attribute.
                                     Defaults to False.
        only_current (bool, optional): If True, retrieves only the current attribute. Defaults
                                       to False.

    Returns:
        dpg.attr__define.Attr or None: The 'idgeneric' attribute of the node if it exists,
                                       otherwise None.

    Note:
        The function first verifies if the provided 'node' is a valid 'dpg.node__define.Node'.
        It then retrieves the 'idgeneric' attribute based on the given parameters. The 'idgeneric'
        attribute typically contains important metadata about an array associated with the node.
        This function is useful for accessing this metadata in a flexible and controlled manner.
    """
    # standard è stato rimosso perchè inutilizzato
    name = dpg.cfg.getArrayDescName()
    if not isinstance(node, dpg.node__define.Node):
        return None
    idgeneric = node.getAttr(
        name,
        reload=reload,
        load_if_changed=load_if_changed,
        only_current=only_current,
        check_date=check_date,
    )
    return idgeneric


def get_numeric_array(
    node,
    linear=None,
    values=None,
    scale=None,
    reload=None,
    unit=None,
    parname=None,
    aux=None,
    pointer=None,
):
    """

    Retrieves numerical array data and associated calibration information from a specified node.

    Args:
        - node: The node from which the array and its associated metadata will be fetched.
        - linear: Option to specify whether the data should be linearized (default is None).
        - values: The values associated with the array (default is None).
        - scale: The scale factor to apply to the values (default is None).
        - reload: Flag to reload data if necessary (default is None).
        - unit: The unit of measurement for the data (default is None).
        - parname: The name of the parameter associated with the array (default is None).
        - aux: Auxiliary data for additional processing (default is None).
        - pointer: The pointer to the array data (default is None).

    Returns:
        - ret: The numerical array and associated calibration data if successful.
    """
    # TODO incompleta
    dim = 0

    pointer = get_array(node, aux=aux, dim=dim, reload=reload)

    if not isinstance(pointer, dpg.node__define.Node):
        return

    ret = dpg.calibration.get_values(
        node,
        to_create=True,
        values=values,
        scale=scale,
        reload=reload,
        unit=unit,
        parname=parname,
    )

    var = pointer  # Dereference the pointer

    var = dpg.calibration.convertData(var, values, linear=linear, scale=scale)


# HACK funzione inutilizzata
def get_array_desc(node):
    """Not Used, to be removed"""
    return get_idgeneric(node)


def get_array_info(
    node,
    reload=False,
    only_current=False,
    load_if_changed=False,
    check_date=False,
    aux=False,
) -> dict:
    """
    Retrieves comprehensive information about an array associated with a node.

    This function gathers detailed information about an array from a specified node.
    It includes various attributes like name, file names, data type, mode, and other
    metadata. The function supports options for data reloading, checking dates, and
    handling auxiliary information.

    Args:
        node: The node from which array information is to be retrieved.
        reload (bool, optional): If True, forces the information to be reloaded. Defaults to False.
        only_current (bool, optional): If True, retrieves only the current information. Defaults to False.
        load_if_changed (bool, optional): If True, loads the information if it has been changed. Defaults to False.
        check_date (bool, optional): If True, checks the date associated with the information. Defaults to False.
        aux (bool, optional): If True, handles auxiliary information. Defaults to False.

    Returns:
        dict: A dictionary containing detailed array information, including 'name', 'lut',
              'filename', 'type', 'mode', 'pathname', 'format', 'endian', 'bitplanes',
              'texture', 'sampled', 'corrected', 'legend', 'str_format', 'date', 'time',
              'origin', 'components', 'exists_sampled', 'tag', 'dim', and 'attr'.

    Note:
        The function gathers information by retrieving the 'idgeneric' attribute of the node
        and then extracting specific details. It handles various scenarios and conditions
        based on the node's attributes. This function is crucial for accessing and
        understanding the metadata associated with array data in a node.
    """

    log_message(
        "Funzione da rimuovere! Utilizzare node.getArrayInfo()",
        level="ERROR",
        all_logs=True,
    )
    return
    bitplanes = None
    str_format = None
    date = None
    time = None

    attr = get_idgeneric(
        node,
        reload=reload,
        load_if_changed=load_if_changed,
        only_current=only_current,
        check_date=check_date,
    )
    name, _, _ = dpg.attr.getAttrValue(attr, "name", "")
    lut, _, _ = dpg.attr.getAttrValue(attr, "lutfile", "")
    legend, _, _ = dpg.attr.getAttrValue(attr, "legendfile", "legend.txt")
    mode, _, _ = dpg.attr.getAttrValue(attr, "mode", "")
    texture, _, _ = dpg.attr.getAttrValue(attr, "texture", 1)
    corrected, _, _ = dpg.attr.getAttrValue(attr, "corrected", 0)
    endian, _, _ = dpg.attr.getAttrValue(attr, "endian", 0)
    sampled, exists_sampled, _ = dpg.attr.getAttrValue(attr, "sampled", 0)
    origin, _, _ = dpg.attr.getAttrValue(attr, "origin", "")
    tmp, _, _ = dpg.attr.getAttrValue(attr, "str_format", "")
    if tmp != "":
        str_format = tmp
    tmp, exists, _ = dpg.attr.getAttrValue(attr, "date", "")
    if exists:
        date = tmp
        time, _, _ = dpg.attr.getAttrValue(attr, "time", "00:00")
    format, _, _ = dpg.attr.getAttrValue(attr, "format", "")
    filename = ""

    components, tag, filename, format, mode = getDataFiles(attr, aux=aux)
    pathname = os.path.dirname(filename)
    if len(pathname) == 0 or len(pathname[0]) <= 1:
        pathname = dpg.tree.getNodePath(node)

    tmp, exists, _ = dpg.attr.getAttrValue(attr, "bitplanes", 8)
    if exists:
        bitplanes = tmp

    dtype, _, _ = dpg.attr.getAttrValue(attr, "type", 0)
    if dtype == 0:
        if tmp > 8:
            dtype = 2
        else:
            dtype = 1
    elif dtype != 4:
        if dtype > 1 and exists:
            calId = dpg.calibration.get_idcalibration(node)
            bitplanes, _, _ = dpg.attr.getAttrValue(calId, "bitplanes", 10)

    dim, _, attr_ind = dpg.attr.getAttrValue(attr, "ncols", 0)
    nlines, _, l_ind = dpg.attr.getAttrValue(attr, "nlines", 0)
    nplanes, _, p_ind = dpg.attr.getAttrValue(attr, "nplanes", 0)

    # if nlines > 0 and l_ind == attr_ind:
    #     dim = [dim, nlines]
    #     if nplanes > 0 and p_ind == attr_ind:
    #         dim = dim + [nplanes]

    if nlines > 0 and l_ind == attr_ind:
        dim = [nlines, dim]
        if nplanes > 0 and p_ind == attr_ind:
            dim = [nplanes] + dim

    out_dict = {}
    out_dict["name"] = name
    out_dict["lut"] = lut
    out_dict["filename"] = filename
    out_dict["type"] = dtype
    out_dict["mode"] = mode
    out_dict["pathname"] = pathname
    out_dict["format"] = format
    out_dict["endian"] = endian
    out_dict["texture"] = texture
    out_dict["sampled"] = sampled
    out_dict["corrected"] = corrected
    out_dict["legend"] = legend
    out_dict["origin"] = origin
    out_dict["components"] = components
    out_dict["exists_sampled"] = exists_sampled
    out_dict["tag"] = tag
    out_dict["dim"] = dim
    out_dict["attr"] = attr

    out_dict["str_format"] = str_format
    out_dict["date"] = date
    out_dict["time"] = time
    out_dict["bitplanes"] = bitplanes

    return out_dict


def formatIsGraphic(format: str):
    """
    Checks if the provided format string corresponds to a graphical format.

    This function determines whether a given format string matches one of the known
    graphical formats. It supports various common image formats such as JPEG, PNG,
    TIFF, etc.

    Args:
        format (str): The format string to be checked.

    Returns:
        bool or None: Returns True if the format is one of the supported graphical
                      formats, False if not, and None if the input is not a string.

    Note:
        The function compares the provided format string, after converting it to upper
        case, with a predefined list of graphical formats. This is useful for identifying
        if a file format is related to images or graphics. The function returns None if
        the input is not a string, indicating invalid or inappropriate format input.
    """
    graphics = ["JPEG", "JPG", "BMP", "PNG", "TIFF", "TIF", "GIF"]
    if isinstance(format, str):
        return format.upper() in graphics
    else:
        return None


def getBidimensionalDim(dim) -> list:
    """
    Determines the bidimensional dimensions from a given dimension input.

    This function analyzes the input dimension data and extracts a bidimensional
    (2D) representation of it. It handles various types of input, including integers,
    lists, and tuples, to determine the appropriate 2D dimensions. The function is
    designed to work with dimensions in 1D, 2D, and 3D, returning a 2D dimension or
    a default [0, 0] if the input is not suitable.

    Args:
        dim (int, list, or tuple): The dimension data to be processed. Can be an integer,
                                   a list, or a tuple, potentially containing nested tuples.

    Returns:
        list: A list of two integers representing the bidimensional dimensions. Returns
              [0, 0] if the input cannot be properly converted to 2D dimensions.

    Note:
        The function first determines the number of dimensions ('nDim') in the input.
        If 'nDim' is less than 2, it returns [0, 0]. If 'nDim' is exactly 2, it returns
        the input as the bidimensional dimensions. For 'nDim' greater than 3, it again
        returns [0, 0]. In cases where 'nDim' is 3, the function further analyzes the
        input to determine the correct 2D dimensions based on the values of the third
        dimension compared to the others.
    """
    # if isinstance(dim, int):
    #     nDim = 1
    # elif isinstance(dim[0], tuple):
    #     nDim = len(dim[0])
    # else:
    #     nDim = len(dim)
    nDim = np.size(dim)
    d3 = -1

    if nDim < 2:
        return np.array([0, 0])
    elif nDim == 2:
        return dim
    elif nDim > 3:
        return np.array([0, 0])

    d3 = dim[2]
    if dim[1] == dim[0]:
        return dim[:2]
    elif d3 <= dim[0]:
        return dim[:2]
    else:
        d3 = dim[0]
        return dim[1:3]


def check_array(
    node, type: int = None, dim: list = None, value=0, data: np.ndarray = None
):
    """TBD"""
    if node is None:
        return

    if len(dim) == 0:
        _, _, dim, _, _, _, _, _ = node.getArrayInfo()
        if not isinstance(dim, list):
            dim = [dim]

    if len(dim) == 0:
        return

    if sum(dim) <= 0:
        return

    if type is None or type == 0:
        _, _, _, _, _, _, _, type = node.getArrayInfo()

    pointer = dpg.array.check_current_pointer(node, type, dim)

    if pointer is not None:
        return pointer

    pointer = np.empty(dim, dtype=dpg.io.type_idl2py(type))

    pointer[:] = value
    dpg.array.set_array(node, pointer)

    log_message("Data correctly created in " + str(node.path) + ".")

    return pointer


def check_current_pointer(node, type, dim):
    """
    Verifies and retrieves the pointer to an array with the specified type and dimensions.

    This function checks whether the array associated with the given node matches the
    specified type and dimensions. If the conditions are met, the function returns the
    pointer to the array; otherwise, it returns `None`.

    Parameters:
        node (Node): The node containing the array to be verified.
        type (int): The expected type of the array.
        dim (list or tuple): The expected dimensions of the array.

    Returns:
        pointer (ndarray or None): The pointer to the array if the type and dimensions
        match; otherwise, `None`.
    """
    pointer, out_dict = dpg.array.get_array(node, silent=True)
    if pointer is None:
        return None
    exDim = out_dict["dim"]
    exType = out_dict["type"]
    if exType != type:
        return None
    if not np.all(np.equal(dim, exDim)):
        return None

    return pointer


def getDataFiles(attr, aux: bool = False):
    """
    Retrieves data file information and associated properties from an attribute.

    This function extracts details about data files, such as filenames, formats, modes,
    and tags, from a given attribute. It handles different data file types including
    auxiliary files, image files, directional data files, and color components.

    Args:
        attr (dpg.attr__define.Attr): The attribute object from which to retrieve the data file information.
        aux (bool, optional): Specifies if auxiliary data should be considered. Defaults to False.

    Returns:
        tuple: A tuple containing the following elements:
               - components: A list of components or single component of the data file.
               - tag (str): The tag associated with the data file.
               - filename (str): The name of the data file.
               - file_format (str): The format of the data file.
               - mode (str or None): The mode associated with the data file, if any.

    Note:
        The function checks for various tags associated with different types of data files,
        such as 'datafile', 'imgfile', 'datafile.red' (for color components), etc. It
        prioritizes auxiliary files if the 'aux' flag is set. The function returns
        detailed file information, which is useful for processing and handling different
        types of data files in a node.
    """
    components = None
    mode = None
    exists = False
    file_format = ""
    filename = ""
    tmp, exists, _ = dpg.attr.getAttrValue(attr, "mode", "")
    if exists:
        mode = tmp

    tmp, exists, _ = dpg.attr.getAttrValue(attr, "format", "")
    if exists:
        file_format = tmp

    tag = "datafile"
    if aux:
        tag = tag + ".aux"
    files, exists, _ = dpg.attr.getAttrValue(attr, tag, "")
    if not exists and aux:
        filename = ""
        return components, tag, filename, file_format, mode
    if exists or attr is None:
        if tmp != "":
            tmp = "." + tmp
        if isinstance(files, list):
            filename = "data" + tmp
        else:
            filename = files
        if not format_is_numeric(file_format):
            file_format = ""
            if isinstance(files, list):
                format = [""] * len(files)
        components = files
        return components, tag, filename, file_format, mode
    # endif
    tag = "datafile.dir"
    file_1, exists, _ = dpg.attr.getAttrValue(attr, tag, "")
    if exists:
        mode = "arrow"
        file_format = "wind"
        tag = [tag, "datafile.speed"]
        file_2, _, _ = dpg.attr.getAttrValue(attr, tag[1], "")
        filename = "wind.dat"
        components = [file_1, file_2]
        return components, tag, filename, file_format, mode

    tag = "imgfile"
    filename, exists, _ = dpg.attr.getAttrValue(attr, tag, "")
    if exists:
        components = filename
        return components, tag, filename, file_format, mode

    tag = "datafile.red"
    file_1, exists, _ = dpg.attr.getAttrValue(attr, tag, "")
    if exists:
        tag = [tag, "datafile.green", "datafile.blue"]
        file_2, _, _ = dpg.attr.getAttrValue(attr, tag[1], "")
        file_3, _, _ = dpg.attr.getAttrValue(attr, tag[1], "")
        components = [file_1, file_2, file_3]
        return components, tag, filename, file_format, mode

    tag = "datafile"
    components = filename
    return components, tag, filename, file_format, mode


def format_is_numeric(format_str: str) -> bool:
    """
    Check if the given format is considered numeric.

    Args:
        format_str (str): The format string to check.

    Returns:
        bool: True if the format is numeric, False otherwise.
    """
    numerics = [
        "DAT",
        "BUFR",
        "SHP",
        "DBF",
        "TIFF",
        "TIF",
        "TXT",
        "VAL",
        "ASCII",
        "HDF",
    ]
    return format_str.upper() in numerics


def create_array(
    node=None,
    pointer=None,
    dtype=None,
    dim=None,
    filename: str = "",
    format=None,
    str_format=None,
) -> np.ndarray:
    """
    Creates a NumPy array with the specified dimensions and data type, and associates it with a node if provided.

    Args:
        node (optional): The node from which array information can be retrieved, or to which the created array will be associated
        pointer (optional): A pointer to an existing array. Defaults to None
        dtype (optional): The data type of the array. If not provided, it will be retrieved from the node. Defaults to None
        dim (tuple, optional): The dimensions of the array. If not provided, it will be retrieved from the node. Defaults to None
        filename (str, optional): The filename associated with the array, used when saving the array. Defaults to an empty string
        format (optional): The format of the array (e.g., 'txt', 'bin'). Defaults to None
        str_format (optional): A string format for saving the array, if applicable. Defaults to None

    Returns:
        np.ndarray: - A NumPy array created with the specified or retrieved dimensions and data type
                    - Returns None if the dimensions are invalid or not provided
    """
    if dim is None:
        _, _, dim, _, _, _, _, _ = node.getArrayInfo()
    if dim is None:
        return None
    if np.size(dim) <= 0:
        return None
    if dtype is None:
        _, _, _, _, _, _, _, dtype = node.getArrayInfo()

    array = np.zeros(dim, dtype=dpg.io.type_idl2py(dtype))
    if isinstance(node, dpg.node__define.Node):
        set_array(
            node, pointer=array, filename=filename, format=format, str_format=str_format
        )
    return array


def copy_array_info(
    fromNode, toNode, only_if_not_exists: bool = False, to_save: bool = False
):
    """
    Copies array information from one node to another, with options to handle existing information and saving.

    This function copies array information, such as data type, dimensions, and bitplanes,
    from a source node to a destination node. It provides an option to copy the information
    only if it does not already exist in the destination node. Additionally, the function
    can save the destination node after copying the information.

    Args:
        fromNode (dpg.node__define.Node): The node from which to copy the array information.
        toNode (dpg.node__define.Node): The node to which the array information is to be copied.
        only_if_not_exists (bool, optional): If True, copies the information only if it does
                                             not already exist in the destination node. Defaults
                                             to False.
        to_save (bool, optional): If True, saves the destination node after copying the information.
                                  Defaults to False.

    Returns:
        None: The function does not return any value.

    Note:
        If 'only_if_not_exists' is True, the function first checks if the 'idgeneric'
        attribute exists in the 'toNode'. If it exists, the function retrieves array
        information from the 'fromNode' and sets it to the 'toNode' using 'set_array_info'.
        If 'only_if_not_exists' is False, the function directly copies the 'idgeneric'
        attribute from the 'fromNode' to the 'toNode' using 'dpg.tree.copyAttr'. The 'to_save'
        option allows for immediate saving of changes to the 'toNode', ensuring the updated
        information is persisted.
    """
    if only_if_not_exists:
        idgeneric = get_idgeneric(toNode, only_current=True, standard=True)
        if idgeneric is not None:
            _, bitplanes, dim, _, _, _, _, dtype = fromNode.getArrayInfo()
            if dim is not None and np.sum(dim) > 0:
                set_array_info(toNode, dtype=dtype, dim=dim, bitplanes=bitplanes)
                if to_save:
                    dpg.tree.saveNode(toNode)
            return
    # endif

    dpg.tree.copyAttr(fromNode, toNode, dpg.cfg.getArrayDescName())
    if to_save:
        dpg.tree.saveNode(toNode)

    return


def copy_array(fromNode, toNode):
    """
    Copies array data from one node to another, including metadata like filename and format.

    This function retrieves array data from a source node and then sets this data to a
    destination node. It also transfers relevant metadata such as the filename and format
    of the array. The array data is then saved to the destination node.

    Args:
        fromNode (dpg.node__define.Node): The node from which the array data is to be copied.
        toNode (dpg.node__define.Node): The node to which the array data is to be copied.

    Returns:
        np.ndarray or None: The array data that was copied to the 'toNode', or None if no
                            data was found in the 'fromNode'.

    Note:
        The function uses 'get_array' to retrieve the array data and its metadata from the
        'fromNode'. If data is found, it then uses 'set_array' to set this data to the
        'toNode', including the filename and format. After setting the data, 'save_array'
        is called to save the changes to the 'toNode'. This function is useful for cloning
        array data and its associated information from one node to another.
    """
    data = None
    data, data_dict = get_array(fromNode)
    filename = data_dict["filename"]
    format = data_dict["format"]
    if data is None:
        return data

    set_array(toNode, pointer=data, filename=filename, format=format)
    save_array(toNode)

    return data
