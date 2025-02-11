import numpy as np
import h5py
import re
import sou_py.dpg as dpg

"""
Funzioni ancora da portare
H5_get_wmo_code
H5_SET_RECURSIVE_GROUP
GET_H5_DATA
H5_WaitFile
H5_OPEN_READ
H5_GET_DATA
"""


def h5_replace_proj_string(projection, p0lat, p0lon):
    """
    Generates a projection string based on the specified projection type and coordinates.

    This function constructs a string that represents a map projection, using the provided
    projection type and latitude/longitude coordinates. The function handles 'MERCATOR' and
    'Transverse Mercator (tmerc)' projections specially. For other projections, it simply
    uses the given type. The projection string also includes the WGS84 ellipsoid model.

    Args:
        projection (str): The type of map projection (e.g., 'mercator', 'tmerc').
        p0lat (float): Latitude in decimal degrees for the projection's central point.
        p0lon (float): Longitude in decimal degrees for the projection's central point.

    Returns:
        str: A string that represents the map projection, ready for use in GIS software.

    Note:
        The function expects the 'projection' argument in any case (upper or lower),
        but internally it is converted to upper case for processing.
    """
    prj = projection.upper()
    str_lat = format(p0lat, ".1f")
    str_lon = format(p0lon, ".1f")

    pos = prj.find("MERCATOR")
    if pos >= 0:
        if pos == 0:
            proj_str = f"+proj=merc +lat_ts={str_lat} +lon_0={str_lon}"
        else:
            proj_str = f"+proj=tmerc +lat_0={str_lat} +lon_0={str_lon}"
    else:
        proj_str = f"+proj={prj} +lat_0={str_lat} +lon_0={str_lon}"

    proj_str += " +ellps=WGS84"

    return proj_str


def h5_replace_quantity(par_name, aux=None, inverse=False):
    """
    Converts a parameter name to its corresponding code or vice versa based on a given mapping.

    This function maps certain meteorological or radar parameter names to their respective
    shorthand codes, or performs the reverse mapping if 'inverse' is True. Additional behavior
    can be triggered with the 'aux' parameter for specific cases.

    Args:
        par_name (str): The original name of the parameter (e.g., 'CZ', 'UZ') or its shorthand code.
        aux (bool, optional): An auxiliary flag used for certain parameters to modify the mapping.
                              Defaults to None.
        inverse (bool, optional): If set to True, the function performs the reverse mapping
                                  (code to name). Defaults to False.

    Returns:
        str: The mapped shorthand code of the parameter, or the full name if 'inverse' is True.

    Note:
        The function expects 'par_name' in any case (upper or lower), but internally it is
        converted to upper case for processing. The mappings are specific to certain meteorological
        or radar parameters and their shorthand codes.
    """
    par_name = par_name.upper()

    if inverse:
        if par_name == "DBZH":
            return "Z"
        if par_name == "TH":
            return "UZ"
        if par_name == "ACRR":
            return "RT"
        if par_name == "RATE":
            return "R"
        if par_name == "QIND":
            return "Quality"
        if par_name == "HGHT":
            return "H"
        return par_name

    if par_name == "CZ":
        return "DBZH"
    if par_name == "UZ":
        return "TH"
    if par_name == "Z":
        return "DBZH"
    if par_name == "K":
        return "KDP"
    if par_name == "W":
        return "WRAD"
    if par_name == "V":
        if aux:
            return "VDIR"
        return "VRAD"
    if par_name == "R":
        return "RATE"
    if par_name == "RT":
        return "ACRR"
    if par_name == "H":
        return "HGHT"
    if par_name == "QUALITY":
        return "QIND"
    if par_name == "AMV":
        if aux:
            return "AMDIR"
        return "AMVEL"

    return par_name


def h5_replace_type(prod_name):
    """
    Maps a given product name to a specific type code based on predefined criteria.

    This function identifies specific substrings within the provided product name and
    maps them to a corresponding type code. Each product name is checked for certain
    keywords, and based on the presence of these keywords, a type code is returned.
    If none of the predefined keywords are found in the product name, a default type
    code is returned.

    Args:
        prod_name (str): The name of the product to be mapped to a type code.

    Returns:
        str: The type code corresponding to the product name. Possible return values
             are 'SURF', 'RR', 'CAPPI', 'MAX', 'ETOP', 'VIL', or 'COMP' as a default.

    Note:
        The function relies on the presence of specific substrings in 'prod_name' to
        determine the type code. The substrings it looks for are 'SRI', 'SRT', 'CAPPI',
        'VMI', 'ETM', and 'VIL'. If none of these are found, 'COMP' is returned by default.
    """
    if "SRI" in prod_name:
        return "SURF"
    if "SRT" in prod_name:
        return "RR"
    if "CAPPI" in prod_name:
        return "CAPPI"
    if "VMI" in prod_name:
        return "MAX"
    if "ETM" in prod_name:
        return "ETOP"
    if "VIL" in prod_name:
        return "VIL"
    return "COMP"


def h5_valid_scalar(value):
    """
    Determines if the given value is a valid scalar.

    This function checks if the input value is a valid scalar by evaluating two conditions.
    If the value is a NumPy string (np.str_), it checks whether the string is non-empty.
    For other data types, it checks if the value is finite using NumPy's `np.isfinite` method.
    This ensures that the value is neither NaN (not a number) nor infinite, which are
    considered invalid for scalar values in many contexts.

    Args:
        value: The value to be checked for validity. Can be of any type, but common
               types are numeric values or strings.

    Returns:
        bool: True if the value is a valid scalar, False otherwise. For a string, valid
              means non-empty. For numeric types, valid means the value is finite.

    Note:
        The function uses NumPy (imported as np), so it is capable of handling types
        specific to NumPy arrays as well as standard Python types.
    """
    if isinstance(value, np.str_):
        return value != ""
    return np.isfinite(value)


def h5_set_tags(group_id, tags, values):
    """
    Sets attribute tags with corresponding values for a specified HDF5 group.

    This function iterates through the given tags and values, setting each tag with its
    corresponding value as an attribute of the specified HDF5 group. The function first
    checks if the number of tags matches the number of values and if there are any tags
    provided. It then validates each value using `h5_valid_scalar` before setting the
    attribute. Attributes are set using the HDF5 file associated with the `group_id`.

    Args:
        group_id (h5py.Group): The HDF5 group to which the attributes will be added.
        tags (list of str): The tags (attribute names) to be added to the group.
        values (list): The values corresponding to each tag, to be set as attributes
                       of the group.

    Returns:
        None: The function does not return a value. It modifies the HDF5 file in place.

    Note:
        The function requires the `h5py` library for working with HDF5 files. It opens
        the file in 'r+' mode, allowing for reading and writing. The function relies on
        `h5_valid_scalar` to ensure that only valid scalar values are set as attributes.
        If the number of tags does not match the number of values, or if no tags are
        provided, the function does nothing.
    """
    """
    # Usage example (DA CONTROLLARE)
    with h5py.File('your_file.h5', 'r+') as file:
        group = file['your_group']
        tags = ['tag1', 'tag2']
        values = ['value1', 'value2']
        h5_set_tags(group, tags, values)
    """
    nT = len(tags)

    if nT <= 0 or len(values) != nT:
        return

    for vvv in range(nT):
        if h5_valid_scalar(values[vvv]):
            with h5py.File(group_id.file.filename, "r+") as h5file:
                attr = h5file[group_id.name].attrs.create(tags[vvv], values[vvv])
                attr.write(values[vvv])


def h5_set_data(
    group_id, data, gzip=1, chunk_dimensions=None, tags=None, values=None, name=None
):
    """
    Saves data to an HDF5 group with optional compression, chunking, and tagging.

    This function writes a given data array to an HDF5 group. It supports optional
    gzip compression, custom chunk dimensions for the dataset, and the addition of
    custom tags (attributes). The function first checks if the provided data array
    is non-trivial (more than one element). It then handles chunking for 2D data
    arrays, sets up the HDF5 dataspace and datatype, and writes the data to the
    group. If provided, tags and their corresponding values are also set for the
    dataset.

    Args:
        group_id (h5py.Group): The HDF5 group to which the data will be written.
        data (np.ndarray): The data array to be saved to the HDF5 group.
        gzip (int, optional): The level of gzip compression to apply to the dataset.
                              Defaults to 1.
        chunk_dimensions (tuple of int, optional): The dimensions for chunking the
                                                   dataset. Defaults to (100, 100)
                                                   for 2D data.
        tags (list of str, optional): The tags (attribute names) to be added to the
                                      dataset. Defaults to None.
        values (list, optional): The values corresponding to each tag. Defaults to None.
        name (str, optional): The name to assign to the dataset within the HDF5 group.
                              If None, 'data' is used as a default name.

    Returns:
        None: The function does not return a value. It modifies the HDF5 group in place.

    Note:
        The function requires `h5py` and `numpy`. It uses `h5py` for HDF5 file
        operations and `numpy` for handling data arrays. It only writes data if the
        array contains more than one element.
    """
    """
    # Usage example (DA CONTROLLARE)
    with h5py.File('your_file.h5', 'a') as file:
        group = file['your_group']
        data = your_data_array
        tags = ['tag1', 'tag2']
        values = ['value1', 'value2']
        h5_set_data(group, data, tags=tags, values=values)
    """
    if len(data) <= 1:
        return

    dim = data.shape
    if dim == 2:
        if chunk_dimensions is None:
            chunk_dimensions = (100, 100)
        if dim[0] < chunk_dimensions[0] or dim[1] < chunk_dimensions[1]:
            tmp = np.empty(chunk_dimensions)

    dataspace_id = h5py.h5s.create_simple(dim)
    dtype = h5py.h5t.py_create(data, logical=1)

    if name is not None:
        attr_id = h5py.h5d.create(group_id.id, name.encode(), dtype.id, dataspace_id.id)
    else:
        attr_id = h5py.h5d.create(
            group_id.id,
            b"data",
            dtype.id,
            dataspace_id.id,
            gzip=gzip,
            chunks=chunk_dimensions,
        )

    attr_id.write(data)

    if tags is not None and values is not None:
        h5_set_tags(attr_id, tags, values)

    attr_id.close()


def h5_execute(pathname, h5_id, current, index=None, n_data=None, aux=None):
    """
    Executes statements from a file within the context of an HDF5 group.

    This function reads and executes statements from a given file. The execution is
    context-sensitive, depending on the name derived from the pathname. If the derived
    name is 'users', the function returns immediately. If the name is 'data', it uses
    the provided 'h5_id' as the group identifier. For other names, it creates a new
    group under 'h5_id'. Each line in the file is treated as a separate statement
    and executed. Errors in execution are caught and a placeholder error message is
    printed.

    Args:
        pathname (str): The path to the file containing statements to be executed.
        h5_id (h5py.Group or similar): The HDF5 group identifier where the data or
                                       groups are to be created or modified.
        current: The current context or state, the exact usage depends on the
                 implementation.
        index (optional): An index or key to specify a particular segment or
                          component. Defaults to None.
        n_data (optional): Additional data or parameters that might be required
                           for the operation. Defaults to None.
        aux (optional): Auxiliary information or flags that could influence the
                        execution. Defaults to None.

    Returns:
        None: The function does not return a value. It performs operations on HDF5
              groups and handles file execution.

    Note:
        The function assumes that the statements in the file are valid Python code
        and relevant to the operation. Errors during execution are caught, but
        currently, only a placeholder error message is printed. Proper error handling
        should be implemented for robustness.
    """
    """
    # Usage example (DA CONTROLLARE)
    with h5py.File('your_file.h5', 'a') as file:
        group = file['your_group']
        h5_execute('your_script.txt', group, current_value, index=your_index, n_data=your_n_data, aux=your_aux)
    """
    name = pathname.rsplit(".", 1)[0]
    if name == "users":
        return

    if name == "data":
        group_id = h5_id
    else:
        group_id = h5_id.create_group(name)

    with open(pathname, "r") as file:
        statements = file.readlines()

    for statement in statements:
        statement = statement.strip()
        try:
            exec(statement)
        except:
            print("ERRORE")  # TODO gestire errore

    if name != "data":
        group_id.close()


def h5_set_groups(group_id, template, node, index=None, n_data=None, aux=None):
    """
    Iterates over items in a template to execute corresponding actions for each item in an HDF5 group.

    This function traverses through all items in a given HDF5 template, accumulating their
    pathnames. For each pathname, it calls the `h5_execute` function, passing along the
    current HDF5 group identifier, node, and any additional parameters. This allows for
    performing operations defined in external files (referred to by pathnames) on specific
    groups within an HDF5 file.

    Args:
        group_id (h5py.Group): The HDF5 group identifier where operations are to be performed.
        template (h5py.File or h5py.Group): The HDF5 file or group serving as a template
                                            for traversal.
        node: The current context or node relevant to the operation, the exact usage
              depends on the implementation.
        index (optional): An index or key to specify a particular segment or component.
                          Defaults to None.
        n_data (optional): Additional data or parameters that might be required for the
                           operation. Defaults to None.
        aux (optional): Auxiliary information or flags that could influence the
                        execution. Defaults to None.

    Returns:
        None: The function does not return a value. It executes operations defined in
              external files on specific groups in the HDF5 file.

    Note:
        The function relies on `h5_execute` to perform the actual execution of statements
        found in the pathnames. It assumes that the template provided is properly structured
        and contains valid pathnames for traversing and executing operations.
    """
    """
    # Usage example (DA CONTROLLARE)
    with h5py.File('your_file.h5', 'a') as file:
        group = file['your_group']
        template_group = file['your_template_group']
        current_value = 42  # replace with the actual value
        h5_set_groups(group, template_group, current_value, index=your_index, n_data=your_n_data, aux=your_aux)
    """
    pathnames = []
    template.visititems(lambda name, obj: pathnames.append(name))

    for pathname in pathnames:
        h5_execute(pathname, group_id, node, index=index, n_data=n_data, aux=aux)


def h5_get_tags(file_id, data_set_name, tags):
    """
    Retrieves the values of specified tags from a dataset within an HDF5 file.

    This function opens an HDF5 file and accesses a specified dataset. It then iterates
    over a list of tags, retrieving the value of each tag from the dataset's attributes.
    If a tag is not found in the dataset's attributes, `None` is appended to the result
    list in place of its value. The function ensures that the file identifier is valid
    and that there are tags to retrieve before proceeding.

    Args:
        file_id (list): A list containing the identifier of the HDF5 file. The list should
                        contain only one element which is the file identifier.
        data_set_name (str): The name of the dataset within the HDF5 file from which to
                             retrieve the tags' values.
        tags (list of str): The tags whose values are to be retrieved from the dataset's
                            attributes.

    Returns:
        list: A list of values corresponding to each tag. If a tag is not present in the
              dataset's attributes, `None` is returned in its place.

    Note:
        The function requires the `h5py` library to interact with HDF5 files. It is assumed
        that `file_id` is a valid identifier for an open-able HDF5 file and that it contains
        exactly one element. Tags are checked in a case-insensitive manner.
    """
    """
    # Usage example (DA CONTROLLARE)
    file_id = ['your_file.h5']
    data_set_name = 'your_data_set'
    tags = ['tag1', 'tag2', 'tag3']
    values = h5_get_tags(file_id, data_set_name, tags)
    print(values)  # Output: [value1, value2, value3]
    """
    if len(file_id) != 1:
        return
    if file_id[0] <= 0:
        return

    n_t = len(tags)
    if n_t <= 0:
        return

    values = []
    with h5py.File(file_id[0], "r") as file:
        group = file[data_set_name]

        for tag in tags:
            tag = tag.upper()
            if tag in group.attrs:
                values.append(group.attrs[tag])
            else:
                values.append(None)

    return values


def h5_parse_projdef(projdef):
    """
    Parses a projection definition string and extracts key parameters.

    This function takes a projection definition string and uses regular expressions
    to extract the projection type, latitude, longitude, and zone number from it.
    If the projection type is 'AEQD' (Azimuthal Equidistant), it is converted to
    'Azimuthal'. The function returns these extracted values. If the input does not
    meet the expected format or length, default values are returned.

    Args:
        projdef (list): A list containing a single string, which is the projection
                        definition to be parsed.

    Returns:
        tuple: A tuple containing four strings: the projection type, latitude, longitude,
               and zone number. If the input list does not contain exactly one element,
               default values ('', '0', '0', '0') are returned.

    Note:
        The function is designed to work with a specific format of projection definition
        string, which includes parameters like '+proj', '+lat_0', '+lon_0', and '+zone'.
        Latitude and longitude are expected to be suffixed with 'N' and 'E' respectively.
        The function relies on the `re` module for regular expression matching.
    """
    """
    # Usage example (DA CONTROLLARE)
    projdef = ['+proj=aeqd +lat_0=40N +lon_0=-100E']
    projection, lat, lon, zone = h5_parse_projdef(projdef)
    print('Projection:', projection)
    print('Latitude:', lat)
    print('Longitude:', lon)
    print('Zone:', zone)
    """
    projection = ""
    lat = "0"
    lon = "0"
    zone = "0"

    if len(projdef) != 1:
        return projection, lat, lon, zone

    projdef = projdef[0]

    zone_match = re.search(r"\+zone=(\d{2})", projdef)
    if zone_match:
        zone = zone_match.group(1)

    proj_match = re.search(r"\+proj=(\w+)", projdef)
    if proj_match:
        projection = proj_match.group(1)

    lat_match = re.search(r"\+lat_0=(\d+\.\d+)N", projdef)
    if lat_match:
        lat = lat_match.group(1)

    lon_match = re.search(r"\+lon_0=(\d+\.\d+)E", projdef)
    if lon_match:
        lon = lon_match.group(1)

    if projection.upper() == "AEQD":
        projection = "Azimuthal"

    return projection, lat, lon, zone
