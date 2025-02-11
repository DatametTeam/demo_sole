import shapefile
import shutil

"""
Funzioni ancora da portare
PRO PrepareToBufr 
FUNCTION CreateShapeEntities 
PRO SET_SHAPE_ATTRIBUTE 
PRO EXPORT_TXT 
PRO EXPORT_BFR 
PRO EXPORT_TIF 
PRO EXPORT_ODM 
PRO EXPORT_ODIM 
PRO EXPORT_HDF 
PRO EXPORT_SHP 
PRO EXPORT_GRAPHIC 
"""


def createShapeAttribute(shape, attr_name, parname=None):
    '''
    Creates an attribute field in a shapefile based on the provided attribute name and optional parameter name.

    Args:
        shape: A shapefile.Writer object where the attribute field will be added.
        attr_name: The name of the attribute field to create.
        parname: Optional parameter name for the field if `attr_name` is not specified.

    Returns:
        - **int**: Returns 1 if the attribute field was successfully created with "VALUE"; otherwise, returns 0 if the attribute was created with "ID" or other names.
    '''


    """
    # Usage example
    shape = shapefile.Writer()
    attr_name = 'Population'
    parname = 'Density'
    createShapeAttribute(shape, attr_name, parname)
    """
    found = 0

    if "VALUE" in attr_name.upper():
        shape.field(attr_name, "F", 5, 6)
        found = 1
        return found

    if attr_name.upper() == "ID":
        shape.field("ID", "N", 3, 6)
        return

    if attr_name:
        shape.field(attr_name, "C", 7, 32)
        return

    if parname:
        shape.field(parname, "F", 5, 6)


def export_txt(prodPath, path, name, format):
    '''
    Copies a specific file from the source directory to the destination directory based on the given format.

    Args:
        prodPath: The path to the source directory containing the files to be copied.
        path: The path to the destination directory where the file will be copied.
        name: The name of the destination file.
        format: The format type that determines which file to copy. Valid values are "SITES" or "PAR".

    Returns:
        - **None**: If the specified file does not exist or the format does not match "SITES" or "PAR", no file is copied.
    '''

    """
    # Usage example
    prodPath = '/path/to/source/'
    path = '/path/to/destination/'
    name = 'output.txt'
    format = 'SITES'
    export_txt(prodPath, path, name, format)
    """
    format = format.upper()

    if format == "SITES":
        sites_path = prodPath + "sites.txt"
        if not shutil.os.path.exists(sites_path):
            return
        shutil.copy(sites_path, path + "/" + name)
        return

    if format == "PAR":
        par_path = prodPath + "VAM.par"
        if not shutil.os.path.exists(par_path):
            return
        shutil.copy(par_path, path + "/" + name)
        return
