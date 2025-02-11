from numbers import Number

import numpy as np

import sou_py.dpg as dpg
from sou_py.dpg.log import log_message

"""
Funzioni ancora da portare
FUNCTION Attr::Init 
PRO Attr::CleanUp 
PRO Attr__Define 
"""


class Attr(object):
    def __init__(
        self,
        name,
        owner=None,
        str_format="",
        format="",
        to_not_save=False,
        file_date=0,
        pointer=None,
    ):
        """
       Initializes an Attr object, replicates a file in a directory.

        Args:
            name (str): The name of the file associated with this Attr object.
            owner (Node): The node (directory) that contains the file. Must not be None.
            str_format (str, optional): String format for saving the file. Defaults to an empty string.
            format (str, optional): The format of the file (e.g., 'txt', 'bin'). Defaults to an empty string.
            to_not_save (bool, optional): If True, the file should not be saved. Defaults to False.
            file_date (int, optional): The date associated with the file (used for versioning). Defaults to 0.
            pointer (np.ndarray, optional): The content of the file, if already loaded. Defaults to None.

        Returns:
            Attr: An instance of the Attr object representing the specified file in a directory.

        Notes:
            - The `name` must be a valid string representing the file name.
            - The `owner` should be an instance of the Node class or a similar structure that holds the directory context.
            - If `format` is not provided, it is inferred from the file extension of `name`.
        """
        if isinstance(owner, type(None)):
            print("Error: owner is None")
            return
        if not isinstance(name, str):
            print("Error: name is not a str")
            return
        if name == "":
            print("Error: name is empty str")
            return

        self.str_format = str_format
        self.to_not_save = to_not_save
        self.file_date = file_date
        self.pointer = pointer  # deve essere un array (nd.array)
        self.owner = owner
        self.name = name
        if format is None:
            format = ""
        if format != "":
            self.format = format
        else:
            try:
                self.format = self.name.split(".")[-1]
            except:
                self.format = format
        # endif
        # self.path = owner.path

        return

    def setProperty(
        self,
        pointer=None,
        format: str = "",
        file_date: int = 0,
        str_format: str = "",
        to_not_save: bool = None,
    ):
        """
        This Method sets specific property for the Attr object. Like the content of file or if it needs to be saved.

        Args:
            pointer (list or dict or np.ndarray): Content of file. Defaults to None.
            format (str): Format of the file. Defaults to ''.
            file_date (int): Date of file. Defaults to 0.
            str_format (str): Format of the string. Defaults to ''.
            to_not_save (bool): If True, Attr is not saved. Defaults to None.

        Returns:
            None
        """
        if isinstance(pointer, list):
            pointer = np.array(pointer)
        if pointer is not None and (
            isinstance(pointer, dict) or isinstance(pointer, np.ndarray)
        ):
            self.pointer = pointer
        if format != "" and isinstance(format, str):
            self.format = format
        if file_date != 0 and isinstance(file_date, Number):
            self.file_date = file_date
        if str_format != "" and isinstance(str_format, str):
            self.str_format = str_format
        if to_not_save is not None:
            self.to_not_save = to_not_save

    def getProperty(self, str_property: str):
        """
        Get method. It returns a given attribute of class Attr.

        Args:
        str_property (str): String. possible values: 'path', 'name', 'format', 'pointer'
                            'file_date', 'str_format', 'to_not_save'.

        Returns:
            bool or str or None
        """
        str_property = str_property.lower()
        if str_property == "path":
            return self.owner.path
        elif str_property == "name":
            return self.name
        elif str_property == "format":
            return self.format
        elif str_property == "pointer":
            return self.pointer
        elif str_property == "file_date":
            return self.file_date
        elif str_property == "str_format":
            return self.str_format
        elif str_property == "to_not_save":
            return self.to_not_save
        else:
            print("WARNING: undefined property in Attr.getProperty")
            return None

    def getInherited(self):
        """
        This method return the 'inherits' attribute if exists. Otherwise, it returns an empty list.

        Returns:
            list of inherits or empty list
        """
        inherit, exists, _ = dpg.attr.getAttrValue(self, "inherits", "")
        if not exists:
            return []
        if isinstance(inherit, list):
            log_message("inherit is a list: TO CHECK", level="ERROR")
            return inherit
        return [inherit]

    def load(self, silent: bool = False, no_palette: bool = False):
        """
        load method. It loads Attr file content.

        Args:
        silent (bool): Boolean, error logs. Defaults to False.
        no_palette (bool): Colour palette for data visualization. Defaults to False.

        Returns:
        lut (str): Colour table for data visualization

        NOTE:
        It writes self.pointer

        """

        path = self.getProperty("path")
        name = self.getProperty("name")
        format = self.getProperty("format")
        lut = None
        names = None

        # qui entra per leggere gli ASCII (.txt)
        if dpg.attr.formatIsAscii(format):
            if self.pointer is None:
                self.pointer, self.file_date = dpg.attr.readAttr(
                    path, name, format=format, file_date=None
                )
            return None

        if not isinstance(self.owner, dpg.node__define.Node):
            print("self.owner non valido! Da gestire")
        _, _, dim, endian, _, _, _, type = self.owner.getArrayInfo()

        self.pointer, palette, coords, file_date, names = dpg.io.read_array(
            path,
            name,
            type,
            dim,
            format,
            endian=endian,
            get_file_date=True,
            silent=silent,
            no_palette=no_palette,
        )

        if self.pointer is not None and file_date is not None:
            self.file_date = file_date
        # endif

        if palette is not None:
            lut = name
            pos = lut.rfind(".")
            if pos > 0:
                lut = lut[:pos] + ".lut"
            else:
                lut += ".lut"
            _ = self.owner.replaceAttrValues(dpg.cfg.getArrayDescName(), "lutfile", lut)
            _ = self.owner.addAttr(lut, palette, format="LUT")
        # endif

        if coords is not None:
            auxfile = name
            pos = auxfile.find(".dbf")
            if pos > -1:
                auxfile = auxfile[:pos] + ".shp" + auxfile[pos + 4 :]
                _ = self.owner.addAttr(auxfile, coords, file_date=file_date)
            else:
                tmp_pointer = self.pointer
                self.pointer = coords
                pos = auxfile.find(".shp")
                if pos > -1 and tmp_pointer is not None:
                    auxfile = auxfile[:pos] + ".dbf" + auxfile[pos + 4 :]
                    _ = self.owner.addAttr(auxfile, tmp_pointer, file_date=file_date)
        # endif
        if names is not None:
            self.owner.addAttr('attr_names', names, file_date=file_date)
        # endif
        return lut

    def get(
        self,
        reload=False,
        check_date=False,
        to_not_load=False,
        silent=True,
        no_palette=False,
        get_lut=False,
        format="",
    ):
        """
        It returns file content of current Attr.
        It returns self.pointer if file content has already been read, otherwise it calls the load method.
        FILE_CHANGED=file_changed is output

        Args:
            reload (bool): Flag used to update information inside Attr.Defaults to False.
            check_date (bool): if True, the date is verified using the compareDate method. Defaults to False.
            to_not_load (bool): if True, return immediately without changing Attr.
            silent (bool): Boolean for error logs. Defaults to True.
            no_palette (bool): colour palette for data visualization. Defaults to False.
            get_lut: N.B. param not used.
            format (str): Format of the file. Defaults to ''.

        Returns:
            file_changed (bool): True if the data inside Attr has been modified, False otherwise.
            lut (str): Colour table for data visualization
            format (str): Format of the file. Defaults to ''.
        """
        if self.format != "":
            format = self.format

        file_changed = 0
        lut = None

        if to_not_load:
            return self, file_changed, lut, format

        if self.pointer is not None and reload == False:
            if check_date:
                path = self.getProperty("path")
                name = self.getProperty("name")
                if isinstance(
                    dpg.attr.compareDate(self.file_date, path, name), type(None)
                ):
                    self.pointer = None

        if self.pointer is None or reload:
            lut = self.load(silent=silent, no_palette=no_palette)
            if self.pointer is not None:
                file_changed = True

        return self, file_changed, lut, format

    def copy(self):
        """
        Creates a copy of the current Attr object.

        This method creates a new instance of the Attr class, copying all attributes
        from the current instance to the new one. The new instance will have the same
        name, owner, string format, file format, save flag, file date, and pointer
        content as the original.

        Returns:
            Attr: A new instance of the Attr class with copied attributes.
        """
        attr = Attr(
            name=self.name,
            owner=self.owner,
            str_format=self.str_format,
            format=self.format,
            to_not_save=self.to_not_save,
            file_date=self.file_date,
            pointer=self.pointer,
        )
        return attr

    def removeTags(self, tags: str):
        """
        Remove the passed tags from a dictionary.

        Args:
            tags (str): Array which contains the tags to remove from the dictionary.

        Returns:
            np.ndarray: a list of the removed elements from the dictionary.

        """

        if not isinstance(self.pointer, dict):
            return
        el = []
        if isinstance(tags, str):
            tags = [tags]
        tag_low = [tag.lower() for tag in tags]
        for key in list(self.pointer.keys()):
            if key.lower() in tag_low:
                el.append(self.pointer.pop(key))
        return el

    def replaceTags(
        self,
        tags,
        values,
        to_add: bool = False,
        clean: bool = False,
        rem_inherits: bool = False,
    ):
        """
        Replaces or adds tags and values in the Attr object's pointer dictionary.

        This method modifies the pointer dictionary by replacing or adding the specified tags
        and their corresponding values. It provides options to add values to existing tags,
        remove existing tags, clean the entire dictionary, and remove inherited tags.

        Args:
            tags (str or list of str): The tags to be replaced or added
            values (any or list of any): The values corresponding to the tags
            to_add (bool, optional): If True, adds values to existing tags instead of replacing. Defaults to False
            clean (bool, optional): If True, cleans the entire dictionary before adding new tags. Defaults to False
            rem_inherits (bool, optional): If True, removes 'inherits' tags from the dictionary. Defaults to False

        Note:
            The tags are converted to uppercase before being added or replaced in the dictionary.
            If the dictionary is cleaned, all existing tags and values are removed before adding new ones.
            If 'inherits' tags are removed, they are removed before adding or replacing other tags.
        """
        if (np.size(tags) == 0) or (np.size(values) == 0):
            return
        attr = self.pointer
        # TODO: da fare check che non sia None
        if isinstance(tags, str):
            tags = [tags]
            values = [values]
        if not isinstance(attr, dict):
            return
        if not to_add:
            dpg.attr.removeTags(attr, tags)
        if rem_inherits:
            dpg.attr.removeTags(attr, ["inherits"])
        if clean:
            dpg.attr.removeTags(attr, list(attr.keys()))
        # endif
        for key, val in zip(tags, values):
            if key in attr.keys():
                if isinstance(attr[key], list):
                    attr[key].append(val)
                else:
                    attr[key] = [attr[key], val]

            else:
                attr[key] = val
        return

    def save(self, alt_path: str = ""):
        """
        Saves the Attr object's data to a file.

        This method writes the content of the Attr object's pointer attribute to a file.
        If the `to_not_save` flag is set to True, the method returns without saving. The
        file is saved to the path specified in the object's properties, or to an alternative
        path if provided.

        Args:
            alt_path (str, optional): An alternative path to save the file. Defaults to an
                                      empty string, which means the file is saved to the
                                      original path

        Note:
            The method retrieves the path, name, and format properties of the object to
            determine the file location and format. If an alternative path is provided, it
            overrides the original path. The content is then written to the file using the
            `dpg.attr.writeAttr` method
        """
        if self.to_not_save:
            return
        path = self.getProperty("path")
        name = self.getProperty("name")
        format = self.getProperty("format")

        if (alt_path != "") and (alt_path is not None):
            path = alt_path

        dpg.attr.writeAttr(
            self.pointer, path, name, format=format, str_format=self.str_format
        )

        return
