import gc
from datetime import datetime

import geopandas

import sou_py.dpg as dpg
import sou_py.dpb as dpb

import os
import numpy as np

from sou_py.dpg.attr__define import Attr
from sou_py.dpg.log import log_message
from sou_py.dpg.map__define import Map

"""
Funzioni ancora da portare
PRO Node__Define 
"""

sharedNodes = []


class Node(dpg.container__define.Container):
    """
    A Node class that extends the Container class, representing an individual element in a data structure or a
    graphical interface.

    This class models a Node as a container, allowing it to hold and manage other objects or data elements.

    The Node class includes methods for adding or removing child elements, navigating the hierarchy,
    updating the content or properties of the node, and other functionalities as required by the context.

    Attributes:
        path (str): Path to the directory, used to initialize the root node.
        name (str): Name of the subdirectory, used together with 'parent' when 'path' is not provided.
        to_not_save (bool): If True, the created node will not be stored in memory.
        shared (bool): If True, adds the created node to the global list of shared nodes.
        map (Map): TBD
        parent (Node): The Node object corresponding to the parent directory. Defaults to None.

    Methods:
        __init__(self, path='', name='', to_not_save=False, shared=False, map=None, parent=None):
                Initializes the container with an optional list of objects.
        getProperty(self, str_property): Retrieves a specified property of the current Node object.
        def addAttr(self, name, pointer=None, format='', file_date=0, str_format='', to_not_save=False):
                Adds or updates an attribute to the Node object.
        getSons(self): Retrieves all child nodes (sons) of the current node.
        getSon(self, name): Retrieves a specific child node of the current node by its name.
        addNode(self, name='', remove_if_exists=False, to_not_save=False, get_exists=False):
                Adds a new child node to the current node, with various configuration options.
        createSubTree(self, only_current=False): Creates a subtree under the current node based on the subdirectories
        of its path.
        getMyAttr(self, name): Retrieves a specific attribute of the current node by name.
        getSingleAttr(self, name, only_current=True, to_not_load=False, format='', reload=False, silent=True,
        no_palette=False, check_date=False):
                Retrieves a single attribute with the specified name from the current node or its ancestors.
        getArrayInfo(self, only_current=False, reload=False, check_date=False, aux=False):
                Retrieves detailed information about an array associated with the current node.
        getAllDescendant(self, only_leaves=False, and_me=False, nodes=None):
                Retrieves all descendant nodes of the current node based on specified criteria.
        getValues(self, reload=False, to_create=False, recompute=False, get_bitplanes=False):
                Retrieves value-related information for the current node based on various criteria.
        getArray(self, reload=False, aux=False, to_not_load=False, check_date=False, silent=False,
        replace_quant=False, quantize=None):
                Retrieves array data and related information for the current node.
        getAttr(self, name, format='', stop_at_first=False, only_current=False, to_not_load=False,
                upper=False, lower=False, check_date=False, reload=False, load_if_changed=False):
                Retrieves an attribute with the specified name and format from the current node or its ancestors.
        quantizeData(self, replace=False, force=False, calib=None): Quantizes data for the current node based on
        calibration parameters.
        save(self, only_current=False, alt_path=''): Saves the current node's data and attributes, optionally
        quantizing and handling alternative paths.
        removeAttr(self, name, delete_file=False): Removes a specified attribute from the node and optionally deletes
        its associated file.
        make_prod_desc(self, desc=False, make=False): Creates or updates a product description attribute for the node
        based on specified criteria.
        getRoot(self, level=False): Retrieves the root node of the current node's hierarchy along with its level.
        def cleanUp(self, directory=False, from_root=False, shared=False): TBD
        detachNode(self): Detaches the current node from its parent in the node hierarchy.
        checkMap(self, map=None, dim=None, par=None, reload=False, reset=False, attr=None):
                Initializes or updates a map object for the node based on various parameters and attributes.
        setProperty(self, to_not_save=None): Sets the 'to_not_save' property of the current node.
        replaceAttrValues(self, name, tags, values): Replaces specific values of an attribute of the current node or
        creates a new attribute if it doesn't exist.
        setArrayInfo(self, filename='', type=None, dim=None, format='', aux=False, date='', time='', name=''):
                Sets or updates the array information attribute of the current node.
        setArray(self, array, filename='', format='', aux=0, no_copy=1):
                Sets or updates the array data for the current node, along with associated metadata.

    Inherits:
        dpg.container__define.Container: Inherits the Container class from the 'dpg' framework,
        leveraging its functionalities and properties.
    """

    def __init__(
            self,
            path: str = "",
            name: str = "",
            to_not_save: bool = False,
            shared: bool = False,
            map=None,
            parent=None,
    ):
        """
        Initializes a new Node object, replicating a directory in a filesystem.

        This constructor initializes a Node object, which represents a directory in a filesystem. The node can be
        initialized with either a direct path to the directory or with a name and a parent node. Additional options
        include whether to store the object in memory and whether to add it to a global list of shared nodes.

        Args:
            path (str, optional): Path to the directory, used to initialize the root node. Defaults to None.
            name (str, optional): Name of the subdirectory, used together with 'parent' when 'path' is not provided.
                Defaults to None.
            parent (Node, optional): The Node object corresponding to the parent directory. Defaults to None.
            to_not_save (bool, optional): If True, the created node will not be stored in memory. Defaults to False.
            shared (bool, optional): If True, adds the created node to the global list of shared nodes. Defaults to
            False.
            map: [Description of the 'map' parameter] TBD

        Returns:
            Node: An instance of the Node object representing the specified directory in the filesystem.
        """
        super().__init__(lst=[])

        self.to_not_save = to_not_save
        self.shared = shared
        self.map = map
        self.parent = parent
        self.path = path

        if isinstance(parent, self.__class__):
            if not isinstance(name, str):
                print("Error: name is not str")
                return
            if name == "":
                print("Error: name is empty str 1")
                return
            son = parent.getSon(name)
            if isinstance(son, self.__class__):
                print("Error: son exists")
                print("Name =", name, "parent.path =", parent.path)
                return
            self.parent = parent
            parent.add(self)
            shared = parent.getProperty("shared")
            path = parent.getProperty("path")
            path = dpg.path.getFullPathName(path, os.path.basename(name))

        else:
            if shared:
                dpg.globalVar.GlobalState.SHARED_TREES.append(self)

        if isinstance(path, dict):
            path = path[list(path.keys())[0]]
        if not isinstance(path, str):
            print("Error: path is not str")
            raise TypeError("path is not str")

        if path == "":
            log_message("Error: path is empty str", 'WARNING')
            return

        # if not os.path.exists(path):
        #    raise FileNotFoundError("No such file or directory: "+path)

        self.to_not_save = to_not_save
        self.shared = shared
        self.map = map
        self.parent = parent
        self.path = dpg.path.checkPathname(path)
        self.name = os.path.basename(self.path)

        return

    def getProperty(self, str_property: str) -> type:
        """
        Retrieves a specified property of the current Node object.

        This method returns the value of a given property of the Node object. The property to be returned
        is specified by the 'str_property' parameter.

        Args:
            str_property (str): Name of the property to be returned. Possible values include 'path',
            'shared', 'map', and 'name'.

        Returns:
            [type]: The value of the requested property. The return type depends on the property being retrieved.
        """
        if not isinstance(str_property, str):
            print("WARNING: undefined property in Node.getProperty")
            return None
        str_property = str_property.lower()
        if str_property == "path":
            return self.path
        elif str_property == "shared":
            return self.shared
        elif str_property == "map":
            return self.map
        elif str_property == "name":
            return os.path.basename(self.path)
        else:
            print("WARNING: undefined property in Node.getProperty")
            return None

    def addAttr(
            self,
            name: str,
            pointer=None,
            format: str = "",
            file_date: int = 0,
            str_format: str = "",
            to_not_save: bool = False,
    ):
        """
        Adds or updates an attribute to the Node object.

        This method either creates a new attribute or updates an existing one in the Node object.
        If 'pointer' is None and no file with the given 'name' exists in the Node's path, the method returns None.
        Otherwise, it either creates a new attribute if it doesn't exist, or updates the existing attribute
        with the provided values.

        Args:
            name (str): The name of the attribute to be added or updated.
            pointer (optional): The pointer to the attribute's data. Defaults to None.
            format (str, optional): The format of the attribute. Defaults to an empty string.
            file_date (int, optional): The file date associated with the attribute. Defaults to 0.
            str_format (str, optional): The string format of the attribute. Defaults to an empty string.
            to_not_save (bool, optional): If True, the attribute is not saved immediately. Defaults to False.

        Returns:
            dpg.attr__define.Attr or None: The newly created or updated attribute, or None if the attribute
            cannot be created and does not exist.

        Note:
            The method checks if the attribute exists and is an instance of 'dpg.attr__define.Attr'. If it does
            not exist or is not an instance, a new attribute is created; otherwise, the existing attribute is updated.
        """
        if pointer is None:
            if not os.path.isfile(os.path.join(self.path, name)):
                return None
        # endif

        oAttr = self.getMyAttr(name)

        if not isinstance(oAttr, dpg.attr__define.Attr):
            oAttr = dpg.attr__define.Attr(
                owner=self,
                name=name,
                pointer=pointer,
                format=format,
                file_date=file_date,
                str_format=str_format,
                to_not_save=to_not_save,
            )
            self.add(oAttr)
        else:
            oAttr.setProperty(
                pointer=pointer,
                format=format,
                file_date=file_date,
                str_format=str_format,
                to_not_save=to_not_save,
            )
        # endelse
        return oAttr

    def getSons(self):
        """
        Retrieves all child nodes (sons) of the current node.

        This method returns a list of child nodes associated with the current node. It fetches the child
        nodes that are instances of the same class as the current node.

        Returns:
            list: A list of child nodes of the current node. The list contains instances of the same class
            as the current node.
        """

        def parse_name(name):
            try:
                # Split into date and time components
                if '.' in name:
                    date_part, time_part = name.split('.')
                    # Parse date and time correctly
                    datetime_obj = datetime.strptime(f"{date_part}.{time_part.zfill(4)}", "%d-%m-%Y.%H%M")
                    return datetime_obj
            except (ValueError, IndexError):
                pass

            # try:
            #     # Try parsing as an integer
            #     return int(name)
            # except ValueError:
            #     pass

            # If not a date or integer, return as a string for lexicographic sorting
            return name

        # Retrieve the sons
        sons = self.get(self.__class__)

        # Sort the sons using the custom parsing function
        sons_sorted = sorted(sons, key=lambda son: parse_name(son.name))

        return sons_sorted

    def getSon(self, name: str):
        """
        Retrieves a specific child node of the current node by its name.

        This method searches for a child node of the current node by a given name. It handles special
        cases where the name represents the current node ('<current>', '.'), the parent node ('..'),
        or a shared tree ('@' prefix). The method returns the corresponding child node if found.
        If the name is not a string, empty, or no matching child node is found, it returns None.

        Args:
            name (str): The name of the child node to retrieve. Special names include '<current>', '.',
            '..', and names starting with '@' for shared trees.

        Returns:
            dpg.node__define.Node or None: The child node matching the specified name, or None if no
            matching node is found or if the name is invalid.

        Note:
            The method performs case-insensitive comparison for the node name. It prints an error message
            and returns None if the 'name' parameter is not a valid string or is empty.
        """
        if not isinstance(name, str):
            print("Error: name is not str")
            return None
        if name == "":
            print("Error: name is empty str 2")
            return None
        if name.startswith("@"):
            return dpg.tree.createTree(name, shared=True)
        if name == "<current>":
            return self
        if name == ".":
            return self
        if name == "..":
            return self.parent

        name = os.path.basename(name)

        upName = name.upper()
        sons = self.getSons()
        for son in sons:
            if isinstance(son, self.__class__):
                nodename = son.getProperty("name")
                if nodename.upper() == upName.upper():
                    return son
            # endif
        # endfor
        return None

    def addNode(
            self,
            name="",
            remove_if_exists: bool = False,
            to_not_save: bool = False,
    ):
        """
        Adds a new child node to the current node, with various configuration options.

        This method creates and adds a new child node with the specified 'name' to the current node. It handles
        cases where a node with the same name already exists, and provides options to remove the existing node,
        not save the new node immediately, and to return the existence status of the node.

        Args:
            name (str): The name of the new child node to be added. If empty or not a string, an error is printed and
            None is returned.
            remove_if_exists (bool, optional): If True, any existing child node with the same name will be removed
                before adding the new node. Defaults to False.
            to_not_save (bool, optional): If True, the new node will not be saved immediately. Defaults to False.
            get_exists (bool, optional): If True, returns a tuple containing the new node and a boolean flag indicating
                whether the node already existed. Defaults to False.

        Returns:
            Node or (Node, bool): The new child node that was added. If 'get_exists' is True, a tuple is returned
            where the first element is the new node and the second element is a boolean flag indicating whether
            the node already existed.

        Note:
            The method performs checks for the validity of the 'name'. If the 'name' is empty or not a string,
            an error message is printed and None is returned. If a node with the same name exists, and
            'remove_if_exists' is False, the existing node is returned.
        """
        if not isinstance(name, str):
            log_message("addNode Error: name passed is not a str", level="WARNING+")
            return None
        if name == "":
            log_message("addNode Error: name is empty")
            return None

        exists = False
        son = self.getSon(name)
        if isinstance(son, self.__class__):
            # print('Warning: Node already exists')
            if not remove_if_exists:
                exists = True
                return son, exists
            dpg.tree.removeNode(son, directory=True)
        son = Node(name=name, parent=self, to_not_save=to_not_save)
        return son, exists

    def createSubTree(self, only_current: bool = False):
        """
        Creates a subtree under the current node based on the subdirectories of its path.

        This method iterates over subdirectories of the current node's path and creates a corresponding
        child node ('son') for each subdirectory. If 'only_current' is False, the method recursively
        calls itself to create subtrees for these child nodes, replicating the directory structure.

        Args:
            only_current (bool, optional): If True, the method only creates direct child nodes for the current
                node without further recursion into subdirectories. Defaults to False.

        Returns:
            None: The function does not return any value.

        Note:
            The subtree creation is dependent on the subdirectories present in the node's path. The method
            uses 'dpg.cfg.getSubDir' to retrieve the list of subdirectories. This function is a recursive
            implementation, creating a complete tree structure replicating the filesystem's hierarchy.
        """

        dirs = dpg.cfg.getSubDir(self.path)
        for subdir in dirs:
            son, _ = self.addNode(subdir)
            if isinstance(son, self.__class__) and (not only_current):
                son.createSubTree(only_current=only_current)
        return

    def getMyAttr(self, name: str):
        """
        Retrieves a specific attribute of the current node by name.

        This method searches for an attribute with the given 'name' among the attributes of the current node.
        The search is case-insensitive. The method returns the attribute if found, or None otherwise. It
        validates that 'name' is a non-empty string before proceeding with the search.

        Args:
            name (str): The name of the attribute to retrieve.

        Returns:
            dpg.attr__define.Attr or None: The attribute with the specified name if found, otherwise None.

        Note:
            If 'name' is not a string or is an empty string, the function prints an error message and returns None.
            The function iterates over the set of attributes associated with the node, performing a case-insensitive
            comparison to find the matching attribute.
        """
        if not isinstance(name, str):
            print("Error: Name is not str")
            return None
        if name == "":
            print("Error: name is empty str")
            return None
        attr_set = self.get(dpg.attr__define.Attr)

        if len(attr_set) == 0:
            return None

        # name = os.path.basename(name)

        upName = name.upper()
        for attr in attr_set:
            if isinstance(attr, dpg.attr__define.Attr):
                nnn = attr.getProperty("name")
                if nnn.upper() == upName:
                    return attr
            # endif
        # endfor
        return None

    def getSingleAttr(
            self,
            name: str,
            only_current: bool = False,
            to_not_load: bool = False,
            format: str = "",
            reload: str = False,
            silent: bool = True,
            no_palette: bool = False,
            check_date: bool = False,
            return_format: bool = False,
            get_owner: bool = False,
            get_file_changed: bool = False,
    ):
        """
        Retrieves a single attribute with the specified name from the current node or its ancestors.

        This method searches for an attribute named 'name' in the current node and, if not found and
        'only_current' is False, continues searching up the node hierarchy. The search includes options
        for loading the attribute, checking its date, and handling its format.

        Args:
            name (str): The name of the attribute to retrieve.
            only_current (bool, optional): If True, the search is limited to the current node. Defaults to True.
            to_not_load (bool, optional): If True, the attribute is not loaded. Defaults to False.
            format (str, optional): The format of the attribute. Defaults to an empty string.
            reload (bool, optional): If True, the attribute is reloaded. Defaults to False.
            silent (bool, optional): If True, suppresses print statements. Defaults to True.
            no_palette (bool, optional): If True, no palette is used in the attribute. Defaults to False.
            check_date (bool, optional): If True, checks the date of the attribute. Defaults to False.

        Returns:
            dpg.attr__define.Attr or None: The attribute object if found, otherwise None.

        Note:
            The method returns None if 'name' is not a valid string, is empty, or if the attribute is not
            found. The method also adds the attribute to the node if it does not exist and 'to_not_load'
            is False.
        """

        if not isinstance(name, str) or name == "":
            # print('Error: Name not valid')
            return None, None

        owner = self
        file_changed = 0

        while isinstance(owner, self.__class__):
            attr = owner.getMyAttr(name)
            if not isinstance(attr, dpg.attr__define.Attr) and (to_not_load == False):
                attr = owner.addAttr(name, format=format)
            if isinstance(attr, dpg.attr__define.Attr):
                attr, file_changed, lut, format = attr.get(
                    format=format,
                    check_date=check_date,
                    reload=reload,
                    to_not_load=to_not_load,
                    silent=silent,
                    no_palette=no_palette,
                )
            if attr is not None or only_current:
                if return_format and get_owner and get_file_changed:
                    return attr, format, owner, file_changed
                elif return_format and get_owner:
                    return attr, format, owner
                elif return_format and get_file_changed:
                    return attr, format, file_changed
                elif get_owner and get_file_changed:
                    return attr, owner, file_changed
                elif return_format:
                    return attr, format
                elif get_owner:
                    return attr, owner
                elif get_file_changed:
                    return attr, file_changed
                else:
                    return attr

            owner = owner.parent

        if return_format and get_owner and get_file_changed:
            return None, None, None, None
        elif (
                (return_format and get_owner)
                or (return_format and get_file_changed)
                or (get_owner and get_file_changed)
        ):
            return None, None, None
        elif return_format or get_owner or get_file_changed:
            return None, None
        else:
            return None

    def getArrayInfo(
            self,
            only_current: bool = False,
            reload: bool = False,
            check_date: bool = False,
            aux: bool = False,
    ):
        """
        Retrieves detailed information about an array associated with the current node.

        This method fetches various properties and metadata related to an array attribute of the current node.
        It includes options to specify how the attribute data is fetched, such as only considering the current
        state, reloading the data, and checking the date. It also has an option to fetch auxiliary file information.

        Args:
            only_current (bool, optional): If True, only considers the current state of the array attribute. Defaults
            to False.
            reload (bool, optional): If True, reloads the array attribute data. Defaults to False.
            check_date (bool, optional): If True, checks the date of the array attribute. Defaults to False.
            aux (bool, optional): If True, retrieves information about an auxiliary file. Defaults to False.

        Returns:
            dict or None: A dictionary containing various properties of the array, such as attribute object,
            lookup table file, endian type, format, dimensions, and others. Returns None if no relevant
            attribute is found.

        Note:
            The method attempts to fetch array information from the array description attribute first,
            and if not found, from the geographic description attribute. It constructs a dictionary of all
            relevant array properties, including file names, format, dimensions, and type.
        """
        bitplanes = None

        # cosa sarebbe load_if_changed? questo flag è presente nel metodo getAttr()
        # TODO: rimane da comprendere questa cosa --> probabilmente da chiedere a Mimmo
        attr = self.getSingleAttr(
            dpg.cfg.getArrayDescName(),
            only_current=only_current,
            reload=reload,
            check_date=check_date,
        )  # TODO load_if_changed ?

        if attr is None:
            attr = self.getSingleAttr(dpg.cfg.getGeoDescName(), only_current=True)
            if attr is None:
                return None, None, None, None, None, None, None, None

        lut, _, _ = dpg.attr.getAttrValue(attr, "lutfile", "")
        endian, _, _ = dpg.attr.getAttrValue(attr, "endian", 0)
        format, _, _ = dpg.attr.getAttrValue(attr, "format", "")
        tmp, exists, _ = dpg.attr.getAttrValue(attr, "bitplanes", 8)
        if exists:
            bitplanes = tmp

        if aux:
            filename, _, _ = dpg.attr.getAttrValue(attr, "datafile.aux", "")
        else:
            filename, _, _ = dpg.attr.getAttrValue(attr, "datafile", "")
            if filename == "":
                filename, _, _ = dpg.attr.getAttrValue(attr, "imgfile", "")
                if filename != "" and (format == "" or format is None):
                    format = filename.split(".")[-1]

        type, _, _ = dpg.attr.getAttrValue(attr, "type", 1)
        ncols, _, attr_ind = dpg.attr.getAttrValue(attr, "ncols", 0)
        nlines, _, l_ind = dpg.attr.getAttrValue(attr, "nlines", 0)
        nplanes, _, p_ind = dpg.attr.getAttrValue(attr, "nplanes", 0)

        # Warning: Abbiamo cambiato l'ordine delle dimensioni perchè in
        # IDL scorrono in maniera diversa. Il nuovo ordine assume (elevation, azimuth, range)
        if ncols > 0:  # and l_ind == attr_ind:
            dim = np.array([nlines, ncols])
            if nplanes > 0:  # and p_ind == attr_ind:
                dim = np.array([nplanes, nlines, ncols])
        else:
            dim = np.array(ncols)

        return attr, bitplanes, dim, endian, filename, format, lut, type

    def getAllDescendant(
            self, only_leaves: bool = False, and_me: bool = False, nodes: list = None
    ) -> list:
        """
        Retrieves all descendant nodes of the current node based on specified criteria.

        This method accumulates all descendant nodes of the current node in a list. It allows for the option
        to include only leaf nodes (nodes without children), include the current node itself in the list,
        and specify an existing list to which the descendant nodes should be added.

        Args:
            only_leaves (bool, optional): If True, only leaf nodes are included in the result. Defaults to False.
            and_me (bool, optional): If True, includes the current node in the list of descendants. Defaults to False.
            nodes (list, optional): An existing list to which descendant nodes will be added. If None, a new list
                is created. Defaults to None.

        Returns:
            list: A list of descendant nodes according to the specified criteria.

        Note:
            The method first checks if the current node has child nodes. If not, it adds the current node to the
            list if 'and_me' or 'only_leaves' is True. It then recursively calls itself on each child node to
            accumulate all descendant nodes. If 'only_leaves' is False, all child nodes are added to the list.
        """
        if nodes is None:
            nodes = []
        sons = self.getSons()

        if len(sons) == 0:
            if and_me:
                nodes.append(self)
                return nodes
            # endif
            if only_leaves:
                nodes.append(self)
            # endif
            return nodes
        # endif

        if not only_leaves:
            nodes.extend(sons)  # HACK perché usare extend e non append?
            if and_me:
                nodes.append(self)
        # endif

        for son in sons:
            if isinstance(son, self.__class__):
                nodes = son.getAllDescendant(nodes=nodes, only_leaves=only_leaves)
        # endfor

        return nodes

    def getValues(
            self,
            reload: bool = False,
            to_create: bool = False,
            recompute: bool = False,
            bitplanes: int = None,
            values=None,
    ) -> dict:
        """
        Retrieves value-related information for the current node based on various criteria.

        This method fetches calibration data and other value-related information for the current node. It
        includes options to reload the data, recompute values, create new values, and fetch bitplane information.

        Args:
            reload (bool, optional): If True, reloads the attribute data. Defaults to False.
            to_create (bool, optional): If True, creates new values if they don't exist. Defaults to False.
            recompute (bool, optional): If True, recomputes the values. Defaults to False.
            get_bitplanes (bool, optional): If True, fetches bitplane information. Defaults to False.

        Returns:
            dict: A dictionary containing various pieces of value-related information, such as calibration data,
            file paths, bitplanes, and value arrays.

        Note:
            The method fetches calibration data and, depending on the flags, other related information. If certain
            conditions are met (like 'recompute' or 'to_create' being True), it may create or recompute attribute
            values. The method constructs and returns a dictionary containing all relevant information.
        """

        calib = self.getAttr(
            dpg.cfg.getValueDescName(), reload=reload, stop_at_first=True
        )
        if calib is None:
            return None, None

        if isinstance(calib, list):
            # calib è sempre una lista con lunghezza pari a 1 stop_at_first=True.
            # Di conseguenza questa condizione risulterà sempre True
            if len(calib) == 1:
                calib = calib[0]

        vfile, _, _ = dpg.attr.getAttrValue(calib, "valuesfile", "values.txt")
        if bitplanes is not None:
            bitplanes, _, _ = dpg.attr.getAttrValue(calib, "bitplanes", default=bitplanes)

        if not recompute:
            vPointer = self.getSingleAttr(
                vfile, only_current=True, format="VAL_256", silent=True, reload=reload
            )
        else:
            vPointer = None
            to_create = True

        if isinstance(vPointer, dpg.attr__define.Attr):
            if values is not None:
                values = vPointer.pointer
            return calib, vPointer.pointer

        if not to_create:
            return calib, None

        if bitplanes is None:
            attr = self.getSingleAttr(dpg.cfg.getArrayDescName(), only_current=True)
            tmp, exists, _ = dpg.attr.getAttrValue(attr, "bitplanes", 8)
            # exist è sempre 'False' dato dal fatto che restituisce 'generic.txt' che non contiene bitplanes
            # e quindi la condizione qui sotto non sarà mai vera.
            if exists:
                bitplanes = tmp

        bitplanes, _, _, slope, _, _, _, _, _, values = (
            dpg.calibration.createArrayValues(calib, bitplanes=bitplanes)
        )

        oAttr = self.addAttr(vfile, pointer=values, format="VAL_256", to_not_save=True)

        return calib, values

    def getArray(
            self,
            reload: bool = False,
            aux: bool = False,
            to_not_load: bool = False,
            check_date: bool = False,
            silent: bool = False,
            replace_quant: bool = False,
            quantize=None,
    ):
        """
        Retrieves array data and related information for the current node.

        This method fetches detailed array data from the current node, including dimensions, file type,
        format, lookup tables, bitplanes, calibration data, and other relevant properties. It includes
        options for reloading data, checking dates, handling auxiliary files, and quantizing data.

        Args:
            reload (bool, optional): If True, reloads the array data. Defaults to False.
            aux (bool, optional): If True, fetches auxiliary file information. Defaults to False.
            to_not_load (bool, optional): If True, the data is not loaded. Defaults to False.
            check_date (bool, optional): If True, checks the date of the array data. Defaults to False.
            silent (bool, optional): If True, suppresses print statements. Defaults to False.
            replace_quant (bool, optional): If True, replaces quantized data. Defaults to False.
            quantize: [Description of the 'quantize' parameter]

        Returns:
            tuple:
                - The pointer to the array data.
                - A dictionary containing various properties of the array such as dimensions, format,
                calibration data, and others.

        Note:
            The method constructs a dictionary of array properties and fills it with data obtained from
            the node's attributes. It handles different data types and conditions, such as quantization
            and invalid values. The method returns None and an empty dictionary if the array data cannot
            be retrieved.
        """
        # input: reload, to_not_load, check_date, aux, silent, replace_quant, quantize
        # output: dim, type, filename, format, values, lut, bitplanes, calib, file_type(qui dtype), file_changed(
        # ancora non gestito)

        out_dict = {}
        out_dict["bitplanes"] = None
        out_dict["dim"] = [0]
        out_dict["filename"] = ''
        out_dict["file_changed"] = 0
        out_dict["file_type"] = 0
        out_dict["format"] = ''
        out_dict["type"] = 0

        attr, bitplanes, _, _, filename, format, lut, file_type = self.getArrayInfo(
            only_current=True, reload=reload, check_date=check_date, aux=aux
        )

        if attr is None:
            return None, out_dict

        pointer_attr, file_changed = self.getSingleAttr(
            filename,
            only_current=True,
            to_not_load=to_not_load,
            reload=reload,
            check_date=check_date,
            silent=silent,
            format=format,
            get_file_changed=True,
        )

        if pointer_attr is None:
            log_message(
                f"Cannot find data in {dpg.tree.getNodePath(node=self)}",
                level="WARNING",
                all_logs=True,
            )
            return None, out_dict
        if pointer_attr.pointer is None:
            return None, out_dict

        dtype_attr = getattr(pointer_attr.pointer, "dtype", None)
        dtype = dpg.io.type_py2idl(
            dtype_attr if dtype_attr is not None else type(pointer_attr.pointer)
        )

        if isinstance(pointer_attr.pointer, np.ndarray) or isinstance(pointer_attr.pointer, geopandas.GeoDataFrame):
            dim = pointer_attr.pointer.shape
        elif isinstance(pointer_attr.pointer, list):
            dim = len(pointer_attr.pointer)
        else:
            log_message("Pointer_attr.pointer type unknown. Check", level='WARNING+')
            dim = (1, 1)

        # if dim[0] == 0:
        #     dim = (1, 1)

        if dtype < 4 or dtype > 11:
            bitplanes, _, _ = dpg.attr.getAttrValue(attr, "bitplanes", 8)
            to_create = True
        else:
            to_create = False

        calib, values = self.getValues(
            reload=reload, to_create=to_create, bitplanes=bitplanes
        )

        if dtype == 4 or dtype == 5:
            fNull, exists, _ = dpg.attr.getAttrValue(calib, "fNull", 0.0)
            if not exists:
                fNull, _, _ = dpg.attr.getAttrValue(calib, "undetect", 0.0)
            if exists:
                ind_null, count_null, _, _ = dpg.values.count_invalid_values(
                    pointer_attr.pointer, f_null=fNull
                )
                if count_null > 0:
                    pointer_attr.pointer[ind_null] = np.nan
                _ = dpg.attr.removeTags(calib, ["fNull", "undetect"])
            if quantize is not None:
                if isinstance(quantize, list):
                    log_message("Error keeping quantize!", level="ERROR")
                    quantize = quantize[0]
                scaling, _, _ = dpg.attr.getAttrValue(quantize, "scaling", 0)
                quant_dict = self.quantizeData(
                    calib=quantize, replace=True, force=scaling
                )
                values = quant_dict["values"]
        else:
            if values is not None:
                if replace_quant:
                    pointer_attr.pointer = values[pointer_attr.pointer]

        dtype_attr = getattr(pointer_attr.pointer, "dtype", None)
        dtype = dpg.io.type_py2idl(
            dtype_attr if dtype_attr is not None else type(pointer_attr.pointer)
        )

        out_dict["bitplanes"] = bitplanes
        out_dict["dim"] = dim
        out_dict["filename"] = filename
        out_dict["file_changed"] = file_changed
        out_dict["file_type"] = file_type
        out_dict["format"] = format
        out_dict["type"] = dtype

        return pointer_attr.pointer, out_dict

    def getAttr(
            self,
            name: str,
            format: str = "",
            stop_at_first: bool = False,
            only_current: bool = False,
            to_not_load: bool = False,
            upper: bool = None,
            lower: bool = False,
            check_date: bool = False,
            reload: bool = False,
            load_if_changed=False,
    ):
        """
        Retrieves an attribute with the specified name and format from the current node or its ancestors.

        This method searches for an attribute named 'name' in the current node and, if not found, continues
        the search up the node hierarchy. The method includes options for handling the attribute's format,
        checking dates, reloading data, and controlling the scope of the search.

        Args:
            name (str): The name of the attribute to retrieve.
            format (str, optional): The format of the attribute. Defaults to an empty string.
            stop_at_first (bool, optional): If True, stops the search at the first matching attribute. Defaults to
            False.
            only_current (bool, optional): If True, considers only the current state of the attribute. Defaults to
            False.
            to_not_load (bool, optional): If True, the attribute data is not loaded. Defaults to False.
            upper (bool, optional): If True, retrieves the attribute with uppermost ownership. Defaults to False.
            lower (bool, optional): If True, retrieves the attribute with lowermost ownership. Defaults to False.
            check_date (bool, optional): If True, checks the date of the attribute. Defaults to False.
            reload (bool, optional): If True, reloads the attribute. Defaults to False.
            load_if_changed (bool, optional): If True, loads the attribute if it has changed. Defaults to False.

        Returns:
            list of dpg.attr__define.Attr: A list of attributes is returned.

        Note:
            The method iterates through the node hierarchy, accumulating inherited attributes if found. It
            handles different conditions based on the provided flags and returns either a single attribute
            or a list of attributes based on the 'upper', 'lower', 'only_current', and 'stop_at_first' options.
        """

        attr = None
        owner = None
        lastOwn = None
        # format = None
        if not isinstance(name, str) or name == "":
            return attr

        nIn = 0
        curr = self
        while isinstance(curr, self.__class__):
            attrTmp, format, own = curr.getSingleAttr(
                name,
                format=format,
                only_current=only_current,
                to_not_load=to_not_load,
                check_date=check_date,
                reload=reload,
                return_format=True,
                get_owner=True,
            )
            if attrTmp is None:
                if upper is not None and upper:
                    owner = lastOwn
                    if isinstance(attr, list) and len(attr) > 0:
                        attr = [attr[-1]]
                return attr
            lastOwn = own
            if nIn == 0:
                owner = own

            if format is None or format.upper() != "TXT" or to_not_load or lower:
                return [attrTmp]

            if isinstance(attrTmp, dpg.attr__define.Attr):
                inherit = attrTmp.getInherited()
                nIn = len(inherit)

            if only_current and nIn <= 0:
                return [attrTmp]

            if attr is None:
                attr = []

            attr.append(attrTmp)

            for inel in inherit:
                attrTmp = curr.getSingleAttr(
                    inel,
                    format=format,
                    reload=reload,
                    check_date=check_date,
                    only_current=True,
                )
                if attrTmp is not None:
                    attr.append(attrTmp)

            if only_current or stop_at_first:
                return attr
            curr = own.parent
        # end while

        if attr is not None and upper and len(attr) > 1:
            attr = [attr[-1]]
        # endif

        return attr

    def quantizeData(self, replace: bool = False, force: bool = False, calib=None):
        """
        Quantizes data for the current node based on calibration parameters.

        This method performs data quantization on the current node's array data. It fetches the array data,
        applies quantization based on calibration parameters, and handles various options like replacing the
        original data, forcing a specific format, and using external calibration data.

        Args:
            replace (bool, optional): If True, replaces the original data with the quantized data. Defaults to False.
            force (bool, optional): If True, forces quantization even if the data format is not 'DAT' or 'TIF'.
            Defaults to False.
            calib: Calibration data to be used for quantization. If None, uses internal calibration data. Defaults to
            None.

        Returns:
            dict: A dictionary containing quantized data and related information, such as data pointers, values,
            and log scale information.

        Note:
            The method retrieves the array description and datafile attributes to fetch the array data. It then
            checks for valid data types and formats before proceeding with quantization. The method returns a
            dictionary with quantized data and additional information. It updates attributes and tags based on
            the 'replace', 'force', and 'calib' parameters.
        """

        out = {}

        attr = self.getSingleAttr(dpg.cfg.getArrayDescName(), only_current=True)
        datafile, _, _ = dpg.attr.getAttrValue(attr, "datafile", "")
        oAttr = self.getSingleAttr(datafile, to_not_load=True, only_current=True)
        if isinstance(oAttr, tuple):
            if any(value is None for value in oAttr):
                return out
        if oAttr is None:
            return out

        out["oAttr"] = oAttr
        out["datafile"] = datafile
        out["quant"] = oAttr.pointer

        dtype_attr = getattr(oAttr.pointer, "dtype", None)
        dtype = dpg.io.type_py2idl(
            dtype_attr if dtype_attr is not None else type(oAttr.pointer)
        )
        # dtype = dpg.io.type_py2idl(oAttr.pointer.dtype)
        if dtype < 4 or dtype > 5:
            return out

        format = oAttr.getProperty("format")
        if (format.upper() != "DAT") and (format.upper() != "TIF"):
            if not force:
                return {}
        # endif

        out_calib = self.getAttr(dpg.cfg.getValueDescName(), only_current=True)
        if out_calib is None:
            if calib is None:
                print("LogMessage", self.path + " ... Cannot find calibration.")
                return {}
            # endif
            out_calib = calib
        else:
            if calib is None:
                calib = out_calib
        # endelse

        out_, out_calib = dpg.calibration.quantizeData(
            oAttr.pointer, calib, out_calib=out_calib
        )
        if not out_:
            return out_
        quant = out_["quant"]
        out["quant"] = out_["quant"]
        values = out_["values"]
        out["values"] = out_["values"]
        log_scale = out_["log_scale"]
        out["log_scale"] = out_["log_scale"]

        if replace:
            oAttr.pointer = quant

        ret = self.addAttr(dpg.cfg.getValueDescName(), out_calib)

        dtype_attr = getattr(quant, "dtype", None)
        dtype = dpg.io.type_py2idl(
            dtype_attr if dtype_attr is not None else type(quant)
        )

        dpg.attr.replaceTags(attr, ["type"], [dtype])

        vName = "values.txt"
        str_format, _, _ = dpg.attr.getAttrValue(out_calib, "str_format", "")
        if replace or log_scale or str_format != "":
            if str_format == "":
                _ = self.addAttr(
                    vName,
                    values,
                    format="VAL",
                    str_format=str_format,
                    to_not_save=False,
                )
                out_calib = dpg.attr.replaceTags(out_calib, "valuesfile", vName)
        else:
            vAttr = self.getSingleAttr(vName, to_not_load=True, only_current=True)
            if not isinstance(vAttr, type(None)):
                vAttr.setProperty(to_not_save=True)
            dpg.attr.removeTags(out_calib, ["valuesfile"])
        # endelse
        return out

    def save(self, only_current: bool = False, alt_path: str = ""):
        """
        Saves the current node's data and attributes, optionally quantizing and handling alternative paths.

        This method saves the current node, including its attributes and quantized data if applicable. It
        provides options to save only the current state of the node and to specify an alternative path for
        saving. The method also handles saving of all associated objects.

        Args:
            only_current (bool, optional): If True, only the current state of the node is saved. Defaults to False.
            alt_path (str, optional): An alternative file path where the node's data will be saved.
                If empty, the default path associated with the node is used. Defaults to an empty string.

        Returns:
            None: The function does not return any value.

        Note:
            The method quantizes data if necessary and saves it to the specified path. It iterates over
            all objects associated with the node, saving each one. The function checks and respects the
            'to_not_save' property of attributes to avoid unwanted saving. If 'only_current' or 'alt_path'
            is specified, the method limits its scope accordingly.
        """

        if self.to_not_save:
            return

        out = self.quantizeData()

        if "quant" in out.keys():
            quant = out["quant"]
        else:
            quant = None
        if "datafile" in out.keys():
            datafile = out["datafile"]
        else:
            datafile = None
        if "oAttr" in out.keys():
            oAttr = out["oAttr"]
        else:
            oAttr = None

        if isinstance(oAttr, dpg.attr__define.Attr) and isinstance(quant, np.ndarray):
            to_not_save = oAttr.to_not_save
            if not to_not_save:
                path = self.path
                if alt_path:
                    path = alt_path
                _ = dpg.attr.save_var(path, datafile, quant)
                oAttr.setProperty(to_not_save=True)
                # endif
        # endif

        if (only_current) or (alt_path):
            classtype = dpg.attr__define.Attr
        else:
            classtype = None

        objs = self.get(classtype)
        for ooo in objs:
            if isinstance(ooo, dpg.attr__define.Attr) or isinstance(
                    ooo, dpg.node__define.Node
            ):
                ooo.save(alt_path=alt_path)
        # endfor

        if isinstance(oAttr, dpg.attr__define.Attr) and isinstance(quant, np.ndarray):
            oAttr.setProperty(to_not_save=to_not_save)

        return

    def removeAttr(self, name: str, delete_file: bool = False):
        """
        Removes a specified attribute from the node and optionally deletes its associated file.

        This method removes an attribute with the given 'name' from the node. If 'delete_file' is True,
        and a file associated with the attribute exists in the node's path, that file is also deleted.

        Args:
            name (str): The name of the attribute to be removed.
            delete_file (bool, optional): If True, the file associated with the attribute is deleted
                if it exists. Defaults to False.

        Returns:
            None: The function does not return any value.

        Note:
            The method retrieves the attribute using 'getMyAttr' and then removes it. If 'delete_file' is
            True, the method checks if a file with the attribute's name exists in the node's path and
            deletes it if present.
        """

        attr = self.getMyAttr(name)
        pathtofile = os.path.join(self.path, name)

        if delete_file and os.path.exists(pathtofile):
            os.remove(pathtofile)

        if isinstance(attr, dpg.attr__define.Attr):
            self.remove(attr)

        return

    def make_prod_desc(self, desc: bool = False, make: bool = False):
        """
        Creates or updates a product description attribute for the node based on specified criteria.

        This method handles the creation or updating of a product description attribute for the current node.
        It allows for the option to create a new description or modify an existing one based on the 'desc'
        and 'make' parameters.

        Args:
            desc (bool or other type, optional): Determines the action to be taken for the product description.
                If True, updates or creates the description. If False or 0, no action is taken. If set to 1,
                updates the description with the value of 'desc'. Defaults to False.
            make (bool, optional): If True, checks for the existence of a 'makedesc' attribute and creates or
                updates the product description based on its value. Defaults to False.

        Returns:
            None: The function does not return any value, but may modify or create a product description attribute.

        Note:
            The function's behavior varies based on the values of 'desc' and 'make'. If 'make' is True,
            it uses the value of a 'makedesc' attribute to create or update the product description.
            If 'desc' is set to 1, the product description is updated with the value of 'desc'.
        """
        if desc:
            tmp = desc
        prodName = dpg.cfg.getProductDescName()
        if make:
            attr = self.getSingleAttr(prodName)
            make, _, _ = dpg.attr.getAttrValue("makedesc", "")
            if len(make) == 0:
                return

        # desc = translateTextTag() commentato per import circolare con tree.py
        if desc != 1:
            return
        if desc == 0:
            tmp = desc
            return
        ret = dpg.tree.replaceAttrValues(
            self, prodName, "desc", desc, only_current=True, to_save=True
        )
        return

    def getRoot(self):
        """
        Retrieves the root node of the current node's hierarchy along with its level.

        This method traverses up the node hierarchy from the current node to find the root node. It optionally
        calculates the level of the current node, which is the number of steps from the current node to the root.

        Returns:
            tuple: A tuple containing:
                - The root node of the current node's hierarchy.
                - The level of the current node if 'level' is True; otherwise, 0.

        Note:
            The level calculation starts from 0 and increments by 1 for each step up the node hierarchy until
            the root is reached. If 'level' is False, the level is returned as 0 regardless of the actual level.
        """

        level = 0

        curr = self
        next = self.parent
        while isinstance(next, Node):
            curr = next
            next = next.parent
            level += 1

        return curr, level

    def obj_destroy(self, obj):
        """
        Destroys the specified object by deleting it. This function removes the object
        from memory, effectively freeing up resources associated with it.

        Parameters:
            obj: The object to be destroyed.

        Returns:
            None.
        """
        del obj

    def cleanUp(
            self, directory: bool = False, from_root: bool = False, shared: bool = False
    ):
        """
        Cleans up the current object, including its child nodes, attributes, and associated map.
        Optionally removes the directory corresponding to the object if certain conditions are met.
        Handles cleanup from either the root node or the current node.

        Parameters:
            directory (bool, optional): If True, attempts to remove the directory associated with the object.
            from_root (bool, optional): If True, performs cleanup starting from the root node (default is False).
            shared (bool, optional): If True, allows cleanup of shared directories; otherwise, shared directories are
            preserved (default is False).

        Returns:
            None.
        """
        if from_root:
            root, _ = self.getRoot()
            sons = root.getSons()
            path = root.getProperty("path")
        else:
            sons = self.getSons()
            path = self.path

        for son in sons:
            if isinstance(son, Node):
                self.obj_destroy(son)

        attr_set = self.get(dpg.attr__define.Attr)
        for attr in attr_set:
            if isinstance(attr, dpg.attr__define.Attr):
                self.obj_destroy(attr)

        if isinstance(self.map, dpg.map__define.Map):
            self.obj_destroy(self.map)

        if directory:
            if len(path) < 10:
                log_message(path + " NOT Removed!", level="WARNING")
            else:
                if self.shared and not shared:
                    log_message(path + " NOT Removed!", level="WARNING")
                else:
                    dpg.utility.delete_path(
                        path
                    )  # FILE_DELETE, path, /RECURSIVE, /ALLOW_NONEXISTENT
        # endif

        if isinstance(self.parent, Node):
            self.parent.remove(self)

        self.Cleanup(self)

    def detachNode(self):
        """
        Detaches the current node from its parent in the node hierarchy.

        This method removes the current node from its parent, effectively detaching it from the node hierarchy.
        The parent node is updated to reflect the removal of this node. If the current node has no parent,
        the function does nothing.

        Returns:
            None: The function does not return any value.

        Note:
            The method checks if the current node has a parent. If a parent exists, the node is removed from
            the parent's children, and the 'parent' attribute of the node is set to None, indicating that
            it no longer belongs to any parent node.
        """
        if self.parent is None:
            return
        self.parent.remove(self)
        self.parent = None

    def checkMap(
            self,
            map: Map = None,
            dim=None,
            par: list = None,
            reload: bool = False,
            reset: bool = False,
            attr=None,
    ):
        """
        Initializes or updates a map object for the node based on various parameters and attributes.

        This method handles the creation or update of a map object associated with the node. It initializes
        or updates the map using geographical and navigational attributes. The method can also use an existing
        map object, dimensions, and parameters as input for the update.

        Args:
            map: An existing map object to be updated. If provided, the method updates this map based on
                the node's attributes. If None, a new map object is created.
            dim (optional): The dimensions to be used for the map. Defaults to None.
            par (list, optional): A list of parameters for the map. Defaults to an empty list.
            reload (bool, optional): If True, reloads the node's geographical attributes. Defaults to False.
            reset (bool, optional): If True, resets the existing map object before updating. Defaults to False.
            attr (optional): Geographical attributes to be used for initializing or updating the map.
                If None, attributes are fetched from the node.

        Returns:
            tuple: A tuple containing:
                - The map object associated with the node.
                - The parameters used for the map.
                - The dimensions of the map.

        Note:
            The method initializes a new map object if none is provided, or updates the provided map with
            new properties and dimensions. It handles geographical and navigational data, including azimuth
            and elevation coordinates. The map's properties are updated based on the node's or the provided
            attributes.
        """

        oMap = None
        if par is None:
            par = np.array([], dtype=np.float32)
        if dim is None:
            dim = np.array([], dtype=int)

        if attr is None:
            attr = self.getAttr(dpg.cfg.getGeoDescName(), reload=reload)

        if self.map is not None:
            if reset or reload:
                self.map.init_map(attr)
            dim, mapProj, par = self.map.getProperty(dim=dim, par=par, map=map)
            return self.map, par, dim, attr

        if not attr:
            return self.map, par, dim, attr

        self.map = dpg.map__define.Map()
        # TODO: task 51 --> mappa non inizializzata
        map = self.map.init_map(attr)
        if isinstance(map, dpg.map__define.Map):
            self.map = map

        _, _, dim, _, filename, format, _, _ = self.getArrayInfo()
        if not isinstance(dim, np.ndarray) and dim is not None:
            dim = np.array(dim)

        if dim is None and filename != "":
            if dpg.array.formatIsGraphic(format):
                pointer_attr = self.getSingleAttr(filename, format=format)
                if isinstance(pointer_attr, dpg.attr__define.Attr):
                    dim = pointer_attr.pointer.shape

        el_coords, eloff, elres, filename, attr = dpg.navigation.get_el_coords(
            self, attr=attr
        )
        az_coords, filename, attr = dpg.navigation.get_az_coords(self, attr=attr)
        dim = dpg.array.getBidimensionalDim(dim)

        par, origin = dpg.navigation.set_geo_par(
            attr, dim=dim, eloff=eloff, elres=elres
        )

        self.map.setProperty(
            dim=dim, par=par, origin=origin, az_coords=az_coords, el_coords=el_coords
        )
        par_tmp = self.map.checkProperty(attr, par)
        if par_tmp is not None:
            par = par_tmp

        return self.map, par, dim, attr

    def setProperty(self, to_not_save: bool = None):
        """
        Sets the 'to_not_save' property of the current node.

        This method allows setting the 'to_not_save' property for the current node, which controls whether
        the node should be saved or not.

        Args:
            to_not_save (bool, optional): The value to set for the 'to_not_save' property. If None, the property
                is not modified. Defaults to None.

        Returns:
            None: The function does not return any value.

        Note:
            This method is used to control the saving behavior of the node. When 'to_not_save' is set to True,
            the node will not be saved; otherwise, it follows the default saving behavior.
        """
        if to_not_save is not None:
            self.to_not_save = to_not_save

    def replaceAttrValues(self, name: str, tags: list, values: list) -> Attr:
        """
        Replaces specific values of an attribute of the current node or creates a new attribute if it doesn't exist.

        This method updates an existing attribute named 'name' by replacing specified tag values. If the attribute
        does not exist, it creates a new attribute with the given name, tags, and values.

        Args:
            name (str): The name of the attribute to update or create.
            tags (list): A list of tag names within the attribute whose values need to be updated.
            values (list): A list of values corresponding to the tags that will replace the current values.

        Returns:
            dpg.attr__define.Attr: The updated or newly created attribute object.

        Note:
            The method first attempts to retrieve an existing attribute with the given name. If found, it updates
            the specified tags with the new values. If the attribute does not exist, it creates a new one with
            the specified tags and values.
        """

        oAttr = self.getMyAttr(name)
        if isinstance(oAttr, dpg.attr__define.Attr):
            oAttr.replaceTags(tags, values)
            return oAttr

        attr = dpg.attr.createAttrStruct(tags, values)
        oAttr, self.addAttr(name, attr, format="txt")

        return oAttr

    def setArrayInfo(
            self,
            filename: str = "",
            type=None,
            dim: list = None,
            format: str = "",
            aux: bool = False,
            date: str = "",
            time: str = "",
            name: str = "",
    ):
        """
        Sets or updates the array information attribute of the current node.

        This method updates the array information attribute with specified details such as filename, type,
        dimensions, format, and other metadata. It handles different dimensions and formats, and adds or
        updates various tags within the attribute.

        Args:
            filename (str, optional): The name of the file associated with the array. Defaults to an empty string.
            type (optional): The type of the array. Defaults to None.
            dim (list, optional): The dimensions of the array. Can be a list of one, two, or three elements.
                Defaults to None.
            format (str, optional): The format of the array. Defaults to an empty string.
            aux (bool, optional): If True, modifies the 'datafile' tag to 'datafile.aux'. Defaults to False.
            date (str, optional): The date associated with the array. Defaults to an empty string.
            time (str, optional): The time associated with the array. Defaults to an empty string.
            name (str, optional): The name associated with the array. Defaults to an empty string.

        Returns:
            None: The function does not return any value.

        Note:
            The method constructs the array description attribute by defining appropriate tags and values based
            on the provided arguments. It handles the removal of redundant tags based on the dimensionality of
            the array. The method uses 'replaceAttrValues' to update the attribute.
        """

        if dim is None:
            dim = []

        if len(dim) == 1:
            values = [str(dim)]
            varnames = ["nlines"]
        if len(dim) == 2:
            values = [str(dim[0]), str(dim[1])]
            varnames = ["nlines", "ncols"]
        if len(dim) == 3:
            values = [str(dim[0]), str(dim[1]), str(dim[2])]
            varnames = ["nplanes", "nlines", "ncols"]

        if filename != "":
            tag = "datafile"
            if aux:
                tag += ".aux"
            varnames.append(tag)
            values.append(filename)

        if type is not None:
            varnames.append("type")
            values.append(str(type))

        if date != "":
            varnames.append("date")
            values.append(date)

        if time != "":
            varnames.append("time")
            values.append(time)

        if format != "":
            varnames.append("format")
            values.append(format)

        if name != "":
            varnames.append("name")
            values.append(name)

        oAttr = self.replaceAttrValues(dpg.cfg.getArrayDescName(), varnames, values)
        if isinstance(oAttr, dpg.attr__define.Attr):
            if len(dim) == 2:
                oAttr.removeTags("nplanes")
            if len(dim) == 1:
                oAttr.removeTags(["nlines", "nplanes"])

        return

    def setArray(
            self,
            array: np.ndarray,
            filename: str = "",
            format: str = "",
            aux: int = 0,
            no_copy: int = 1,
    ):
        """
        Sets or updates the array data for the current node, along with associated metadata.

        This method assigns an array to the current node, optionally copying it, and updates or creates
        the necessary attributes to describe the array, including filename, format, dimensions, and type.
        It also handles auxiliary file naming and format specification.

        Args:
            array (np.ndarray): The array to be set for the current node.
            filename (str, optional): The name of the file associated with the array. If empty, a name is
                generated or fetched from existing attributes. Defaults to an empty string.
            format (str, optional): The format of the array data. Defaults to 'dat' if empty.
            aux (int, optional): If nonzero, treats the file as an auxiliary file. Defaults to 0.
            no_copy (int, optional): If nonzero, the array is not copied and the original array is used.
                Defaults to 1.

        Returns:
            None: The function does not return any value.

        Note:
            The method updates the array description attribute with the array's dimensions, type, and file name.
            It handles the creation of a new filename based on the node's name if no filename is provided.
            The method also determines the array type and dimensions for the attribute update.
        """
        if array is None:
            return
        if len(array) <= 1:
            return

        if filename == "" or filename is None:
            attr = self.getSingleAttr(dpg.cfg.getArrayDescName(), only_current=True)
            if isinstance(attr, dpg.attr__define.Attr):
                if aux:
                    filename, _, _ = dpg.attr.getAttrValue(attr, "datafile.aux", "")
                else:
                    filename, _, _ = dpg.attr.getAttrValue(attr, "datafile", "")
            if filename == "" or filename is None:
                filename = self.getProperty("name")
                if aux:
                    filename += ".aux"
                else:
                    filename += ".dat"

        if format == "" or format is None:
            format = "dat"

        if no_copy:
            pointer = array
        else:
            pointer = array.copy()

        dim = pointer.shape

        dtype_attr = getattr(pointer, "dtype", None)
        dtype = dpg.io.type_py2idl(
            dtype_attr if dtype_attr is not None else type(pointer)
        )

        self.addAttr(filename, pointer=pointer, format=format)
        self.setArrayInfo(
            filename=filename, type=dtype, dim=dim, format=format, aux=aux
        )

        return

    def getParent(self):
        """
        Returns: parent node
        """
        return self.parent
