import os
import platform
import shutil
import errno
import enum
import stat
import sys

import sou_py.dpg as dpg
from sou_py.dpg.attr__define import Attr
from sou_py.dpg.log import log_message

"""
Funzioni ancora da portare
FUNCTION CheckTag 
FUNCTION GetAllAttrNames 
FUNCTION IDL_rv_translate_tag 
PRO IDL_rv_add_log 
PRO IDL_rv_close_log 
PRO IDL_rv_open_log 
PRO RenameAttr 
FUNCTION AttachNode             // UNUSED
FUNCTION FindParent             // UNUSED
FUNCTION GetBaseName            // UNUSED
FUNCTION GetLinks               // UNUSED
FUNCTION GetNodeLevel           // UNUSED
FUNCTION GetNSubLevels          // UNUSED
FUNCTION GetSonName             // UNUSED
FUNCTION GetVars                // UNUSED
FUNCTION IDL_rv_get_curr_tree   // UNUSED
FUNCTION IsExternNode           // UNUSED
FUNCTION PathToCheck            // UNUSED
FUNCTION ReplaceAttr            // UNUSED
PRO FreeNode                    // UNUSED
PRO IDL_rv_get_NLogs            // UNUSED
PRO MoveNode                    // UNUSED
"""


def updateTree(tree, only_current: bool = False):
    """
    Updates the specified tree by creating a subtree.

    This function serves as a wrapper for the `createSubTree` function, calling it with
    the provided `tree` and `only_current` parameters. It effectively updates the `tree`
    by creating a new subtree within it.

    Args:
        tree: The tree to be updated.
        only_current (bool, optional): A flag to indicate whether only the current state
            of the tree should be considered in creating the subtree. Defaults to False.

    Returns:
        None: The function does not return a value but updates the tree structure.

    Note:
        The actual implementation of the tree update logic is handled by the `createSubTree`
        function. This function serves primarily as a convenient interface.
    """
    createSubTree(tree, only_current=only_current)


def copyMemory(node, path):
    """
    Recursively copies the memory of a node and its sub-nodes to a specified path.

    This function checks if the given node is an instance of 'dpg.node__define.Node'. If so, it
    recursively traverses all sub-nodes (sons) of the given node. For each sub-node, it constructs
    a new path by appending the sub-node's 'name' property to the given path and then calls itself
    to copy the memory of the sub-node to this new path. Finally, it saves the node's data to the
    specified path.

    Args:
        node: The node whose memory is to be copied.
        path (str): The file system path where the node's memory is to be copied.

    Returns:
        None: The function does not return any value.

    Note:
        The function will return immediately and do nothing if the 'node' is not an instance of
        'dpg.node__define.Node'. It performs its operation recursively on all sub-nodes of the given node.
    """
    if not isinstance(node, dpg.node__define.Node):
        return
    sons = node.getSons()
    for nnn in sons:
        newPath = os.path.join(path, nnn.getProperty("name"))
        copyMemory(nnn, newPath)
    # endfor
    node.save(alt_path=path)
    return


def copyDir(
        fromPath: str,
        toPath: str,
        overwrite: bool = True,
        just_here: bool = False,
        to_create: bool = False,
) -> int:
    """
    Copy contents from one directory to another.

    Args:
        fromPath (str): The source directory path.
        toPath (str): The destination directory path.
        overwrite (bool): If True, overwrite existing files in the destination. Defaults to True.
        just_here (bool): If True, copy only the contents of the directory, not subdirectories.

    Returns:
        int: The error count during file copy operations.
    """
    error = 0
    if os.path.exists(toPath):
        basefolder = os.path.basename(fromPath)
        toPath = os.path.join(toPath, basefolder)
    try:
        shutil.copytree(fromPath, toPath, dirs_exist_ok=overwrite)
    except Exception as e:
        print(f"Error copying directory {fromPath} to {toPath}: {e}")
        error += 1

    # if just_here:
    #     if os.path.isdir(toPath):
    #         log_message("JUST HERE NOT IMPLEMENTED", level='ERROR', all_logs=True)
    #         sys.exit()
    #         if not overwrite:
    #             toPath = os.path.dirname(toPath)
    #             overwrite = True

    return error


def copyAllFiles(
        fromPath: str,
        toPath: str,
        overwrite: bool = False,
        mode: int = None,
        no_dat: bool = False,
) -> int:
    """
    Copy all files from one directory to another.

    Args:
        from_path (str): The source directory path.
        to_path (str): The destination directory path.
        overwrite (bool): If True, overwrite existing files in the destination. Defaults to False.
        mode (int, optional): The mode (permissions) to set for the copied files, specified as an octal number.
        no_dat (bool): If True, do not copy .dat files. Defaults to False.

    Returns:
        int: The error count during file copy operations.
    """
    error = 0

    path = dpg.path.checkPathname(fromPath)

    if not os.path.isdir(path):
        print("LogMessage", "Cannot Find " + path)
        return error + 1
    if not os.path.isdir(toPath):
        try:
            os.makedirs(toPath)
        except OSError as e:
            if e.errno != errno.EEXIST:
                print(f"Could not create directory {toPath}")
                return error
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    for file_path in files:
        filePathLower = file_path.lower()
        if no_dat and filePathLower.endswith(".dat"):
            continue
        if os.path.isdir(file_path):
            continue
        if not overwrite and os.path.exists(os.path.join(toPath, file_path)):
            continue
        try:
            shutil.copy2(os.path.join(path, file_path), toPath)
            # endfor
            if mode:
                changeFilesMode(toPath, mode)
        except Exception as e:
            error += 1
            print(f"Error copying {file_path} to {toPath}: {e}")
    return error


def copyNode(
        source,
        dest=None,
        only_current: bool = False,
        overwrite: bool = False,
        toPath: str = "",
        from_memory: bool = False,
        add: bool = False,
        mode=None,
):
    """
    Copies a node to a specified destination, with options for various copy modes and behaviors.

    This function performs a copy operation of a node (`source`) to a destination. The destination can be
    another node or a path specified in `toPath`. The function handles different scenarios based on the
    provided arguments, such as copying only current files, copying from memory, adding the source node
    under the destination path, and changing file modes after copying.

    Args:
        source: The source node to be copied.
        dest (optional): The destination node where the source will be copied. Defaults to None.
        only_current (bool, optional): If True, only current files are copied. Defaults to False.
        overwrite (bool, optional): If True, allows overwriting of existing files. Defaults to False.
        toPath (str, optional): The file system path to which the node should be copied. Defaults to ''.
        from_memory (bool, optional): If True, copies the node from memory. Defaults to False.
        add (bool, optional): If True, adds the source node under the destination path. Defaults to False.
        mode (optional): The mode to be applied to the files after copying. Defaults to None.

    Returns:
        int: 0 if the copy operation is successful, -1 if it fails due to invalid source/destination types
        or if `toPath` is empty when `dest` is not a node.

    Note:
        The function conducts various checks and operations based on the arguments provided, and it may
        return early with an error code (-1) if certain conditions are not met. It also delegates specific
        copy tasks to other functions like `copyAllFiles`, `copyMemory`, and `copyDir` based on the
        arguments.
    """

    err = 0
    if not isinstance(source, dpg.node__define.Node) or (
            (not isinstance(dest, dpg.node__define.Node)) and (toPath == "")
    ):
        err = -1
        return err
    fromPath = source.getProperty("path")
    if toPath == "":
        toPath = dest.getProperty("path")
    if only_current:
        copyAllFiles(fromPath, toPath, overwrite=overwrite)
        return
    # endif
    if from_memory:
        if add:
            toPath = os.path.join(toPath, source.getProperty("name"))
        copyMemory(source, toPath)
    else:
        if toPath.find(fromPath) == 0:
            toPath, err = copySonsToNode(
                source, dest, and_files=True, overwrite=overwrite, to_path=toPath
            )
        else:
            if add:
                toPath = os.path.join(toPath, dest.getProperty("name"))
            # in questo punto viene crata la cartella LAURO per il sampling come copia e incolla..
            # e poi dove viene modificato il parameters.txt?
            err = copyDir(fromPath, toPath, overwrite=overwrite, just_here=True)
        # endelse
    # endelse
    if err != 0:
        return

    if mode:
        changeFilesMode(toPath, mode)
    updateTree(dest)
    return


def replaceAttrValues(
        node,
        name: str,
        varnames: list,
        values: list,
        to_owner: bool = False,
        to_add: bool = False,
        rem_inherits: bool = False,
        only_current: bool = False,
        to_save: bool = False,
) -> Attr:
    """
    Replaces attribute values of a node with new values, and optionally saves the changes.

    This function replaces the values of specified variables in an attribute of a node. If the
    attribute does not exist or if the node is not an instance of 'dpg.node__define.Node', a new
    attribute is created. The function allows for the addition of new variables, removal of inherited
    variables, and the option to save the attribute after modification.

    Args:
        node: The node whose attribute values are to be replaced.
        name (str): The name of the attribute to be modified.
        varnames (list): A list of variable names whose values are to be replaced.
        values (list): A list of new values corresponding to the variable names in 'varnames'.
        to_owner (bool, optional): If True, ensures that the attribute is owned by the node.
            Defaults to False.
        to_add (bool, optional): If True, adds new variables to the attribute. Defaults to False.
        rem_inherits (bool, optional): If True, removes inherited variables from the attribute.
            Defaults to False.
        only_current (bool, optional): If True, only affects the current state of the attribute.
            Defaults to False.
        to_save (bool, optional): If True, saves the attribute after modifications. Defaults to False.

    Returns:
        dpg.attr__define.Attr: The modified attribute object.

    Note:
        The function handles various conditions like non-existence of the attribute and type checking of
        the node and attribute. It also deals with ownership transfer of the attribute if required.
    """

    # TODO controllare se il paramentro ind va tenuto
    attr = getSingleAttr(node, name, only_current=only_current)
    pointer = dpg.attr.createAttrStruct(varnames, values)

    if not isinstance(attr, dpg.attr__define.Attr) and isinstance(
            node, dpg.node__define.Node
    ):
        attr = node.addAttr(name, pointer=pointer)
        if (to_save) and isinstance(attr, dpg.attr__define.Attr):
            attr.save()
        return attr

    if (not to_owner) and (attr.owner != node):
        _, _ = copyAttr(attr.owner, node, name)
        attr = getSingleAttr(node, name)
        owner = node
    # endif

    # il problema dell'aggiunta del "format = dat" è che in questo punto viene passato "varnames"
    # che tra le altre cose ha la parola "format"
    dpg.attr.replaceTags(
        attr, varnames, values, to_add=to_add, rem_inherits=rem_inherits
    )
    if to_save and isinstance(attr, dpg.attr__define.Attr):
        attr.save()
    return attr


def createNode(parent=None, path: str = "", shared: bool = False):
    """
    Creates and returns a new node with specified parameters.

    This function initializes a new node of type 'dpg.node__define.Node'. The new node can be optionally
    set to have a parent, a specific file system path, and a shared status.

    Args:
        parent (optional): The parent of the new node. Defaults to None.
        path (str, optional): The file system path for the new node. Defaults to an empty string.
        shared (bool, optional): Indicates whether the node is shared. Defaults to False.

    Returns:
        dpg.node__define.Node: The newly created node.

    Note:
        The function directly creates a new node using the 'dpg.node__define.Node' class constructor
        with the provided arguments.
    """
    node = dpg.node__define.Node(path=path, parent=parent, shared=shared)
    return node


def createAttr(node, name: str, varnames: list, values: list, format: str = "") -> Attr:
    """
    Creates an attribute for a given node with specified parameters.

    This function creates an attribute for a node, provided the node is an instance of
    'dpg.node__define.Node' and the attribute name is valid (not None or empty). It initializes
    the attribute with given variable names and their corresponding values. The format for the
    attribute can also be specified.

    Args:
        node: The node to which the attribute will be added.
        name (str): The name of the attribute to be created.
        varnames (list): A list of variable names for the attribute.
        values (list): A list of values corresponding to the variable names in 'varnames'.
        format (str, optional): The format of the attribute. Defaults to an empty string.

    Returns:
        pData: The attribute structure if the attribute is successfully created, None otherwise.

    Note:
        The function returns None if the node is not an instance of 'dpg.node__define.Node' or if the
        attribute name is invalid. The attribute structure is created using 'dpg.attr.createAttrStruct'
        and the attribute is added to the node using 'addAttr'.
    """
    if not isinstance(node, dpg.node__define.Node):
        return None
    if name is None or name == "":
        return None

    pData = dpg.attr.createAttrStruct(varnames, values)
    oAttr = addAttr(node, name, pData=pData, format="TXT")
    if not isinstance(oAttr, dpg.attr__define.Attr):
        pData = None

    return oAttr


def createRoot(path: str = "", shared: bool = False):
    """
    Creates and returns a root node with specified parameters.

    This function serves as a convenience wrapper for creating a root node. It calls the
    'createNode' function with the provided path and shared status, thereby initializing
    a new root node.

    Args:
        path (str, optional): The file system path for the root node. Defaults to an empty string.
        shared (bool, optional): Indicates whether the root node is shared. Defaults to False.

    Returns:
        dpg.node__define.Node: The newly created root node.

    Note:
        The function delegates the creation of the node to the 'createNode' function, specifying
        that the new node is a root node by not assigning a parent.
    """
    return createNode(path=path, shared=shared)


def createSubTree(node, only_current: bool = False):
    """
    Creates a subtree from a given node.

    This function is responsible for creating a subtree from the specified node. The subtree is
    created based on the current state of the node if 'only_current' is True. The function performs
    the subtree creation only if the provided 'node' is an instance of 'dpg.node__define.Node'.

    Args:
        node: The node from which the subtree is to be created.
        only_current (bool, optional): If True, the subtree is created based on the current state
            of the node. Defaults to False.

    Returns:
        None: The function does not return any value.

    Note:
        The function will not perform any action and will return immediately if the provided 'node'
        is not an instance of 'dpg.node__define.Node'.
    """
    if isinstance(node, dpg.node__define.Node):
        node.createSubTree(only_current=only_current)
    else:
        return
    return


def findSharedNode(path: str):
    """
    Searches for and returns a shared node matching a specified path.

    This function looks through all shared nodes to find a node with a path matching the provided 'path'.
    The search is conducted in two stages: first, it checks the paths of top-level shared nodes; if no
    match is found, it then checks the paths of all descendant nodes within each shared node. The function
    ensures that only instances of 'dpg.node__define.Node' are considered in the search.

    Args:
        path (str): The file system path to search for among shared nodes.

    Returns:
        dpg.node__define.Node or None: The node matching the provided path, if found; otherwise, None.

    Note:
        The function returns None if no matching node is found. It uses 'dpg.path.checkPathname' to
        standardize the path format before comparison.
    """
    sPath = dpg.path.checkPathname(path, with_separator=False)

    for ttt in dpg.globalVar.GlobalState.SHARED_TREES:
        if isinstance(ttt, dpg.node__define.Node):
            currPath = ttt.getProperty("path")
            if os.path.samefile(sPath, currPath):
                return ttt
            # endif
    # endfor

    for ttt in dpg.globalVar.GlobalState.SHARED_TREES:
        if isinstance(ttt, dpg.node__define.Node):
            nodes = ttt.getAllDescendant()
            for node in nodes:
                if isinstance(node, dpg.node__define.Node):
                    currPath = node.getProperty("path")
                    if sPath == currPath:
                        return node
                # endif
            # endfor
        # endif
    # endfor

    return None


def createTree(path, shared: bool = False, only_root: bool = False):
    """
    Creates a tree structure starting from a root node at a specified path.

    This function creates a tree with a root node at the given path. If the 'shared' flag is True,
    it first attempts to find an existing shared node with the same path. If such a node is found,
    it is returned as the root of the tree. Otherwise, a new root node is created. The function also
    has the option to create only the root node without its subtree if 'only_root' is set to True.

    Args:
        path (str): The file system path where the root node of the tree will be created.
        shared (bool, optional): If True, searches for an existing shared node with the same path before
            creating a new node. Defaults to False.
        only_root (bool, optional): If True, only the root node is created without creating its subtree.
            Defaults to False.

    Returns:
        dpg.node__define.Node: The root node of the newly created tree.

    Note:
        If a shared node with the same path is found, it is returned without creating a new tree.
        Otherwise, a new root node is created and, unless 'only_root' is True, a subtree is also created
        under this root.
    """
    if shared:
        node = findSharedNode(path)
        if isinstance(node, dpg.node__define.Node):
            return node
    # endif
    root = createRoot(path, shared=shared)  # root è un oggetto Node
    createSubTree(root, only_current=only_root)
    return root


def getParents(node):
    """
    Retrieves a list of parent nodes for a given node.

    This function constructs a list of all parent nodes of the specified node, traversing up the
    hierarchy until no further parent nodes are found. The function only operates if the provided
    'node' is an instance of 'dpg.node__define.Node'.

    Args:
        node: The node for which parent nodes are to be found.

    Returns:
        list of dpg.node__define.Node or None: A list containing all parent nodes of the given node,
        in order from the closest parent to the furthest. Returns None if the provided node is not
        an instance of 'dpg.node__define.Node'.

    Note:
        The function will return None immediately if 'node' is not an instance of 'dpg.node__define.Node'.
        The parent nodes are added to the list in the order they are found, starting from the immediate
        parent and moving up the hierarchy.
    """
    parents = []
    if not isinstance(node, dpg.node__define.Node):
        return None
    next = node.parent
    while isinstance(next, dpg.node__define.Node):
        parents.append(next)
        next = next.parent
    # endwhile
    return parents


def findAttr(
        node,
        name: str,
        attrSet: list = None,
        where_found: list = None,
        all: bool = False,
        down: bool = False,
        lower: bool = False,
        to_not_load: bool = False,
):
    """
    Searches for and retrieves attributes with a specified name from a node and its relatives.

    This function finds attributes named 'name' in a given node, its descendants, and optionally its
    ancestors. The search behavior can be tailored with various flags: searching in all descendants,
    only in leaf nodes, in lower nodes, and whether to include parent nodes. The function also allows
    for the option to not load the attributes during the search.

    Args:
        node: The node from which the search begins.
        name (str): The name of the attribute to search for.
        attrSet (list, optional): A list to store found attributes. Defaults to an empty list if False.
        where_found (list, optional): A list to store the nodes where attributes are found. Defaults to an
            empty list if False.
        all (bool, optional): If True, searches all descendants. Defaults to False.
        down (bool, optional): If True, includes parent nodes in the search. Defaults to False.
        lower (bool, optional): If True, searches in lower nodes. Defaults to False.
        to_not_load (bool, optional): If True, does not load the attributes. Defaults to False.

    Returns:
        tuple: A tuple containing two elements:
            1. attrSet (list): A list of found attributes.
            2. where_found (list): A list of nodes where the attributes were found.

    Note:
        The function's behavior changes based on the combination of 'all', 'down', and 'lower' flags.
        If 'all' is False, the search stops at the first found attribute. If 'lower' is True, it forces
        'all' to True and 'only_leaves' to False. The function returns a tuple of two lists: one with
        found attributes and another with the nodes where these attributes were found.
    """
    parents = []

    if all:
        only_leaves = False
        only_current = True
        if not down:
            parents = getParents(node)
    else:
        only_leaves = True
        only_current = False
    # endif
    if lower:
        only_leaves = True
        only_current = False
        all = True
        if down:
            parents = getParents(node)
    # endif

    if isinstance(node, dpg.node__define.Node):
        nodes = node.getAllDescendant(only_leaves=only_leaves)
    else:
        nodes = []

    if len(nodes) > 0 and node not in nodes:
        nodes.append(node)
    else:
        nodes = [node]

    if isinstance(parents, list) and len(parents) > 0:
        nodes.extend(parents)

    if not isinstance(attrSet, list):
        attrSet = []
    if not isinstance(where_found, list):
        where_found = []
    for nnode in nodes:
        if isinstance(nnode, dpg.node__define.Node):
            attr = nnode.getAttr(
                name, only_current=only_current, lower=lower, to_not_load=to_not_load
            )
        else:
            attr = None
        if attr is not None and isinstance(where_found, list):
            if isinstance(attr, list):
                # attr è una lista
                attrSet.extend(attr)
                owners = [x.owner for x in attr]
                where_found.extend(owners)
            else:
                # attr non è una lista
                attrSet.append(attr)
                where_found.append(attr.owner)

            if not all:
                return attrSet  # TODO siamo sicuri?
    # endfor
    return attrSet, where_found


def removeNode(
        node, directory: bool = False, from_root: bool = False, shared: bool = False
):
    """
    Remove a node from a tree structure, optionally removing the corresponding directory from the filesystem.

    Args:
        node (dpg.node__define.Node): the node to be removed
        directory (bool): flag indicating if the directory associated with the node should also be removed
        from_root (bool): flag indicating if the removal should be from the root of the tree (not used in this
        implementation)
        shared (bool): flag indicating if the node is shared (not used in this implementation)

    Returns:
        None
    """

    if isinstance(node, dpg.node__define.Node):
        if directory:
            log_message(f"Remove Node {node.path}")
            if os.path.isdir(node.path):
                shutil.rmtree(node.path)
        if node.parent:
            node.parent.lst.remove(node)
        dpg.globalVar.GlobalState.SHARED_TREES = [shr_tree for shr_tree in dpg.globalVar.GlobalState.SHARED_TREES if
                                                  shr_tree != node]


def removeNodes(nodes, directory: bool = False):
    """
    Removes a list of nodes, with an option to remove their directories.

    This function iterates over a list of nodes and removes each one using the 'removeNode' function.
    It offers the option to also remove the directory associated with each node.

    Args:
        nodes (list): A list of nodes to be removed.
        directory (bool, optional): If True, the directories associated with the nodes are also
            removed. Defaults to False.

    Returns:
        None: The function does not return any value.

    Note:
        The actual removal of each node is handled by the 'removeNode' function, which this function
        calls for each node in the provided list.
    """
    for node in nodes:
        removeNode(node, directory)
    return


def getNodePath(node, sep: str = "") -> str:
    """
    Retrieves the file system path of a given node, with an option to modify the path separator.

    This function returns the file system path of a node. If the 'sep' parameter is specified, it replaces
    the default path separator in the node's path with the provided separator.

    Args:
        node: The node whose path is to be retrieved.
        sep (str, optional): The desired path separator to be used in the returned path.
            If False, the default separator is used. Defaults to False.

    Returns:
        str: The file system path of the node. Returns an empty string if the node is not an instance
        of 'dpg.node__define.Node'.

    Note:
        The function checks if the provided 'node' is an instance of 'dpg.node__define.Node'. If not,
        it returns an empty string. The path's separator is replaced only if 'sep' is not empty string and
        differs from the current path separator.
    """
    if not isinstance(node, dpg.node__define.Node):
        return ""
    path = node.getProperty("path")

    if sep is None or not isinstance(sep, str) or sep == "":
        return path

    curr_sep = os.path.sep
    if curr_sep != sep:
        path = path.replace(curr_sep, sep)

    return path


def removeTree(tree: list, directory: bool = False, shared: bool = False):
    """
    Removes an entire tree structure, including optionally the directories and shared nodes.

    This function iterates over all nodes in the provided 'tree' and removes each one using the
    'removeNode' function. It allows for the option to also remove the directories associated with
    each node and to specify if the nodes are shared.

    Args:
        tree (list): A list representing the tree structure, where each element is a node to be removed.
        directory (bool, optional): If True, the directories associated with the nodes are also removed.
            Defaults to False.
        shared (bool, optional): If True, indicates that the nodes are shared and should be treated as such
            during removal. Defaults to False.

    Returns:
        None: The function does not return any value.

    Note:
        The removal of each node is handled by the 'removeNode' function, which this function calls for
        each node in the provided tree. The 'from_root' parameter in 'removeNode' is set to True, indicating
        that the removal process should start from the root of the tree.
    """
    if tree is not None:
        # Comportamento senza reversed
        #  [node1, node2, node3, ...] -> removeNode(node1) -> [node2, node3, ..]
        # in questo modo avanza poi alla seconda posizione ma l'elemento della seconda posizione si è spostato in prima
        if not isinstance(tree, list):
            tree = [tree]
        for node in reversed(tree):
            removeNode(node, directory=directory, from_root=True, shared=shared)
    return


def translateTextTag(
        node, texts, strings, only_current=False, exact=False, alt_node=False
):
    """
    TBD
    """
    # TODO
    pass


def findSon(node, name: str):
    """
    Finds and returns the first descendant node with a specified name from a given node.

    This function searches for a descendant node named 'name' within the specified 'node'. It
    uses the 'findAllDescendant' function to locate all descendant nodes with the given name.
    The function then returns the first node from the list of found nodes. If 'findAllDescendant'
    returns a single node instead of a list, that node is returned directly.

    Args:
        node: The node from which to start the search for the descendant.
        name (str): The name of the descendant node to find.

    Returns:
        dpg.node__define.Node or None: The first descendant node with the specified name, or None
        if no such node is found.

    Note:
        The function returns None if 'findAllDescendant' does not find any nodes with the specified name.
        If 'findAllDescendant' returns a single node (not a list), that node is returned as the result.
    """
    nodes = findAllDescendant(node, name)
    if not nodes:
        return None
    elif isinstance(nodes, dpg.node__define.Node):
        return nodes
    elif isinstance(nodes, list):
        return nodes[0]
    else:
        # Condizione non raggiungibile (in teoria)
        return None


def findAllDescendant(node, name: str = ""):
    """
    Finds and returns all descendant nodes of a given node that match a specified name or criteria.

    This function retrieves all descendants of the specified 'node' and filters them based on the 'name'
    parameter. The 'name' parameter can be a string or a list of strings, where each string represents
    a criterion for selecting descendant nodes. Special criteria such as '.', '..', and '<current>' are
    handled uniquely. The function returns a list of nodes that match the given criteria.

    Args:
        node: The node from which the descendant search will begin.
        name (str or list, optional): The name(s) or special criteria to filter descendant nodes.
            Defaults to an empty string which implies no filtering.

    Returns:
        list of dpg.node__define.Node or None: A list of descendant nodes that match the specified criteria.
        Returns None if no matching nodes are found or if the provided 'node' is not an instance of
        'dpg.node__define.Node'.

    Note:
        Special name criteria include:
        - '.' or '<current>': Refers to the node itself.
        - '<all>': Returns all direct children of the node.
        - '..': Refers to the parent of the node.
        If 'name' is a list, the function searches for each name in the list and returns a list of nodes
        where each element corresponds to the respective name in the 'name' list.
    """
    if not isinstance(node, dpg.node__define.Node):
        return None

    desc = node.getAllDescendant(and_me=True)
    if len is None or len(desc) == 0:
        return None

    nodes = []
    toFind = 0

    if isinstance(name, list):
        toFind = len(name)
    if isinstance(name, str) and name != "":
        toFind = 1

    if toFind == 0:
        return None
    if toFind == 1:
        if name == "." or name == "<current>":
            nodes.append(node)
            return nodes
        if name == "<all>":
            return node.getSons()
        if name == "..":
            nodes.append(node.parent)
            return nodes
        ind, count, byName = compareNodeName(desc, name, parent=node)
        if count > 0:
            if byName == 0:
                ind = ind[0]
                count = 1
            nodes = [desc[i] for i in ind]
            if len(nodes) == 1:
                nodes = nodes[0]
        else:
            log_message(f"Cannot find node {name} @ node {node.path}")
        return nodes

    nodes = [None] * toFind
    for nnn in range(nodes):
        if name[nnn] == ".":
            nodes[nnn] = node
        elif name[nnn] == "..":
            nodes[nnn] = node.parent
        elif name[nnn] == "<current>":
            nodes[nnn] = node
        else:
            ind, count, _ = compareNodeName(desc, name[nnn])
            if count > 0:
                nodes[nnn] = desc[ind[0]]
            else:
                """LOG"""
                # LogMessage, 'Cannot Find Node ' + name[nnn]
        # endif
    # endfor

    return nodes


def getSons(node, ind: bool = False):
    """
    Retrieves the child nodes (sons) of a given node.

    This function returns the child nodes of the specified 'node'. The function only operates if the
    provided 'node' is an instance of 'dpg.node__define.Node'.

    Args:
        node: The node from which child nodes are to be retrieved.
        ind (bool, optional): Defaults to False.

    Returns:
        list of dpg.node__define.Node or None: A list containing the child nodes of the given node, or
        None if the provided node is not an instance of 'dpg.node__define.Node'.

    Note:
        The function will return None immediately if 'node' is not an instance of 'dpg.node__define.Node'.
        The 'ind' parameter is present but not utilized in the current implementation of the function.
    """
    if not isinstance(node, dpg.node__define.Node):
        return None
    return node.getSons()


def compareNodeName(nodes, name: str, parent=None):
    """
    Compares a given name with the names of a list of nodes and identifies matching nodes.

    This function checks if the provided 'name' matches the name of any node in the 'nodes' list.
    It considers different scenarios: the name being a path, a relative path, or just a node name.
    The function returns indices of nodes that match the name, the count of such nodes, and a flag
    indicating whether the comparison was by name or path.

    Args:
        nodes (list): A list of nodes to compare the name with.
        name (str): The name or path to be compared against the nodes' names.
        parent (optional): The parent node, used when calculating relative paths. Defaults to None.

    Returns:
        tuple:
            - ind (int or list of int): The index (or indices) of nodes in the 'nodes' list that match the 'name'.
            - count (int): The number of nodes that matched the 'name'.
            - byName (int): A flag indicating if the comparison was by name (1) or path (0).

    Note:
        The function handles different name formats and converts all comparisons to uppercase for
        case-insensitivity. It also handles scenarios where the 'name' is a path or a relative path,
        adjusting the comparison logic accordingly.
    """
    """
    OLD INFO
    The provided IDL code is trying to determine the appropriate separator for file paths (sep) 
    by examining realName, and then it sets up a string array names based on the number of nodes nNodes.
    The code first attempts to find a forward slash / in realName. 
    If not found, it checks for a backslash \ at position 1. 
    If found, it sets sep to backslash and pos to 0. 
    If not found again, it sets sep to backslash and searches for it. 
    If no separator is found, byName is set to 1, and names is created as an empty string array with length nNodes.
    """
    byName = 0
    ind = -1
    realName = dpg.path.checkPathname(name)
    if len(nodes) == 0:
        count = 0
        return ind, count, byName
    # TODO forse è il caso di mettere un controllo di tipo su realName
    sep = "/"
    pos = realName.find(sep)

    if pos < 0:
        pos = realName.find("\\")
        if pos == 1:
            pos = 0
            sep = "\\"

    if pos < 0:
        sep = "\\"
        pos = realName.find(sep)

    if pos < 0:
        byName = 1

    names = [""] * len(nodes)

    for nnn, node in enumerate(nodes):
        if pos == 0:
            currName = getNodePath(node, sep=sep)
        elif pos > 0:
            currName = getRelativePath(node, parent=parent, sep=sep)
        else:
            currName = getNodeName(node)  # TODO errore, vedi implementazione funzione.
        names[nnn] = currName

    currName = realName
    if pos >= 0:
        currName = checkLastSep(currName, sep)
    ind = [i for i, name in enumerate(names) if name == currName]

    if not ind:
        names.append(".")
        ind = [i for i, name in enumerate(names) if name == currName]

    return ind, len(ind), byName


def getRelativePath(node, parent, sep="") -> str:
    """
    Computes and returns the relative path of a node with respect to a specified parent node.

    This function calculates the relative path from the 'parent' node to the 'node'. It determines the
    full path of both 'node' and 'parent', and then computes the relative path of 'node' with respect
    to 'parent'. If the 'parent' is not specified or is not an instance of 'dpg.node__define.Node',
    the function calculates the relative path from the root node of 'node'.

    Args:
        node: The node for which the relative path is to be computed.
        parent: The parent node with respect to which the relative path is calculated.
        sep (str, optional): The path separator to use in the relative path. If False,
            uses the default path separator. Defaults to False.

    Returns:
        str: The relative path of 'node' with respect to 'parent'.

    Note:
        If 'node' does not lie within the subtree of 'parent', the function constructs a relative path
        from the nearest ancestor of 'node' that is in the same subtree as 'parent'. This relative path
        includes the names of all intermediate nodes up to 'node', separated by the specified or default
        path separator.
    """
    currPath = getNodePath(node, sep=sep)
    if isinstance(parent, dpg.node__define.Node):
        parentPath = getNodePath(parent, sep=sep)
    else:
        root_node, _ = getRoot(node)
        parentPath = getNodePath(root_node, sep=sep)
    pos = currPath.find(parentPath)
    if pos != 0:
        sep = os.path.sep
        spare = getNodeName(node)
        ppp = node.parent
        rrr, _ = getRoot(node)
        while isinstance(node, dpg.node__define.Node) and ppp != rrr:
            spare = os.path.join(getNodeName(ppp), spare)
            ppp = ppp.parent
        return spare

    spare = currPath[len(parentPath):]
    if spare == "":
        spare = "."
    return spare


def getNodeName(node) -> str:
    """
    Retrieves the name property of a given node.

    This function returns the 'name' property of the specified 'node', provided the 'node' is an
    instance of 'dpg.node__define.Node'. If the 'node' is not an instance of this class, the function
    returns an empty string.

    Args:
        node: The node from which to retrieve the name property.

    Returns:
        str: The name property of the node if it is an instance of 'dpg.node__define.Node'; otherwise,
        an empty string.

    Note:
        The function relies on the 'getProperty' method of the node to retrieve the 'name' property.
    """
    if isinstance(node, dpg.node__define.Node):
        return node.getProperty("name")
    else:
        return ""


def checkLastSep(path: str, sep: str) -> str:
    """
    Ensures that a given path string ends with a specific separator.

    This function checks if the provided 'path' string ends with the given 'sep' character. If the path
    does not end with 'sep', the function appends 'sep' to the end of the path. If 'path' already ends
    with 'sep', it is returned unchanged.

    Args:
        path (str): The file system path to be checked.
        sep (str): The separator character to ensure at the end of the path.

    Returns:
        str: The modified path string, guaranteed to end with the specified separator.

    Note:
        The function uses 'rfind' to locate the last occurrence of the separator in the path. If this
        occurrence is not at the end of the string, the separator is appended.
    """
    pos = path.rfind(sep)
    if pos == len(path) - 1:
        return path
    return path + sep


# La funzione restituisce:
# - la root del nodo
# - il livello del nodo input
def getRoot(node):
    """
    Retrieves the root node and level of a given node.

    This function returns the root node of the specified 'node', along with the level of the given 'node'
    within the tree. If the 'level' parameter is True, it also returns the depth level of 'node' from the
    root. If 'node' is not an instance of 'dpg.node__define.Node', the function returns None for the root
    node and sets the level to -1.

    Args:
        node: The node for which the root node and level are to be determined.

    Returns:
        tuple:
            - The root node of the given 'node'.
            - The depth level of 'node' from the root if 'level' is True; otherwise, False.

    Note:
        The function relies on the 'getRoot' method of the node to retrieve the root node and level.
        It returns (None, -1) if 'node' is not an instance of 'dpg.node__define.Node'.
    """
    if not isinstance(node, dpg.node__define.Node):
        level = -1
        return None, level

    return node.getRoot()


def addAttr(
        node,
        name: str,
        pData,
        format=None,
        file_date=None,
        str_format: str = None,
        to_save: bool = None,
):
    """
    Adds an attribute to a specified node and optionally saves it.

    This function adds an attribute to the given 'node' using the provided data ('pData') and additional
    optional parameters like 'format', 'file_date', and 'str_format'. If 'to_save' is True, the newly
    created attribute is also saved.

    Args:
        node: The node to which the attribute will be added.
        name (str): The name of the attribute to be added.
        pData: The data pointer for the attribute.
        format (optional): The format of the attribute. Defaults to None.
        file_date (optional): The date associated with the attribute. Defaults to None.
        str_format (optional): The string format for the attribute. Defaults to None.
        to_save (bool, optional): If True, saves the attribute after adding it. Defaults to None.

    Returns:
        dpg.attr__define.Attr or None: The attribute object that was added to the node, or None
        if the node is not an instance of 'dpg.node__define.Node'.

    Note:
        The attribute is only saved if 'to_save' is True and the created attribute is an instance
        of 'dpg.attr__define.Attr'. The function returns None if 'node' is not an instance of
        'dpg.node__define.Node'.
    """
    if not isinstance(node, dpg.node__define.Node):
        return None
    oAttr = node.addAttr(
        name, pointer=pData, format=format, file_date=file_date, str_format=str_format
    )
    if isinstance(oAttr, dpg.attr__define.Attr) and (to_save):
        oAttr.save()
    return oAttr


def addNode(
        parent,
        name: str,
        remove_if_exists: bool = None,
        path=None,
        to_not_save: bool = False,
):
    """
    Adds a new node to a specified parent node, with various configuration options.

    This function creates a new node with the specified 'name' under the 'parent' node. It offers additional
    options such as removing an existing node with the same name, not saving the node immediately, and
    returning the existence status of the node.

    Args:
        parent: The parent node to which the new node will be added.
        name (str): The name of the new node to be added.
        remove_if_exists (bool): If True, any existing node with the same name will be removed before adding the new
        node. Defaults to None.
        path (Path): The path for the new node.
        to_not_save (bool): If True, the node is not saved immediately after being added. Defaults to None.
        get_exists (bool): If True, the function also returns a flag indicating whether the node already existed.
        Defaults to False.

    Returns:
        dpg.node__define.Node or (dpg.node__define.Node, bool): The new node that was added. If 'get_exists'
        is True, a tuple is returned where the first element is the new node and the second element is a
        boolean flag indicating whether the node already existed.

    Note:
        The function checks if 'parent' is an instance of 'dpg.node__define.Node'. It returns None if this
        is not the case. The 'path' parameter is present but not currently utilized in the function.
    """

    if not isinstance(parent, dpg.node__define.Node):
        # if parent is None:
        #     log_message(f"Can't add node parent! It is None.", 'ERROR')

        return None, None

    node, exists = parent.addNode(
        name,
        remove_if_exists=remove_if_exists,
        to_not_save=to_not_save,
    )
    return node, exists


"""
; NAME:
; FindNode
;
; :Description:
;    Torna un nodo apprtenente all'albero di un nodo qualsiasi dato un path.
;
; :Params:
;    node
;    path
;
; :Keywords:
;    COUNT
;
"""


def findNode(node, path: str, count: int = 0):
    """
    Finds and returns a specific descendant node based on a given path, starting from a specified node.

    This function searches for a descendant node that matches the provided 'path', starting from the
    specified 'node'. It first retrieves the root of the given 'node' and then uses the 'findAllDescendant'
    function to find all descendants matching the 'path'. If exactly one matching node is found, it is
    returned; otherwise, None is returned.

    Args:
        node (node__define.Node): The node from which to start the search.
        path (str): The path used to identify the target node.
        count (integer): Defaults to 0.

    Returns:
        dpg.node__define.Node: The node matching the specified path if exactly one match is found; otherwise, None.

    Note:
        The function only returns a node if there is exactly one match for the provided path. The 'count'
        parameter is present but not utilized in the current implementation of the function.
    """

    if not isinstance(node, dpg.node__define.Node):
        return None
    root = node.getRoot()
    nodes = findAllDescendant(root, path)
    if nodes is None or len(nodes) != 1:
        return None
    else:
        return nodes[0]


def saveNode(node, only_current: bool = False, alt_path: str = ""):
    """
    Saves the state of a given node, with options for current state and alternative path.

    This function saves the specified 'node' to its associated file. It provides options to save only the
    current state of the node and to specify an alternative path for saving the node's file.

    Args:
        node: The node to be saved.
        only_current (bool, optional): If True, only the current state of the node is saved.
            Defaults to False.
        alt_path (str, optional): An alternative file path where the node's file will be saved.
            If empty, the default path associated with the node is used. Defaults to an empty string.

    Returns:
        None: The function does not return any value.

    Note:
        The function performs the save operation only if 'node' is an instance of 'dpg.node__define.Node'.
        If 'node' is not of this type, the function does nothing and returns.
    """
    if not isinstance(node, dpg.node__define.Node):
        return
    node.save(only_current=only_current, alt_path=alt_path)
    return


def removeAttrValues(
        node,
        name: str,
        varnames: list,
        to_owner: bool = False,
        owner=None,
        format: str = "txt",
        to_save: bool = False,
) -> bool:
    """
    Removes specified values from an attribute of a node and optionally saves the changes.

    This function removes values corresponding to 'varnames' from an attribute named 'name' of the given
    'node'. It provides options to handle attribute ownership and to save the attribute after the removal.
    The function operates only if the attribute exists and is in the specified 'format' (default is 'txt').

    Args:
        node: The node from which attribute values are to be removed.
        name (str): The name of the attribute whose values are to be removed.
        varnames (list): A list of variable names to be removed from the attribute.
        to_owner (bool, optional): If True, ensures that the operation is performed on the attribute owned
            by the node. Defaults to False.
        owner (optional): The actual owner of the attribute, if different from 'node'. Defaults to None.
        format (str, optional): The format of the attribute. Defaults to 'txt'.
        to_save (bool, optional): If True, saves the attribute after removing the values. Defaults to False.

    Returns:
        bool: True if the removal was successful, False otherwise.

    Note:
        The function first checks for the existence of the attribute and its format. If the 'to_owner' flag
        is False and an 'owner' is specified, the attribute is first copied from the 'owner' to the 'node'
        before the removal operation. The function returns False if the attribute does not exist or if its
        format is not 'TXT'.
    """
    attr = getSingleAttr(node, name, format=format)

    if attr is None:
        return False

    if str(format).upper() != "TXT":
        return False

    # TODO owner deve essere restituito da getSingleAttr e portato in output
    # Check if to_owner is set and owner is different from node
    if not to_owner and owner and owner != node:
        # Copy the attribute from owner to node
        _, _ = copyAttr(owner, node, name)
        # Get the updated attribute
        attr = getSingleAttr(node, name)

    # Remove specified tags from the attribute
    _ = dpg.attr.removeTags(attr, varnames)

    # Check if to_save is set and the attribute object is valid
    if to_save and isinstance(attr, dpg.attr__define.Attr):
        attr.save()

    return True


def getSingleAttr(
        node,
        name: str,
        owner=None,
        only_current: bool = False,
        reload: bool = False,
        format: bool = "",
        to_not_load: bool = False,
        no_palette: bool = False,
        check_date: bool = False,
        file_changed=None,
):
    """
    Retrieves a single attribute of a node based on the specified criteria.

    This function fetches a single attribute identified by 'name' from the specified 'node'. It provides
    various options for how the attribute is retrieved, such as whether to get only the current value,
    whether to reload the attribute, and specific format considerations.

    Args:
        node: The node from which the attribute is to be retrieved.
        name (str): The name of the attribute to retrieve.
        owner (optional): The owner of the attribute. This parameter is present but not used in the function.
        only_current (bool, optional): If True, only the current value of the attribute is retrieved.
            Defaults to False.
        reload (bool, optional): If True, the attribute is reloaded. Defaults to False.
        format (str, optional): The format of the attribute to be retrieved. Defaults to an empty string.
        to_not_load (bool, optional): If True, the attribute is not loaded. Defaults to False.
        no_palette (bool, optional): If True, no palette is used in attribute retrieval. Defaults to False.
        check_date (bool, optional): If True, checks the date of the attribute. Defaults to False.
        file_changed (optional): This parameter is present but not used in the function.

    Returns:
        dpg.attr__define.Attr or None: The requested attribute, or None if the 'name' is not a valid
        string or if 'node' is not an instance of 'dpg.node__define.Node'.

    Note:
        The function returns None if either 'name' is not a valid string or 'node' is not an instance
        of 'dpg.node__define.Node'. The 'owner' and 'file_changed' parameters are included but not
        utilized in the current implementation.
    """

    """
    ind -> non utilizzato da node.getSingleAttr
    Da verificare l'utilità dei parametri
    to_not_load
    no_palette
    check_date 
    file_changed
    """
    if not isinstance(name, str) or name == "":
        return None

    if not isinstance(node, dpg.node__define.Node):
        return None

    return node.getSingleAttr(
        name,
        only_current=only_current,
        to_not_load=to_not_load,
        no_palette=no_palette,
        format=format,
        reload=reload,
        check_date=check_date,
    )


def copyAttr(
        from_node,
        to_node,
        name: str,
        append: bool = False,
        format=None,
        only_current: bool = False,
        reload: bool = False,
        to_save: bool = False,
        dest_name: str = None,
):
    """
    Copies an attribute from one node to another, with various options for handling the copy.

    This function copies an attribute identified by 'name' from 'from_node' to 'to_node'. It allows
    for appending to an existing attribute, specifying a different destination name, and various other
    options like format, saving the attribute, and handling current or reloaded states.

    Args:
        from_node (node__define.Node): The node from which the attribute is to be copied.
        to_node (node__define.Node): The node to which the attribute will be copied.
        name (str): The name of the attribute to be copied.
        append (bool): If True, appends the copied attribute to an existing attribute in 'to_node'. Defaults to False.
        format (optional): The format of the attribute. Defaults to None.
        only_current (bool): If True, only the current state of the attribute is considered. Defaults to False.
        reload (bool): If True, reloads the attribute from 'from_node' before copying. Defaults to False.
        to_save (bool): If True, saves the attribute after copying. Defaults to False.
        dest_name (str): The destination name for the attribute in 'to_node'. If None, uses 'name'. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - The result of adding the attribute to 'to_node'.
            - The new attribute object created by the copy.

    Note:
        The function performs validity checks on 'to_node' and 'attr'. If any conditions fail, it returns
        (None, None). It also handles the removal of the 'inherits' tag from the new attribute and checks
        for name and node equality to avoid redundant operations.
    """

    """
    input param: from_node, to_node, name, append, format, only_current,
                 reload, to_save, dest_name
    output param: ret, new
    """

    if not node_valid(to_node):
        return None, None

    attrList = getAttr(
        from_node, name, only_current=only_current, reload=reload, format=format
    )
    if attrList is None:  # TODO può essere un singolo oggetto o una lista
        return None, None

    if format is None:
        format = attrList[0].getProperty("format")

    if dest_name is None:
        dest_name = name

    if dest_name == name and to_node == from_node:
        return None, None

    new_attr = dpg.attr.mergeAttr(attrList)
    if new_attr is None:
        return None, None

    # Remove 'inherits' tag
    _ = dpg.attr.removeTags(new_attr, "inherits")

    if append:
        existing_attr = getSingleAttr(to_node, name, only_current=True)
        if existing_attr is not None:
            new_attr = dpg.attr.mergeAttr([existing_attr, new_attr])

    ret = addAttr(to_node, dest_name, new_attr.pointer, format=format, to_save=to_save)

    return ret, new_attr


def node_valid(node) -> bool:
    """
    Checks if the provided object is a valid instance of 'dpg.node__define.Node'.

    This function determines whether the given 'node' is an instance of the specified Node class
    in the 'dpg.node__define' module. It is a utility function used to validate node objects.

    Args:
        node: The object to be checked for validity as a Node instance.

    Returns:
        bool: True if 'node' is an instance of 'dpg.node__define.Node', False otherwise.
    """
    return isinstance(node, dpg.node__define.Node)


def getAttr(
        node,
        name: str,
        owner=None,
        ind=None,
        stop_at_first: bool = False,
        only_current: bool = False,
        to_not_load: bool = False,
        upper: bool = False,
        lower: bool = False,
        check_date: bool = False,
        format=None,
        reload: bool = False,
        load_if_changed: bool = False,
):
    """
    Retrieves an attribute from a node based on various criteria.

    This function obtains an attribute identified by 'name' from the specified 'node', with additional
    options to control the retrieval process. These options include whether to stop at the first match,
    consider only the current state of the attribute, avoid loading the attribute, and other format-specific
    and loading options.

    Args:
        node: The node from which the attribute is to be retrieved.
        name (str): The name of the attribute to retrieve.
        owner (optional): The owner of the attribute. This parameter is present but not used in the function.
        ind (optional): An index parameter. This parameter is present but not used in the function.
        stop_at_first (bool, optional): If True, stops the search at the first matching attribute. Defaults to False.
        only_current (bool, optional): If True, considers only the current state of the attribute. Defaults to False.
        to_not_load (bool, optional): If True, avoids loading the attribute. Defaults to False.
        upper (bool, optional): If True, transforms attribute names to uppercase before comparison. Defaults to False.
        lower (bool, optional): If True, transforms attribute names to lowercase before comparison. Defaults to False.
        check_date (bool, optional): If True, checks the date of the attribute. Defaults to False.
        format (optional): The format of the attribute. Defaults to None.
        reload (bool, optional): If True, reloads the attribute. Defaults to False.
        load_if_changed (bool, optional): If True, loads the attribute if it has changed. Defaults to False.

    Returns:
        list of dpg.attr__define.Attr or None: The retrieved attributes list, or None if 'node' is not an instance of
        'dpg.node__define.Node' or if the attribute is not found.

    Note:
        The 'owner' and 'ind' parameters are included but not utilized in the current implementation.
        The function performs various checks based on the provided flags to control the attribute retrieval process.
    """
    if not isinstance(node, dpg.node__define.Node):
        return None

    return node.getAttr(
        name,
        stop_at_first=stop_at_first,
        only_current=only_current,
        # owner=owner TODO controllare se owner è utile o meno
        to_not_load=to_not_load,
        upper=upper,
        lower=lower,
        check_date=check_date,
        format=format,
        reload=reload,
        load_if_changed=load_if_changed,
    )


def addPointer(p_set, new_el) -> list:
    """
    Adds a new element to a list, initializing the list if necessary.

    This function appends a new element 'new_el' to the list 'p_set'. If 'p_set' is None or uninitialized,
    the function initializes it as a list containing 'new_el'.

    Args:
        p_set: The list to which the new element is to be added. If None, a new list is created.
        new_el: The new element to be added to the list.

    Returns:
        list: The updated list after adding 'new_el'.

    Note:
        The function ensures that 'p_set' is a list before appending 'new_el'. If 'p_set' is None,
        it initializes 'p_set' as a new list with 'new_el' as its first element.
    """
    # Check if the list is None or uninitialized
    if p_set is None:
        return [new_el]

    # Append the new element to the list
    p_set.append(new_el)

    return p_set


def searchSpecialChar(text: str) -> int:
    """
    Searches for the first occurrence of any special character in a given text.

    This function scans through 'text' to find the first occurrence of any predefined special character.
    The special characters considered are: '$', ':', '.', '@', '/', '\\', '-'. It returns the position
    of the first special character found. If none of these characters are present in 'text', the function
    returns -1.

    Args:
        text (str): The text string in which to search for special characters.

    Returns:
        int: The index of the first occurrence of a special character in 'text', or -1 if none are found.

    Note:
        The function searches for the characters sequentially and returns the position of the first one
        it encounters. If no special characters from the defined set are found in 'text', -1 is returned.
    """

    """
    - Python has a built-in method for strings called find, which is used instead of IDL's strpos. 
        The find method returns the lowest index of the substring (if found). 
        If not found, it returns -1.
    - The condition to check if ppp is greater than or equal to 0 is retained, 
        as Python's find method also returns -1 if the substring is not found.
    - The check if posS is greater than the length of the text to return -1 is also maintained in the Python version.
    - The special characters array is modified to be more Pythonic, 
        replacing special = [...] with special_chars = [...].
    - The for loop is modified to iterate directly over the special characters, 
        rather than using an index, making the code more Pythonic.
    """
    if not isinstance(text, str):
        return -1
    special_chars = ["$", ":", ".", "@", "/", "\\", "-"]
    pos = len(text)  # Initial position set to length of the text

    # Search for each special character in the text
    for char in special_chars:
        p = text.find(char)
        if 0 <= p < pos:
            pos = p

    # If no special character is found, return -1
    if pos == len(text):
        return -1

    return pos


def searchAlternateTag(text: str, alt: list = None) -> int:
    """
    Splits the input text by the '|' character and optionally extends a list with the split parts.

    This function searches for the '|' character in 'text' and splits the text into segments based on this
    character. If the 'alt' list is provided, it extends this list with the segments. The function returns
    the count of segments found in 'text'.

    Args:
        text (str): The text to be split based on the '|' character.
        alt (list, optional): A list to be extended with the split segments of 'text'. If None, no extension
            occurs. Defaults to None.

    Returns:
        int: The number of segments obtained by splitting 'text' by the '|' character.

    Note:
        If the '|' character is not found in 'text', the function returns 0 and 'alt' remains unchanged.
        The function modifies the 'alt' list in place if it is provided.
    """
    if not "|" in text:
        return 0

    alt_tags = text.split("|")
    count = len(alt_tags)

    if alt is not None:
        alt.extend(alt_tags)

    return count


def getSon(node, name: str = ''):
    """
    Retrieves a child node with a specific name from the given node.

    This function returns a child node named 'name' of the specified 'node', provided that 'node' is an
    instance of 'dpg.node__define.Node'. If the 'node' is not an instance of this class or if no child
    node with the specified name exists, the function returns None.

    Args:
        node: The parent node from which the child node is to be retrieved.
        name (str): The name of the child node to retrieve.

    Returns:
        dpg.node__define.Node or None: The child node with the specified name, or None if either the
        parent node is not of the correct type or no child with the specified name is found.

    Note:
        The function checks the type of 'node' and only attempts to retrieve the child node if 'node'
        is an instance of 'dpg.node__define.Node'.
    """
    if isinstance(node, dpg.node__define.Node):
        return node.getSon(name)
    return None


def getBrothers(node):
    """
    Retrieves the sibling nodes of a given node.

    This function finds and returns the siblings (or 'brothers') of the specified 'node'. It first
    identifies the parent of 'node' and then retrieves all child nodes (sons) of this parent. The
    siblings of 'node' are all these child nodes excluding 'node' itself.

    Args:
        node: The node whose siblings are to be retrieved.

    Returns:
        tuple: A tuple containing two elements:
            - An integer representing the count of sibling nodes.
            - A list of sibling nodes.

    Note:
        If 'node' has no siblings (i.e., it is an only child or an error occurs in fetching siblings),
        the function returns (0, []). The function assumes that the 'parent' and 'sons' properties are
        correctly implemented in the node's class.
    """
    brothers = []
    parent = node.parent
    sons = getSons(parent)

    if len(sons) <= 1:
        return 0, None

    siblings = [s for s in sons if s != node]
    brothers.extend(siblings)

    return len(siblings), brothers


def getParentSonName(node, parent) -> str:
    """
    Retrieves the name of the node that is a direct child of the specified parent in the node's ancestry.

    This function traverses the ancestry of the given 'node' to find the first ancestor that is a direct
    child of the specified 'parent'. It returns the name of this ancestor node. If 'parent' is an immediate
    parent of 'node', the function returns the name of 'node' itself.

    Args:
        node: The node whose ancestry is to be traversed.
        parent: The target parent node in the ancestry.

    Returns:
        str: The name of the direct child node of 'parent' in the ancestry of 'node'. If 'parent' is
        not found in the ancestry, or if 'node' is not valid, the function returns an empty string.

    Note:
        The function uses 'node_valid' to check the validity of nodes during traversal. It stops traversing
        when it reaches 'parent' or when there are no more valid ancestors to consider.
    """
    curr = node
    next_node = node.parent

    while node_valid(next_node) and next_node != parent:
        curr = next_node
        next_node = next_node.parent

    return getNodeName(curr)


def isDescendant(node, parent) -> bool:
    """
    Determines whether a given node is a descendant of a specified parent node.

    This function checks if the 'node' is a descendant of the 'parent' node. It first compares the
    'node' directly with the 'parent'. If they are not the same, the function retrieves all parent
    nodes of 'node' and checks if 'parent' is among them.

    Args:
        node: The node to check for being a descendant.
        parent: The node to check against as the potential ancestor.

    Returns:
        bool: True if 'node' is a descendant of 'parent', False otherwise.

    Note:
        The function returns True if 'node' and 'parent' are the same. If 'node' has no parents or
        the 'parent' node is not found among its ancestors, it returns False.
    """
    if node == parent:
        return True

    parents = getParents(node)

    if len(parents) == 0:
        return False

    if not parents:
        return False

    return parent in parents


def isSharedNode(node) -> bool:
    """
    Checks if a given node is marked as shared.

    This function determines whether the specified 'node' is a shared node by checking its 'shared' property.
    It first verifies if 'node' is an instance of 'dpg.node__define.Node'.

    Args:
        node: The node to check for the shared status.

    Returns:
        bool: True if 'node' is an instance of 'dpg.node__define.Node' and is marked as shared,
        False otherwise.

    Note:
        The function returns False if 'node' is not an instance of 'dpg.node__define.Node' or if the
        'shared' property of 'node' indicates it is not shared.
    """
    if not isinstance(node, dpg.node__define.Node):
        return False

    return node.getProperty("shared")


def detachNode(node) -> bool:
    """
    Detaches a given node from its parent.

    This function detaches the specified 'node' from its parent in the node hierarchy, provided that
    'node' is an instance of 'dpg.node__define.Node'.

    Args:
        node: The node to be detached from its parent.

    Returns:
        bool: True if the node is successfully detached, False if 'node' is not an instance of
        'dpg.node__define.Node' or the detachment operation fails.

    Note:
        The function checks if 'node' is an instance of 'dpg.node__define.Node' before attempting
        to detach it. If 'node' is not a valid instance, the function returns False without performing
        any action.
    """
    if not isinstance(node, dpg.node__define.Node):
        return False
    node.detachNode()


def getSharedNode(path: str):
    """
    Retrieves a shared node based on a specified path.

    This function uses the 'createTree' function to get or create a shared node at the given 'path'. If a
    shared node already exists at the path, it is returned; otherwise, a new shared node is created at
    this location.

    Args:
        path (str): The file system path where the shared node is located or to be created.

    Returns:
        dpg.node__define.Node: The shared node located or created at the specified path.

    Note:
        The function delegates to 'createTree' with the 'shared' parameter set to True, indicating that
        the node should be treated as shared. It is designed to ensure that a shared node at the specified
        path is always returned, whether it pre-exists or is newly created.
    """
    return createTree(path, shared=True)


def nodeIsEmpty(node) -> bool:
    """
    Checks if a given node is empty.

    This function determines whether the specified 'node' is empty. A node is considered empty if it
    has no child nodes (sons) and no associated array data. The function first checks for the presence
    of child nodes. If no child nodes are found, it then checks for the presence of array data associated
    with the node.

    Args:
        node: The node to be checked for emptiness.

    Returns:
        bool: False if the node has child nodes or array data, True otherwise.

    Note:
        The function uses 'getSons' to check for child nodes and 'dpg.array.get_array' to check for array data.
        If either child nodes or array data are present, the node is considered not empty.
    """
    sons = getSons(node)

    if len(sons) > 0:
        return False

    array, _ = dpg.array.get_array(node)

    return array is not None


def getAllDescendant(
        node, nodes: list = None, only_leaves: bool = False, and_me: bool = False
):
    """
    Retrieves all descendant nodes of a given node based on specified criteria.

    This function collects all descendants of the specified 'node'. It allows for options to include
    only leaf nodes, include the node itself, and to specify an existing list to which the descendant
    nodes should be added.

    Args:
        node: The node from which to gather descendants.
        nodes (list, optional): An existing list to which descendant nodes will be added. If None,
            a new list is created. Defaults to None.
        only_leaves (bool, optional): If True, only leaf nodes (nodes without children) are included
            in the results. Defaults to False.
        and_me (bool, optional): If True, the 'node' itself is included in the list of descendants.
            Defaults to False.

    Returns:
        list of dpg.node__define.Node or None: A list containing the descendant nodes according to the
        specified criteria. Returns None if 'node' is not an instance of 'dpg.node__define.Node'.

    Note:
        The function checks if 'node' is an instance of 'dpg.node__define.Node'. If not, it returns None.
        The 'nodes' parameter allows the function to add the descendants to an existing list, which can
        be useful for recursive calls or aggregating results from multiple nodes.
    """
    if not isinstance(node, dpg.node__define.Node):
        return None
    return node.getAllDescendant(only_leaves=only_leaves, and_me=and_me, nodes=nodes)


def removeAttr(node, name: str = "", delete_file: bool = False) -> bool:
    """
    Removes an attribute from a given node and optionally deletes the associated file.

    This function removes an attribute named 'name' from the specified 'node'. It also provides an option
    to delete the file associated with the attribute, if any. The function only operates if 'node' is an
    instance of 'dpg.node__define.Node'.

    Args:
        node: The node from which the attribute is to be removed.
        name (str, optional): The name of the attribute to be removed. If empty, no action is taken.
            Defaults to an empty string.
        delete_file (bool, optional): If True, the file associated with the attribute is also deleted.
            Defaults to False.

    Returns:
        bool: True if the operation is successful, False if 'node' is not an instance of
        'dpg.node__define.Node' or if the name is empty.

    Note:
        The function performs the removal operation only if 'node' is a valid instance and 'name' is
        not empty. The option to delete the associated file provides a way to clean up related data
        files if necessary.
    """
    if not isinstance(node, dpg.node__define.Node):
        return False
    node.removeAttr(name, delete_file=delete_file)
    return True


def changeFilesMode(to_path: str, mode: int):
    """
    Change the mode (permissions) of a file or all files within a directory.

    Args:
        to_path (str): The path to the file or directory.
        mode (int): The new mode (permissions) to set, which should be an octal number (e.g., 0o755).

    Returns:
        None
    """

    # Check if the OS is not Windows
    if os.name == "nt":
        return

    # Check for valid input
    if not isinstance(to_path, str) or not isinstance(mode, int):
        return

    path = dpg.path.checkPathname(to_path)

    # Change mode for a single file
    if os.path.isfile(path):
        os.chmod(path, mode)
    # If it's a directory, change mode for the directory and all files within it
    elif os.path.isdir(path):
        os.chmod(path, mode)  # Change mode for the directory
        for root, dirs, files in os.walk(path):
            for name in files:
                file_path = os.path.join(root, name)
                os.chmod(file_path, mode)


def copySonsToNode(
        from_node,
        to_node,
        and_files: bool = False,
        overwrite: bool = False,
        to_path: str = "",
):
    """
    Copies all child nodes (sons) from one node to another, with options for copying files and overwriting.

    This function copies all child nodes of 'from_node' to 'to_node'. It allows for the copying of
    associated files and provides an option to overwrite existing nodes and files in the destination.
    If 'to_path' is not specified, the function uses the path of 'to_node' as the destination path.

    Args:
        from_node: The node from which child nodes are to be copied.
        to_node: The node to which the child nodes will be copied.
        and_files (bool, optional): If True, associated files of the child nodes are also copied. Defaults to False.
        overwrite (bool, optional): If True, existing child nodes and files in the destination are overwritten.
            Defaults to False.
        to_path (str, optional): The file system path to which child nodes and files will be copied.
            Defaults to an empty string, which uses the path of 'to_node'.

    Returns:
        tuple: A tuple containing:
            - The path to which the child nodes and files have been copied.
            - An error code (0 for success, -1 for failure in copying, -2 if 'to_node' is invalid).

    Note:
        The function performs a series of checks and operations, including directory creation, node copying,
        and file handling. It returns an error code indicating the status of the operation, with different
        codes for various failure conditions.
    """
    # TODO controllare input e output
    error = 0
    if not isinstance(from_node, dpg.node__define.Node):
        error = -1
        return to_path, error

    sons = from_node.getSons()
    if to_path == "" or to_path is None:
        to_path = getNodePath(to_node)
    if not os.path.exists(to_path):
        os.makedirs(to_path)
    for son in sons:
        from_path = getNodePath(son)
        err = copyDir(from_path, to_path)
        overwrite = True
        if err != 0:
            error = -1
            return to_path, error
    if and_files:
        from_path = getNodePath(from_node)
        err = copyAllFiles(from_path, to_path, overwrite=overwrite)
        if err != 0:
            error = -1
    if overwrite:
        if not isinstance(to_node, dpg.node__define.Node):
            error = -2
            return to_path, error
        tmp_sons = to_node.getSons()
        for son in tmp_sons:
            removeNode(son)
    if len(sons) > 0:
        updateTree(to_node)
    return to_path, error
