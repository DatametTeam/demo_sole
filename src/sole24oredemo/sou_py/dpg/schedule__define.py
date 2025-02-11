import multiprocessing
import os
import threading
import time
import traceback
from pathlib import Path
import sou_py.dpg as dpg
import sou_py.dpb as dpb
import sou_py.preprocessing as preprocessing
import sou_py.products as products
import numpy as np
import shutil

from sou_py.dpg.log import log_message
from sou_py.dpg.scheduler__define import Scheduler

"""
Funzioni ancora da portare
FUNCTION IDL_rv_find_proc 
FUNCTION IDL_rv_get_schedule_path 
FUNCTION IDL_rv_get_system 
PRO IDL_rv_remove_prod_list 
PRO IDL_rv_update_prod_list 
"""

lock = multiprocessing.Lock()


class Schedule(object):
    def __init__(
            self,
            out_path: str,
            site=None,
            output_tree=None,
            input_tree=None,
            name: str = None,
            schedule_path: str = None,
            scheduler: Scheduler = None,
    ):
        """
        Initializes an object Schedule for managing, including setting up input/output trees,
        and configuring scheduler properties like site, name, and schedule path.

        Args:
            out_path (str): The path where output files and schedule information will be stored.
            site (optional): The site identifier. Defaults to None.
            output_tree (optional): The output tree structure. Defaults to None.
            input_tree (optional): The input tree structure, if required. Defaults to None.
            name (str, optional): The name of the schedule. If None, the basename of `out_path` is used.
            schedule_path (str, optional): The path to the schedule. Defaults to None.
            scheduler (Scheduler, optional): A scheduler containing the date and time for scheduling operations.
                                             Defaults to None.

        Returns:
            None
        """
        # if sites == []:
        #    pass
        # sites = getLocalSites()
        # end
        self.counter = 0
        self.STOP_SEQ = 0
        self.output_tree = output_tree
        self.input_tree = input_tree
        self.outPath = out_path
        self.scheduler = scheduler
        self.site = site

        if name is None:
            self.name = os.path.basename(out_path)
        else:
            self.name = name  # nome della schedula
        self.schedule_path = schedule_path

        self.createTree(path=os.path.join(out_path, self.name), io_flag="output")
        prods = self.output_tree.getSons()
        for prod in prods:
            self.removeTemplateFiles(prod, root=True)
            dpg.times.set_time(prod, date=self.scheduler.date, time=self.scheduler.time)
            dpg.times.set_time(
                prod,
                date=self.scheduler.date,
                time=self.scheduler.time,
                to_save=True,
                nominal=True,
            )

    def createTree(
            self,
            path: str,
            io_flag: str = "input",
            shared: bool = False,
            only_root: bool = False,
    ):
        """
        This method is used to instantiate a tree structure starting from a root node at a specified path. 'io_flag'
        define which tree should be created, then the method dpg.tree.createTree is used.

        Args:
            path (str): The file system path where the root node of the tree will be created.
            io_flag (str, optional): Flag used to determine which tree should be created. Defaults to 'input'.
            shared (bool, optional): If True, searches for an existing shared node with the same path before
            creating a new node. Defaults to False.
            only_root (bool, optional): If True, only the root node is created without creating its subtree.
            Defaults to False.

        Returns:
            dpg.node__define.Node or None
        """
        if io_flag == "input":
            self.input_tree = dpg.tree.createTree(
                path, shared=shared, only_root=only_root
            )
            return self.input_tree
        elif io_flag == "output":
            self.output_tree = dpg.tree.createTree(
                path, shared=shared, only_root=only_root
            )
            return self.output_tree
        else:
            return None

    # def removeTree(self, io_flag):
    #     if io_flag == 'input':
    #         self.input_tree = None
    #     elif io_flag == 'output':
    #
    #     self.root = None
    #     return

    def removeTemplateFiles(self, node, root: bool = False):
        """
        The following method starts from a given node and removes all the indicated attributes and possibly
        associated files. If the node is not a root node it does the same for all its descendants.

        Args:
            node (Node): Node from which to derive its descendants.
            root (bool, optional): True if the selected node is a root node. Defaults to false.

        Returns:
            None
        """
        filesToDelete = [
            dpg.cfg.getProcDescName(),
            "phase1.txt",
            "phase2.txt",
            dpg.cfg.getParDescName(gui=True),
            dpg.cfg.getProdListName(),
            dpg.cfg.getProdListName(interactive=True),
        ]

        if root:
            filesToDelete.extend(
                [
                    dpg.cfg.getItemDescName(),
                    dpg.cfg.getModelDescName(),
                    dpg.cfg.getGeoDescName(),
                    dpg.cfg.getValueDescName(),
                ]
            )
            nodes = [node]
        else:
            nodes = node.getAllDescendant(and_me=True)
        # endif

        for nnn in nodes:
            for fff in filesToDelete:
                nnn.removeAttr(name=fff, delete_file=True)
            # endfor
        # endfor
        return

    def schedule_check(self, site: str, raw_path: str):
        """
        Checks the status of the scheduling process for a given site and raw path.

        Args:
            site: The name of the site to check.
            raw_path: The path to the raw data associated with the schedule.

        Returns:
            - **int**: Returns 1 if the check passes, or 0 if the check fails or if the scheduling process is already
            marked as done.
        """

        # SCHEDULE_CHECK, schedule, site, rawPath, CHECK=check, INTERACTIVE=interactive

        check = 0
        if site == "":
            return check
        check = 1
        if raw_path is None:
            return check

        attr = dpg.attr.loadAttr(
            os.path.join(self.outPath, self.name), dpg.cfg.getScheduleDescName()
        )
        check_file, _, _ = dpg.attr.getAttrValue(attr, "check_file", "", prefix=site)
        check_raw, _, _ = dpg.attr.getAttrValue(attr, "check_raw", 0, prefix=site)

        if check_file == "":
            if raw_path == "" and check_raw > 0:
                check = 0
            return check
        # endif

        attr = dpg.attr.loadAttr(raw_path, check_file)
        done, _, _ = dpg.attr.getAttrValue(attr, self.name, "")

        if done == "done":
            check = 0
        else:
            dpg.attr.saveAttr(raw_path, check_file, [self.name], ["done"], append=True)
            # SaveAttr(rawPath, check_file, self.name, 'done', append=True)
        return check

    def duplicateProductNode(
            self,
            schProd,
            outProd,
            rawPath: str = None,
            site_name: str = "",
            date: str = "",
            time: str = "",
            no_log: bool = False,
    ) -> bool:
        """
        Duplicates a folder and its sub-folders, linking them to another location.

        This function duplicates the specified source node (`schProd`) and links it to a
        new child node under `outProd`. It also sets various attributes and properties for
        the new node, including the raw data path, date, time, and site name.

        Args:
            schProd (dpg.node__define.Node): The source node to be duplicated.
            outProd (dpg.node__define.Node): The output node where the new node will be added.
            rawPath (str, optional): The raw data path to set for the new node. Defaults to None.
            site_name (str, optional): The name of the new child node to be added. If empty or invalid,
                                       an error is printed and the function returns False.
            date (str, optional): The date to set for the new node. Defaults to an empty string.
            time (str, optional): The time to set for the new node. Defaults to an empty string.
            no_log (bool, optional): If True, logging is suppressed. Defaults to False.

        Returns:
            bool: True if the duplication and linking were successful, False otherwise.
        """
        if not isinstance(schProd, dpg.node__define.Node):
            return False
        if not isinstance(outProd, dpg.node__define.Node):
            return False

        node = schProd
        siteNode, exists = outProd.addNode(site_name)
        if exists:
            if not no_log:
                outName = outProd.parent.getProperty("name")
                msg = site_name + " already processed for schedule " + outName
                print("IDL_rv_add_log", msg)  # , /WARNING, PROCNAME=''
            # endif
            return False
        # endif
        dpg.tree.copyNode(node, siteNode, overwrite=False)  # , ERR=err, MODE='666'o

        if rawPath is not None and rawPath != '':
            dpg.radar.set_raw_path(siteNode, rawPath)

        attr = siteNode.getSingleAttr(dpg.cfg.getMosaicDescName(), only_current=True)
        if isinstance(attr, dpg.attr__define.Attr):
            siteNode.removeAttr(dpg.cfg.getMosaicDescName(), delete_file=True)
            siteNode.removeAttr(dpg.cfg.getProductDescName(), delete_file=True)
        # endif

        siteNode.removeAttr(
            dpg.cfg.getProdListName(), delete_file=True
        )  # cancella file products.txt da cartella output
        siteNode.removeAttr(
            dpg.cfg.getProdListName(interactive=True), delete_file=True
        )  # cancella file int_products.txt da cartella output

        dpg.times.set_time(siteNode, date=date, time=time, site_name=site_name)
        dpg.times.set_time(
            siteNode, date=date, time=time, site_name=site_name, nominal=True
        )
        if not no_log:
            outName = outProd.getProperty("name")
            log_message(
                f"Added site {site_name} to product {outName}",
                all_logs=False,
                general_log=True,
            )
        # endif
        return True

    def addSiteToSchedule(
            self,
            siteName: str,
            date: str = "",
            time: str = "",
            rawPath: dict = "",
            no_log: str = False,
    ):
        """
        Duplicates each product as it finds it, creates a tree identical to that of the schedule except that it is
        called with the name of the site.

        Args:
            siteName (str): The name of the child node to retrieve.
            date (str, optional): The date to set for the node. Defaults to "".
            time (str,optional): The time to set for the node. Defaults to "".
            rawPath (dict, optional): A dict containing the raw path. Defaults to "".
            no_log (bool, optional): Show logs if True. Defaults to False.

        Returns:
            None
        """
        if siteName == "":
            return

        schTree = self.createTree(path=self.schedule_path, io_flag="input")
        prods = self.output_tree.getSons()
        if len(prods) == 0:
            if not no_log:
                msg = siteName + " schedule " + self.outPath + " not compatible!"
                print("IDL_rv_add_log", msg)  # , /FATAL_ERROR, PROCNAME='')
            # endif
            # self.removeTree()
            return
        # endif

        ret = 0
        outPath = self.output_tree.path

        for ppp in prods:
            son = schTree.getSon(ppp.name)
            if self.duplicateProductNode(
                    schProd=son,
                    outProd=ppp,
                    rawPath=rawPath,
                    site_name=siteName,
                    date=date,
                    time=time,
                    no_log=no_log,
            ):
                ret = ret + 1
        # endfor

        if not no_log:
            msg = "Found " + str(ret) + " products in " + self.outPath
            # print(msg)  # , PROCNAME=''
        # endif
        # self.removeTree()
        return

    def find_proc(self, site, prefix: bool = False):
        """
        This method retrieves all child nodes from the output_tree. Subsequently, for each node in the list it searches
        for attributes with the name of the process descriptor file, this is repeated in all descendant nodes where the
        attributes are present. Then, the attributes list is sorted according to priority field. At the end, a
        list containing all the attributes and a list of nodes where the attributes were found is returned.

        Args:
            site (str): The name of the child node to retrieve.
            prefix (bool, optional): If True, uses the prefixed version of the 'priority' attribute for sorting.
                                     Defaults to False.

        Returns:
            int or tuple(list,list): if the list containing the child nodes is not empty, two lists are returned, all
            the attributes found and the nodes containing them. Otherwise, 0 is returned.
        """
        sons = self.output_tree.getSons()

        if len(sons) == 0:
            return 0
        procFile = dpg.cfg.getProcDescName()
        attrSet = []
        nodeSet = []
        for ppp, son in enumerate(sons):
            siteNode = son.getSon(site)
            attr, node = dpg.tree.findAttr(siteNode, procFile, down=True, all=True)
            attrSet.extend(attr)
            nodeSet.extend(node)
        sorted_indexes = dpg.radar.sort_attr(attrSet, prefix)

        nodes = [nodeSet[i] for i in sorted_indexes]
        attrs = [attrSet[i] for i in sorted_indexes]
        return nodes, attrs

    def schedule_init(
            self,
            site: str,
            date: str = "",
            time: str = "",
            rawPath: str = "",
            no_log: bool = False,
    ):
        """
        Initializes the scheduling process by setting up directories, creating the output tree, and adding the site
        to the schedule.

        Args:
            site: The name of the site to initialize.
            date: Optional date for the schedule (default is an empty string).
            time: Optional time for the schedule (default is an empty string).
            rawPath: Optional raw path to associate with the schedule (default is an empty string).
            no_log: If True, disables logging (default is False).

        Returns:
            - **None**
        """

        if self.outPath == "":
            return

        schName = self.name
        self.site = site
        dir = dpg.path.checkPathname(self.outPath)
        out_dir = os.path.join(self.outPath, self.name)

        if not os.path.isdir(out_dir):
            self.scheduler.copySchedule(self.schedule_path, dir)

        self.createTree(path=out_dir, io_flag="output")

        if site is None or site == "":
            return

        self.addSiteToSchedule(
            site, date=date, time=time, rawPath=rawPath, no_log=False
        )

        if os.path.isfile(
                os.path.join(self.outPath, site + ".log")
        ):  # err gt 0 then begin
            msg = "Another process is trying to process the schedule ... ignored."
            # print("IDL_rv_add_log, msg, /DATE, PROCNAME='', /NEW_LINE")
            # self.removeTree()
        else:
            if isinstance(rawPath, str) and (rawPath != ""):
                log_message(f"Current Raw Path: {rawPath}")
        return

    def stop_sequence(
            self, remove_node: bool = False, remove_all: bool = False, jump=None
    ):
        """
        This function assign a value to STOP_SEQUENCE parameter based on different flags. This value is used in the
        execution of a schedule to determine which operations to perform.

        Args:
            remove_node (bool, optional): If true, STOP_SEQ is equal to 2.
            remove_all (bool, optional): If true, STOP_SEQ is equal to 3.
            jump (optional): if None, STOP_SEQ is equal to 1.

        Returns:
            None
        """
        if not jump:
            self.STOP_SEQ = 1
        if remove_node:
            self.STOP_SEQ = 2
        if remove_all:
            self.STOP_SEQ = 3

    def schedule_exec(self, site: str, phase: int):
        """
        Executes phase [F] of a schedule.
        For each phase_[F].txt file found in the schedule tree, all the statements contained in it are executed.

        Args:
            site (str): Site to be processed.
            phase (int): Phase (1 or 2)

        Returns:
            None
        """

        if site is None or site == "":
            return

        phTag = "phase" + str(phase).strip()
        nodeSet, attrSet = self.find_proc(site=site, prefix=phTag)

        # StartSubProgress, nNodes  # Progress bar??
        nodesToRemove = []

        for nnn9, current in enumerate(nodeSet):
            try:
                # updateProgress
                t1 = time.time()
                phFile, _, _ = dpg.attr.getAttrValue(attrSet[nnn9], phTag, "")
                toRemove9 = phase
                JUMP_SEQ = 0
                self.STOP_SEQ = 0

                if len(phFile) > 0:

                    log_message(f"--" * 50, all_logs=True, general_log=False)
                    path_parts = current.path.split(os.sep)
                    if path_parts[-9] == "RADAR":
                        if current.name == "Quality":
                            product_name = current.parent.parent.name
                        else:
                            product_name = current.parent.name
                    else:
                        product_name = path_parts[-1]
                    log_message(f"Starting {phFile} for product {product_name}", all_logs=True, general_log=False)
                    log_message(f"Executing following statements: ", all_logs=False, general_log=False)

                    attr = attrSet[nnn9]
                    toRemove9, _, _ = dpg.attr.getAttrValue(attr, "toRemove", 0, prefix=site)

                    filePath = os.path.join(current.path, phFile)
                    if os.path.isfile(filePath):

                        text_file = open(filePath, "r")
                        statements = text_file.readlines()
                        text_file.close()

                        nStat9 = len(statements)
                        for sss9, statement in enumerate(statements):
                            if len(statement) > 0 and not statement.startswith(";"):
                                log_message(f"{statement}", level="INFO", all_logs=False, general_log=False)
                                exec(statement)
                                # if statVoid eq 0 then ManageError
                                if self.STOP_SEQ > 0:
                                    log_message(
                                        f"Sequence stopped @ statement: {statement} @ node "
                                        f"{dpg.tree.getNodePath(current)}", level="WARNING", all_logs=False,
                                        general_log=True)
                                    break
                                if JUMP_SEQ < 0:
                                    log_message(f"Sequence jumped", level="WARNING", all_logs=False, general_log=True, )
                                    sss9 += JUMP_SEQ
                                    JUMP_SEQ = 0
                        # endif
                    # dpg.array.set_array(current, data=my_data)
                    # dpb.dpb.put_data(current, var=self.my_data, no_copy=1)
                if self.STOP_SEQ == 3:
                    dpg.tree.removeNodes(nodeSet, directory=True)
                    return

                if toRemove9 == phase or toRemove9 == 4 or self.STOP_SEQ == 2:
                    nodesToRemove.append(current)

                self.LR_HR_exec(current, site)

                if current.name == "Quality":
                    node_name = f"{current.parent.parent.name} -> {current.name}"
                else:
                    node_name = (
                        current.name
                        if current.parent.name == site
                        else current.parent.name
                    )
                log_message(
                    f"Schedule on site {site} with node {node_name} executed in: "
                    f"{round((time.time() - t1), 4)} seconds",
                    all_logs=True, )

            except Exception as e:
                stack_trace = traceback.format_exc()
                log_message(
                    f"Exception occurred at {site}\n{stack_trace}",
                    level="EXCEPTION",
                    newlines=True,
                    all_logs=False,
                    general_log=True,
                )

        for node in nodesToRemove:
            dpg.tree.removeNode(node, directory=True)
        return

    def find_site_pos(self, path: str, site: str) -> int:
        """
        Finds the position of a specified site within a file path.

        Args:
            path (str): The file path to search.
            site (str): The name of the site to locate within the path.

        Returns:
            int: The index of the site in the path if found, otherwise -1.
        """
        path_parts = path.split(os.sep)
        for i, part in enumerate(path_parts):
            if part.lower() == site.lower():
                return i
        return -1

    def build_node_name(self, path_parts: list, site_index: int) -> str:
        """
        Constructs a node name by concatenating relevant path parts.

        Args:
            path_parts (list): A list of directory names from the file path.
            site_index (int): The index of the site within the path parts.

        Returns:
            str: The constructed node name.
              - If the site is the last part, returns the preceding directory name.
              - Otherwise, concatenates the previous directory with the remaining path parts.
        """
        if site_index == len(path_parts) - 1:
            return path_parts[site_index - 1]
        node_name = path_parts[site_index - 1]
        for part in path_parts[site_index:]:
            node_name += f" -> {part}"
        return node_name

    def LR_HR_exec(self, current, site: str):
        try:
            path_parts = current.path.split(os.sep)
            site_pos = self.find_site_pos(current.path, site)

            if site_pos == -1:
                return

            node_name = self.build_node_name(path_parts, site_pos)

            # log_message(f"{node_name}", all_logs=True, general_log=False)

        except Exception as e:
            stack_trace = traceback.format_exc()
            log_message(
                f"Exception occurred at {site}\n{stack_trace}",
                level="EXCEPTION",
                newlines=True,
                all_logs=False,
                general_log=True,
            )

    def schedule_end(
            self, site: str, phase: int, interactive: bool = False, export: bool = False
    ):  # ,  NPRODS=nProds
        """
        Finalizes the scheduling process by managing product nodes and optionally triggering export operations.

        Args:
            site: The name of the site to process.
            phase: The phase of the scheduling process (e.g., 1 or 2).
            interactive: Whether the process should be interactive (default is False).
            export: Whether to export the results after processing (default is True).

        Returns:
            - **int**: The number of product nodes processed.
        """

        prods = self.output_tree.getSons()
        valids = 0
        for ppp, current in enumerate(prods):
            node = current.getSon(site)
            if isinstance(node, dpg.node__define.Node):
                self.removeTemplateFiles(node)
                dpg.tree.saveNode(node)

            sons = dpg.tree.getSons(current)
            directory = True if phase == 2 else False
            if len(sons) > 0:
                directory = False
                valids = valids + 1
                if phase > 1:
                    current.make_prod_desc()
                    self.update_prod_list(current.path, interactive=interactive)
            # if directory:
            #     pathtofile = os.path.join(current.path, site)
            #     if os.path.exists(pathtofile):
            #         os.remove(pathtofile)
            # del current
            dpg.tree.removeNode(current, directory=directory)

        filePath = self.output_tree.path
        if phase == 2:
            (Path(filePath) / 'phase2.end').touch()

        if valids > 0 and export:
            print("Executing distribution.com")
            os.system(
                os.path.join(self.scheduler.rv_home, "target", "distribution.com ")
                + filePath
            )

        # dpg.tree.removeTree(self)
        return len(prods)

    def update_prod_list(self, prodPath: str, interactive: bool = False, reset=None):
        """
        Updates the product list stored in the given path. This method manages the
        product list files, ensuring that the list is properly updated, sorted, and
        respects the maximum number of products allowed.

        Args:
            prodPath (str): The file system path where products are stored.
            interactive (bool, optional): If True, the function will return the name of
                the product list file specific to interactive mode. Defaults to False.
            reset (optional): If not None, the product list will be reset to contain only the provided product path.

        Returns:
            None
        """
        if prodPath == "":
            return
        if not os.path.isdir(prodPath):
            return
        prodName = os.path.basename(prodPath)
        schedule = os.path.basename(os.path.dirname(prodPath))
        list_path = self.scheduler.getSchedulePath()
        list_path = os.path.join(list_path, schedule, prodName)
        if not os.path.isdir(list_path):
            return
        prod_desc = dpg.attr.loadAttr(list_path, dpg.cfg.getProductDescName())
        max_frames, _, _ = dpg.attr.getAttrValue(prod_desc, "max_frames", 12)

        prod_list, n_prods, list_name = self.getProdList(
            list_path, interactive=interactive
        )
        if not isinstance(prod_list, list):
            prod_list = [prod_list]

        if reset is None and n_prods > 0:
            pos = prodPath.find("20")
            ind = [i for i, p in enumerate(prod_list) if p == prodPath]
            count = len(ind)
            if count <= 0 and pos > 0:
                prod_list.append(prodPath)
                prod_list = sorted(prod_list)
                if len(prod_list) > max_frames:
                    prod_list.pop(0)
                if prod_list[-1] != prodPath:
                    prod_list = [prodPath]
        else:
            prod_list = [prodPath]

        tags = ["path" for v in prod_list]
        f = open(os.path.join(list_path, list_name), "w")
        for k, v in zip(tags, prod_list):
            f.write(k + " = " + v + "\n")
        f.close()

    def getProdList(
            self, path: str = None, interactive: bool = False, pathname: str = None
    ):
        """
        Retrieves the product list from the specified path. This method loads the
        product list attribute and returns its value, the length of the list, and
        the list name.

        Args:
            path (str, optional): The file system path where the product list is stored.
                Defaults to None.
            interactive (bool, optional): If True, the function will use the interactive
                mode product list name. Defaults to False.
            pathname (str, optional): The full path to the file. Defaults to None.

        Returns:
            tuple: A tuple containing:
                - attr_value: The value of the 'path' attribute from the
                  product list.
                - len_attr (int): The length of the product list.
                - list_name (str): The name of the product list file.
        """

        list_name = dpg.cfg.getProdListName(interactive=interactive)
        attr = dpg.attr.loadAttr(path, list_name, pathname=pathname)
        attr_value, exist, _ = dpg.attr.getAttrValue(attr, "path", "")
        if exist:
            len_attr = len(attr_value) if isinstance(attr_value, list) else 1
            return attr_value, len_attr, list_name
        else:
            return None, 0, list_name

    def get_curr_volume(
            self,
            prodId,
            moment: str,
            linear: bool = False,
            reload: bool = False,
            projected: bool = False,
    ):
        """
        This function allows access to the latest sampled volume.

        Args:
            prodId (Node):  Node of current product.
            moment (str):   Physical quantity required.
            linear (bool):  Keyword that enables linear scaling conversion.
            reload (bool):  If true, the data is read back from the file.
            projected (bool): Keyword that enables the projection of the beams onto the ground,
                            equivalent to a further sampling in the range of a factor cos(elev).

        Returns:
            tuple:
                - main: Variable that will contain the node associated with the volume.
                - var: Variable to which the data array will be associated.

        """

        raw_tree = dpg.access.getRawTree(prodId, sampled=True, reload=reload)
        if raw_tree is None:
            return np.array([0]), None
        main = dpg.access.find_volume(raw_tree, moment)
        var = None

        if not isinstance(main, dpg.node__define.Node):
            return None, None

        var = dpb.dpb.get_data(main, var, numeric=True, linear=linear)

        if projected:
            var = dpb.dpb.project_volume(main, var)
        return var, main

    def get_volumes(
            prodId, moment: str = None, any_flag: bool = False, raw_tree=None, measure=None
    ):
        """
        Returns the root node of a high-resolution scan (one or more quantities).

        Args:
            prodId: Current node of a generic product.

        Returns:
            volId: Node (or list of nodes) representing the root of a scan.

        """
        if moment is not None and moment != "":
            measure = moment

        if measure is None:
            measure, _, _ = dpg.radar.get_par(
                prodId,
                "measure",
                "",
            )  # /START_WITH)
            if measure is None:
                if not any_flag:
                    return
                measure = "CZ"
        # endif

        # global_par = {}
        # dpg.utility.init_global_par_dict(global_par)

        if isinstance(measure, list):
            len_measure = len(measure)
            volId = [None] * len_measure
            for idx in range(len_measure):
                volId[idx] = dpg.access.find_volume(
                    raw_tree, measure[idx], prod_id=prodId
                )
        elif isinstance(measure, str):
            # measure = [measure]
            volId = dpg.access.find_volume(raw_tree, measure, prod_id=prodId)
            len_measure = 1
        else:
            return None

        if len_measure == 1:
            if not isinstance(volId, dpg.node__define.Node):
                if any_flag and raw_tree is not None:
                    volId = raw_tree

        return volId


def get_schedule_path(prodId):
    """
    Retrieves the schedule file path for a given node, along with its owner and system information.

    This function is responsible for constructing the file path for the 'schedule.txt' file based on
    the node's attributes and system information. It fetches the corresponding attributes
    from the provided node, retrieves necessary information, and assembles the schedule path by
    combining the relevant directory, schedule name, and production name.

    Args:
        prodId (Node): The node object from which the schedule attribute is to be retrieved.

    Returns:
        tuple: A tuple containing:
            - path (str): The full path to the schedule file based on the node's information.
            - owner (str): The owner of the retrieved schedule attribute.
            - system (str): The system information corresponding to the node.

        Returns None if the system information is not available.
    """
    attr = dpg.tree.getSingleAttr(prodId, dpg.cfg.getScheduleDescName())
    owner = attr.owner  # obtain owner

    active, _, _ = dpg.attr.getAttrValue(attr, "active", 0)

    system, _, _ = dpg.attr.getAttrValue(attr, "system", "")
    if system == "" or system is None:
        return None

    path = dpg.rpk.getSchedulePath(system)
    schedule = dpg.tree.getNodeName(owner)
    prodName = dpg.tree.getParentSonName(prodId, owner)
    path = os.path.join(path, schedule, prodName)

    return path, owner, system
