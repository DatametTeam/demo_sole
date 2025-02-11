import traceback

import numpy as np
import pyproj

import sou_py.dpg as dpg
from sou_py.dpg.attr__define import Attr
from sou_py.dpg.log import log_message

"""
Funzioni ancora da portare
PRO Map::GetSpace 
PRO Map::GetSpaceInfo 
PRO Map::Warp 
PRO Map__Define 
"""

from audioop import getsample
from operator import truediv


# from os import getxattr


class Map(object):
    def __init__(self):
        """
        Map object initialization.
        nPar    = ...
        dim     = ...
        par     = ...
        origin  = ...
        pAz     = ...
        pEl     = ...
        mapProj = ...
        """
        self.nPar = 0
        self.dim = [0, 0]
        self.par = [0.0] * 12
        self.origin = [0.0] * 3
        self.pAz = None
        self.pEl = None
        self.mapProj = None
        self.attr = None

        self.up_name = ""
        self.p0lat = 0.
        self.p0lon = 0.
        self.uv_box = None
        self.projection = 0

    def init_map(self, attr: Attr, projection: str = ""):
        """
        Initializes the map projection and associated parameters. If the projection is not defined, it is retrieved from
        the attributes along with information about latitude and longitude. If a global variable containing the map
        projection is present, it is assigned to this instance; otherwise, it is initialized and appended to a global
        variable called. Various parameters are then read from `attr` and added to the list `par`,
        which includes [coff, cres, loff, lres, hoff, hres]. Finally, the initialized map object is returned.

        Args:
            attr (Attr): An instance of the class `attr_define.Attr` containing attribute values.
            projection (str, optional): The projection type of the map. Defaults to ''.

        Returns:
            self or None: The initialized map object with updated projection and parameters,
                          or None if the projection is invalid or set to 'NONE'.
        """
        p0lat = 0
        p0lon = 0

        if projection == "":
            projection, _, _ = dpg.attr.getAttrValue(attr, "projection", "")
            orig_lat, _, _ = dpg.attr.getAttrValue(attr, "orig_lat", 0.0)
            orig_lon, existsOrig, _ = dpg.attr.getAttrValue(attr, "orig_lon", 0.0)
            p0lat, _, _ = dpg.attr.getAttrValue(attr, "prj_lat", 0.0)
            p0lon, existsp0lon, _ = dpg.attr.getAttrValue(attr, "prj_lon", 0.0)
            if not existsp0lon:
                p0lat = orig_lat
                p0lon = orig_lon
            rangeRes, _, _ = dpg.attr.getAttrValue(attr, "rangeRes", 0.0)
            hres, _, _ = dpg.attr.getAttrValue(attr, "hRes", 0.0)
            if rangeRes > 0.0 or hres > 0.0:
                if existsOrig:
                    p0lat = orig_lat
                    p0lon = orig_lon
                if projection == "":
                    projection = dpg.cfg.getDefaultProjection()
            if projection == "":
                coordfile, _, _ = dpg.attr.getAttrValue(attr, "coordfile", "")
                if coordfile != "":
                    projection = "latlon"

        if projection == "" or projection.upper() == "NONE":
            return None

        shared_map = dpg.map.findSharedMap(
            projection=projection, p0lat=p0lat, p0lon=p0lon
        )
        if shared_map:
            self.mapProj = shared_map.mapProj

        else:
            proj = dpg.map.map_proj_init(attr, proj_name=projection)
            self.mapProj = proj
            dpg.globalVar.GlobalState.SHARED_MAPS.append(self)

        self.p0lon = p0lon
        self.p0lat = p0lat
        self.projection = projection
        self.up_name = projection

        ncols, ncols_exists, _ = dpg.attr.getAttrValue(attr, "ncols", 0)
        if ncols_exists:
            self.dim[0] = ncols
        nlines, nlines_exists, _ = dpg.attr.getAttrValue(attr, "nlines", 0)
        if nlines_exists:
            self.dim[1] = nlines
        coff, coff_exists, _ = dpg.attr.getAttrValue(attr, "coff", float("nan"))
        if coff_exists:
            self.par[0] = coff
        cres, cres_exists, _ = dpg.attr.getAttrValue(attr, "cres", 0.0)
        if cres_exists:
            self.par[1] = cres
        loff, loff_exists, _ = dpg.attr.getAttrValue(attr, "loff", float("nan"))
        if loff_exists:
            self.par[2] = loff
        lres, lres_exists, _ = dpg.attr.getAttrValue(attr, "lres", 0.0)
        if lres_exists:
            self.par[3] = lres
            if self.nPar < 4:
                self.nPar = 4
        hoff, hoff_exists, _ = dpg.attr.getAttrValue(attr, "hoff", 0.0)
        if hoff_exists:
            self.par[4] = hoff
        hres, hres_exists, _ = dpg.attr.getAttrValue(attr, "hres", 0.0)
        if hres_exists:
            self.par[5] = hres
            if self.nPar < 5:
                self.nPar = 6

        return self

    def destroy(self):
        """
        Cleans up and releases resources used by the Map object.

        This method sets the attributes `pAz` and `pEl` to None, effectively releasing
        any resources or references they may hold

        Note:
            This method does not delete the Map object itself, but it prepares it by clearing its internal references
        """
        if self.pAz is not None:
            self.pAz = None
        if self.pEl is not None:
            self.pEl = None

    def cleanUp(self):
        """
        This method invoke destroy. See the destroy for further information.

        NOTE:
            This method will be deprecated in future releases.

        """
        self.destroy()

    def getProperty(
            self,
            par: list = None,
            dim: list = None,
            origin: list = None,
            az_coords: list = None,
            map=None,
            el_coords=None,
    ):
        """
        Return information about: dim, mapProj and par if mapProj is present, None otherwise.

        Args:
            par (list): Parameters list.
            dim (list): Dimension of the map.
            origin (list): List of coordinates for the site origin.
            az_coords (list): Azimuth coordinates.
            map (Map): Specific map.
            el_coords (list): Elevation coordinates.

        Returns:
            None or tuple (list,Proj,list):
                - dim (list): Dimension of the map.
                - mapProj (Proj): PROJ-based coordinate operation.
                - par (list): Parameters list.
        """
        if par is not None:
            if self.nPar >= 4:
                par = self.par[0: self.nPar]

        if dim is not None:
            dim = self.dim

        if origin is not None:
            origin = self.origin

        if az_coords is not None:
            az_coords = self.pAz

        if el_coords is not None:
            el_coords = self.pEl

        if self.mapProj is None:
            return dim, None, par

        mapProj = self.mapProj
        name, proj, p0Lat, p0Lon = dpg.map.getMapName(map)

        return dim, mapProj, par

    def setProperty(self, par: list, az_coords, el_coords, origin: list, dim: list):
        """
        Sets the properties of the Map object.

        This method updates various attributes of the Map object based on the provided
        parameters. It sets the azimuth and elevation coordinates, the origin point, the
        number of parameters, and the dimensions of the map.

        Args:
            par (list):        A list of parameter values to set.
            az_coords:         The azimuth coordinates to set.
            el_coords:         The elevation coordinates to set.
            origin (list):     A list representing the origin point. It should have either 2 or 3 elements.
            dim (list):        A list representing the dimensions of the map. It should have at least 2 elements.

        Note:
            - If the length of the origin list is 2, it sets only the first two elements of the origin attribute.
            - If the length of the origin list is 3, it sets all three elements of the origin attribute.
            - The method ensures that the number of parameters (nPar) is updated based on the length of the `par` list.
            - The dimensions are set to the first two elements of the `dim` list.
        """

        self.pAz = az_coords
        self.pEl = el_coords

        if len(origin) == 3:
            self.origin = origin
        if len(origin) == 2:
            self.origin[0:2] = origin

        nPar = len(par)

        if len(par) >= 4:
            self.nPar = nPar
            self.par[0:nPar] = par

        if len(dim) >= 2:
            self.dim = dim[:2]

        return

    def checkProperty(self, attr: list, par: list = None):
        """
        Ensures that the object has valid and correctly calculated parameters based on current attributes and
        configurations.

        Args:
            attr (list): List of attributes.
            par (list, optional): List of parameters associated with the attributes.

        Returns:
            list or None: The updated list of parameters if modifications are made, otherwise `par` as is.
        """

        if self.par[1] != 0 and self.par[3] != 0:
            return

        if not self.mapProj:
            return

        cfac, _, _ = dpg.attr.getAttrValue(attr, "cfac", 0.0)
        if cfac != 0.0:
            self.par[1] = 65536000000.0 / (7.2 * cfac)
            lfac, _, _ = dpg.attr.getAttrValue(attr, "lfac", 0.0)
            self.par[3] = 65536000000.0 / (7.2 * lfac)
            if par is not None:
                if self.nPar >= 4:
                    par = self.par[0: self.nPar]
            return par

        isotropic, _, _ = dpg.attr.getAttrValue(attr, "isotropic", 0)

        latRange, lonRange, reverse = dpg.map.getLLRange(attr)
        if latRange is None and lonRange is None:
            return par

        y, x = dpg.map.latlon_2_yx(latRange, lonRange, self)

        if reverse:
            y = y[::-1]

        box = [x[0], y[0], x[1], y[1]]
        par = dpg.map.get_par_from_box(box=box, dim=self.dim, isotropic=isotropic)
        self.par[0:3] = par

        return par

    def get_map_info(self):
        """
        Retrieves the projection type and reference point coordinates of the map.

        Returns:
            tuple (str, float, float):
                - The projection type of the map.
                - The latitude of the reference point of the map.
                - The longitude of the reference point of the map.
        """
        return self.projection, self.p0lat, self.p0lon
