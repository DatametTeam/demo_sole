import numpy as np

import sou_py.dpg as dpg
import sou_py.dpb as dpb
import sou_py.preprocessing as pre


def getTextureTest(
        currScanId,
        linear=False,
        inverse=False,
        up_thresh=None,
        up_spread=None,
        down_thresh=None,
        down_spread=None,
        as_is=False,
):
    """
    Processes data from a given node to compute and return a texture map, with options
    for linear scaling, trapezoidal transformation, and inversion of the result.

    The function retrieves data from a node identified by `currScanId`, applies
    various transformations like texture calculation, optional trapezoidal
    transformation, and inversion. If `as_is` is set to True, the function skips
    further processing after the texture computation and returns the raw texture.

    Args:
        currScanId (Node): Identifier for the scan or node from which data will
            be retrieved.
        linear (bool, optional): If True, the data is processed linearly. Defaults to False.
        inverse (bool, optional): If True, inverts the final data (1 - data).
            Defaults to False.
        up_thresh (float, optional): Upper threshold for the trapezoidal transformation.
            If None, the default threshold will be used.
        up_spread (float, optional): Spread value for the upper threshold in the
            trapezoidal transformation.
        down_thresh (float, optional): Lower threshold for the trapezoidal transformation.
            If None, the default threshold will be used.
        down_spread (float, optional): Spread value for the lower threshold in the
            trapezoidal transformation.
        as_is (bool, optional): If True, returns the texture data as computed,
            without applying further transformations like the trapezoidal
            or inversion steps. Defaults to False.

    Returns:
        numpy.ndarray or int: Returns the processed texture data as a numpy array.
        If the input data has dimensions less than or equal to 1, or invalid,
        it returns 0.
    """
    minVal = 0.0
    data, minVal, _ = dpb.dpb.get_data(
        currScanId, numeric=True, linear=linear, getMinMaxVal=True
    )
    dim = data.shape
    if len(dim) <= 1:
        return 0
    if dim[1] <= 1:
        return 0

    data = dpg.prcs.texture(data, minVal=minVal)

    if as_is:
        return data

    data = dpg.prcs.trapez(data, down_thresh, up_thresh, down_spread, up_spread)

    if inverse:
        data = 1 - data

    return data

    pass


def texturetest(prodId, attr=None, moment=None, subtract=False, no_update=False):
    """
    Processes radar volume data and applies texture-based transformations to each scan, with options
    for updating, subtracting, and handling the transformed output.

    Args:
        prodId (Node): radar data to be processed.
        attr (str, optional): Attribute used to find to Remove parameter. Default is None.
        moment (str, optional): Moment or specific time to consider. Defaults to None, in which case it retrieves the
                                default "measure" value.
        subtract (bool, optional): If True, subtracts the computed coefficient from the quality values. Defaults to
        False.
        no_update (bool, optional): If True, the function will skip updating the data with new values
                                    and return the raw texture maps. Defaults to False.

    Returns:
        None: If no scans are found, the function will return early without modification.
    """
    max_el, _ = dpb.dpb.get_par(prodId, "max_el", default=10.0)
    if max_el < 0:
        return

    up_thresh, _ = dpb.dpb.get_par(prodId, "up_thresh", default=10.0)
    up_spread, _ = dpb.dpb.get_par(prodId, "up_spread", default=10.0)
    down_thresh, _ = dpb.dpb.get_par(prodId, "down_thresh", default=0.0)
    down_spread, _ = dpb.dpb.get_par(prodId, "down_spread", default=0.0)
    linear, _ = dpb.dpb.get_par(prodId, "linear", default=0)
    inverse, _ = dpb.dpb.get_par(prodId, "inverse", default=0)

    if moment is None:
        moment, _ = dpb.dpb.get_par(prodId, "moment", default="")
    if moment == "":
        moment, _ = dpb.dpb.get_par(prodId, "measure", default="CZ")

    toRemove, _ = dpb.dpb.get_par(prodId, "toRemove", default=1, attr=attr)

    volId = dpb.dpb.get_volumes(prodId, moment=moment)
    scans_dict = dpb.dpb.get_scans(volId, max_el=max_el)
    scans = scans_dict["scans"]
    scan_dim = scans_dict["scan_dim"]
    coord_set = scans_dict["coord_set"]

    if len(scans) <= 0:
        return

    dim = [len(scans)] + scan_dim

    outPointer, par, _ = dpg.radar.check_in(
        node_in=scans[0], node_out=prodId, dim=dim, type=4, filename="volume.dat"
    )
    if outPointer is None:
        return

    nEl = dim[-1] * dim[-2]
    maxVal = 100.0

    out = np.array(())
    for sss, scan in enumerate(scans):
        textureMap = getTextureTest(
            scan,
            linear=linear,
            inverse=inverse,
            up_thresh=up_thresh,
            up_spread=up_spread,
            down_thresh=down_thresh,
            down_spread=down_spread,
            as_is=no_update,
        )

        if len(textureMap.shape) > 1:
            if not no_update:
                textureMap *= maxVal
                updated = pre.quality.quality(
                    prodId=prodId,
                    update=True,
                    test_name="TextureTest" + moment,
                    maxVal=maxVal,
                    elevation=coord_set[sss],
                    subtract=subtract,
                )
                if toRemove and updated == 0:
                    del outPointer
                    return
            if no_update or not toRemove:
                out = dpg.warp.warp_map(
                    scan, prodId, source_data=textureMap, regular=True
                )  # Map transformation function that remaps data from a source map to a destination map.

        if np.size(out) == nEl:
            outPointer[sss, :, :] = out
        else:
            outPointer[sss, :, :] = np.nan

    if not no_update and toRemove == 1:
        del outPointer
        return

    dpg.radar.check_out(outId=prodId, pointer=outPointer, par=par, el_coords=coord_set)  # update a series of
    # information on the ode prodId

    return
