import os.path

import sou_py.dpg as dpg
import sou_py.dpb as dpb


def radargauge(prodId, name, update=False, do_all=False, maxBox=None, medianBox=None, optim=None, replicate=None,
               split=None):
    if update:
        if do_all:
            nodes = dpg.tree.findAllDescendant(dpg.tree.getRoot(prodId), name)
            for sss in nodes:
                if sss != prodId:  # TODO: controllare
                    prefix = dpg.tree.getNodeName(sss.parent.name)
                    suffix, prefix, _ = dpg.radar.get_par(prodId, 'suffix', '')
                    path = name + suffix
                    radargauge_update(sss, path, maxBox=maxBox, medianBox=medianBox)
            return
        if not os.path.isdir('/data1/SENSORS/SUMMARY/' + name):  # TODO: mimmo?
            return
        date, time, _ = dpg.times.get_time(prodId)
        time, _, mm = dpg.times.checkTime(time)
        if mm != 0:
            return
        radargauge_update(prodId, name, optim=optim, maxBox=maxBox, medianBox=medianBox, replicate=replicate, split=split)
        return

    # TODO: da finire ma non ora

