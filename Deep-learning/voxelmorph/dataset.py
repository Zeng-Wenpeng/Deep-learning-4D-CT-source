

import numpy as np
from torch.utils.data import Dataset

from generators import *
import voxelmorph as vxm  # nopep8


class AnatomicalDataset(Dataset):
    def __init__(self, idxes, atlas, add_feat_axis, seg_supervised=False, bidir=False):
        self.idxes = idxes
        self.atlas = atlas
        self.bidir = bidir
        self.add_feat_axis = add_feat_axis
        self.seg_supervised = seg_supervised

    def __getitem__(self, item):
        """
        Generator for scan-to-atlas registration.

        TODO: This could be merged into scan_to_scan() by adding an optional atlas
        argument like in semisupervised().

        Parameters:
            vol_names: List of volume files to load, or list of preloaded volumes.
            atlas: Atlas volume data.
            bidir: Yield input image as output for bidirectional models. Default is False.
            batch_size: Batch size. Default is 1.
            no_warp: Excludes null warp in output list if set to True (for affine training).
                Default is False.
            segs: Load segmentations as output, for supervised training. Forwarded to the
                internal volgen generator. Default is None.
            kwargs: Forwarded to the internal volgen generator.
        """

        atlas = vxm.py.utils.load_volfile(self.atlas, np_var='vol',
                                          add_batch_axis=True, add_feat_axis=self.add_feat_axis)
        scan = vxm.py.utils.load_volfile(self.idxes[item], np_var='vol',
                                  add_batch_axis=True, add_feat_axis=self.add_feat_axis)
        if not self.seg_supervised:
            outvols = [atlas, scan] if self.bidir else [atlas]
        else:
            seg = res[1]
            outvols = [seg, scan] if self.bidir else [seg]
        if not no_warp:
            outvols.append(zeros)
        return (invols, outvols)

