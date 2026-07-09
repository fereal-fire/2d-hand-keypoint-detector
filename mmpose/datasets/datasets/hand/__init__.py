# Copyright (c) OpenMMLab. All rights reserved.
from .freihand_dataset import FreiHandDataset
from .hand_coco_wholebody_dataset import HandCocoWholeBodyDataset
from .interhand2d_dataset import InterHand2DDataset
from .interhand3d_dataset import InterHand3DDataset
from .onehand10k_dataset import OneHand10KDataset
from .panoptic_hand2d_dataset import PanopticDataset
from .rhd2d_dataset import Rhd2DDataset
from .synthmocap_hand2d_dataset import SynthMocapHandDataset

from .dexs0_dataset import Dexs0HandDataset
from .freihand_hamer_dataset import FreihandHamerHandDataset
from .halpe_dataset import HalpeHandDataset
from .interhand26m_dataset import InterHand26MDataset
from .mtc_dataset import MTCHandDataset
from .h2o3d_dataset import H2O3DHandDataset
from .ho3d_dataset import HO3DHandDataset
from .mpiinzsl_dataset import MPIINZSLHandDataset
from .rhd_dataset import RHDHandDataset
from .mtc_dataset import MTCHandDataset

__all__ = [
    'FreiHandDataset', 'InterHand2DDataset', 'InterHand3DDataset',
    'OneHand10KDataset', 'PanopticDataset', 'Rhd2DDataset',
    'HandCocoWholeBodyDataset', 'SynthMocapHandDataset', 'Dexs0HandDataset',
    'FreihandHamerHandDataset', 'HalpeHandDataset', 'InterHand26MDataset',
    'MTCHandDataset', 'H2O3DHandDataset', 'HO3DHandDataset', 
    'MPIINZSLHandDataset', 'RHDHandDataset', 'MTCHandDataset'
]
