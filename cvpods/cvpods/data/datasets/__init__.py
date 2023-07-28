#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2019-2021 Megvii Inc. All rights reserved.

from .bdd100k import Bdd100kDataset
from .city import CityDataset
from .city_7cls import City7clsDataset
from .city_car import CityCarDataset
from .citypersons import CityPersonsDataset
from .cityscapes import CityScapesDataset
from .coco import COCODataset
from .crowdhuman import CrowdHumanDataset
from .foggy import FoggyDataset
from .imagenet import ImageNetDataset
from .imagenetlt import ImageNetLTDataset
from .kitti import KITTIDataset
from .lvis import LVISDataset
from .objects365 import Objects365Dataset
from .sim10k import Sim10kDataset
from .torchvision_datasets import CIFAR10Dataset, STL10Datasets
from .voc import VOCDataset
from .widerface import WiderFaceDataset

__all__ = [k for k in globals().keys() if not k.startswith("_")]
