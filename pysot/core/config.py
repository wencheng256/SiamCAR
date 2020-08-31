# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from yacs.config import CfgNode as CN

__C = CN()

cfg = __C

__C.META_ARC = "siamcar_r50"

__C.CUDA = True

# ------------------------------------------------------------------------ #
# Training options
# ------------------------------------------------------------------------ #
__C.TRAIN = CN()

# Anchor Target
__C.TRAIN.EXEMPLAR_SIZE = 127

__C.TRAIN.SEARCH_SIZE = 255

__C.TRAIN.OUTPUT_SIZE = 25

__C.TRAIN.RESUME = ''

__C.TRAIN.PRETRAINED = ''

__C.TRAIN.LOG_DIR = './logs'

__C.TRAIN.SNAPSHOT_DIR = './snapshot'

__C.TRAIN.EPOCH = 20

__C.TRAIN.START_EPOCH = 0

__C.TRAIN.BATCH_SIZE = 32

__C.TRAIN.NUM_WORKERS = 1

__C.TRAIN.MOMENTUM = 0.9

__C.TRAIN.WEIGHT_DECAY = 0.0001

__C.TRAIN.CLS_WEIGHT = 1.0

__C.TRAIN.LOC_WEIGHT = 2.0

__C.TRAIN.CEN_WEIGHT = 1.0

__C.TRAIN.PRINT_FREQ = 20

__C.TRAIN.LOG_GRADS = False

__C.TRAIN.GRAD_CLIP = 10.0

__C.TRAIN.BASE_LR = 0.005

__C.TRAIN.LR = CN()

__C.TRAIN.LR.TYPE = 'log'

__C.TRAIN.LR.KWARGS = CN(new_allowed=True)

__C.TRAIN.LR_WARMUP = CN()

__C.TRAIN.LR_WARMUP.WARMUP = True

__C.TRAIN.LR_WARMUP.TYPE = 'step'

__C.TRAIN.LR_WARMUP.EPOCH = 5

__C.TRAIN.LR_WARMUP.KWARGS = CN(new_allowed=True)

__C.TRAIN.NUM_CLASSES = 2

__C.TRAIN.NUM_CONVS = 4

__C.TRAIN.PRIOR_PROB = 0.01

__C.TRAIN.LOSS_ALPHA = 0.25

__C.TRAIN.LOSS_GAMMA = 2.0

# ------------------------------------------------------------------------ #
# Dataset options
# ------------------------------------------------------------------------ #
__C.DATASET = CN(new_allowed=True)

# Augmentation
# for template
__C.DATASET.TEMPLATE = CN()

# for detail discussion
__C.DATASET.TEMPLATE.SHIFT = 4

__C.DATASET.TEMPLATE.SCALE = 0.05

__C.DATASET.TEMPLATE.BLUR = 0.0

__C.DATASET.TEMPLATE.FLIP = 0.0

__C.DATASET.TEMPLATE.COLOR = 1.0

__C.DATASET.SEARCH = CN()

__C.DATASET.SEARCH.SHIFT = 64

__C.DATASET.SEARCH.SCALE = 0.18
# __C.DATASET.SEARCH.SCALE = 0

__C.DATASET.SEARCH.BLUR = 0.0

__C.DATASET.SEARCH.FLIP = 0.0

__C.DATASET.SEARCH.COLOR = 1.0

# for detail discussion
__C.DATASET.NEG = 0.0

__C.DATASET.GRAY = 0.0

__C.DATASET.NAMES = ('VID', 'COCO', 'DET', 'YOUTUBEBB')

__C.DATASET.VID = CN()
__C.DATASET.VID.ROOT = 'vid/crop511'
__C.DATASET.VID.ANNO = 'vid/train.json'
__C.DATASET.VID.FRAME_RANGE = 200
__C.DATASET.VID.NUM_USE = 200000  # repeat until reach NUM_USE

__C.DATASET.GOT = CN()
__C.DATASET.GOT.ROOT = 'got/crop511'
__C.DATASET.GOT.ANNO = 'got/train.json'
__C.DATASET.GOT.FRAME_RANGE = 300
__C.DATASET.GOT.NUM_USE = 200000

__C.DATASET.TEMPLE = CN()
__C.DATASET.TEMPLE.ROOT = 'Temple/crop511'
__C.DATASET.TEMPLE.ANNO = 'Temple/train.json'
__C.DATASET.TEMPLE.FRAME_RANGE = 200
__C.DATASET.TEMPLE.NUM_USE = 50000


__C.DATASET.OTB = CN()
__C.DATASET.OTB.ROOT = 'otb/crop511'
__C.DATASET.OTB.ANNO = 'otb/train.json'
__C.DATASET.OTB.FRAME_RANGE = 200
__C.DATASET.OTB.NUM_USE = 5000


__C.DATASET.LASOT = CN()
__C.DATASET.LASOT.ROOT = 'Lasot/crop511'
__C.DATASET.LASOT.ANNO = 'Lasot/train.json'
__C.DATASET.LASOT.FRAME_RANGE = 800
__C.DATASET.LASOT.NUM_USE = 400000

__C.DATASET.YOUTUBEBB = CN()
__C.DATASET.YOUTUBEBB.ROOT = 'yt_bb/crop511'
__C.DATASET.YOUTUBEBB.ANNO = 'yt_bb/train.json'
__C.DATASET.YOUTUBEBB.FRAME_RANGE = 20
__C.DATASET.YOUTUBEBB.NUM_USE = 200000  # use all not repeat

__C.DATASET.COCO = CN()
__C.DATASET.COCO.ROOT = 'coco/crop511'
__C.DATASET.COCO.ANNO = 'coco/train2017.json'
__C.DATASET.COCO.FRAME_RANGE = 1
__C.DATASET.COCO.NUM_USE = 200000

__C.DATASET.DET = CN()
__C.DATASET.DET.ROOT = 'det/crop511'
__C.DATASET.DET.ANNO = 'det/train.json'
__C.DATASET.DET.FRAME_RANGE = 1
__C.DATASET.DET.NUM_USE = 200000

__C.DATASET.FT1 = CN()
__C.DATASET.FT1.ROOT = 'finetune1/crop511'
__C.DATASET.FT1.ANNO = 'finetune1/train.json'
__C.DATASET.FT1.FRAME_RANGE = 1
__C.DATASET.FT1.NUM_USE = 30000

__C.DATASET.FT2 = CN()
__C.DATASET.FT2.ROOT = 'finetune2/crop511'
__C.DATASET.FT2.ANNO = 'finetune2/train.json'
__C.DATASET.FT2.FRAME_RANGE = 200
__C.DATASET.FT2.NUM_USE = 50000

__C.DATASET.VOT = CN()
__C.DATASET.VOT.ROOT = 'VOT2016/crop511'
__C.DATASET.VOT.ANNO = 'VOT2016/train.json'
__C.DATASET.VOT.FRAME_RANGE = 1
__C.DATASET.VOT.NUM_USE = 30000

__C.DATASET.FT = CN()
__C.DATASET.FT.ROOT = 'finetune/crop511'
__C.DATASET.FT.ANNO = 'finetune/train.json'
__C.DATASET.FT.FRAME_RANGE = 1
__C.DATASET.FT.NUM_USE = 30000

__C.DATASET.VOTFULL = CN()
__C.DATASET.VOTFULL.ROOT = 'VOT_FULL/crop511'
__C.DATASET.VOTFULL.ANNO = 'VOT_FULL/train.json'
__C.DATASET.VOTFULL.FRAME_RANGE = 200
__C.DATASET.VOTFULL.NUM_USE = -1

__C.DATASET.VOT2018 = CN()
__C.DATASET.VOT2018.ROOT = 'VOT2018/crop511'
__C.DATASET.VOT2018.ANNO = 'VOT2018/train.json'
__C.DATASET.VOT2018.FRAME_RANGE = 1
__C.DATASET.VOT2018.NUM_USE = 30000

__C.DATASET.VIDEOS_PER_EPOCH = 600000 #600000
# ------------------------------------------------------------------------ #
# Backbone options
# ------------------------------------------------------------------------ #
__C.BACKBONE = CN()

# Backbone type, current only support resnet18,34,50;alexnet;mobilenet
__C.BACKBONE.TYPE = 'res50'

__C.BACKBONE.KWARGS = CN(new_allowed=True)

# Pretrained backbone weights
__C.BACKBONE.PRETRAINED = ''

# Train layers
__C.BACKBONE.TRAIN_LAYERS = ['layer2', 'layer3', 'layer4']

# Layer LR
__C.BACKBONE.LAYERS_LR = 0.1

# Switch to train layer
__C.BACKBONE.TRAIN_EPOCH = 10

# ------------------------------------------------------------------------ #
# Adjust layer options
# ------------------------------------------------------------------------ #
__C.ADJUST = CN()

# Adjust layer
__C.ADJUST.ADJUST = True

__C.ADJUST.KWARGS = CN(new_allowed=True)

# Adjust layer type
__C.ADJUST.TYPE = "AdjustAllLayer"

# ------------------------------------------------------------------------ #
# RPN options
# ------------------------------------------------------------------------ #
__C.CAR = CN()

# RPN type
__C.CAR.TYPE = 'MultiCAR'

__C.CAR.KWARGS = CN(new_allowed=True)

# ------------------------------------------------------------------------ #
# Tracker options
# ------------------------------------------------------------------------ #
__C.TRACK = CN()

__C.TRACK.TYPE = 'SiamCARTracker'

# Scale penalty
__C.TRACK.PENALTY_K = 0.04

# Window influence
__C.TRACK.WINDOW_INFLUENCE = 0.44

# Interpolation learning rate
__C.TRACK.LR = 0.4

# Exemplar size
__C.TRACK.EXEMPLAR_SIZE = 127

# Instance size
__C.TRACK.INSTANCE_SIZE = 255

# Context amount
__C.TRACK.CONTEXT_AMOUNT = 0.5

__C.TRACK.STRIDE = 8


__C.TRACK.SCORE_SIZE = 25

__C.TRACK.hanming = True

__C.TRACK.NUM_K = 2

__C.TRACK.NUM_N = 1

__C.TRACK.REGION_S = 0.1

__C.TRACK.REGION_L = 0.44


# ------------------------------------------------------------------------ #
# HP_SEARCH parameters
# ------------------------------------------------------------------------ #
__C.HP_SEARCH = CN()

__C.HP_SEARCH.OTB50 = [0.15, 0.1, 0.4]

__C.HP_SEARCH.GOT10K = [0.6, 0.04, 0.1]

__C.HP_SEARCH.UAV123 = [0.39, 0.04, 0.37]

__C.HP_SEARCH.LaSOT = [0.33, 0.04, 0.4]
