import os
import argparse
import os
import numpy as np
import torch
import json
import time
import utils.data_loaders
from easydict import EasyDict as edict
from importlib import import_module
from pprint import pprint
from manager import Manager

import math
from models.crapcn import CRAPCN, CRAPCN_d
# from models.IC_crapcn import IC_crapcn_pcn
from models.IC_crapcn_d import IC_crapcn_pcn
# from models.SeedFormer import seedformer_pcn
# from models.IC_SeedFormer import seedformer_pcn
# from models.PointAttN import Model
# from models.IC_PointAttN import Model
# from models.ODGNet import UpTrans_pcn
# from models.IC_SnowflakeNet import SnowflakeNet
# from models.SnowflakeNet import SnowflakeNet
# from models.AnchorFormer import AnchorFormer
# from models.IC_AnchorFormer import AnchorFormer
# from models.IC_SymmCompletion import SymmCompletion
# from models.SymmCompletion import SymmCompletion
TRAIN_NAME = os.path.splitext(os.path.basename(__file__))[0]


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--desc', type=str, default='Training/Testing CRA-PCN', help='description')
parser.add_argument('--net_model', type=str, default='model', help='Import module.')
parser.add_argument('--plantpcom_test', dest='plantpcom_test', help='Test neural networks', action='store_true')
parser.add_argument('--inference', dest='inference', help='Inference for benchmark', action='store_true')
parser.add_argument('--output', type=int, default=True, help='Output testing results.')
parser.add_argument('--pretrained', type=str, default='', help='Pretrained path for testing.')
args = parser.parse_args()

def PlantConfig():
    __C = edict()
    cfg = __C

    # ================= Dataset =================
    __C.DATASETS = edict()

    __C.DATASETS.PLANT3D = edict()
    __C.DATASETS.PLANT3D.JSON_PATH                       = "dataset/PlantPCom/Plant3d_com.json"
    __C.DATASETS.PLANT3D.PARTIAL_POINTS_PATH             = "dataset/PlantPCom/%s/partial/%s/%s/%s_%02d.pcd"
    __C.DATASETS.PLANT3D.COMPLETE_POINTS_PATH            = "dataset/PlantPCom/%s/complete/%s/%s.pcd"
    __C.DATASETS.PLANT3D.N_RENDERINGS = 20


    __C.DATASET = edict()
    __C.DATASET.TRAIN_DATASET                            = 'Plant3D'
    __C.DATASET.TEST_DATASET                             = 'Plant3D'
    __C.DATASET.VAL_DATASET                              = 'Plant3D'


    __C.CONST = edict()
    __C.CONST.NUM_WORKERS                                = 8
    __C.CONST.N_INPUT_POINTS                             = 2048
    __C.CONST.DEVICE                                     = '0'


    __C.DIR = edict()
    __C.DIR.OUT_PATH                                     = 'results/crapcn_plantpcom'
    __C.DIR.TEST_PATH                                    = 'plantpcom_test/plantpcom'


    __C.NETWORK = edict()
    __C.NETWORK.UPSAMPLE_FACTORS                         = [2, 2, 1, 8]


    __C.TRAIN                                        = edict()
    __C.TRAIN.BATCH_SIZE                             = 32
    __C.TRAIN.N_EPOCHS                               = 100
    __C.TRAIN.SAVE_FREQ                              = 25
    __C.TRAIN.LEARNING_RATE                          = 0.001
    __C.TRAIN.LR_MILESTONES                          = [50, 100, 150, 200, 250]
    __C.TRAIN.LR_DECAY_STEP                          = 50
    __C.TRAIN.WARMUP_STEPS                           = 200
    __C.TRAIN.WARMUP_EPOCHS                          = 20
    __C.TRAIN.GAMMA                                  = .5
    __C.TRAIN.BETAS                                  = (.9, .999)
    __C.TRAIN.WEIGHT_DECAY                           = 0.001
    __C.TRAIN.LR_DECAY                               = 150

    # ================= 测试 =================
    __C.TEST = edict()
    __C.TEST.METRIC_NAME                                 = 'ChamferDistance'

    return cfg


def train_net(cfg):
    torch.backends.cudnn.benchmark = True


    train_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](cfg)
    val_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.VAL_DATASET](cfg)

    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset_loader.get_dataset(
        utils.data_loaders.DatasetSubset.TRAIN),
                                                    batch_size=cfg.TRAIN.BATCH_SIZE,
                                                    num_workers=cfg.CONST.NUM_WORKERS,
                                                    collate_fn=utils.data_loaders.collate_fn,
                                                    pin_memory=True,
                                                    shuffle=True,
                                                    drop_last=False)
    val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset_loader.get_dataset(
        utils.data_loaders.DatasetSubset.VAL),
                                                  batch_size=cfg.TRAIN.BATCH_SIZE,
                                                  num_workers=cfg.CONST.NUM_WORKERS//2,
                                                  collate_fn=utils.data_loaders.collate_fn,
                                                  pin_memory=True,
                                                  shuffle=False)

    # Set up folders for logs and checkpoints
    timestr = time.strftime('_Log_%Y_%m_%d_%H_%M_%S', time.gmtime())
    cfg.DIR.OUT_PATH = os.path.join(cfg.DIR.OUT_PATH, TRAIN_NAME+timestr)
    cfg.DIR.CHECKPOINTS = os.path.join(cfg.DIR.OUT_PATH, 'checkpoints')
    cfg.DIR.LOGS = cfg.DIR.OUT_PATH
    print('Saving outdir: {}'.format(cfg.DIR.OUT_PATH))
    if not os.path.exists(cfg.DIR.CHECKPOINTS):
        os.makedirs(cfg.DIR.CHECKPOINTS)

    # save config file
    pprint(cfg)
    config_filename = os.path.join(cfg.DIR.LOGS, 'config.json')
    with open(config_filename, 'w') as file:
        json.dump(cfg, file, indent=4, sort_keys=True)

    # Save Arguments
    torch.save(args, os.path.join(cfg.DIR.LOGS, 'args_training.pth'))

    model = CRAPCN() # or 'CRAPCN_d'
    # model = IC_crapcn_pcn()
    # model = seedformer_pcn()
    # model = Model(dataset='pcn')
    # model = UpTrans_pcn()
    # model = IC_odgnet_pcn()
    # model = SnowflakeNet()
    # model = AnchorFormer()
    # model = SymmCompletion()
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    # load existing model
    if 'WEIGHTS' in cfg.CONST:
        print('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        model.load_state_dict(checkpoint['model'])
        print('Recover complete. Current epoch = #%d; best metrics = %s.' % (checkpoint['epoch_index'], checkpoint['best_metrics']))


    ##################
    # Training Manager
    ##################

    manager = Manager(model, cfg)

    # Start training
    manager.train(model, train_data_loader, val_data_loader, cfg)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    # Check python version
    seed = 1128
    set_seed(seed)
    
    print('cuda available ', torch.cuda.is_available())
    cfg = PlantConfig()
    train_net(cfg)