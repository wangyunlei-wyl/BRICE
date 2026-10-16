import os
import argparse
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
TRAIN_NAME = os.path.splitext(os.path.basename(__file__))[0]



#  Arguments 
parser = argparse.ArgumentParser()
parser.add_argument('--desc', type=str, default='Training/Testing CRA-PCN', help='description')
parser.add_argument('--net_model', type=str, default='model', help='Import module.')
parser.add_argument('--plantpcom_test', dest='plantpcom_test', help='Test neural networks', action='store_true')
parser.add_argument('--inference', dest='inference', help='Inference for benchmark', action='store_true')
parser.add_argument('--output', type=int, default=True, help='Output testing results.')
parser.add_argument('--pretrained', type=str, default='', help='Pretrained path for testing.')
args = parser.parse_args()

# Configuration for PCN
def PCNConfig():

    __C                                              = edict()
    cfg                                              = __C

    #
    # Dataset Config
    #
    __C.DATASETS                                         = edict()
    __C.DATASETS.PLANT3D = edict()
    __C.DATASETS.PLANT3D.JSON_PATH                       = "dataset/PlantPCom/Plant3d_com.json"
    __C.DATASETS.PLANT3D.PARTIAL_POINTS_PATH             = "dataset/PlantPCom/%s/partial/%s/%s/%s_%02d.pcd"
    __C.DATASETS.PLANT3D.COMPLETE_POINTS_PATH            = "dataset/PlantPCom/%s/complete/%s/%s.pcd"
    __C.DATASETS.PLANT3D.N_RENDERINGS = 20

    # ================= Dataset 选择 =================
    __C.DATASET = edict()
    __C.DATASET.TRAIN_DATASET                            = 'Plant3D'
    __C.DATASET.TEST_DATASET                             = 'Plant3D'
    __C.DATASET.VAL_DATASET                             = 'Plant3D'

    #
    # Constants
    #
    __C.CONST                                        = edict()

    __C.CONST.NUM_WORKERS                            = 8
    __C.CONST.N_INPUT_POINTS                         = 2048

    #
    # Directories
    #

    __C.DIR                                          = edict()
    __C.DIR.OUT_PATH                                 = None
    __C.DIR.TEST_PATH                                = 'results/plantpcom'
    __C.CONST.DEVICE                                 = '1'
    __C.CONST.WEIGHTS                                = '/checkpoints/ckpt-best.pth' # 'ckpt-best.pth'  # specify a path to run plantpcom_test and inference


    #
    # Network
    #
    __C.NETWORK                                      = edict()
    __C.NETWORK.UPSAMPLE_FACTORS                     = [1, 2, 4, 8] # 16384
    __C.NETWORK.KP_EXTENTS                           = [0.1, 0.1, 0.05, 0.025] # 16384

    # Train
    #
    __C.TRAIN                                        = edict()
    __C.TRAIN.BATCH_SIZE                             = 1
    __C.TRAIN.N_EPOCHS                               = 400
    __C.TRAIN.SAVE_FREQ                              = 25
    __C.TRAIN.LEARNING_RATE                          = 0.001
    __C.TRAIN.LR_MILESTONES                          = [50, 100, 150, 200, 250]
    __C.TRAIN.LR_DECAY_STEP                          = 50
    __C.TRAIN.WARMUP_STEPS                           = 200
    __C.TRAIN.WARMUP_EPOCHS                          = 20
    __C.TRAIN.GAMMA                                  = .5
    __C.TRAIN.BETAS                                  = (.9, .999)
    __C.TRAIN.WEIGHT_DECAY                           = 0
    __C.TRAIN.LR_DECAY                               = 150

    #
    # Test
    #
    __C.TEST                                         = edict()
    __C.TEST.METRIC_NAME                             = 'ChamferDistance'


    return cfg



def test_net(cfg):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    ########################
    # Load Train/Val Dataset
    ########################

    val_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.VAL_DATASET](cfg)

    val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset_loader.get_dataset(
                                                    utils.data_loaders.DatasetSubset.VAL),
                                                  batch_size=1,
                                                  num_workers=cfg.CONST.NUM_WORKERS,
                                                  collate_fn=utils.data_loaders.collate_fn,
                                                  pin_memory=True,
                                                  shuffle=False)

    # Set up folders for logs and checkpoints
    #cfg.DIR.TEST_PATH = os.path.join(cfg.DIR.TEST_PATH, cfg.DIR.PRETRAIN)
    cfg.DIR.RESULTS = os.path.join(cfg.DIR.TEST_PATH, 'results')
    cfg.DIR.LOGS = cfg.DIR.TEST_PATH
    print('Saving outdir: {}'.format(cfg.DIR.TEST_PATH))
    if not os.path.exists(cfg.DIR.RESULTS):
        os.makedirs(cfg.DIR.RESULTS)


    #######################
    # Prepare Network Model
    #######################

    # model = CRAPCN() # CRAPCN_d
    model = IC_crapcn_pcom()
    # model = Model('pcn')
    # model = seedformer_pcn()
    # model = IC_odgnet_pcn()
    # model = UpTrans_pcn()
    # model = SymmCompletion()
    # model = SnowflakeNet()
    # model = AnchorFormer()
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    # load pretrained model
    print('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
    checkpoint = torch.load(cfg.CONST.WEIGHTS)

    model.load_state_dict(checkpoint['model'])

    manager = Manager(model, cfg)

    manager.test_pcom(cfg, model, val_data_loader, outdir=cfg.DIR.RESULTS if args.output else None)
        

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    from models.crapcn import CRAPCN, CRAPCN_d
    from models.IC_crapcn import IC_crapcn_pcom
    # from models.IC_crapcn_d import IC_crapcn_pcom
    # from models.SeedFormer import seedformer_pcn
    # from models.PointAttN import Model
    # from models.IC_PointAttN import Model
    # from models.IC_SeedFormer import seedformer_pcn
    # from models.IC_odgnet import IC_odgnet_pcn
    from models.ODGNet import UpTrans_pcn
    # from models.SymmCompletion import SymmCompletion
    from models.IC_SymmCompletion import SymmCompletion
    # from models.IC_SnowflakeNet import SnowflakeNet
    # from models.SnowflakeNet import SnowflakeNet
    # from models.AnchorFormer import AnchorFormer
    from models.IC_AnchorFormer import AnchorFormer

    print('cuda available ', torch.cuda.is_available())
    cfg = PCNConfig()
    test_net(cfg)
