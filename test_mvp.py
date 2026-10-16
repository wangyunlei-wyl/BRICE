import os
import argparse
import numpy as np
import h5py
import torch.utils.data as data
from easydict import EasyDict as edict
from utils.mvp_utils import *
from manager import Manager_MVP

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--desc', type=str, default='Training/Testing MVP', help='description')
parser.add_argument('--net_model', type=str, default='model', help='Import module.')
parser.add_argument('--plantpcom_test', dest='plantpcom_test', help='Test neural networks', action='store_true')
parser.add_argument('--inference', dest='inference', help='Inference for benchmark', action='store_true')
parser.add_argument('--output', type=int, default=True, help='Output testing results.')
parser.add_argument('--pretrained', type=str, default='', help='Pretrained path for testing.')
args = parser.parse_args()


# MVP dataset
# input: 2048
# output: 2048
class MVP_CP(data.Dataset):
    def __init__(self, prefix="train"):
        if prefix=="train":
            self.file_path = '/dataset/MVP/MVP_Train_CP.h5'
        elif prefix=="val":
            self.file_path = '/dataset/MVP/MVP_Test_CP.h5'
        elif prefix=="plantpcom_test":
            self.file_path = '/dataset/MVP/MVP_ExtraTest_Shuffled_CP.h5'
        else:
            raise ValueError("ValueError prefix should be [train/val/plantpcom_test] ")

        self.prefix = prefix

        input_file = h5py.File(self.file_path, 'r')
        self.input_data = np.array(input_file['incomplete_pcds'][()])

        print(self.input_data.shape)

        if prefix != "plantpcom_test":
            self.gt_data = np.array(input_file['complete_pcds'][()])
            self.labels = np.array(input_file['labels'][()])
            print(self.gt_data.shape, self.labels.shape)


        input_file.close()
        self.len = self.input_data.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        partial = torch.from_numpy((self.input_data[index]))

        if self.prefix != "plantpcom_test":
            complete = torch.from_numpy((self.gt_data[index // 26]))
            label = (self.labels[index])
            return label, partial, complete
        else:
            return partial


# Configuration for MVP
def MVPConfig():
    __C = edict()
    cfg = __C

    #
    # Dataset Config
    #
    __C.DATASETS = edict()
    __C.DATASETS.COMPLETION3D = edict()
    __C.DATASETS.COMPLETION3D.CATEGORY_FILE_PATH = './datasets/Completion3D.json'
    __C.DATASETS.COMPLETION3D.PARTIAL_POINTS_PATH = '/path/to/datasets/Completion3D/%s/partial/%s/%s.h5'
    __C.DATASETS.COMPLETION3D.COMPLETE_POINTS_PATH = '/path/to/datasets/Completion3D/%s/gt/%s/%s.h5'
    __C.DATASETS.SHAPENET = edict()
    __C.DATASETS.SHAPENET.CATEGORY_FILE_PATH = './datasets/ShapeNet.json'
    __C.DATASETS.SHAPENET.N_RENDERINGS = 8
    __C.DATASETS.SHAPENET.N_POINTS = 2048

    __C.DATASETS.SHAPENET.PARTIAL_POINTS_PATH = '../data/PCN/%s/partial/%s/%s/%02d.pcd'
    __C.DATASETS.SHAPENET.COMPLETE_POINTS_PATH = '../data/PCN/%s/complete/%s/%s.pcd'

    #
    # Dataset
    #
    __C.DATASET = edict()
    # Dataset Options: Completion3D, ShapeNet, ShapeNetCars, Completion3DPCCT
    __C.DATASET.TRAIN_DATASET = 'ShapeNet'
    __C.DATASET.TEST_DATASET = 'ShapeNet'

    #
    # Constants
    #
    __C.CONST = edict()

    __C.CONST.NUM_WORKERS = 8
    __C.CONST.N_INPUT_POINTS = 2048

    #
    # Directories
    #

    __C.DIR = edict()
    # __C.DIR.OUT_PATH = 'results/crapcn_mvp'
    __C.DIR.TEST_PATH = 'results/mvp_test/IC_crapcn_mvp'
    __C.CONST.DEVICE = '0'

    #
    # Network
    #
    __C.NETWORK = edict()
    __C.NETWORK.UPSAMPLE_FACTORS = [2, 2, 1, 8]  # 16384
    __C.NETWORK.KP_EXTENTS = [0.1, 0.1, 0.05, 0.025]  # 16384
    #
    # Train
    #
    __C.TRAIN = edict()
    __C.TRAIN.BATCH_SIZE = 2
    __C.TRAIN.N_EPOCHS = 5
    __C.TRAIN.SAVE_FREQ = 25
    __C.TRAIN.LEARNING_RATE = 0.0001
    __C.TRAIN.LR_MILESTONES = [50, 100, 150, 200, 250]
    __C.TRAIN.LR_DECAY_STEP = 50
    __C.TRAIN.WARMUP_STEPS = 200
    __C.TRAIN.WARMUP_EPOCHS = 20
    __C.TRAIN.GAMMA = .5
    __C.TRAIN.BETAS = (.9, .999)
    __C.TRAIN.WEIGHT_DECAY = 0.001
    __C.TRAIN.LR_DECAY = 150

    #
    # Test
    #
    __C.TEST = edict()
    __C.TEST.METRIC_NAME = 'ChamferDistance'

    return cfg


def test_net(cfg, model, ckpt):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    ########################
    # Load Train/Val Dataset
    ########################

    dataset_test = MVP_CP(prefix='val')
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=32, shuffle=False, num_workers=8)


    # Set up folders for logs and checkpoints
    #cfg.DIR.TEST_PATH = os.path.join(cfg.DIR.TEST_PATH, cfg.DIR.PRETRAIN)
    cfg.DIR.RESULTS = os.path.join(cfg.DIR.TEST_PATH, 'results')
    cfg.DIR.LOGS = cfg.DIR.TEST_PATH
    print('Saving outdir: {}'.format(cfg.DIR.TEST_PATH))
    if not os.path.exists(cfg.DIR.RESULTS):
        os.makedirs(cfg.DIR.RESULTS)

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    # load pretrained model
    print('Recovering from %s ...' % (ckpt))
    checkpoint = torch.load(ckpt)

    model.load_state_dict(checkpoint['model'])

    manager = Manager_MVP(model, cfg)

    manager.test(cfg, model, dataloader_test, outdir=cfg.DIR.RESULTS if args.output else None)
        

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



if __name__ == '__main__':

    # from models.crapcn import CRAPCN_mvp, CRAPCN_mvp_d
    from models.IC_crapcn import IC_crapcn_mvp
    # from models.SnowflakeNet import SnowflakeNet
    # from models.SeedFormer import seedformer_mvp
    # from models.AnchorFormer import AnchorFormer
    # from models.PointAttN_pcn import Model
    # from models.ODGNet import UpTrans_mvp
    # from models.SymmCompletion import SymmCompletion
    from models.IC_odgnet import IC_odgnet_mvp
    # from models.IC_PointAttN import Model
    # from models.IC_SeedFormer import seedformer_mvp
    # from models.IC_SymmCompletion import SymmCompletion
    # from models.IC_AnchorFormer import AnchorFormer

    #######################
    # Prepare Network Model
    #######################

    model = IC_crapcn_mvp()

    ckpt = 'results/IC_crapcn_mvp/train_mvp_Log_2026_03_06_02_49_08/checkpoints/ckpt-best.pth'

    print('cuda available ', torch.cuda.is_available())
    cfg = MVPConfig()
    test_net(cfg, model, ckpt)

