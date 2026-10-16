import json
import logging
import numpy as np
import random
import torch.utils.data.dataset
import open3d as o3d
import utils.data_transforms
from enum import Enum, unique
from tqdm import tqdm
from utils.io import IO
import os
import torch.utils.data as data


label_mapping = {
    3: '03001627',
    6: '04379243',
    5: '04256520',
    1: '02933112',
    4: '03636649',
    2: '02958343',
    0: '02691156',
    7: '04530566'
}


@unique
class DatasetSubset(Enum):
    TRAIN = 0
    TEST = 1
    VAL = 2


def collate_fn(batch):
    taxonomy_ids = []
    model_ids = []
    data = {}

    for sample in batch:
        taxonomy_ids.append(sample[0])
        model_ids.append(sample[1])
        _data = sample[2]
        for k, v in _data.items():
            if k not in data:
                data[k] = []
            data[k].append(v)

    for k, v in data.items():
        data[k] = torch.stack(v, 0)

    return taxonomy_ids, model_ids, data


code_mapping = {
    'plane': '02691156',
    'cabinet': '02933112',
    'car': '02958343',
    'chair': '03001627',
    'lamp': '03636649',
    'couch': '04256520',
    'table': '04379243',
    'watercraft': '04530566',
}


def read_ply(file_path):
    pc = o3d.io.read_point_cloud(file_path)
    ptcloud = np.array(pc.points)
    return ptcloud


class Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, options, file_list, transforms=None):
        self.options = options
        self.file_list = file_list
        self.transforms = transforms
        self.cache = dict()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = {}
        rand_idx = -1

        # random select one sample per shape for training
        if 'n_renderings' in self.options:
            rand_idx = random.randint(0, self.options['n_renderings'] -
                                      1) if self.options['shuffle'] else 0

        # load required data
        for ri in self.options['required_items']:
            file_path = sample['%s_path' % ri]
            if type(file_path) == list:
                file_path = file_path[rand_idx]
            # print(file_path)
            data[ri] = IO.get(file_path).astype(np.float32)

        # apply transforms
        if self.transforms is not None:
            data = self.transforms(data)

        return sample['taxonomy_id'], sample['model_id'], data


class ShapeNetDataLoader(object):
    def __init__(self, cfg):
        self.cfg = cfg

        # Load the dataset indexing file
        self.dataset_categories = []
        with open(cfg.DATASETS.SHAPENET.CATEGORY_FILE_PATH) as f:
            self.dataset_categories = json.loads(f.read())

    def get_dataset(self, subset):
        n_renderings = self.cfg.DATASETS.SHAPENET.N_RENDERINGS if subset == DatasetSubset.TRAIN else 1
        file_list = self._get_file_list(self.cfg, self._get_subset(subset), n_renderings)
        transforms = self._get_transforms(self.cfg, subset)
        return Dataset({
            'n_renderings': n_renderings,
            'required_items': ['partial_cloud', 'gtcloud'],
            'shuffle': subset == DatasetSubset.TRAIN
        }, file_list, transforms)

    def _get_transforms(self, cfg, subset):
        if subset == DatasetSubset.TRAIN:
            return utils.data_transforms.Compose([{
                'callback': 'RandomSamplePoints',
                'parameters': {
                    'n_points': cfg.CONST.N_INPUT_POINTS
                },
                'objects': ['partial_cloud']
            }, {
                'callback': 'RandomMirrorPoints',
                'objects': ['partial_cloud', 'gtcloud']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial_cloud', 'gtcloud']
            }])
        else:
            return utils.data_transforms.Compose([{
                'callback': 'RandomSamplePoints',
                'parameters': {
                    'n_points': cfg.CONST.N_INPUT_POINTS
                },
                'objects': ['partial_cloud']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial_cloud', 'gtcloud']
            }])

    def _get_subset(self, subset):
        if subset == DatasetSubset.TRAIN:
            return 'train'
        elif subset == DatasetSubset.VAL:
            return 'val'
        else:
            return 'test'

    def _get_file_list(self, cfg, subset, n_renderings=1):
        """Prepare file list for the dataset"""
        file_list = []

        for dc in self.dataset_categories:
            logging.info('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']))
            samples = dc[subset]

            for s in tqdm(samples, leave=False):
                file_list.append({
                    'taxonomy_id':
                    dc['taxonomy_id'],
                    'model_id':
                    s,
                    'partial_cloud_path': [
                        cfg.DATASETS.SHAPENET.PARTIAL_POINTS_PATH % (subset, dc['taxonomy_id'], s, i)
                        for i in range(n_renderings)
                    ],
                    'gtcloud_path':
                    cfg.DATASETS.SHAPENET.COMPLETE_POINTS_PATH % (subset, dc['taxonomy_id'], s),
                })

        logging.info('Complete collecting files of the dataset. Total files: %d' % len(file_list))
        return file_list



class ShapeNetCarsDataLoader(ShapeNetDataLoader):
    """
    ShapeNet only on car category
    """
    def __init__(self, cfg):
        super(ShapeNetCarsDataLoader, self).__init__(cfg)

        # Remove other categories except cars
        self.dataset_categories = [
            dc for dc in self.dataset_categories
            if dc['taxonomy_id'] == '02958343'
        ]


class Completion3DDataLoader(object):
    """
    Completion3D: get dataset file list
    """
    def __init__(self, cfg):
        self.cfg = cfg

        # Load the dataset indexing file
        self.dataset_categories = []
        with open(cfg.DATASETS.COMPLETION3D.CATEGORY_FILE_PATH) as f:
            self.dataset_categories = json.loads(f.read())

    def get_dataset(self, subset):
        file_list = self._get_file_list(self.cfg, self._get_subset(subset))
        transforms = self._get_transforms(self.cfg, subset)
        required_items = [
            'partial_cloud'
        ] if subset == DatasetSubset.TEST else ['partial_cloud', 'gtcloud']

        return Dataset(
            {
                'required_items': required_items,
                'shuffle': subset == DatasetSubset.TRAIN
            }, file_list, transforms)

    def _get_transforms(self, cfg, subset):
        if subset == DatasetSubset.TRAIN:
            return utils.data_transforms.Compose([{
                'callback': 'RandomSamplePoints',
                'parameters': {
                    'n_points': cfg.CONST.N_INPUT_POINTS
                },
                'objects': ['partial_cloud']
            }, {
                'callback':
                'RandomMirrorPoints',
                'objects': ['partial_cloud', 'gtcloud']
            }, {
                'callback':
                'ScalePoints',
                'parameters': {
                    'scale': 0.85
                },
                'objects': ['partial_cloud', 'gtcloud']
            }, {
                'callback':
                'ToTensor',
                'objects': ['partial_cloud', 'gtcloud']
            }])
        elif subset == DatasetSubset.VAL:
            return utils.data_transforms.Compose([{
                'callback':
                'ScalePoints',
                'parameters': {
                    'scale': 0.85
                },
                'objects': ['partial_cloud', 'gtcloud']
            }, {
                'callback':
                'ToTensor',
                'objects': ['partial_cloud', 'gtcloud']
            }])
        else:
            return utils.data_transforms.Compose([{
                'callback': 'ToTensor',
                'objects': ['partial_cloud']
            }])

    def _get_subset(self, subset):
        if subset == DatasetSubset.TRAIN:
            return 'train'
        elif subset == DatasetSubset.VAL:
            return 'val'
        else:
            return 'plantpcom_test'

    def _get_file_list(self, cfg, subset):
        """Prepare file list for the dataset"""
        file_list = []

        # Collect file list
        for dc in self.dataset_categories:
            logging.info('Collecting files of Taxonomy [ID=%s, Name=%s]' %
                         (dc['taxonomy_id'], dc['taxonomy_name']))
            samples = dc[subset]

            for s in tqdm(samples, leave=False):
                file_list.append({
                    'taxonomy_id':
                    dc['taxonomy_id'],
                    'model_id':
                    s,
                    'partial_cloud_path':
                    cfg.DATASETS.COMPLETION3D.PARTIAL_POINTS_PATH %
                    (subset, dc['taxonomy_id'], s),
                    'gtcloud_path':
                    cfg.DATASETS.COMPLETION3D.COMPLETE_POINTS_PATH %
                    (subset, dc['taxonomy_id'], s),
                })

        logging.info(
            'Complete collecting files of the dataset. Total files: %d' %
            len(file_list))
        return file_list


class Completion3DPCCTDataLoader(Completion3DDataLoader):
    """
    Dataset Completion3D containing only plane, car, chair, table
    """
    def __init__(self, cfg):
        super(Completion3DPCCTDataLoader, self).__init__(cfg)

        # Remove other categories except couch, chairs, car, lamps
        cat_set = {'02691156', '03001627', '02958343',
                   '04379243'}  # plane, chair, car, table
        # cat_set = {'04256520', '03001627', '02958343', '03636649'}
        self.dataset_categories = [
            dc for dc in self.dataset_categories
            if dc['taxonomy_id'] in cat_set
        ]


class KittiDataLoader(object):
    """
    KITTI: extracted car objects
    """
    def __init__(self, cfg):
        self.cfg = cfg

        # Load the dataset indexing file
        self.dataset_categories = []
        with open(cfg.DATASETS.KITTI.CATEGORY_FILE_PATH) as f:
            self.dataset_categories = json.loads(f.read())

    def get_dataset(self, subset):
        file_list = self._get_file_list(self.cfg, self._get_subset(subset))
        transforms = self._get_transforms(self.cfg, subset)
        required_items = ['partial_cloud', 'bounding_box']

        return Dataset({
            'required_items': required_items,
            'shuffle': False
        }, file_list, transforms)

    def _get_subset(self, subset):
        if subset == DatasetSubset.TRAIN:
            return 'train'
        elif subset == DatasetSubset.VAL:
            return 'val'
        else:
            return 'plantpcom_test'

    def _get_transforms(self, cfg, subset):
        return utils.data_transforms.Compose([{
            'callback':
            'NormalizeObjectPose',
            'parameters': {
                'input_keys': {
                    'ptcloud': 'partial_cloud',
                    'bbox': 'bounding_box'
                }
            },
            'objects': ['partial_cloud', 'bounding_box']
        }, {
            'callback': 'RandomSamplePoints',
            'parameters': {
                'n_points': cfg.CONST.N_INPUT_POINTS
            },
            'objects': ['partial_cloud']
        }, {
            'callback':
            'ToTensor',
            'objects': ['partial_cloud', 'bounding_box']
        }])

    def _get_file_list(self, cfg, subset):
        """Prepare file list for the dataset"""
        file_list = []

        # Collect file list
        for dc in self.dataset_categories:
            logging.info('Collecting files of Taxonomy [ID=%s, Name=%s]' %
                         (dc['taxonomy_id'], dc['taxonomy_name']))
            samples = dc[subset]

            for s in tqdm(samples, leave=False):
                file_list.append({
                    'taxonomy_id':
                    dc['taxonomy_id'],
                    'model_id':
                    s,
                    'partial_cloud_path':
                    cfg.DATASETS.KITTI.PARTIAL_POINTS_PATH % s,
                    'bounding_box_path':
                    cfg.DATASETS.KITTI.BOUNDING_BOX_FILE_PATH % s,
                })

        logging.info(
            'Complete collecting files of the dataset. Total files: %d' %
            len(file_list))
        return file_list


# ShapeNet-55/34
# Ref: https://github.com/yuxumin/PoinTr/blob/master/datasets/ShapeNet55Dataset.py
class ShapeNet55Dataset(torch.utils.data.dataset.Dataset):
    """
    ShapeNet55 dataset: return complete clouds, partial clouds are generated online
    """
    def __init__(self, options, file_list, transforms=None):
        self.options = options
        self.file_list = file_list
        self.transforms = transforms
        self.cache = dict()

    def __len__(self):
        return len(self.file_list)

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = {}

        for ri in self.options['required_items']:
            file_path = sample['%s_path' % ri]
            data[ri] = IO.get(file_path).astype(np.float32)
            # shapenet55
            data[ri] = self.pc_norm(data[ri])
            data[ri] = torch.from_numpy(data[ri]).float()

        if self.transforms is not None:
            data = self.transforms(data)

        return sample['taxonomy_id'], sample['model_id'], data


class ShapeNet55DataLoader(object):
    """
    ShapeNet55: get dataset file list
    """
    def __init__(self, cfg):
        self.cfg = cfg

    def get_dataset(self, subset):
        file_list = self._get_file_list(self.cfg, self._get_subset(subset))
        transforms = None
        return ShapeNet55Dataset(
            {
                'required_items': ['gtcloud'],
                'shuffle': subset == DatasetSubset.TRAIN
            }, file_list, transforms)

    def _get_subset(self, subset):
        if subset == DatasetSubset.TRAIN:
            return 'train'
        else:
            return 'plantpcom_test'

    def _get_file_list(self, cfg, subset):
        """Prepare file list for the dataset"""

        # Load the dataset indexing file
        with open(
                os.path.join(cfg.DATASETS.SHAPENET55.CATEGORY_FILE_PATH,
                             subset + '.txt'), 'r') as f:
            lines = f.readlines()

        # Collect file list
        file_list = []
        for line in lines:
            line = line.strip()
            taxonomy_id = line.split('-')[0]
            model_id = line.split('-')[1].split('.')[0]
            file_list.append({
                'taxonomy_id':
                taxonomy_id,
                'model_id':
                model_id,
                'gtcloud_path':
                cfg.DATASETS.SHAPENET55.COMPLETE_POINTS_PATH % (line),
            })

        print('Complete collecting files of the dataset. Total files: %d' %
              len(file_list))
        return file_list


class UnlabeledHybrid(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.pc_path = config.PC_PATH
        self.subset = config.subset
        self.npoints = config.N_POINTS

        self.data_list_file = os.path.join(
            self.data_root, f'{self.subset}.txt')
        test_data_list_file = os.path.join(self.data_root, 'plantpcom_test.txt')

        self.sample_points_num = config.npoints
        self.whole = config.get('whole')

        print(f'[DATASET] sample out {self.sample_points_num} points')
        print(f'[DATASET] Open file {self.data_list_file}')
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()
        if self.whole:
            with open(test_data_list_file, 'r') as f:
                test_lines = f.readlines()
            print('[DATASET] Open file {test_data_list_file}')
            lines = test_lines + lines
        self.file_list = []
        for line in lines:
            line = line.strip()
            taxonomy_id = ''
            model_id = ''
            self.file_list.append({
                'taxonomy_id': taxonomy_id,
                'model_id': model_id,
                'file_path': line
            })
        print('[DATASET] {len(self.file_list)} instances were loaded')

        self.permutation = np.arange(self.npoints)

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc

    def random_sample(self, pc, num):
        permutation = np.arange(pc.shape[0])
        np.random.shuffle(permutation)
        pc = pc[permutation[:num]]
        return pc

    def __getitem__(self, idx):
        sample = self.file_list[idx]

        data = IO.get(os.path.join(
            self.pc_path, sample['file_path'])).astype(np.float32)

        data = self.random_sample(data, self.sample_points_num)
        data = self.pc_norm(data)
        data = torch.from_numpy(data).float()
        # sample['taxonomy_id'] and sample['model_id'] are not utilized
        return sample['taxonomy_id'], sample['model_id'], data

    def __len__(self):
        return len(self.file_list)


class Plant3DComDataLoader(object):
    def __init__(self, cfg):
        self.cfg = cfg

        with open(cfg.DATASETS.PLANT3D.JSON_PATH) as f:
            self.dataset_categories = json.load(f)


    # def get_dataset(self, subset):
    #     n_renderings = self.cfg.DATASETS.PLANT3D.N_RENDERINGS
    #
    #     file_list = self._get_file_list(
    #         self.cfg,
    #         self._get_subset(subset),
    #         n_renderings
    #     )
    #
    #     transforms = self._get_transforms(self.cfg, subset)
    #
    #     # =====  统计 partial 数量 =====
    #     total_partial = 0
    #     for item in file_list:
    #         total_partial += len(item['partial_cloud_path'])
    #
    #     print(f"[{subset}]")
    #     print(f"完整点云数量: {len(file_list)}")
    #     print(f"partial点云总数量: {total_partial}")
    #     print("====================================")
    #
    #     return Dataset({
    #         'n_renderings': n_renderings,
    #         'required_items': ['partial_cloud', 'gtcloud'],
    #         'shuffle': subset == DatasetSubset.TRAIN
    #     }, file_list, transforms)


    #   全部加载
    def get_dataset(self, subset):

        file_list = self._get_file_list(
            self.cfg,
            self._get_subset(subset),
            self.cfg.DATASETS.PLANT3D.N_RENDERINGS
        )

        transforms = self._get_transforms(self.cfg, subset)

        print(f"[{subset}]")
        print(f"训练样本数量（partial-level）: {len(file_list)}")
        print("====================================")

        return Dataset({
            'required_items': ['partial_cloud', 'gtcloud'],
            'shuffle': subset == DatasetSubset.TRAIN
        }, file_list, transforms)


    def _get_subset(self, subset):
        if subset == DatasetSubset.TRAIN:
            return 'train'
        elif subset == DatasetSubset.VAL:
            return 'val'
        else:
            return 'test'

    def _get_transforms(self, cfg, subset):
        if subset == DatasetSubset.TRAIN:
            return utils.data_transforms.Compose([
                {
                    'callback': 'RandomSamplePoints',
                    'parameters': {'n_points': cfg.CONST.N_INPUT_POINTS},
                    'objects': ['partial_cloud']
                },
                {
                    'callback': 'RandomMirrorPoints',
                    'objects': ['partial_cloud', 'gtcloud']
                },
                {
                    'callback': 'ToTensor',
                    'objects': ['partial_cloud', 'gtcloud']
                }
            ])
        else:
            return utils.data_transforms.Compose([
                {
                    'callback': 'RandomSamplePoints',
                    'parameters': {'n_points': cfg.CONST.N_INPUT_POINTS},
                    'objects': ['partial_cloud']
                },
                {
                    'callback': 'ToTensor',
                    'objects': ['partial_cloud', 'gtcloud']
                }
            ])


    # 随机抽取
    # def _get_file_list(self, cfg, subset, n_renderings):
    #     file_list = []
    #
    #     for dc in self.dataset_categories:
    #         taxonomy_id = dc['taxonomy_id']
    #         samples = dc[subset]
    #
    #         logging.info(f'Collecting [{taxonomy_id}] subset={subset}')
    #
    #         for s in tqdm(samples, leave=False):
    #
    #             # === 20个partial路径 ===
    #             partial_paths = [
    #                 cfg.DATASETS.PLANT3D.PARTIAL_POINTS_PATH %
    #                 (subset, taxonomy_id, s, s, i)
    #                 for i in range(n_renderings)
    #             ]
    #
    #             file_list.append({
    #                 'taxonomy_id': taxonomy_id,
    #                 'model_id': s,
    #                 'partial_cloud_path': partial_paths,
    #                 'gtcloud_path':
    #                     cfg.DATASETS.PLANT3D.COMPLETE_POINTS_PATH %
    #                     (subset, taxonomy_id, s),
    #             })
    #
    #     logging.info(f'Total files: {len(file_list)}')
    #     return file_list

    # 全部加载
    def _get_file_list(self, cfg, subset, n_renderings):
        file_list = []

        for dc in self.dataset_categories:
            taxonomy_id = dc['taxonomy_id']
            samples = dc[subset]
            logging.info(f'Collecting [{taxonomy_id}] subset={subset}')
            for s in tqdm(samples, leave=False):
                # 每个partial单独作为一个样本
                for i in range(n_renderings):
                    partial_path = cfg.DATASETS.PLANT3D.PARTIAL_POINTS_PATH % \
                                   (subset, taxonomy_id, s, s, i)

                    file_list.append({
                        'taxonomy_id': taxonomy_id,
                        'model_id': f"{s}_{i}",  # 唯一ID
                        'partial_cloud_path': partial_path,  # ❗不是list
                        'gtcloud_path':
                            cfg.DATASETS.PLANT3D.COMPLETE_POINTS_PATH %
                            (subset, taxonomy_id, s),
                    })

        logging.info(f'Total samples (partial-level): {len(file_list)}')
        return file_list



# //////////////////////////////////////////// = Dataset Loader Mapping = //////////////////////////////////////////// #

DATASET_LOADER_MAPPING = {
    'Completion3D': Completion3DDataLoader,
    'Completion3DPCCT': Completion3DPCCTDataLoader,
    'ShapeNet': ShapeNetDataLoader,
    'ShapeNetCars': ShapeNetCarsDataLoader,
    'KITTI': KittiDataLoader,
    'ShapeNet55': ShapeNet55DataLoader,
    'Plant3D': Plant3DComDataLoader,
}  # yapf: disable



if __name__ == "__main__":
    import torch
    from easydict import EasyDict as edict

    # 构造 cfg
    cfg = edict()

    cfg.CONST = edict({
        "N_INPUT_POINTS": 2048
    })

    cfg.DATASETS = edict()
    cfg.DATASETS.PLANT3D = edict({
        "JSON_PATH" : "/media/ps/Data/WYL/dataset/PlantPCom/Plant3d_com.json",
        # partial（注意多了 index）
        "PARTIAL_POINTS_PATH" : "/media/ps/Data/WYL/dataset/PlantPCom/%s/partial/%s/%s/%s_%02d.pcd",
        # gt
        "COMPLETE_POINTS_PATH" : "/media/ps/Data/WYL/dataset/PlantPCom/%s/complete/%s/%s.pcd",
        # 关键参数
        "N_RENDERINGS": 20
    })

    cfg.DATASETS.SHAPENET = edict({
        "CATEGORY_FILE_PATH" : "/media/ps/Data/WYL/CRA-PCN-main/datasets/ShapeNet.json",
        # partial（注意多了 index）
        "PARTIAL_POINTS_PATH" : "/media/ps/Data/WYL/dataset/PCN/%s/partial/%s/%s/%02d.pcd",
        # gt
        "COMPLETE_POINTS_PATH" : "/media/ps/Data/WYL/dataset/PCN/%s/complete/%s/%s.pcd",
        # 关键参数
        "N_RENDERINGS": 8
    })


    # 创建 dataloader
    loader = Plant3DComDataLoader(cfg)
    # loader = ShapeNetDataLoader(cfg)

    dataset = loader.get_dataset(DatasetSubset.TRAIN)

    print("Dataset size:", len(dataset))

    # 取一个样本
    taxonomy_id, model_id, data = dataset[0]

    print("taxonomy_id:", taxonomy_id)
    print("model_id:", model_id)
    print("partial shape:", data['partial_cloud'].shape)
    print("gt shape:", data['gtcloud'].shape)

    # 测试 DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn
    )

    for batch in dataloader:
        taxonomy_ids, model_ids, data = batch
        print(taxonomy_ids, model_ids)
        print("Batch partial:", data['partial_cloud'].shape)
        print("Batch gt:", data['gtcloud'].shape)
        break