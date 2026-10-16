import os
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
import time
from tqdm import tqdm
import utils.data_loaders
import utils.helpers
from utils.average_meter import AverageMeter
from utils.metrics import Metrics
from utils.schedular import GradualWarmupScheduler
from utils.loss_utils_clamp import get_loss_clamp, get_loss_mvp
from utils.ply import read_ply, write_ply
import pointnet_utils.pc_util as pc_util
from utils.mvp_utils import *
from PIL import Image

class Manager:

    # Initialization methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, model, cfg):
        """
        Initialize parameters and start training/testing
        :param model: network object
        :param cfg: configuration object
        """

        ############
        # Parameters
        ############
        
        # Epoch index
        self.epoch = 0

        # Create the optimizers
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                           lr=cfg.TRAIN.LEARNING_RATE,
                                           weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                                           betas=cfg.TRAIN.BETAS)

        # lr scheduler
        # self.lr_decays = {i: 0.1 ** (1 / cfg.TRAIN.LR_DECAY) for i in range(1, cfg.TRAIN.N_EPOCHS+1)}
        # self.scheduler_steplr = StepLR(self.optimizer, step_size=cfg.TRAIN.LR_DECAY_STEP, gamma=cfg.TRAIN.GAMMA)
        # self.lr_scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=cfg.TRAIN.WARMUP_STEPS,
        #                                       after_scheduler=self.scheduler_steplr)
        self.scheduler_steplr = StepLR(self.optimizer, step_size=1, gamma=0.1 ** (1 / cfg.TRAIN.LR_DECAY))
        self.lr_scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=cfg.TRAIN.WARMUP_EPOCHS,
                                              after_scheduler=self.scheduler_steplr)

        # record file
        self.train_record_file = open(os.path.join(cfg.DIR.LOGS, 'training.txt'), 'w')
        self.test_record_file = open(os.path.join(cfg.DIR.LOGS, 'testing.txt'), 'w')

        # eval metric
        self.best_metrics = float('inf')
        self.best_epoch = 0


    # Record functions
    def train_record(self, info, show_info=True):
        if show_info:
            print(info)
        if self.train_record_file:
            self.train_record_file.write(info + '\n')
            self.train_record_file.flush()

    def test_record(self, info, show_info=True):
        if show_info:
            print(info)
        if self.test_record_file:
            self.test_record_file.write(info + '\n')
            self.test_record_file.flush()


    def train(self, model, train_data_loader, val_data_loader, cfg):

        init_epoch = 0
        steps = 0

        # training record file
        print('Training Record:')
        self.train_record('n_itr, cd_pc, cd_p1, cd_p2, cd_p3, partial_matching')
        print('Testing Record:')
        self.test_record('#epoch cdc cd1 cd2 partial_matching | cd3 | #best_epoch best_metrics')

        # Training Start
        for epoch_idx in range(init_epoch + 1, cfg.TRAIN.N_EPOCHS + 1):

            self.epoch = epoch_idx

            # timer
            epoch_start_time = time.time()

            model.train()

            # Update learning rate
            self.lr_scheduler.step()

            # total cds
            total_cd_pc = 0
            total_cd_p1 = 0
            total_cd_p2 = 0
            total_cd_p3 = 0
            total_partial = 0

            batch_end_time = time.time()
            n_batches = len(train_data_loader)
            learning_rate = self.optimizer.param_groups[0]['lr']
            for batch_idx, (taxonomy_ids, model_ids, data) in enumerate(train_data_loader):
                for k, v in data.items():
                    data[k] = utils.helpers.var_or_cuda(v)

                # unpack data
                partial = data['partial_cloud']
                gt = data['gtcloud']

                pcds_pred = model(partial)

                loss_total, losses, gts = get_loss_clamp(pcds_pred, partial, gt, sqrt=True)

                self.optimizer.zero_grad()
                loss_total.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                self.optimizer.step()

                cd_pc_item = losses[0].item() * 1e3
                total_cd_pc += cd_pc_item
                cd_p1_item = losses[1].item() * 1e3
                total_cd_p1 += cd_p1_item
                cd_p2_item = losses[2].item() * 1e3
                total_cd_p2 += cd_p2_item
                cd_p3_item = losses[3].item() * 1e3
                total_cd_p3 += cd_p3_item
                partial_item = losses[4].item() * 1e3
                total_partial += partial_item
                n_itr = (epoch_idx - 1) * n_batches + batch_idx

                # training record
                message = '{:d} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(n_itr, cd_pc_item, cd_p1_item, cd_p2_item, cd_p3_item, partial_item)
                self.train_record(message, show_info=False)

                # if steps <= cfg.TRAIN.WARMUP_STEPS:
                #     self.lr_scheduler.step()
                #     steps += 1

            # avg cds
            avg_cdc = total_cd_pc / n_batches
            avg_cd1 = total_cd_p1 / n_batches
            avg_cd2 = total_cd_p2 / n_batches
            avg_cd3 = total_cd_p3 / n_batches
            avg_partial = total_partial / n_batches

            # Update learning rate
            # if self.epoch in self.lr_decays:
            #     for param_group in self.optimizer.param_groups:
            #         param_group['lr'] *= self.lr_decays[self.epoch]
            # else:
            #     raise ValueError('Epoch exceeds max limit!')
            epoch_end_time = time.time()

            # Training record
            self.train_record(
                '[Epoch %d/%d] LearningRate = %f EpochTime = %.3f (s) Losses = %s' %
                (epoch_idx, cfg.TRAIN.N_EPOCHS, learning_rate, epoch_end_time - epoch_start_time, ['%.4f' % l for l in [avg_cdc, avg_cd1, avg_cd2, avg_cd3, avg_partial]]))

            # Validate the current model
            cd_eval = self.validate(cfg, model=model, val_data_loader=val_data_loader)

            # Save checkpoints
            if cd_eval < self.best_metrics:
                self.best_epoch = epoch_idx
                file_name = 'ckpt-best.pth' if cd_eval < self.best_metrics else 'ckpt-epoch-%03d.pth' % epoch_idx
                output_path = os.path.join(cfg.DIR.CHECKPOINTS, file_name)
                torch.save({
                    'epoch_index': epoch_idx,
                    'best_metrics': cd_eval,
                    'model': model.state_dict()
                }, output_path)

                print('Saved checkpoint to %s ...' % output_path)
                if cd_eval < self.best_metrics:
                    self.best_metrics = cd_eval

        # training end
        self.train_record_file.close()
        self.test_record_file.close()


    def validate(self, cfg, model=None, val_data_loader=None, outdir=None):
        # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
        torch.backends.cudnn.benchmark = True

        # Switch models to evaluation mode
        model.eval()

        n_samples = len(val_data_loader)
        test_losses = AverageMeter(['cdc', 'cd1', 'cd2', 'cd3', 'partial_matching'])
        test_metrics = AverageMeter(Metrics.names())

        # Start testing
        for model_idx, (taxonomy_id, model_id, data) in enumerate(val_data_loader):
            taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
            #model_id = model_id[0]

            with torch.no_grad():
                for k, v in data.items():
                    data[k] = utils.helpers.var_or_cuda(v)

                # unpack data
                partial = data['partial_cloud']
                gt = data['gtcloud']

                # forward
                pcds_pred = model(partial.contiguous())
                loss_total, losses, _ = get_loss_clamp(pcds_pred, partial, gt, sqrt=True)

                # get metrics
                cdc = losses[0].item() * 1e3
                cd1 = losses[1].item() * 1e3
                cd2 = losses[2].item() * 1e3
                cd3 = losses[3].item() * 1e3
                partial_matching = losses[4].item() * 1e3

                _metrics = [cd3]
                test_losses.update([cdc, cd1, cd2, cd3, partial_matching])
                test_metrics.update(_metrics)

        # Record testing results
        message = '#{:d} {:.4f} {:.4f} {:.4f} {:.4f} | {:.4f} | #{:d} {:.4f}'.format(self.epoch, test_losses.avg(0), test_losses.avg(1), test_losses.avg(2), test_losses.avg(4), test_losses.avg(3), self.best_epoch, self.best_metrics)
        self.test_record(message)

        return test_losses.avg(3)


    def test(self, cfg, model=None, test_data_loader=None, outdir=None):
        # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
        torch.backends.cudnn.benchmark = True

        # Switch models to evaluation mode
        model.eval()

        n_samples = len(test_data_loader)
        test_losses = AverageMeter(['cdc', 'cd1', 'cd2', 'cd3', 'partial_matching'])
        test_metrics = AverageMeter(Metrics.names())
        category_metrics = dict()
        category_models = dict()

        # Start testing
        for model_idx, (taxonomy_id, model_id, data) in enumerate(test_data_loader):
            taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
            #model_id = model_id[0]

            with torch.no_grad():
                for k, v in data.items():
                    data[k] = utils.helpers.var_or_cuda(v)

                partial = data['partial_cloud']
                gt = data['gtcloud']

                b, n, _ = partial.shape

                pcds_pred = model(partial.contiguous())
                loss_total, losses, _ = get_loss_clamp(pcds_pred, partial, gt, sqrt=True)

                cdc = losses[0].item() * 1e3
                cd1 = losses[1].item() * 1e3
                cd2 = losses[2].item() * 1e3
                cd3 = losses[3].item() * 1e3
                partial_matching = losses[4].item() * 1e3

                _metrics = [cd3]
                test_losses.update([cdc, cd1, cd2, cd3, partial_matching])

                test_metrics.update(_metrics)
                if taxonomy_id not in category_metrics:
                    category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
                category_metrics[taxonomy_id].update(_metrics)

                # output to file
                if outdir:
                    if not os.path.exists(os.path.join(outdir, taxonomy_id)):
                        os.makedirs(os.path.join(outdir, taxonomy_id))
                    if not os.path.exists(os.path.join(outdir, taxonomy_id+'_images')):
                        os.makedirs(os.path.join(outdir, taxonomy_id+'_images'))
                    # save pred, gt, partial pcds 
                    pred = pcds_pred[-1]
                    for mm, model_name in enumerate(model_id):
                        output_file = os.path.join(outdir, taxonomy_id, model_name)
                        write_ply(output_file + '_pred.ply', pred[mm, :].detach().cpu().numpy(), ['x', 'y', 'z'])
                        write_ply(output_file + '_gt.ply', gt[mm, :].detach().cpu().numpy(), ['x', 'y', 'z'])
                        write_ply(output_file + '_partial.ply', partial[mm, :].detach().cpu().numpy(), ['x', 'y', 'z'])
                        # record
                        if taxonomy_id not in category_models:
                            category_models[taxonomy_id] = []
                        category_models[taxonomy_id].append((model_name, cd3))
                        # output img files
                        img_filename = os.path.join(outdir, taxonomy_id+'_images', model_name+'.jpg')
                        output_img = pc_util.point_cloud_three_views(pred[mm, :].detach().cpu().numpy(), diameter=7)
                        output_img = (output_img*255).astype('uint8')
                        im = Image.fromarray(output_img)
                        im.save(img_filename)


        # Record category model cds
        if outdir:
            for cat in list(category_models.keys()):
                model_tuples = category_models[cat]
                with open(os.path.join(outdir, cat+'.txt'), 'w') as record_file:
                    # sort by cd
                    for tt in sorted(model_tuples, key=lambda x:x[1]):
                        record_file.write('{:s}  {:.4f}\n'.format(tt[0], tt[1])) 

        # # Print testing results
        # print('============================ TEST RESULTS ============================')
        # print('Taxonomy', end='\t')
        # print('#Sample', end='\t')
        # for metric in test_metrics.items:
        #     print(metric, end='\t')
        # print()

        # for taxonomy_id in category_metrics:
        #     print(taxonomy_id, end='\t')
        #     print(category_metrics[taxonomy_id].count(0), end='\t')
        #     for value in category_metrics[taxonomy_id].avg():
        #         print('%.4f' % value, end='\t')
        #     print()

        # print('Overall', end='\t\t\t')
        # for value in test_metrics.avg():
        #     print('%.4f' % value, end='\t')
        # print('\n')

        # print('Epoch ', self.epoch, end='\t')
        # for value in test_losses.avg():
        #     print('%.4f' % value, end='\t')
        # print('\n')

        # Record category results
        self.train_record('============================ TEST RESULTS ============================')
        self.train_record('Taxonomy\t#Sample\t' + '\t'.join(test_metrics.items))

        for taxonomy_id in category_metrics:
            message = '{:s}\t{:d}\t'.format(taxonomy_id, category_metrics[taxonomy_id].count(0)) 
            message += '\t'.join(['%.4f' % value for value in category_metrics[taxonomy_id].avg()])
            self.train_record(message)

        self.train_record('Overall\t\t' + '\t'.join(['%.4f' % value for value in test_metrics.avg()]))

        # record testing results
        message = '#{:d} {:.4f} {:.4f} {:.4f} {:.4f} | {:.4f} | #{:d} {:.4f}'.format(self.epoch, test_losses.avg(0), test_losses.avg(1), test_losses.avg(2), test_losses.avg(4), test_losses.avg(3), self.best_epoch, self.best_metrics)
        self.test_record(message)

        return test_losses.avg(3)


    def test_pcom(self, cfg, model=None, test_data_loader=None, outdir=None):
        # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
        torch.backends.cudnn.benchmark = True

        # Switch models to evaluation mode
        model.eval()

        # plantpcom dataset categories
        cat_name = ['tomato', 'tobacco', 'sorghum', 'Arabidopsis', 'maize']

        # Initialize PCN-style meters for MVP metrics
        metric_names = ['cd_p', 'cd_t', 'f1']
        test_metrics = AverageMeter(metric_names)
        category_metrics = dict()
        category_models = dict()

        for i, data in tqdm(enumerate(test_data_loader), total=len(test_data_loader), desc='Testing'):
            taxonomy_id_batch, model_id_batch, data_dict = data

            with torch.no_grad():
                inputs = data_dict['partial_cloud'].float().cuda()
                gt = data_dict['gtcloud'].float().cuda()
                output = model(inputs)[-1]
                cd_p, cd_t, f1 = calc_cd(output, gt, calc_f1=True)

                b = inputs.size(0)
                for b_idx in range(b):
                    taxonomy_id = taxonomy_id_batch[b_idx]
                    model_name = model_id_batch[b_idx]

                    val_cd_p = cd_p[b_idx].item() * 1000.0
                    val_cd_t = cd_t[b_idx].item() * 1000.0
                    val_f1 = f1[b_idx].item()

                    _metrics = [val_cd_p, val_cd_t, val_f1]

                    # Update global metrics
                    test_metrics.update(_metrics)

                    # Update category metrics
                    if taxonomy_id not in category_metrics:
                        category_metrics[taxonomy_id] = AverageMeter(metric_names)
                    category_metrics[taxonomy_id].update(_metrics)

                    # Output to file
                    if outdir:
                        # pass
                        if not os.path.exists(os.path.join(outdir, taxonomy_id)):
                            os.makedirs(os.path.join(outdir, taxonomy_id))
                        if not os.path.exists(os.path.join(outdir, taxonomy_id + '_images')):
                            os.makedirs(os.path.join(outdir, taxonomy_id + '_images'))

                        # Convert to numpy
                        pred_np = output[b_idx, :].detach().cpu().numpy()
                        gt_np = gt[b_idx, :].detach().cpu().numpy()
                        partial_np = inputs[b_idx, :].detach().cpu().numpy()

                        output_file = os.path.join(outdir, taxonomy_id, model_name)

                        # Save PLY
                        write_ply(output_file + '_pred.ply', pred_np, ['x', 'y', 'z'])
                        write_ply(output_file + '_gt.ply', gt_np, ['x', 'y', 'z'])
                        write_ply(output_file + '_partial.ply', partial_np, ['x', 'y', 'z'])

                        # Record for txt
                        if taxonomy_id not in category_models:
                            category_models[taxonomy_id] = []
                        category_models[taxonomy_id].append((model_name, val_cd_p))

                        # Save rendered images
                        img_filename = os.path.join(outdir, taxonomy_id + '_images', model_name + '.jpg')
                        output_img = pc_util.point_cloud_three_views(pred_np, diameter=3)  # 3, 5, 7
                        output_img = (output_img * 255).astype('uint8')
                        im = Image.fromarray(output_img)
                        im.save(img_filename)

        # Record category model cds to txt
        if outdir:
            for cat in list(category_models.keys()):
                model_tuples = category_models[cat]
                with open(os.path.join(outdir, cat + '.txt'), 'w') as record_file:
                    # Sort by cd_p
                    for tt in sorted(model_tuples, key=lambda x: x[1]):
                        record_file.write('{:s}  {:.4f}\n'.format(tt[0], tt[1]))

        # ---------------------------
        #    Logging Results
        # ---------------------------
        self.train_record('============================ TEST RESULTS ============================')
        header = '{:<15} {:<10}'.format('Taxonomy', '#Sample') + ''.join(['{:>12}'.format(m) for m in metric_names])
        self.train_record(header)

        for taxonomy_id in category_metrics:
            meter = category_metrics[taxonomy_id]
            sample_count = meter.count(0)
            avg_vals = meter.avg()

            message = '{:<15} {:<10d}'.format(taxonomy_id, int(sample_count))
            message += ''.join(['{:>12.4f}'.format(value) for value in avg_vals])
            self.train_record(message)

        overall_avgs = test_metrics.avg()
        message_overall = '{:<15} {:<10}'.format('Overall', '')  # Overall行不显示样本数，留空
        message_overall += ''.join(['{:>12.4f}'.format(value) for value in overall_avgs])
        self.train_record(message_overall)

        message_summary = '#{:d} {:.4f} {:.4f} {:.4f} | {:.4f} | #{:d} {:.4f}'.format(
            self.epoch,
            overall_avgs[0], overall_avgs[1], overall_avgs[2],
            overall_avgs[0],
            self.best_epoch, self.best_metrics
        )
        self.test_record(message_summary)
        return overall_avgs[1]



class Manager_MVP:
    def __init__(self, model, cfg):


        ############
        # Parameters
        ############
        
        # Epoch index
        self.epoch = 0

        # Create the optimizers
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                           lr=cfg.TRAIN.LEARNING_RATE,
                                           weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                                           betas=cfg.TRAIN.BETAS)

        # lr scheduler
        # self.lr_decays = {i: 0.1 ** (1 / cfg.TRAIN.LR_DECAY) for i in range(1, cfg.TRAIN.N_EPOCHS+1)}
        # self.scheduler_steplr = StepLR(self.optimizer, step_size=cfg.TRAIN.LR_DECAY_STEP, gamma=cfg.TRAIN.GAMMA)
        # self.lr_scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=cfg.TRAIN.WARMUP_STEPS,
        #                                       after_scheduler=self.scheduler_steplr)
        self.scheduler_steplr = StepLR(self.optimizer, step_size=1, gamma=0.1 ** (1 / cfg.TRAIN.LR_DECAY))
        self.lr_scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=cfg.TRAIN.WARMUP_EPOCHS,
                                              after_scheduler=self.scheduler_steplr)

        # record file
        self.train_record_file = open(os.path.join(cfg.DIR.LOGS, 'training.txt'), 'w')
        self.test_record_file = open(os.path.join(cfg.DIR.LOGS, 'testing.txt'), 'w')

        # eval metric
        self.best_metrics = float('inf')
        self.best_epoch = 0


    # Record functions
    def train_record(self, info, show_info=True):
        if show_info:
            print(info)
        if self.train_record_file:
            self.train_record_file.write(info + '\n')
            self.train_record_file.flush()

    def test_record(self, info, show_info=True):
        if show_info:
            print(info)
        if self.test_record_file:
            self.test_record_file.write(info + '\n')
            self.test_record_file.flush()


    def train(self, model, train_data_loader, val_data_loader, cfg):

        init_epoch = 0
        steps = 0

        # training record file
        print('Training Record:')
        self.train_record('n_itr, cd_pc, cd_p1, cd_p2, cd_p3, partial_matching')
        print('Testing Record:')
        self.test_record('#cd_p cd_t f1 | #best_metrics')
        cd_best  = 1e6
        # Training Start
        for epoch_idx in range(init_epoch + 1, cfg.TRAIN.N_EPOCHS + 1):

            self.epoch = epoch_idx

            # timer
            epoch_start_time = time.time()

            model.train()

            # Update learning rate
            self.lr_scheduler.step()

            # total cds
            total_cd_pc = 0
            total_cd_p1 = 0
            total_cd_p2 = 0
            total_cd_p3 = 0
            total_partial = 0

            batch_end_time = time.time()
            n_batches = len(train_data_loader)
            learning_rate = self.optimizer.param_groups[0]['lr']
            for batch_idx, (label, partial, gt) in enumerate(train_data_loader):
                #for k, v in data.items():
                #    data[k] = utils.helpers.var_or_cuda(v)

                # unpack data
                partial = partial.float().cuda()
                gt = gt.float().cuda()

                pcds_pred = model(partial)
                # pcds_pred, loss_en = model(partial)

                loss_total, losses, gts = get_loss_mvp(pcds_pred, partial, gt, sqrt=True)

                # loss_total = loss_total + loss_en

                self.optimizer.zero_grad()
                loss_total.backward()
                self.optimizer.step()

                cd_pc_item = losses[0].item() * 1e3
                total_cd_pc += cd_pc_item
                cd_p1_item = losses[1].item() * 1e3
                total_cd_p1 += cd_p1_item
                cd_p2_item = losses[2].item() * 1e3
                total_cd_p2 += cd_p2_item
                cd_p3_item = losses[3].item() * 1e3
                total_cd_p3 += cd_p3_item
                partial_item = losses[4].item() * 1e3
                total_partial += partial_item
                n_itr = (epoch_idx - 1) * n_batches + batch_idx

                # training record
                message = '{:d} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(n_itr, cd_pc_item, cd_p1_item, cd_p2_item, cd_p3_item, partial_item)
                self.train_record(message, show_info=False)

                # if steps <= cfg.TRAIN.WARMUP_STEPS:
                #     self.lr_scheduler.step()
                #     steps += 1

            # avg cds
            avg_cdc = total_cd_pc / n_batches
            avg_cd1 = total_cd_p1 / n_batches
            avg_cd2 = total_cd_p2 / n_batches
            avg_cd3 = total_cd_p3 / n_batches
            avg_partial = total_partial / n_batches

            epoch_end_time = time.time()

            # Training record

            # save checkpoint 
            now_cd = self.validate(cfg, model, val_data_loader)

            if now_cd < self.best_metrics:
                self.best_metrics = now_cd
                file_name = 'ckpt-best.pth' 
                output_path = os.path.join(cfg.DIR.CHECKPOINTS, file_name)
                torch.save({
                    'epoch_index': epoch_idx,
                    'model': model.state_dict()
                }, output_path)
            
            self.train_record(
                '[Epoch %d/%d] LearningRate = %f EpochTime = %.3f (s) Losses = %s cd_best = %f' %
                (epoch_idx, cfg.TRAIN.N_EPOCHS, learning_rate, epoch_end_time - epoch_start_time,
                 ['%.4f' % l for l in [avg_cdc, avg_cd1, avg_cd2, avg_cd3, avg_partial]], self.best_metrics))
        # training end
        self.train_record_file.close()
        self.test_record_file.close()

    def validate(self, cfg, model=None, val_data_loader=None):

        torch.backends.cudnn.benchmark = True
        metrics = ['cd_p', 'cd_t', 'f1']
        # test_losses = AverageMeter(['cd_p', 'cd_t', 'f1'])
        test_loss_meters = {m: AverageValueMeter() for m in metrics}
        model.eval()

        with torch.no_grad():
            for data in val_data_loader:
                label, inputs_cpu, gt_cpu = data

                inputs = inputs_cpu.float().cuda()
                gt = gt_cpu.float().cuda()

                output = model(inputs)[-1]
                cd_p, cd_t, f1 = calc_cd(output, gt, calc_f1=True)

                cd_p = cd_p * 1e3
                cd_t = cd_t * 1e4

                # test_losses.update([cd_p, cd_t, f1])

                # result_dict = dict()
                # result_dict['cd_p'] = cd_p
                # result_dict['cd_t'] = cd_t
                # result_dict['f1'] = f1
                #
                # for k, v in test_loss_meters.items():
                #     v.update(result_dict[k].mean().item())
                test_loss_meters['cd_p'].update(cd_p.mean().item())
                test_loss_meters['cd_t'].update(cd_t.mean().item())
                test_loss_meters['f1'].update(f1.mean().item())

        # Record testing results
        message = '# cd_p: {:.4f} cd_t:{:.4f} f1:{:.4f}  # best_cd_t:{:.4f}'.format(test_loss_meters['cd_p'].avg,
                                                                                    test_loss_meters['cd_t'].avg,
                                                                                    test_loss_meters['f1'].avg,
                                                                                    self.best_metrics)
        self.test_record(message)
        current_time_str = time.strftime('current_time_str：%Y_%m_%d_%H_%M_%S', time.gmtime())
        self.test_record(current_time_str)

        return test_loss_meters['cd_t'].avg
        
    def test(self, cfg, model=None, test_data_loader=None, outdir=None):
        # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
        torch.backends.cudnn.benchmark = True

        # Switch models to evaluation mode
        model.eval()

        # MVP dataset categories
        cat_name = ['airplane', 'cabinet', 'car', 'chair', 'lamp', 'sofa', 'table', 'watercraft',
                    'bed', 'bench', 'bookshelf', 'bus', 'guitar', 'motorbike', 'pistol', 'skateboard']

        # Initialize PCN-style meters for MVP metrics
        metric_names = ['cd_p', 'cd_t', 'f1']
        test_metrics = AverageMeter(metric_names)
        category_metrics = dict()
        category_models = dict()

        for i, data in tqdm(enumerate(test_data_loader), total=len(test_data_loader), desc='Testing'):
            if len(data) == 4:
                label, model_id_batch, inputs_cpu, gt_cpu = data
            else:
                label, inputs_cpu, gt_cpu = data
                model_id_batch = [f'batch{i}_idx{b}' for b in range(inputs_cpu.size(0))]
            with torch.no_grad():
                inputs = inputs_cpu.float().cuda()
                gt = gt_cpu.float().cuda()

                # Forward pass (MVP net(inputs) returns a list, [-1] is the final refined output)
                output = model(inputs)[-1]

                cd_p, cd_t, f1 = calc_cd(output, gt, calc_f1=True)

                b = inputs.size(0)
                for b_idx in range(b):
                    lbl_int = int(label[b_idx])
                    taxonomy_id = cat_name[lbl_int]
                    model_name = model_id_batch[b_idx]

                    val_cd_p = cd_p[b_idx].item() * 10000.0
                    val_cd_t = cd_t[b_idx].item() * 1000.0
                    val_f1 = f1[b_idx].item()

                    _metrics = [val_cd_p, val_cd_t, val_f1]

                    # Update global metrics
                    test_metrics.update(_metrics)

                    # Update category metrics
                    if taxonomy_id not in category_metrics:
                        category_metrics[taxonomy_id] = AverageMeter(metric_names)
                    category_metrics[taxonomy_id].update(_metrics)

                    # Output to file
                    if outdir:
                        # pass
                        if not os.path.exists(os.path.join(outdir, taxonomy_id)):
                            os.makedirs(os.path.join(outdir, taxonomy_id))
                        if not os.path.exists(os.path.join(outdir, taxonomy_id + '_images')):
                            os.makedirs(os.path.join(outdir, taxonomy_id + '_images'))

                        # Convert to numpy
                        pred_np = output[b_idx, :].detach().cpu().numpy()
                        gt_np = gt[b_idx, :].detach().cpu().numpy()
                        partial_np = inputs[b_idx, :].detach().cpu().numpy()

                        output_file = os.path.join(outdir, taxonomy_id, model_name)

                        # Save PLY
                        write_ply(output_file + '_pred.ply', pred_np, ['x', 'y', 'z'])
                        write_ply(output_file + '_gt.ply', gt_np, ['x', 'y', 'z'])
                        write_ply(output_file + '_partial.ply', partial_np, ['x', 'y', 'z'])

                        # Record for txt
                        if taxonomy_id not in category_models:
                            category_models[taxonomy_id] = []
                        category_models[taxonomy_id].append((model_name, val_cd_p))

                        # Save rendered images
                        img_filename = os.path.join(outdir, taxonomy_id + '_images', model_name + '.jpg')
                        output_img = pc_util.point_cloud_three_views(pred_np, diameter=3)  # 3, 5 或 7
                        output_img = (output_img * 255).astype('uint8')
                        im = Image.fromarray(output_img)
                        im.save(img_filename)

        # Record category model cds to txt
        if outdir:
            for cat in list(category_models.keys()):
                model_tuples = category_models[cat]
                with open(os.path.join(outdir, cat + '.txt'), 'w') as record_file:
                    # Sort by cd_p
                    for tt in sorted(model_tuples, key=lambda x: x[1]):
                        record_file.write('{:s}  {:.4f}\n'.format(tt[0], tt[1]))

                        # ---------------------------
        #    Logging Results
        # ---------------------------
        self.train_record('============================ TEST RESULTS ============================')

        header = '{:<15} {:<10}'.format('Taxonomy', '#Sample') + ''.join(['{:>12}'.format(m) for m in metric_names])
        self.train_record(header)


        for taxonomy_id in category_metrics:
            meter = category_metrics[taxonomy_id]
            sample_count = meter.count(0)
            avg_vals = meter.avg()

            message = '{:<15} {:<10d}'.format(taxonomy_id, int(sample_count))
            message += ''.join(['{:>12.4f}'.format(value) for value in avg_vals])
            self.train_record(message)

        overall_avgs = test_metrics.avg()
        message_overall = '{:<15} {:<10}'.format('Overall', '')  # Overall行不显示样本数，留空
        message_overall += ''.join(['{:>12.4f}'.format(value) for value in overall_avgs])
        self.train_record(message_overall)

        message_summary = '#{:d} {:.4f} {:.4f} {:.4f} | {:.4f} | #{:d} {:.4f}'.format(
            self.epoch,
            overall_avgs[0], overall_avgs[1], overall_avgs[2],
            overall_avgs[0],
            self.best_epoch, self.best_metrics
        )
        self.test_record(message_summary)

        return overall_avgs[1]