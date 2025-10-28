# -*- encoding: utf-8 -*-
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import yaml
import torch
import importlib
import faulthandler
import torch.nn as nn
import shutil
import inspect
import time
from collections import OrderedDict

faulthandler.enable()
import utils
import numpy as np
import os
import pdb
import sys
import copy
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from libs.slr_eval.wer_calculation import evaluate
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from .log_manager import LogManager

class ExperimentManager:
    # 类变量
    arg = None
    rng = None
    device = None
    dataset = {}
    data_loader = {}
    gloss_dict = None
    model = None
    optimizer = None
    kernel_sizes = None
    
    @classmethod
    def init(cls, arg):
        cls.arg = arg
        if cls.arg.random_fix:
            cls.rng = utils.RandomState(seed=cls.arg.random_seed)
        cls.device = utils.GpuDataParallel()
        cls.dataset = {}
        cls.data_loader = {}
        cls.gloss_dict = np.load(cls.arg.dataset_info['dict_path'], allow_pickle=True).item()
        cls.arg.model_args['num_classes'] = len(cls.gloss_dict) + 1
        cls.model, cls.optimizer = cls.loading()

    @classmethod
    def start(cls):
        if cls.arg.phase == 'train':
            best_dev = 100.0
            best_epoch = 0
            total_time = 0
            epoch_time = 0
            LogManager.info('Parameters:\n{}\n'.format(str(vars(cls.arg))))
            seq_model_list = []
            for epoch in range(cls.arg.optimizer_args['start_epoch'], cls.arg.num_epoch):
                save_model = epoch % cls.arg.save_interval == 0
                eval_model = epoch % cls.arg.eval_interval == 0
                epoch_time = time.time()
                # train end2end model
                seq_train(cls.data_loader['train'], cls.model, cls.optimizer,
                          cls.device, epoch, loss_weights=cls.arg.loss_weights)
                if eval_model:
                    dev_wer = seq_eval(cls.arg, cls.data_loader['dev'], cls.model, cls.device,
                                       'dev', epoch, cls.arg.work_dir)
                    LogManager.info("Dev WER: {:05.2f}".format(dev_wer))
                if dev_wer < best_dev:
                    best_dev = dev_wer
                    best_epoch = epoch
                    model_path = "{}_best_model.pt".format(cls.arg.work_dir)
                    cls.save_model(epoch, model_path)
                    LogManager.info('Save best model')
                LogManager.info('Best_dev: {:05.2f}, Epoch : {}'.format(best_dev, best_epoch))
                if save_model:
                    model_path = "{}dev_{:05.2f}_epoch{}_model.pt".format(cls.arg.work_dir, dev_wer, epoch)
                    seq_model_list.append(model_path)
                    cls.save_model(epoch, model_path)
                epoch_time = time.time() - epoch_time
                total_time += epoch_time
                LogManager.info('Epoch {} costs {} mins {} seconds'.format(epoch, int(epoch_time)//60, int(epoch_time)%60))
            LogManager.info('Training costs {} hours {} mins {} seconds'.format(int(total_time)//60//60, int(total_time)//60%60, int(total_time)%60))
        elif cls.arg.phase == 'test':
            if cls.arg.load_weights is None:
                LogManager.info('Please appoint --weights.')
            LogManager.info('Model:   {}.'.format(cls.arg.model))
            LogManager.info('Weights: {}.'.format(cls.arg.load_weights))
            # train_wer = seq_eval(cls.arg, cls.data_loader["train_eval"], cls.model, cls.device,
            #                      "train", 6667, cls.arg.work_dir, cls.recoder)
            dev_wer = seq_eval(cls.arg, cls.data_loader["dev"], cls.model, cls.device,
                               "dev", 6667, cls.arg.work_dir, cls.recoder)
            test_wer = seq_eval(cls.arg, cls.data_loader["test"], cls.model, cls.device,
                                "test", 6667, cls.arg.work_dir, cls.recoder)
            LogManager.log('Evaluation Done.\n')

    @classmethod
    def run_inference(cls, video_data, video_length):
        cls.model.eval()
        with torch.no_grad():
            ret_dict = cls.model(video_data, video_length)
        return ret_dict['recognized_sents']

    @classmethod
    def save_model(cls, epoch, save_path):
        torch.save({
            'epoch': epoch,
            'model_state_dict': cls.model.state_dict(),
            'optimizer_state_dict': cls.optimizer.state_dict(),
            'scheduler_state_dict': cls.optimizer.scheduler.state_dict(),
            'rng_state': cls.rng.save_rng_state(),
        }, save_path)

    @classmethod
    def loading(cls):
        cls.device.set_device ( cls.arg.device )
        print("Loading model")
        model_class = import_class(cls.arg.model)
        model = model_class(
            **cls.arg.model_args,
            gloss_dict=cls.gloss_dict,
            loss_weights=cls.arg.loss_weights,
        )
        # shutil.copy2(inspect.getfile(model_class), cls.arg.work_dir)
        optimizer = utils.Optimizer(model, cls.arg.optimizer_args)

        if cls.arg.load_weights:
            cls.load_model_weights(model, cls.arg.load_weights)
        elif cls.arg.load_checkpoints:
            cls.load_checkpoint_weights(model, optimizer)
        model = cls.model_to_device(model)
        cls.kernel_sizes = model.conv1d.kernel_size
        print("Loading model finished.")
        cls.load_data()
        return model, optimizer

    @classmethod
    def model_to_device(cls, model):
        model = model.to(cls.device.output_device)
        if len(cls.device.gpu_list) > 1:
            model.conv2d = nn.DataParallel(
                model.conv2d,
                device_ids=cls.device.gpu_list,
                output_device=cls.device.output_device)
        model.cuda()
        return model

    @classmethod
    def load_model_weights(cls, model, weight_path):
        state_dict = torch.load(weight_path)
        if len(cls.arg.ignore_weights):
            for w in cls.arg.ignore_weights:
                if state_dict.pop(w, None) is not None:
                    print('Successfully Remove Weights: {}.'.format(w))
                else:
                    print('Can Not Remove Weights: {}.'.format(w))
        weights = cls.modified_weights(state_dict['model_state_dict'], False)
        # weights = cls.modified_weights(state_dict['model_state_dict'])
        model.load_state_dict(weights, strict=True)

    @staticmethod
    def modified_weights(state_dict, modified=False):
        state_dict = OrderedDict([(k.replace('.module', ''), v) for k, v in state_dict.items()])
        if not modified:
            return state_dict
        modified_dict = dict()
        return modified_dict

    @classmethod
    def load_checkpoint_weights(cls, model, optimizer):
        cls.load_model_weights(model, cls.arg.load_checkpoints)
        state_dict = torch.load(cls.arg.load_checkpoints)

        if len(torch.cuda.get_rng_state_all()) == len(state_dict['rng_state']['cuda']):
            print("Loading random seeds...")
            cls.rng.set_rng_state(state_dict['rng_state'])
        if "optimizer_state_dict" in state_dict.keys():
            print("Loading optimizer parameters...")
            optimizer.load_state_dict(state_dict["optimizer_state_dict"])
            #for k, v in optimizer.state_dict().items():
            #    if torch.is_tensor(v):
            #        optimizer.state_dict()[k] = v.cuda()
        if "scheduler_state_dict" in state_dict.keys():
            print("Loading scheduler parameters...")
            optimizer.scheduler.load_state_dict(state_dict["scheduler_state_dict"])

        cls.arg.optimizer_args['start_epoch'] = state_dict["epoch"] + 1
        # cls.recoder.print_log(f"Resuming from checkpoint: epoch {cls.arg.optimizer_args['start_epoch']}")

    @classmethod
    def load_data(cls):
        print("Loading data")
        cls.feeder = import_class(cls.arg.feeder)
        dataset_list = zip(["train", "train_eval", "dev", "test"], [True, False, False, False]) if 'phoenix' in cls.arg.dataset else zip(["train", "dev"], [True, False])
        for idx, (mode, train_flag) in enumerate(dataset_list):
            arg = cls.arg.feeder_args
            arg["prefix"] = cls.arg.dataset_info['dataset_root']
            arg["mode"] = mode.split("_")[0]
            arg["transform_mode"] = train_flag
            arg['dataset'] = cls.arg.dataset
            cls.dataset[mode] = cls.feeder(gloss_dict=cls.gloss_dict, kernel_size=cls.kernel_sizes, **arg)
            cls.data_loader[mode] = cls.build_dataloader(cls.dataset[mode], mode, train_flag)
        print("Loading data finished.")

    @classmethod
    def build_dataloader(cls, dataset, mode, train_flag):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=cls.arg.batch_size if mode == "train" else cls.arg.test_batch_size,
            shuffle=train_flag,
            drop_last=train_flag,
            num_workers=cls.arg.num_worker,  # if train_flag else 0
            collate_fn=cls.feeder.collate_fn,
            pin_memory=True,
        )



def import_class(name):
    components = name.rsplit('.', 1)
    mod = importlib.import_module(components[0])
    mod = getattr(mod, components[1])
    return mod




def seq_train(loader, model, optimizer, device, epoch_idx, loss_weights=None):
    model.train()
    loss_value = []
    total_loss_dict = {}    # dict of all types of loss
    for k in loss_weights.keys(): 
        total_loss_dict[k] = 0
    clr = [group['lr'] for group in optimizer.optimizer.param_groups]
    scaler = GradScaler()
    for batch_idx, data in enumerate(tqdm(loader)):
        vid = device.data_to_device(data[0])
        vid_lgt = device.data_to_device(data[1])
        label = device.data_to_device(data[2])
        label_lgt = device.data_to_device(data[3])
        optimizer.zero_grad()
        with autocast():
            ret_dict = model(vid, vid_lgt, label=label, label_lgt=label_lgt)
            loss, loss_dict = model.criterion_calculation(ret_dict, label, label_lgt)
        if np.isinf(loss.item()) or np.isnan(loss.item()):
            LogManager.info('loss is nan')
            LogManager.info(str(data[1])+'  frames')
            LogManager.info(str(data[3])+'  glosses')
            continue
        scaler.scale(loss).backward()
        scaler.step(optimizer.optimizer)
        scaler.update()
        loss_value.append(loss.item())
        for item, value in loss_dict.items():
            total_loss_dict[item] += value
        if batch_idx % 200 == 0:
            LogManager.info(
                '\tEpoch: {}, Batch({}/{}) done. Loss: {:.8f}  lr:{:.6f}'
                    .format(epoch_idx, batch_idx, len(loader), loss.item(), clr[0]))
            for item, value in total_loss_dict.items():
                LogManager.info(f'\t Mean {item} loss: {value/200:.5f}')
            total_loss_dict = {}
            for k in loss_weights.keys():
                total_loss_dict[k] = 0
        del ret_dict
        del loss
    optimizer.scheduler.step()
    LogManager.info('\tMean training loss: {:.10f}.'.format(np.mean(loss_value)))
    return loss_value


def seq_eval(cfg, loader, model, device, mode, epoch, work_dir):
    model.eval()
    total_sent = []
    total_info = []
    #save_file = {}
    stat = {i: [0, 0] for i in range(len(loader.dataset.dict))}
    for batch_idx, data in enumerate(tqdm(loader)):
        vid = device.data_to_device(data[0])
        vid_lgt = device.data_to_device(data[1])
        label = device.data_to_device(data[2])
        label_lgt = device.data_to_device(data[3])
        with torch.no_grad():
            ret_dict = model(vid, vid_lgt, label=label, label_lgt=label_lgt)

        total_info += [file_name.split("|")[0] for file_name in data[-1]]
        total_sent += ret_dict['recognized_sents']
    try:
        write2file(work_dir + "output-hypothesis-{}.ctm".format(mode), total_info, total_sent)
        ret = evaluate(prefix=work_dir, mode=mode, output_file="output-hypothesis-{}.ctm".format(mode),
                       evaluate_dir=cfg.dataset_info['evaluation_dir'],
                       evaluate_prefix=cfg.dataset_info['evaluation_prefix'],
                       output_dir="epoch_{}_result/".format(epoch))
    except:
        LogManager.error(f"Unexpected error: {sys.exc_info()[0]}")
        ret = "Percent Total Error       =  100.00%   (ERROR)"
        return float(ret.split("=")[1].split("%")[0])
    finally:
        pass
    LogManager.info("Epoch {}, {} {}".format(epoch, mode, ret))
    return float(ret.split("=")[1].split("%")[0])

def write2file(path, info, output):
    filereader = open(path, "w")
    for sample_idx, sample in enumerate(output):
        for word_idx, word in enumerate(sample):
            filereader.writelines(
                f"{info[sample_idx]} 1 {word_idx * 1.0 / 100:.2f} {(word_idx + 1) * 1.0 / 100:.2f} {word[0]}\n"
            )



