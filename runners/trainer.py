import numpy as np
import torch
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.pipeline import Pipeline
from datasets.dataloader import Collection_Loader
from models.util_learner import CriterionPoint, CriterionPointLine, Optimizer
from tqdm import tqdm
torch.manual_seed(0)

class Trainer():
    def __init__(self, args, cfg):
        self.args = args
        print(f"[INFOR] Model: {cfg.regressor.name}")
        self.log_name = str(args.dataset) + "_" + str(args.scene) + "_" + str(cfg.regressor.name)
        self.pipeline = Pipeline(cfg)
        self.criterion = CriterionPointLine(cfg.train.loss.reprojection, cfg.train.num_iters)\
              if 'pl2map' in cfg.regressor.name else CriterionPoint(cfg.train.loss.reprojection,
                                                                    cfg.train.num_iters)
        self.device = torch.device(f'cuda:{args.cudaid}' if torch.cuda.is_available() else 'cpu')
        
        # to device
        self.pipeline.to(self.device)
        self.criterion.to(self.device)

        # dataloader
        train_collection = Collection_Loader(args, cfg, mode="train")
        print("[INFOR] Loaded data collection")
        self.train_loader = torch.utils.data.DataLoader(train_collection, batch_size=cfg.train.batch_size,
                                                        shuffle=cfg.train.loader_shuffle, num_workers=cfg.train.loader_num_workers, 
                                                        pin_memory=True)
        
        self.length_train_loader = len(self.train_loader)
        self.epochs = int(cfg.train.num_iters / self.length_train_loader)
        print(f"[INFOR] Total epochs: {self.epochs}")
        self.optimizer = Optimizer(self.pipeline.regressor.parameters(), self.epochs, **cfg.optimizer)
        
        if self.args.checkpoint:
            # load checkpoint and resume training
            self.start_epoch = self.pipeline.load_checkpoint(self.args.outputs, self.log_name)
            # self.start_epoch = 2024
            self.lr = self.optimizer.adjust_lr(self.start_epoch)
        else:
            self.start_epoch = 0
            self.lr = self.optimizer.lr
        self.train_log = Train_Log(args, cfg, self.length_train_loader, self.start_epoch, self.epochs)
        

    def train(self):
        print("[INFOR] Start training")
        for epoch in range(self.start_epoch, self.epochs):
            if self.train_log.is_save_checkpoint():
                self.pipeline.save_checkpoint(self.args.outputs, self.log_name, epoch) # overwrite(save) checkpoint per epoch
            for batch_idx, (data, target) in enumerate(tqdm(self.train_loader)):
                iters = epoch*self.length_train_loader + batch_idx
                loss,_ = step_fwd(self.pipeline, self.device, data, target, iters,
                                   self.criterion, self.optimizer, train=True)
                self.train_log.update(epoch, batch_idx, loss, self.lr)
            self.lr = self.optimizer.adjust_lr(epoch) # adjust learning rate
            self.train_log.show(epoch) # show loss per epoch
        # self.pipeline.save_checkpoint(self.args.outputs, self.log_name, epoch, True)    


def step_fwd(model, device, data, target=None, iteration=2500000,
             criterion=None, optim=None, train=False):
    """
    A training/validation step."""
    if train: 
        assert criterion is not None
        assert target is not None
    for k,v in data.items():
        if isinstance(v,list):
            continue
        data[k] = data[k].to(device)
    if target is not None:
        for k,_ in target.items():
                target[k] = target[k].to(device)
    output = model(data)
    loss = None
    if train:
        loss = criterion(output, target, iteration)
        if optim is not None:
            optim.learner.zero_grad()
            loss[0].backward()
            optim.learner.step() 
    return loss, output

class Train_Log():
    def __init__(self, args, cfg, length_loader, start_epoch, total_epoch=0) -> None:
        self.args = args
        self.cfg = cfg
        self.total_epoch = total_epoch
        self.log_interval = cfg.train.log_interval
        self.length_train_loader = length_loader
        self.vis_env = str(args.dataset) + "_" + str(args.scene) + \
              "_" + str(cfg.regressor.name) +"_"+ str(args.experiment_version)
        self.showloss = ShowLosses(total_epoch=self.total_epoch)
        self.list_fignames = ['total_loss', 'point_loss', 'point_uncer_loss', 
                              'line_loss', 'line_uncer_loss', 'points_prj_loss',
                                'lines_prj_loss', 'penalty_linedepth', 'penalty_linelength', 'learning_rate']
        if self.args.visdom:
            from visdom import Visdom
            print("[INFOR] Visdom is used for log visualization")
            self.vis = Visdom()
            for name in self.list_fignames:
                self.add_fig(name, start_epoch)

    def add_fig(self, name, start_epoch):
        self.vis.line(X=np.asarray([start_epoch]), Y=np.zeros(1), win=name,
                            opts={'legend': [name], 'xlabel': 'epochs',
                                'ylabel': name}, env=self.vis_env)
    def update_fig(self, idx, epoch_count, value):
        name = self.list_fignames[idx]
        self.vis.line(X=np.asarray([epoch_count]), Y=np.asarray([value]), win=name,
                        update='append', env=self.vis_env)
        
    def update(self, epoch, batch_idx, loss, lr):
        self.showloss.update(loss)
        self.lr = lr
        if self.args.visdom:
            if batch_idx % self.log_interval == 0:
                n_iter = epoch*self.length_train_loader + batch_idx
                epoch_count = float(n_iter)/self.length_train_loader
                l = len(self.list_fignames)
                for idx in range(l-1):
                    self.update_fig(idx, epoch_count, loss[idx].item())
                self.update_fig(l-1, epoch_count, lr)

    def show(self, epoch):
        self.showloss.show(epoch, self.lr)
    def is_save_checkpoint(self):
        his_epoch_loss = self.showloss.dict_losses[0].his_epoch_loss
        if len(his_epoch_loss) == 0:
            return False
        if min(his_epoch_loss) >= his_epoch_loss[-1]:
            return True
        else:
            return False

class His_Loss():
    def __init__(self)->None:
        self.his_epoch_loss = []
        self.temp_batch_loss = []
    def update_loss(self, loss):
        self.temp_batch_loss.append(loss)
    def show(self):
        avg_loss = np.mean(self.temp_batch_loss)
        self.his_epoch_loss.append(avg_loss)
        self.temp_batch_loss = [] # reset
        return avg_loss

class ShowLosses():
    # for debugging, showing all losses if needed
    def __init__(self, list_display=[True, True, True, True, True, True, True], total_epoch=0):
        '''
        corresponding to show following losses:
        ['total_loss', 'point_loss', 'point_uncer_loss', 
                              'line_loss', 'line_uncer_loss', 'points_prj_loss',
                                'lines_prj_loss']
        '''
        self.list_display = list_display
        self.length = len(self.list_display)
        self.names = ['Avg total loss', 'A.P.L', 'A.P.U.L', 'A.L.L', 'A.L.U.L', 'A.P.P.L', 'A.P.L.L']
        # A.P.L means average point loss, A.P.P.L means average point projection loss, etc.
        self.create_dict_losses()
        self.total_epoch = total_epoch

    def create_dict_losses(self):
        self.dict_losses = {}
        for i in range(self.length):
            if self.list_display[i]:
                self.dict_losses[i] = His_Loss()
    
    def update(self, loss):
        for k,_ in self.dict_losses.items():
            self.dict_losses[k].update_loss(loss[k].item())


    def show(self, epoch, lr=0.0):
        content = f"Epoch {epoch}/{self.total_epoch} | "
        for k,_ in self.dict_losses.items():
            avg_loss = self.dict_losses[k].show()
            content += self.names[k] + f": {avg_loss:.5f} | "
        content = content + f"lr: {lr:.6f}"
        print(content)

    