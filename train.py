import argparse
import warnings
from datetime import datetime

import torch
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, DeviceStatsMonitor
from dataset.afgsa_data import AFGSADataModule
from dataset.ybc_data import *
from models.DenoisingMamba.v3 import DenoisingMambaModel
from models.Demc.v1 import DEMCModel

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.set_float32_matmul_precision('high')  # 计算精度


def parse_args():
    # 传入参数
    arg = argparse.ArgumentParser()
    arg.add_argument("--random_seed", type=int, default=20010408)
    arg.add_argument("--img_size", type=int, default=120)
    arg.add_argument("--debug", type=bool, default=False)
    arg.add_argument("--dataset", type=str, default=r"/home/yujiajing0408/PycharmProjects/MD/datas")
    arg.add_argument("--data_name", type=str, default="kjl")
    arg.add_argument("--data_type", type=str, default="h5")
    arg.add_argument("--data_clip",type=bool, default=False)
    arg.add_argument("--device", type=str, default="auto")
    arg.add_argument("--epochs", type=int, default=50)
    arg.add_argument("--lrMilestone", type=int, default=50)
    arg.add_argument("--batch", type=int, default=8)
    arg.add_argument("--loss_rate", type=tuple, default=(0.65, 0.35))
    arg.add_argument("--lr", type=float, default=2.5e-3)
    arg.add_argument("--model", type=str, default="AFGM-v3") # 模型名字-版本
    arg.add_argument("--model_path", type=str, default=None)
    arg.add_argument("--mode", type=str, default="train")
    arg.add_argument("--log_path", type=str, default="/media/yujiajing0408/Data/MD_logs")
    # 返回解析
    args = arg.parse_args()
    return args


def builds(args):
    model = None
    data_module = None
    # 模型
    m = args.model.split('-')
    assert len(m) == 2,"model参数应该为(名字-版本)"
    model_name = m[0]
    model_version = m[1]
    if model_name == "DM":
        model = DenoisingMambaModel(
            img_size=args.img_size,
            loss_weights=args.loss_weights,
            epochs=args.epochs,
            lrMilestone=args.lrMilestone,
            lr=args.lr,
            mix_rule=args.mix_rules,
            save_img_time=5,
            if_abs_pos_embed=True)

    elif model_name == "DEMC":
        model = DEMCModel(
            img_size=args.img_size,
            epochs=args.epochs,
            lrMilestone=args.lrMilestone,
            lr=args.lr,
            loss_weights=args.loss_weights,
            aux_in_channels=7,
            aux_out_channels=32,
            base_channels=32,
            save_img_time=20,
        )
    else:
        assert "未知模型"
    # 数据集
    data_dir = rf'{args.dataset}/{args.data_name}'
    if args.data_name == "kjl":
        data_module = AFGSADataModule(
            data_dir=data_dir,
            batch_size=args.batch,
            num_workers=1,
            pin_memory=True,
            img_size=args.img_size,
            preprocess=False
        )
    elif args.data_name == "ybc":
        data_module = YBCDataModule(
            data_dir="/media/yujiajing0408/Study/YCB",
            npy_dir="/media/yujiajing0408/Data/YCB_NP",
            batch_size=args.batch,
            spp=SPP_8,
            num_workers=20,
            sample_size=(128,128),
            need_reinhard = True,
            need_dispose_normal=True,
            pin_memory=True,
        )
    return model, data_module


def train(args):
    random.seed(args.random_seed)
    # 设置PyTorch的随机种子
    torch.manual_seed(args.random_seed)
    # 设置CUDA的随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
    # 设置numpy的随机种子
    np.random.seed(args.random_seed)
    # 保存节点回调函数
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename=args.model + '-{epoch}-{val_loss:.6f}',
        save_top_k=2,
        mode='min',
        save_last=True
    )
    # 初始化损失权重
    args.loss_weights = {
        'loss_s': args.loss_rate[0],
        'loss_g': args.loss_rate[1],
    }
    now = datetime.now()
    version = f"[{now.year}-{str(now.month).rjust(2, '0')}-{now.day}:{now.hour}]-|epoch={args.epochs}|-|lr={args.lr}|-|loss_rate={args.loss_rate[0]}:{args.loss_rate[1]}|"
    # 日志设置
    logger = TensorBoardLogger(args.log_path if args.log_path != '' else 'tb_logs', name=f"{args.model}-{args.data_name}",
                               version=version, log_graph=True)
    # 训练器配置
    trainer = Trainer(max_epochs=args.epochs,
                      log_every_n_steps=1,
                      logger=logger,
                      fast_dev_run=args.debug,
                      callbacks=[checkpoint_callback],
                      profiler="simple"
                      )

    model,data_module = builds(args)
    if args.model_path is not None:
        print("继续训练：{}".format(args.model_path))
        trainer.fit(model, data_module, ckpt_path=args.model_path)
    else:
        trainer.fit(model, data_module)
    print('完成训练！')

if __name__ == '__main__':
    # 参数
    torch.autograd.set_detect_anomaly(True)
    args = parse_args()
    # args.model_path = r''
    args.epochs = 1500
    args.model = "{}-{}".format("DEMC","v3")
    # args.model_path = '/media/yujiajing0408/Data/MD_logs/AFGM-v3-ybc/[2025-02-4:3]-|epoch=1000|-|lr=0.002|-|loss_rate=0.65:0.35|_32SPP|PSNR-01:29:11/checkpoints/last.ckpt'
    args.data_name = "ybc"
    args.data_type = "npz"
    # args.mix_rules = {
    #     "noisy":(0,3),
    #     "aux":(3,23)
    # } if args.data_name == "ybc" else None
    args.mix_rules = {
        "noisy":(0,3),
        "aux":(3,9)
    }
    args.lr = 2.3e-3
    # args.lr = 1e-4
    args.lrMilestone = 30
    args.batch = 64
    args.img_size = 120
    # args.debug = True
    # 训练
    train(args)