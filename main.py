import argparse
import torch
import os
import transformers
from pytorch_pretrained_bert import BertAdam
from tqdm import tqdm
from train import train
from utils import MyDataset, CollateFn
from torch.utils.data import DistributedSampler, DataLoader
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, required=False, help='设置使用哪些显卡')
    parser.add_argument('--tokenized_data_path', default='CrawlText/data/Shuihu/tokenized/', type=str, required=False,
                        help='tokenized语料存放位置')
    parser.add_argument('--epochs', default=5, type=int, required=False, help='训练循环')
    parser.add_argument('--batch_size', default=8, type=int, required=False, help='训练batch size')
    parser.add_argument('--lr', default=1.5e-4, type=float, required=False, help='学习率')
    parser.add_argument('--warmup_steps', default=2000, type=int, required=False, help='warm up步数')
    parser.add_argument('--log_step', default=1, type=int, required=False, help='多少步汇报一次loss，设置为gradient accumulation的整数倍')
    parser.add_argument('--stride', default=128, type=int, required=False, help='训练时取训练数据的窗口步长')
    parser.add_argument('--gradient_accumulation', default=1, type=int, required=False, help='梯度积累')
    parser.add_argument('--fp16', action='store_true', help='混合精度')
    parser.add_argument('--fp16_opt_level', default='O1', type=str, required=False)
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--num_pieces', default=100, type=int, required=False, help='将训练语料分成多少份')
    parser.add_argument('--min_length', default=128, type=int, required=False, help='最短收录文章长度')
    parser.add_argument('--output_dir', default='model/', type=str, required=False, help='模型输出路径')
    parser.add_argument('--pretrained_model', default='D:/PythonProjects/语言模型/gpt2-chinese-distill',
                        type=str, required=False, help='模型训练起点路径')
    parser.add_argument('--writer_dir', default='tensorboard_summary/', type=str, required=False, help='Tensorboard路径')
    parser.add_argument('--segment', action='store_true', help='中文以词为单位')
    parser.add_argument('--bpe_token', action='store_true', help='subword')

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    print('using device:', args.device)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device  # 此处设置程序使用哪些显卡
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '2385'
    args.world_size = len(args.device.split(','))
    args.model_config = transformers.GPT2Config.from_pretrained(args.pretrained_model)
    args.n_ctx = args.model_config.n_ctx
    print('args:\n' + args.__repr__())
    print('config:\n' + args.model_config.to_json_string())
    # 启动world_size个进程进行训练，每个进程使用一个gpu
    torch.multiprocessing.spawn(train_proc, nprocs=args.world_size, args=(args,))


def train_proc(index, args):
    dist.init_process_group(backend='nccl',
                            init_method='env://',
                            world_size=args.world_size,
                            rank=index)
    cur_device = 'cuda:' + str(index)
    print(cur_device)
    torch.manual_seed(0)
    # model setting
    # 通过 data parallel 来加速，batch被平摊到各个卡中
    model = transformers.GPT2LMHeadModel.from_pretrained(args.pretrained_model)
    model.to(cur_device)
    model.train()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[index])

    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    print('number of parameters: {}'.format(num_parameters))

    # data setting

    train_dataset = MyDataset(tokenized_data_path=args.tokenized_data_path,
                              context_length=args.n_ctx,
                              stride=args.stride)
    train_sampler = DistributedSampler(dataset=train_dataset,
                                       rank=index,
                                       num_replicas=args.world_size)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  sampler=train_sampler,
                                  collate_fn=CollateFn(device=cur_device).collate_fn)

    total_steps = int(len(train_dataloader) / args.batch_size * args.epochs / args.gradient_accumulation
                      * args.world_size)
    print('total steps = {}'.format(total_steps))

    # optimizer setting
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.lr,
                         warmup=0.05,
                         t_total=total_steps)

    # scheduler setting
    # scheduler = transformers.WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=total_steps)

    train(model, optimizer, train_dataloader, None, args.device, args)
    # if fp16:
    #     try:
    #         from apex import amp
    #     except ImportError:
    #         raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    #     model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)


if __name__ == '__main__':
    main()
