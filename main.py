import argparse
import torch
import os
import transformers
from pytorch_pretrained_bert import BertAdam
from tqdm import tqdm
from train import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0', type=str, required=False, help='设置使用哪些显卡')
    parser.add_argument('--tokenized_data_path', default='CrawlText/data/Shuihu/tokenized/', type=str, required=False,
                        help='tokenized语料存放位置')
    parser.add_argument('--epochs', default=5, type=int, required=False, help='训练循环')
    parser.add_argument('--batch_size', default=8, type=int, required=False, help='训练batch size')
    parser.add_argument('--lr', default=1.5e-4, type=float, required=False, help='学习率')
    parser.add_argument('--warmup_steps', default=2000, type=int, required=False, help='warm up步数')
    parser.add_argument('--log_step', default=1, type=int, required=False, help='多少步汇报一次loss，设置为gradient accumulation的整数倍')
    parser.add_argument('--stride', default=768, type=int, required=False, help='训练时取训练数据的窗口步长')
    parser.add_argument('--gradient_accumulation', default=1, type=int, required=False, help='梯度积累')
    parser.add_argument('--fp16', action='store_true', help='混合精度')
    parser.add_argument('--fp16_opt_level', default='O1', type=str, required=False)
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--num_pieces', default=100, type=int, required=False, help='将训练语料分成多少份')
    parser.add_argument('--min_length', default=128, type=int, required=False, help='最短收录文章长度')
    parser.add_argument('--output_dir', default='model/', type=str, required=False, help='模型输出路径')
    parser.add_argument('--pretrained_model', default='', type=str, required=False, help='模型训练起点路径')
    parser.add_argument('--writer_dir', default='tensorboard_summary/', type=str, required=False, help='Tensorboard路径')
    parser.add_argument('--segment', action='store_true', help='中文以词为单位')
    parser.add_argument('--bpe_token', action='store_true', help='subword')


    args = parser.parse_args()
    print('args:\n' + args.__repr__())

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.device  # 此处设置程序使用哪些显卡

    model_config = transformers.GPT2Config.from_pretrained('uer/gpt2-distil-chinese-cluecorpussmall')
    print('config:\n' + model_config.to_json_string())

    args.n_ctx = model_config.n_ctx
    print('using device:', args.device)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    model = transformers.GPT2LMHeadModel.from_pretrained('uer/gpt2-distil-chinese-cluecorpussmall')
    model.to(args.device)
    model.train()

    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    print('number of parameters: {}'.format(num_parameters))

    full_len = 0
    print('calculating total steps')
    for i in tqdm(range(args.num_pieces)):
        with open(args.tokenized_data_path + 'tokenized_train_{}.txt'.format(i), 'r') as f:
            full_len += len([int(item) for item in f.read().strip().split()])
    total_steps = int(full_len / args.stride * args.epochs / args.batch_size / args.gradient_accumulation)
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
                         t_total=total_steps
                         )

    # scheduler = transformers.WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=total_steps)

    train(model, optimizer, None, None, args.device, args)
    # if fp16:
    #     try:
    #         from apex import amp
    #     except ImportError:
    #         raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    #     model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)


if __name__ == '__main__':
    main()