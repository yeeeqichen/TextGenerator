import torch
import os
import random
import numpy as np
from datetime import datetime


def train(model, optimizer, tb_writer, scheduler, device, args):
    print('starting training')
    overall_step = 0
    running_loss = 0
    for epoch in range(args.epochs):
        print('epoch {}'.format(epoch + 1))
        now = datetime.now()
        print('time: {}'.format(now))
        x = np.linspace(0, args.num_pieces - 1, args.num_pieces, dtype=np.int32)
        random.shuffle(x)
        piece_num = 0
        for i in x:
            with open(args.tokenized_data_path + 'tokenized_train_{}.txt'.format(i), 'r') as f:
                line = f.read().strip()
            tokens = line.split()
            tokens = [int(token) for token in tokens]
            start_point = 0
            samples = []
            while start_point < len(tokens) - args.n_ctx:
                samples.append(tokens[start_point: start_point + args.n_ctx])
                start_point += args.stride
            if start_point < len(tokens):
                samples.append(tokens[len(tokens)-args.n_ctx:])
            random.shuffle(samples)
            for step in range(len(samples) // args.batch_size):  # drop last

                #  prepare data
                batch = samples[step * args.batch_size: (step + 1) * args.batch_size]
                batch_inputs = []
                for ids in batch:
                    int_ids = [int(x) for x in ids]
                    batch_inputs.append(int_ids)
                batch_inputs = torch.tensor(batch_inputs).long().to(device)

                #  forward pass
                outputs = model.forward(input_ids=batch_inputs, labels=batch_inputs)
                loss, logits = outputs[:2]

                if args.gradient_accumulation > 1:
                    loss = loss / args.gradient_accumulation

                #  loss backward
                # if fp16:
                #     with amp.scale_loss(loss, optimizer) as scaled_loss:
                #         scaled_loss.backward()
                #         torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
                # else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                #  optimizer step
                if (overall_step + 1) % args.gradient_accumulation == 0:
                    running_loss += loss.item()
                    optimizer.step()
                    optimizer.zero_grad()
                    # scheduler.step()
                if (overall_step + 1) % args.log_step == 0:
                    # tb_writer.add_scalar('loss', loss.item() * args.gradient_accumulation, overall_step)
                    print('now time: {}:{}. Step {} of piece {} of epoch {}, loss {}'.format(
                        datetime.now().hour,
                        datetime.now().minute,
                        step + 1,
                        piece_num,
                        epoch + 1,
                        running_loss * args.gradient_accumulation / (args.log_step / args.gradient_accumulation)))
                    running_loss = 0
                overall_step += 1
            piece_num += 1

        print('saving model for epoch {}'.format(epoch + 1))
        if not os.path.exists(args.output_dir + 'model_epoch{}'.format(epoch + 1)):
            os.mkdir(args.output_dir + 'model_epoch{}'.format(epoch + 1))
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(args.output_dir + 'model_epoch{}'.format(epoch + 1))
        # torch.save(scheduler.state_dict(), output_dir + 'model_epoch{}/scheduler.pt'.format(epoch + 1))
        # torch.save(optimizer.state_dict(), output_dir + 'model_epoch{}/optimizer.pt'.format(epoch + 1))
        print('epoch {} finished'.format(epoch + 1))

        then = datetime.now()
        print('time: {}'.format(then))
        print('time for one epoch: {}'.format(then - now))

    print('training finished')
    if not os.path.exists(args.output_dir + 'final_model'):
        os.mkdir(args.output_dir + 'final_model')
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(args.output_dir + 'final_model')
    # torch.save(scheduler.state_dict(), output_dir + 'final_model/scheduler.pt')
    # torch.save(optimizer.state_dict(), output_dir + 'final_model/optimizer.pt')

