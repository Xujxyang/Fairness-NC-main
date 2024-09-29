# File name: main.py
# Description:
# Author: Marion Bartl
# Date: 18-5-20


import argparse
import math
import random
import time

import numpy as np
import pandas as pd
import torch
from nltk import sent_tokenize
from torch.utils.data import TensorDataset, RandomSampler, DataLoader
from transformers import BertTokenizer, BertForMaskedLM, AdamW, get_linear_schedule_with_warmup

from bias_utils.utils import model_evaluation, mask_tokens, input_pipeline, format_time, statistics
import os

from torch import Tensor
from typing import List, Optional, Tuple, Union

from torch.utils.tensorboard import SummaryWriter

def process_batch(
    embeds: Tensor,
    batch: Tensor,
    masks: Tensor,
    stats_device: Union[str, torch.device] = "cpu",
) -> Tuple[Tensor, Tensor]:
    """Construct a uniform batch of sequences.
    model: CausalLM to to make token predictions on sequences.
    batch: list of original sequences.
    stats_device: which device (cpu/gpu) on which to infer.
    """
    # import pdb; pdb.set_trace()
    # distil model [16, 512, 768]
    # embeds = output.last_hidden_state
    # embeds = batch.logits
    if masks is not None:
        embeds = torch.unsqueeze(masks.to(embeds.device), -1) * embeds

    # offset by one for the next word prediction
    Y = batch[:, 1:].to(stats_device)  #[16, 511]
    X = embeds[:, :-1].to(stats_device) #[16, 511, 768]
    if masks is not None:
        idx = masks[:, 1:].bool()
        Y, X = Y[idx], X[idx]   #[8176], [8176, 768]

    return X.squeeze(), Y.squeeze()



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', help='provide language, either EN or DE', required=True)
    parser.add_argument('--eval', help='.tsv file with sentences for bias evaluation (BEC-Pro or transformed EEC)', required=True)
    parser.add_argument('--tune', help='.tsv file with sentences for fine-tuning (GAP flipped)', required=False)
    parser.add_argument('--out', help='output directory + filename', required=True)
    parser.add_argument('--model', help='which BERT model to use', required=False)
    parser.add_argument('--batch', help='fix batch-size for fine-tuning', required=False, default=16)
    parser.add_argument('--seed', required=False, default=42)
    parser.add_argument('--save_path', required=False, default='cased_nc3_lr_2e-5_all_100')
    parser.add_argument('--learning_rate', required=False, default=2e-5)
    parser.add_argument('--adam_epsilon', required=False, default=1e-8)
    args = parser.parse_args()
    return args

def save_model(processed_model, epoch, tokenizer, args):
    lr, eps = args.learning_rate, args.adam_epsilon
    output_dir = './model_save/{}/epoch_{}/'.format(args.save_path, epoch)

    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Saving model to %s" % output_dir)

    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = processed_model.module if hasattr(processed_model, 'module') else processed_model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    torch.save([epoch, lr, eps], os.path.join(output_dir, 'training_args.bin'))

def fine_tune(model, dataloader, epochs, tokenizer, device):
    model.to(device)
    model.train()

    # ##### NEXT part + comments from tutorial:
    # https://colab.research.google.com/drive/1Y4o3jh3ZH70tl6mCd76vz_IxX23biCPP#scrollTo=oCYZa1lQ8Jn8&forceEdit=true
    # &sandboxMode=true
    # Note: AdamW is a class from the huggingface transformers library (as opposed to pytorch) I
    # believe the 'W' stands for 'Weight Decay fix'
    print(args.learning_rate)
    optimizer = AdamW(model.parameters(),
                      lr=args.learning_rate,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps=args.adam_epsilon)  # args.adam_epsilon  - default is 1e-8.


    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)
    # Store the average loss after each epoch so we can plot them.
    loss_values = []

    writer = SummaryWriter('./logs/cased-2e-5-all-100')
    
    import json
    # import sys
    # sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from lib.collapse import Statistics
    C, D = 28996, 768 #28996 30522
    stats = Statistics(C, D, args.device)

    indices = []
    file_path = '/opt/data/private/NC/BERT-ASE/ase_word_indices_gender_cased.json'
    with open(file_path, 'r', encoding='utf-8') as file:
        data_json = json.load(file)
    for category in data_json.values():
        for index in category.values():
            if index is not None:
                indices.append(index)
    indices = list(set(indices))
    indices = sorted(indices)
    # import pdb; pdb.set_trace()

    for epoch_i in range(0, epochs):
        print('')
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_loss = 0

        for step, batch in enumerate(dataloader):

            # Progress update every 40 batches.
            global_step = epoch_i * len(dataloader) + step
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(dataloader), elapsed))

            # mask inputs so the model can actually learn something
            b_input_ids, b_labels = mask_tokens(batch[0], tokenizer)
            b_input_ids = b_input_ids.to(device)
            b_labels = b_labels.to(device)
            b_input_mask = batch[1].to(device)

            model.zero_grad()

            # import pdb; pdb.set_trace()
            outputs = model(b_input_ids,
                            attention_mask=b_input_mask,
                            labels=b_labels)  #masked_lm_labels=b_labels

            # The call to `model` always returns a tuple, so we need to pull the
            # loss value out of the tuple.
            loss = outputs[0]

            total_loss += loss.item()
            # NC3
            X, Y = process_batch(outputs.hidden_states[12].detach(), b_input_ids, b_input_mask, torch.cuda.current_device())
            stats.collect_means(X.detach(), Y.detach(), b_input_ids.size(0))
            W = list(model.cls.predictions.decoder.parameters())[0]
            indices_have = [idx for idx in indices if stats.counts[idx] != 0]
            sims = stats.similarity(W, indices_have)
            loss_nc3 = sims.std()

            name_loss_nc3 = "loss_nc3_" + str(torch.cuda.current_device())
            writer.add_scalar(name_loss_nc3, loss_nc3, global_step)

            name_loss_ori = "name_loss_ori_" + str(torch.cuda.current_device())
            writer.add_scalar(name_loss_ori, loss, global_step)

            loss = loss + 100 * loss_nc3

            name_loss_total = "loss_total_" + str(torch.cuda.current_device())
            writer.add_scalar(name_loss_total, loss, global_step)
            # Perform a backward pass to calculate the gradients.
            loss.backward()
            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the 'exploding gradients' problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)

        # torch.exp: Returns a new tensor with the exponential of the elements of the input tensor.
        # perplexity = torch.exp(torch.tensor(avg_train_loss))

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)

        save_model(model, epoch_i+1, tokenizer, args)

        print('')
        print('  Average training loss: {0:.2f}'.format(avg_train_loss))
        print('  Training epoch took: {:}'.format(format_time(time.time() - t0)))

    print('Fine-tuning complete!')

    return model


if __name__ == '__main__':

    args = parse_arguments()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # check if GPU is available
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device('cpu')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # determine which pre-trained BERT model to use:
    # if none is provided, use the provided default case for the respective language
    if args.model is None:
        if args.lang == 'EN':
            pretrained_model = 'bert-base-uncased'
        elif args.lang == 'DE':
            pretrained_model = 'bert-base-german-dbmdz-cased'
        else:
            raise ValueError('language could not be understood. Use EN or DE.')
    else:
        pretrained_model = args.model

    print('-- Prepare evaluation data --')
    # import the evaluation data; data should be a tab-separated dataframe
    eval_data = pd.read_csv(args.eval, sep='\t')

    # eval_data = eval_data[:100]

    print('-- Import BERT model --')
    tokenizer = BertTokenizer.from_pretrained(pretrained_model)
    # set up the model
    model = BertForMaskedLM.from_pretrained(pretrained_model,
                                            output_attentions=False,
                                            output_hidden_states=True)

    print('-- Calculate associations before fine-tuning --')
    st = time.time()
    # calculate associations before fine-tuning
    pre_associations = model_evaluation(eval_data, tokenizer, model, device)

    et = time.time()
    print('Calculation took {0:.2f} minutes'.format((et - st) / 60))
    # Add the associations to dataframe
    eval_data = eval_data.assign(Pre_Assoc=pre_associations)

    print('-- Import fine-tuning data --')
    if args.tune:
        if 'gap' in args.tune:
            tune_corpus = pd.read_csv(args.tune, sep='\t')
            tune_data = []
            for text in tune_corpus.Text:
                tune_data += sent_tokenize(text)
        else:
            raise ValueError('Can\'t deal with other corpora besides GAP yet.')

        # make able to handle
        # tune_data = tune_data[:5]

        # as max_len get the smallest power of 2 greater or equal to the max sentence length
        max_len_tune = max([len(sent.split()) for sent in tune_data])
        pos = math.ceil(math.log2(max_len_tune))
        max_len_tune = int(math.pow(2, pos))
        print('Max len tuning: {}'.format(max_len_tune))

        # get tokens and attentions tensor for fine-tuning data
        tune_tokens, tune_attentions = input_pipeline(tune_data, tokenizer, max_len_tune)
        assert tune_tokens.shape == tune_attentions.shape

        # set up Dataloader
        batch_size = int(args.batch)
        train_data = TensorDataset(tune_tokens, tune_attentions)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

        print('-- Set up model fine-tuning --')
        epochs = 3
        # import pdb; pdb.set_trace()
        model = fine_tune(model, train_dataloader, epochs, tokenizer, device)

        print('-- Calculate associations after fine-tuning --')
        # calculate associations after fine-tuning
        post_associations = model_evaluation(eval_data, tokenizer, model, device)

        # add associations to dataframe
        eval_data = eval_data.assign(Post_Assoc=post_associations)

    else:
        print('No Fine-tuning today.')

    # save df+associations in out-file (to be processed in R)
    eval_data.to_csv(args.out + '_' + args.lang + '.csv', sep='\t', encoding='utf-8', index=False)

    if 'Prof_Gender' in eval_data.columns:
        # divide by gender of person term
        eval_m = eval_data.loc[eval_data.Prof_Gender == 'male']
        eval_f = eval_data.loc[eval_data.Prof_Gender == 'female']

        print('-- Statistics Before --')
        statistics(eval_f.Pre_Assoc, eval_m.Pre_Assoc)
        if args.tune:
            print('-- Statistics After --')
            statistics(eval_f.Post_Assoc, eval_m.Post_Assoc)
        print('End code.')
    else:
        print('Statistics cannot be printed, code ends here.')
