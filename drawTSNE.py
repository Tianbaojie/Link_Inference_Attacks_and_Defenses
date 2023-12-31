from doctest import TestResults
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns
import argparse
import os
import logging
import random
import pickle
import numpy as np
from tqdm import tqdm, trange
import gc

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter

from transformers import BertConfig, BertTokenizer, BertTokenizerFast
from transformers import XLMRobertaConfig, XLMRobertaTokenizer, XLMRobertaTokenizerFast
from transformers import AdamW, get_linear_schedule_with_warmup

from model import BertABSATagger, XLMRABSATagger
from seq_utils import compute_metrics_absa,compute_metrics_ate,compute_metrics_ner
from data_utils import XABSAKDDataset
from data_utils import build_or_load_dataset, get_tag_vocab, write_results_to_log 
from defineLoss import getLoss

logger = logging.getLogger(__name__)


MODEL_CLASSES = {
    'bert': (BertConfig, BertABSATagger, BertTokenizerFast),
    'mbert': (BertConfig, BertABSATagger, BertTokenizerFast),
    'xlmr': (XLMRobertaConfig, XLMRABSATagger, XLMRobertaTokenizerFast)
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def init_args():
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument("--tfm_type", default='mbert', type=str, required=True,
                        help="The base transformer, selected from: [bert, mbert, xlmr]")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to the pre-trained model or shortcut name")
    parser.add_argument("--exp_type", default='supervised', type=str, required=True,
                        help="Experiment type, selected from: [supervised, zero_shot, ...]")

    # source/target data and languages
    parser.add_argument("--data_dir", default='./data/', type=str, required=True, help="Base data dir")
    parser.add_argument("--src_lang", default='en', type=str, required=True, help="source language")
    parser.add_argument("--tgt_lang", default='fr', type=str, required=True, help="target language")
    parser.add_argument("--data_select", default=1.0, type=float, help="ratio of the selected data to train, 1 is to use all data")
    parser.add_argument('--tagging_schema', type=str, default='BIEOS')
    parser.add_argument('--label_path', type=str, default='')
    parser.add_argument("--ignore_cached_data", action='store_true')
    parser.add_argument("--train_data_sampler", type=str, default='random')
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")

    # Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_drawTSNE", action='store_true', help="Whether to run drawTSNE.")
    parser.add_argument("--do_distill", action='store_true', help="Whether to run knowledge distillation.")
    parser.add_argument("--trained_teacher_paths", type=str,help="path of the trained model")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev/test set.")
    parser.add_argument("--eval_begin_end", default="15-19", type=str)

    # train configs
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--freeze_bottom_layer", default=-1, type=int)
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=20.0, type=float, 
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--train_begin_saving_step", type=int, default=10000, help="Starting point of evaluation.")
    parser.add_argument("--train_begin_saving_epoch", type=int, default=10, help="Starting point of evaluation.")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=100,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--tag2idx',help="tag2idx")
    parser.add_argument('--idx2tag',help="idx2tag")
    parser.add_argument('--Lambda',type=float, default=0,help="loss weight related parameter")
    parser.add_argument('--task', type=str, default='absa',help="task=absa or ate")
    parser.add_argument('--cle_type', type=str, default='classLoss',help="cle=aspectAlignment,languageAlignment")
    # distributed training
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument('--hidden_size',type=int,default=768,help='model hidden size')
    args = parser.parse_args()

    # set up output dir: './outputs/mbert-en-fr-zero_shot/'
    output_dir = f"outputs/{args.tfm_type}-{args.task}-{args.src_lang}-{args.tgt_lang}-{args.exp_type}-{args.Lambda}-{args.cle_type}-{args.model_name_or_path}"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    args.output_dir = output_dir

    return args
def gettSNEData(args, eval_dataset, model, idx2tag, mode, step=None):
    """
    Perform evaluation on a given `eval_datset` 
    """
    eval_output_dir = args.output_dir
    # eval_dataset, eval_evaluate_label_ids = load_and_cache_examples(args, args.task_name, tokenizer, mode=mode)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    eval_loss, num_eval_steps = 0.0, 0
    preds, pred_labels, gold_labels = None, None, None
    results = {}
    embeddings=[]
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        with torch.no_grad():
            inputs = {'input_ids':      batch['input_ids'].to(args.device),
                      'attention_mask': batch['attention_mask'].to(args.device),
                      'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None,
                      'labels':         batch['labels'].to(args.device)}
            outputs = model(**inputs)
            # logits: (bsz, seq_len, label_size)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.mean().item()
        embedding = outputs[-1].detach().cpu().numpy()
        embeddings.extend(embedding)


        num_eval_steps += 1

        if preds is None:
            preds = logits.cpu().numpy()
            gold_labels = inputs['labels'].cpu().numpy()
        else:
            preds = np.append(preds, logits.cpu().numpy(), axis=0)
            gold_labels = np.append(gold_labels, inputs['labels'].cpu().numpy(), axis=0)
    
    eval_loss = eval_loss / num_eval_steps
    # argmax operation over the last dimension
    pred_labels = np.argmax(preds, axis=-1)
    if args.task=='absa':
        result, ground_truth, predictions = compute_metrics_absa(pred_labels, gold_labels, idx2tag, args.tagging_schema)
    if args.task=='ate':
        result, ground_truth, predictions = compute_metrics_ate(pred_labels, gold_labels, idx2tag, args.tagging_schema)
    if args.task=='ner':
        result, ground_truth, predictions = compute_metrics_ner(pred_labels, gold_labels, idx2tag, args.tagging_schema)
    
    result['eval_loss'] = eval_loss
    results.update(result)

    if mode == 'test':
        file_to_write= {'results': results, 'labels': ground_truth, 'preds': predictions}
        file_name_to_write = f'{args.saved_model_dir}/{args.src_lang}-{args.tgt_lang}-preds-{step}.pickle'
        pickle.dump(file_to_write, open(file_name_to_write, 'wb'))
        logger.info(f"Write predictions to {file_name_to_write}")
    mask=gold_labels[gold_labels>-100]
    embeddings=torch.tensor(embeddings)
    gold_labels=torch.tensor(gold_labels)
    pred_labels=torch.tensor(pred_labels)
    result={'embeddings':embeddings.view(-1,args.hidden_size),'gold_labels':gold_labels.view(-1),'pred_labels':pred_labels.view(-1)}
    return result
def drawTSNE(args,model,trainDataset,testDataset):
    trainResults= gettSNEData(args, trainDataset, model, args.idx2tag, mode='dev')
    testResults = gettSNEData(args, testDataset, model, args.idx2tag, mode='dev')
    '''
    nums=[]
    for tag in [key for key in args.tag2idx.keys()]:
        num=sum(list(testResults['gold_labels']==args.tag2idx[tag]))
        nums.append(num)
    print(nums)
    num=min(nums)
    print(num)
    '''
    embeddingTrain=[]
    goldY=[]
    predY=[]
    numTag=0
    goldYTrain=[]
    for tag in [key for key in args.tag2idx.keys()]:
        mask=trainResults['gold_labels']==args.tag2idx[tag]
        num=sum(mask)
        num=int(sum(mask)*0.3)
        if num<1:
            num=sum(mask)
        embeddingTrain.append(trainResults['embeddings'][mask].view(-1,args.hidden_size)[:num,:])
        numTag=numTag+1
        goldYTrain.extend([tag]*num)
    embeddingTest=[]
    for tag in [key for key in args.tag2idx.keys()]:
        mask=testResults['gold_labels']==args.tag2idx[tag]
        num=sum(mask)
        embeddingTest.append(testResults['embeddings'][mask].view(-1,args.hidden_size))
        goldY.extend([tag]*num)
        predY.extend(testResults['pred_labels'][mask].view(-1))
    embeddingTrain=torch.cat(embeddingTrain).view(-1,args.hidden_size)
    embeddingTest=torch.cat(embeddingTest).view(-1,args.hidden_size)
    #t-SNE
    embeddings=torch.cat([embeddingTrain,embeddingTest]).view(-1,args.hidden_size)

    tsne = TSNE(init='pca',perplexity=50,random_state=20,n_jobs=32)
    x = tsne.fit_transform(embeddings)
    xTrain=x[:embeddingTrain.shape[0],:]
    xTest=x[embeddingTrain.shape[0]:,:]
    pathName=f"outputs/{args.tfm_type}-{args.task}-{args.src_lang}-{args.tgt_lang}-{args.exp_type}-{args.Lambda}-{args.cle_type}-{args.model_name_or_path}/"
    print('tsne image saved in ',pathName)
    plt.figure()
    sns.scatterplot(xTrain[:,0], xTrain[:,1], hue=goldYTrain, legend='full', palette=sns.color_palette("bright",numTag))
    plt.ylim(-100,100)
    plt.xlim(-100,100)
    plt.savefig(pathName+'tsne_train.png')

    plt.figure()
    sns.scatterplot(xTest[:,0], xTest[:,1], hue=goldY, legend='full', palette=sns.color_palette("bright",numTag))
    plt.ylim(-100,100)
    plt.xlim(-100,100)
    plt.savefig(pathName+'tsne_test_gold.png')

    plt.figure()
    sns.scatterplot(xTest[:,0], xTest[:,1], hue=[args.idx2tag[int(i)] for i in predY], legend='full')
    plt.ylim(-100,100)
    plt.xlim(-100,100)
    plt.savefig(pathName+'tsne_test_pred.png')

    plt.figure()
    sns.scatterplot(xTrain[:,0], xTrain[:,1], hue=goldYTrain, legend='full', palette=sns.color_palette("bright",numTag))
    sns.scatterplot(xTest[:,0], xTest[:,1], hue=['gold'+i for i in goldY], legend='full', palette=sns.color_palette("bright",numTag))
    sns.scatterplot(xTest[:,0], xTest[:,1], hue=['pred'+args.idx2tag[int(i)] for i in predY], legend='full')
    plt.ylim(-100,100)
    plt.xlim(-100,100)
    plt.savefig(pathName+'tsne_all.png')

def get_optimizer_grouped_parameters(args, model, no_grad=None):
    no_decay = ["bias", "LayerNorm.weight"]
    if no_grad is not None:
        logger.info(" The frozen parameters are:")
        for n, p in model.named_parameters():
            p.requires_grad = False if any(nd in n for nd in no_grad) else True
            if not p.requires_grad:
                logger.info("   Freeze: %s", n)
        logger.info(" The parameters to be fine-tuned are:")
        for n, p in model.named_parameters():
            if p.requires_grad:
                logger.info("   Fine-tune: %s", n)
    else:
        for n, p in model.named_parameters():
            if not p.requires_grad:
                assert False, "parameters to update with requires_grad=False"

    outputs = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
         "weight_decay": 0.0}
    ]
    return outputs

def train(args, train_dataset, model, tokenizer,kind=-1):
    """ Train the model """

    if args.local_rank in [-1, 0]:
       tb_writer = SummaryWriter()

    # prepare dataloader
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    if args.train_data_sampler == 'random':
        train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    else:
        train_sampler = SequentialSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    # compute total update steps
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # prepare optimizer and schedule (linear warmup and decay)
    if args.freeze_bottom_layer >= 0:
        no_grad = ["embeddings"] + ["layer." + str(layer_i) + "." for layer_i in range(12) if layer_i < args.freeze_bottom_layer]
    else:
        no_grad = None
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(args, model, no_grad=no_grad)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Learning rate = %f", args.learning_rate)
    logger.info("  Model saved path = %s", args.output_dir)

    global_step = 0
    train_loss, logging_loss = 0.0, 0.0
    model.zero_grad()

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for n_epoch, _ in enumerate(train_iterator):
        epoch_train_loss = 0.0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            # batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids':      batch['input_ids'].to(args.device),
                      'attention_mask': batch['attention_mask'].to(args.device),
                      'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None,
                      'labels':         batch['labels'].to(args.device)}
            ouputs = model(**inputs)
            loss = ouputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            train_loss += loss.item()
            epoch_train_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                # update trainable parameters every gradient_accumulation_steps
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and global_step % args.save_steps == 0 and global_step >= args.train_begin_saving_step:
                    # Save model checkpoint per each N steps after X steps 
                    if 'kd_m' not in args.exp_type:
                        output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    else:
                        output_dir = os.path.join(args.output_dir, f'checkpoint-{str(kind)}-{global_step}')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    print(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model 
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break
        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break

        logger.info(f"Current epoch train loss: {epoch_train_loss:.5f}")

        # save a checkpoint when each epoch ends after a specific epoch
        '''
        n_epoch_name = n_epoch
        if n_epoch_name >= args.train_begin_saving_epoch: 
            output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(n_epoch_name))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model 
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir, 'training_args.bin'))
            logger.info("Saving model checkpoint to %s", output_dir)
        '''

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, train_loss / global_step
def trainCLE(args, train_dataset, model, tokenizer,kind=-1):
    """ Train the model """

    if args.local_rank in [-1, 0]:
       tb_writer = SummaryWriter()

    # prepare dataloader
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    if args.train_data_sampler == 'random':
        train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    else:
        train_sampler = SequentialSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    # compute total update steps
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # prepare optimizer and schedule (linear warmup and decay)
    if args.freeze_bottom_layer >= 0:
        no_grad = ["embeddings"] + ["layer." + str(layer_i) + "." for layer_i in range(12) if layer_i < args.freeze_bottom_layer]
    else:
        no_grad = None
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(args, model, no_grad=no_grad)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Learning rate = %f", args.learning_rate)
    logger.info("  Model saved path = %s", args.output_dir)

    global_step = 0
    train_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    test_dataset = build_or_load_dataset(args, tokenizer, mode='test')

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for n_epoch, _ in enumerate(train_iterator):
        epoch_train_loss = 0.0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            # batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids':      batch['input_ids'].to(args.device),
                      'attention_mask': batch['attention_mask'].to(args.device),
                      'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None,
                      'labels':         batch['labels'].to(args.device)}
            ouputs = model(**inputs)
            losses = ouputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            
            embeddings=ouputs[-1]
            labels=inputs['labels']
            maskes=inputs['attention_mask']
            #infoWeight=1/(np.exp(0.015*(global_step-args.Lambda*t_total))+1)
            infoWeight=np.exp(-args.Lambda*(global_step/t_total))
            loss=getLoss(args,losses,infoWeight,embeddings,labels,maskes)


            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            train_loss += loss.item()
            epoch_train_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                # update trainable parameters every gradient_accumulation_steps
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and global_step % args.save_steps == 0 and global_step >= args.train_begin_saving_step:
                    # Save model checkpoint per each N steps after X steps 
                    if 'kd_m' not in args.exp_type:
                        output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    else:
                        output_dir = os.path.join(args.output_dir, f'checkpoint-{str(kind)}-{global_step}')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    print(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model 
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break
        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break

        logger.info(f"Current epoch train loss: {epoch_train_loss:.5f}")
        test_result = evaluate(args, test_dataset, model, args.idx2tag, mode='train', step=global_step)
        print(test_result)
        # save a checkpoint when each epoch ends after a specific epoch
        '''
        n_epoch_name = n_epoch
        if n_epoch_name >= args.train_begin_saving_epoch: 
            output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(n_epoch_name))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model 
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir, 'training_args.bin'))
            logger.info("Saving model checkpoint to %s", output_dir)
        '''

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, train_loss / global_step


def train_kd(args, train_dataset, model, tokenizer):
    """ Train the model with the soft labels """
    # prepare dataloader
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    # compute total update steps
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    if args.freeze_bottom_layer >= 0:
        no_grad = ["embeddings"] + ["layer." + str(layer_i) + "." for layer_i in range(12) if layer_i < args.freeze_bottom_layer]
    else:
        no_grad = None
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(args, model, no_grad=no_grad)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    # Train!
    logger.info("***** Running training with soft labels *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Learning rate = %f", args.learning_rate)
    logger.info("  Model saved path = %s", args.output_dir)

    global_step = 0
    train_loss, logging_loss = 0.0, 0.0
    model.zero_grad()

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for n_epoch, _ in enumerate(train_iterator):
        epoch_train_loss = 0.0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            # batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids':      batch['input_ids'].to(args.device),
                      'attention_mask': batch['attention_mask'].to(args.device),
                      'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None,
                      'teacher_probs':  batch['teacher_probs'].to(args.device)}
            ouputs = model(**inputs)
            loss = ouputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            train_loss += loss.item()
            epoch_train_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                # update trainable parameters every gradient_accumulation_steps
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and global_step % args.save_steps == 0 and global_step >= args.train_begin_saving_step:
                    # Save model checkpoint per each N steps after 1000 steps 
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model 
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break
        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break

        logger.info(f"Current epoch train loss: {epoch_train_loss:.5f}")

    return global_step, train_loss / global_step

def evaluate(args, eval_dataset, model, idx2tag, mode, step=None):
    """
    Perform evaluation on a given `eval_datset` 
    """
    eval_output_dir = args.output_dir
    # eval_dataset, eval_evaluate_label_ids = load_and_cache_examples(args, args.task_name, tokenizer, mode=mode)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    eval_loss, num_eval_steps = 0.0, 0
    preds, pred_labels, gold_labels = None, None, None
    results = {}

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        with torch.no_grad():
            inputs = {'input_ids':      batch['input_ids'].to(args.device),
                      'attention_mask': batch['attention_mask'].to(args.device),
                      'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None,
                      'labels':         batch['labels'].to(args.device)}
            outputs = model(**inputs)
            # logits: (bsz, seq_len, label_size)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.mean().item()

        num_eval_steps += 1

        if preds is None:
            preds = logits.cpu().numpy()
            gold_labels = inputs['labels'].cpu().numpy()
        else:
            preds = np.append(preds, logits.cpu().numpy(), axis=0)
            gold_labels = np.append(gold_labels, inputs['labels'].cpu().numpy(), axis=0)
    
    eval_loss = eval_loss / num_eval_steps
    # argmax operation over the last dimension
    pred_labels = np.argmax(preds, axis=-1)
    if args.task=='absa':
        result, ground_truth, predictions = compute_metrics_absa(pred_labels, gold_labels, idx2tag, args.tagging_schema)
    if args.task=='ate':
        result, ground_truth, predictions = compute_metrics_ate(pred_labels, gold_labels, idx2tag, args.tagging_schema)
    if args.task=='ner':
        result, ground_truth, predictions = compute_metrics_ner(pred_labels, gold_labels, idx2tag, args.tagging_schema)
    
    result['eval_loss'] = eval_loss
    results.update(result)

    if mode == 'test':
        file_to_write= {'results': results, 'labels': ground_truth, 'preds': predictions}
        file_name_to_write = f'{args.saved_model_dir}/{args.src_lang}-{args.tgt_lang}-preds-{step}.pickle'
        pickle.dump(file_to_write, open(file_name_to_write, 'wb'))
        logger.info(f"Write predictions to {file_name_to_write}")
    """
    output_eval_file = os.path.join(eval_output_dir, "%s_results.txt" % mode)
    with open(output_eval_file, "w") as writer:
        #logger.info("***** %s results *****" % mode)
        for key in sorted(result.keys()):
            if 'eval_loss' in key:
                logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
        #logger.info("***** %s results *****" % mode)
    """
    return results
def get_teacher_model_path(args):
    if 'mtl' in args.exp_type:
        one_tgt_lang = 'fr'
        saved_model_dir = f"outputs/{args.tfm_type}-{args.task}-{args.src_lang}-{one_tgt_lang}-{args.exp_type}"
    
    elif args.exp_type == 'zero_shot':
        saved_model_dir = f"outputs/{args.tfm_type}-{args.task}-{args.src_lang}-{args.src_lang}-supervised"
        if not os.path.exists(saved_model_dir):
            raise Exception("No trained models can be found!")
    
    else:
        saved_model_dir = args.output_dir
    
    args.saved_model_dir = saved_model_dir

    # retrieve all the saved checkpoints for model selection
    all_checkpoints = []
    saved_model_dir=args.saved_model_dir
    for f in os.listdir(saved_model_dir):
        sub_dir = os.path.join(saved_model_dir, f)
        if os.path.isdir(sub_dir):
            all_checkpoints.append(sub_dir)
    logger.info(f"We will perform validation on the following checkpoints: {all_checkpoints},to get get_teacher_model_path")
    
    # load the dev and test dataset
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.tfm_type]
    config = config_class.from_pretrained(all_checkpoints[0])
    tokenizer = tokenizer_class.from_pretrained(all_checkpoints[0])
    logger.info("Load DEV dataset...")
    dev_dataset = build_or_load_dataset(args, tokenizer, mode='dev')

    if 'kd_s' in args.exp_type:
        best_checkpoint =''
        best_f1 = -999999.0
        for checkpoint in all_checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoint) > 1 else ""
            # only perform evaluation at the specific epochs
            eval_begin, eval_end = args.eval_begin_end.split('-')
            if True:
                # reload the model and conduct inference
                logger.info(f"\nLoad the trained model from {checkpoint}...")
                model = model_class.from_pretrained(checkpoint, config=config)
                model.to(args.device)
                dev_result = evaluate(args, dev_dataset, model, args.idx2tag, mode='dev')
                if args.task=='absa':
                    # regard the micro-f1 as the criteria of model selection
                    metrics = 'micro_f1'
                elif args.task=='ate':
                    # regard the f1 as the criteria of model selection
                    metrics = 'f1'
                if dev_result[metrics] > best_f1:
                    best_f1 = dev_result[metrics]
                    best_checkpoint = checkpoint
        return best_checkpoint
    elif 'kd_m' in args.exp_type:
        best_checkpoints = []
        best_f1, best_checkpoint, best_global_step = -999999.0, None, None
        for kind in range(3):
            best_checkpoint =''
            best_f1 = -999999.0
            for checkpoint in all_checkpoints:
                global_step = checkpoint.split('-')[-1] if len(checkpoint) > 1 else ""
                i = checkpoint.split('-')[-2] if len(checkpoint) > 1 else -1
                if int(i)!=kind:
                    continue
                # only perform evaluation at the specific epochs
                eval_begin, eval_end = args.eval_begin_end.split('-')
                if True:
                    # reload the model and conduct inference
                    logger.info(f"\nLoad the trained model from {checkpoint}...")
                    model = model_class.from_pretrained(checkpoint, config=config)
                    model.to(args.device)

                    dev_result = evaluate(args, dev_dataset, model, args.idx2tag, mode='dev')
                    if args.task=='absa':
                        # regard the micro-f1 as the criteria of model selection
                        metrics = 'micro_f1'
                    elif args.task=='ate':
                        # regard the f1 as the criteria of model selection
                        metrics = 'f1'
                    if dev_result[metrics] > best_f1:
                        best_f1 = dev_result[metrics]
                        best_checkpoint = checkpoint
            best_checkpoints.append(best_checkpoint)
        return best_checkpoints

def get_teacher_probs(args, dataset, model_class, teacher_model_path):
    teacher_model = model_class.from_pretrained(teacher_model_path)
    teacher_model.to(args.device)
    
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # compute logits for the dataset using the model!
    logger.info(f"***** Compute logits for [{args.tgt_lang}] using the model {teacher_model_path} *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    preds = None
    teacher_model.eval()
    for batch in eval_dataloader:
        with torch.no_grad():
            inputs = {'input_ids':      batch['input_ids'].to(args.device),
                      'attention_mask': batch['attention_mask'].to(args.device),
                      'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None,
                      'labels':         None}
            outputs = teacher_model(**inputs)
            logits = outputs[0]

        # nb_eval_steps += 1
        preds = logits.detach() if preds is None else torch.cat((preds, logits.detach()), dim=0) # dataset_len x max_seq_len x label_len

    preds = torch.nn.functional.softmax(preds, dim=-1)
    
    return preds


def get_multi_teacher_probs(args, dataset, model_class):
    teacher_paths = args.trained_teacher_paths 
    
    # obtain all preds
    all_preds = []
    for teacher_path in teacher_paths:
        preds = get_teacher_probs(args, dataset, model_class, teacher_path)
        all_preds.append(preds)
    
    logger.info("Fuse the soft labels from three pre-trained models")
    combined_preds = 1/3 * all_preds[0] + 1/3 * all_preds[1] + 1/3 * all_preds[2]

    return combined_preds

    
def main():
    # --------------------------------
    # Prepare the tags, env etc.
    args = init_args()
    print("\n", "="*30, f"NEW EXP ({args.src_lang} -> {args.tgt_lang} for {args.exp_type}", "="*30, "\n")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(f"Output directory ({args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome.")

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        os.environ['MASTER_ADDR'] = args.MASTER_ADDR
        os.environ['MASTER_PORT'] = args.MASTER_PORT
        torch.distributed.init_process_group(backend='nccl', rank=args.local_rank, world_size=1)
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    # not using 16-bits training
    logger.info(f"Process rank: {args.local_rank}, device: {device}, n_gpu: {args.n_gpu}") 
    logger.info(f"Distributed training: {bool(args.local_rank != -1)}, 16-bits training: False")

    # Set seed
    set_seed(args)

    # Set up task and the label
    tag_list, tag2idx, idx2tag = get_tag_vocab(args.task, args.tagging_schema, args.label_path)
    num_tags = len(tag_list)
    args.num_labels = num_tags
    args.tag2idx=tag2idx
    args.idx2tag=idx2tag
    logger.info(f"Perform {args.task} task with tagging_schema {args.tagging_schema}, label list being {tag_list} (n_labels={num_tags})")

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    # hard-code lr and bs based on parameter search
    if args.tfm_type == 'mbert':
        lr, batch_size = 5e-5, 16  
    elif args.tfm_type == 'xlmr':
        lr, batch_size = 4e-5, 25
    # args.learning_rate = lr
    # args.per_gpu_train_batch_size = batch_size
    logger.info(f"We hard-coded set lr={args.learning_rate} and bs={args.per_gpu_train_batch_size}")
    
    # -----------------------------------------------------------------
    # Training process (train a model using the data and save the model)
    #args.do_train=False
    if args.do_train:
        logger.info("\n\n***** Prepare to conduct training  *****\n")

        # Set up model (from pre-trained tfms)
        args.tfm_type = args.tfm_type.lower() 
        logger.info(f"Load pre-trained {args.tfm_type} model from `{args.model_name_or_path}`")
        logger.info(f"We set max_seq_length={args.max_seq_length}")
        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.tfm_type]
        config = config_class.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path,
            num_labels=num_tags, id2label=idx2tag, label2id=tag2idx
        )
        args.hidden_size=config.hidden_size
        # logger.info(f"config info: \n {config}")
        tokenizer = tokenizer_class.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
            do_lower_case=args.do_lower_case
        )
        # config.absa_type = args.absa_type  # type of task-specific layer, 'linear'
        if 'Kd_m' not in args.exp_type:
            model = model_class.from_pretrained(args.model_name_or_path, config=config)
            model.to(args.device)

            # Distributed and parallel training
            if args.local_rank != -1:
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                                output_device=args.local_rank,
                                                                find_unused_parameters=True)
            elif args.n_gpu > 1:
                model = torch.nn.DataParallel(model)

        # load the training dataset
        if args.exp_type=='supervised':
            logger.info("Prepare training examples...")
            train_dataset = build_or_load_dataset(args, tokenizer, mode='train')
            if args.tfm_type=='mbert':
                logger.info("***training Example ***")
                for i in range(2):
                    tokens=tokenizer.convert_ids_to_tokens(train_dataset[i]['input_ids'])
                    logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
                    logger.info("input_ids: %s", " ".join([str(x) for x in train_dataset[i]['input_ids'].tolist()]))
                    logger.info("input_mask: %s", " ".join([str(x) for x in train_dataset[i]['attention_mask'].tolist()]))
                    logger.info("segment_ids: %s", " ".join([str(x) for x in train_dataset[i]['token_type_ids'].tolist()]))
                    logger.info("label_ids: %s", " ".join([str(x) for x in train_dataset[i]['labels'].tolist()]))
            # begin training!
            _, _ = train(args, train_dataset, model, tokenizer)
        elif args.exp_type=='cle_supervised':
            logger.info("Prepare training examples...")
            train_dataset = build_or_load_dataset(args, tokenizer, mode='train')
            if args.tfm_type=='mbert':
                logger.info("***training Example ***")
                for i in range(2):
                    tokens=tokenizer.convert_ids_to_tokens(train_dataset[i]['input_ids'])
                    logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
                    logger.info("input_ids: %s", " ".join([str(x) for x in train_dataset[i]['input_ids'].tolist()]))
                    logger.info("input_mask: %s", " ".join([str(x) for x in train_dataset[i]['attention_mask'].tolist()]))
                    logger.info("segment_ids: %s", " ".join([str(x) for x in train_dataset[i]['token_type_ids'].tolist()]))
                    logger.info("label_ids: %s", " ".join([str(x) for x in train_dataset[i]['labels'].tolist()]))
            else:
                logger.info("***training Example ***")
                for i in range(2):
                    tokens=tokenizer.convert_ids_to_tokens(train_dataset[i]['input_ids'])
                    logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
                    logger.info("input_mask: %s", " ".join([str(x) for x in train_dataset[i]['attention_mask'].tolist()]))
                    logger.info("label_ids: %s", " ".join([str(x) for x in train_dataset[i]['labels'].tolist()]))
            # begin training!
            _, _ = trainCLE(args, train_dataset, model, tokenizer)
        elif 'kd_m' in args.exp_type:
            for kind in range(3):
                model = model_class.from_pretrained(args.model_name_or_path, config=config)
                model.to(args.device)

                # Distributed and parallel training
                if args.local_rank != -1:
                    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                                    output_device=args.local_rank,
                                                                    find_unused_parameters=True)
                elif args.n_gpu > 1:
                    model = torch.nn.DataParallel(model)
                logger.info(f"------------Train kind ={kind}---------------------------")
                logger.info("Prepare training examples...")

                if 'cle' in args.exp_type:
                    train_datasets = build_or_load_dataset(args, tokenizer, mode='train',kind=kind)
                    _, _ = trainCLE(args, train_datasets, model, tokenizer,kind=kind)
        
                else:
                    train_datasets = build_or_load_dataset(args, tokenizer, mode='train',kind=kind)
                    _, _ = train(args, train_datasets, model, tokenizer,kind=kind)
        elif args.exp_type=='cleacs':
            logger.info("Prepare training examples...")
            train_datasets = build_or_load_dataset(args, tokenizer, mode='train')
            # begin training!
            _, _ = trainCLE(args, train_datasets, model, tokenizer)
        
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.mkdir(args.output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            torch.save(args, os.path.join(args.output_dir, 'training_args.bin')) 
    # -----------------------------------------------------------------
    # Evaluation process (whether it is supervised setting or zero-shot)
    best_checkpoint='./outputs/bert-ner-en-en-supervised-2.5-all-bert-base-cased/checkpoint-100'
    if args.do_eval:
        if args.task=='absa':
            # regard the micro-f1 as the criteria of model selection
            metrics = 'micro_f1'
        else:
            # regard the f1 as the criteria of model selection
            metrics = 'f1'
        exp_type = args.exp_type
        logger.info("\n\n***** Prepare to conduct evaluation *****\n")
        logger.info(f"We are evaluating for *{args.tgt_lang}* under *{args.exp_type}* setting...")
        
        dev_results, test_results,target_train_results ,source_test_results= {}, {},{},{}
        best_f1, best_checkpoint, best_global_step = -999999.0, 0, 0
        all_checkpoints, global_steps = [], []

        # find the dir containing trained model, different dirs under different settings
        # if the model is multilingual, we will only use one target language for the output dir
        if 'mtl' in exp_type:
            one_tgt_lang = 'fr'
            #need fix
            saved_model_dir = f"outputs/{args.tfm_type}-{args.task}-{args.src_lang}-{one_tgt_lang}-{exp_type}"
        
        elif exp_type == 'zero_shot':

            #need fix
            saved_model_dir = f"outputs/{args.tfm_type}-{args.task}-{args.src_lang}-{args.src_lang}-supervised"
            if not os.path.exists(saved_model_dir):
                raise Exception("No trained models can be found!")
        
        else:
            saved_model_dir = args.output_dir
        
        args.saved_model_dir = saved_model_dir
        # print(args.saved_model_dir)

        # retrieve all the saved checkpoints for model selection
        for f in os.listdir(saved_model_dir):
            sub_dir = os.path.join(saved_model_dir, f)
            if os.path.isdir(sub_dir):
                all_checkpoints.append(sub_dir)
        logger.info(f"We will perform validation on the following checkpoints: {all_checkpoints}")
        
        # load the dev and test dataset
        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.tfm_type]
        config = config_class.from_pretrained(all_checkpoints[0])
        args.hidden_size=config.hidden_size
        tokenizer = tokenizer_class.from_pretrained(all_checkpoints[0])
        logger.info("Load DEV dataset...")
        dev_dataset = build_or_load_dataset(args, tokenizer, mode='dev')

        logger.info("Load targte TRAIN dataset...")
        target_train_dataset = build_or_load_dataset(args, tokenizer, mode='target train')

        logger.info("Load target TEST dataset...")
        test_dataset = build_or_load_dataset(args, tokenizer, mode='test')

        logger.info("Load source TEST dataset...")
        source_test_dataset = build_or_load_dataset(args, tokenizer, mode='source test')
        if args.tfm_type=='mbert':
            logger.info("***targtet TEST Example ***")
            for i in range(2):
                tokens=tokenizer.convert_ids_to_tokens(test_dataset[i]['input_ids'])
                logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
                logger.info("input_ids: %s", " ".join([str(x) for x in test_dataset[i]['input_ids'].tolist()]))
                logger.info("input_mask: %s", " ".join([str(x) for x in test_dataset[i]['attention_mask'].tolist()]))
                logger.info("segment_ids: %s", " ".join([str(x) for x in test_dataset[i]['token_type_ids'].tolist()]))
                logger.info("label_ids: %s", " ".join([str(x) for x in test_dataset[i]['labels'].tolist()]))
        for checkpoint in all_checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoint) > 1 else ""
            if 'kd_m' in args.exp_type:
                kind=checkpoint.split('-')[-2] if len(checkpoint) > 1 else ""
                if '0' <=kind[-1]<='2':
                    continue
            # only perform evaluation at the specific epochs
            eval_begin, eval_end = args.eval_begin_end.split('-')

            if int(eval_begin) <= int(global_step) < int(eval_end):
                global_steps.append(global_step)

                # reload the model and conduct inference
                logger.info(f"\nLoad the trained model from {checkpoint}...")
                model = model_class.from_pretrained(checkpoint, config=config)
                model.to(args.device)

                dev_result = evaluate(args, dev_dataset, model, idx2tag, mode='dev')
                if dev_result[metrics] > best_f1:
                    best_f1 = dev_result[metrics]
                    best_checkpoint = checkpoint
                    best_global_step = global_step

                # add the global step to the name of these metrics for recording
                # 'micro_f1' --> 'micro_f1_1000'
                dev_result = dict((k + '_{}'.format(global_step), v) for k, v in dev_result.items())
                dev_results.update(dev_result)
                targte_train_result = evaluate(args, target_train_dataset, model, idx2tag, mode='test', step=global_step)
                targte_train_result = dict((k + '_{}'.format(global_step), v) for k, v in targte_train_result.items())
                target_train_results.update(targte_train_result)

                test_result = evaluate(args, test_dataset, model, idx2tag, mode='test', step=global_step)
                test_result = dict((k + '_{}'.format(global_step), v) for k, v in test_result.items())
                test_results.update(test_result)

                source_test_result = evaluate(args, source_test_dataset, model, idx2tag, mode='test', step=global_step)
                source_test_result = dict((k + '_{}'.format(global_step), v) for k, v in source_test_result.items())
                source_test_results.update(source_test_result)
    
        # print test results over last few steps
        logger.info(f"\n\nThe best checkpoint is {best_checkpoint}")
        best_step_metric = f"{metrics}_{best_global_step}"
        print(f"F1 scores on source language test set: {source_test_results[best_step_metric]:.4f}")
        print(f"F1 scores on target language train set: {target_train_results[best_step_metric]:.4f}")
        print(f"F1 scores on target language test set: {test_results[best_step_metric]:.4f}")

        print("\n* Results *:  Dev  /  source test  /  target Train  /  target Test  \n")
        if args.task=='absa':
            metric_names = ['micro_f1', 'precision', 'recall', 'eval_loss']
        else:
            metric_names = ['f1', 'precision', 'recall', 'eval_loss']
        for gstep in global_steps:
            print(f"Step-{gstep}:")
            for name in metric_names:
                name_step = f'{name}_{gstep}'
                print(f"{name:<10}: {dev_results[name_step]:.4f}/{source_test_results[name_step]:.4f} / {target_train_results[name_step]:.4f}/{test_results[name_step]:.4f}", sep='  ')
            print()

        results_log_dir = './results_log'
        if not os.path.exists(results_log_dir):
            os.mkdir(results_log_dir)
        log_file_path = f"{results_log_dir}/{args.tfm_type}-{args.exp_type}-{args.src_lang}-{args.tgt_lang}-{args.Lambda}-{args.cle_type}-{args.model_name_or_path}.txt"
        write_results_to_log(log_file_path, source_test_results[best_step_metric],target_train_results[best_step_metric],test_results[best_step_metric], args, dev_results, source_test_results,target_train_results,test_results, global_steps)
    
    if args.do_drawTSNE:
        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.tfm_type]
        config = config_class.from_pretrained(best_checkpoint)
        args.hidden_size=config.hidden_size
        model = model_class.from_pretrained(best_checkpoint, config=config)
        model.to(args.device)
        tokenizer = tokenizer_class.from_pretrained(best_checkpoint)
        trainDataset = build_or_load_dataset(args, tokenizer, mode='train')
        testDataset = build_or_load_dataset(args, tokenizer, mode='test')
        logger.info("\n\n***** Prepare to draw T-SNE *****\n")
        drawTSNE(args,model,trainDataset,testDataset)


if __name__ == '__main__':
    main()
    