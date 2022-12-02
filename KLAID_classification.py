import argparse
from tqdm import tqdm
from pathlib import Path
import utils
import torch
import numpy as np
import random
from classification_albef import ALBEF
from classification_albef_n_class import ALBEF_n_class
import time
import os
from vit import interpolate_pos_embed
import torch.backends.cudnn as cudnn
from transformers import BertTokenizer, BertTokenizerFast, BertModel, BertForMaskedLM
import datetime
from torchvision import transforms
import torchvision.transforms as T
import torch.nn.functional as F
from dataset import KLAID_Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from scheduler import create_scheduler
from sklearn.metrics import f1_score,roc_auc_score
from utils import *
import copy
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# json 개체를 파일이름으로 깔끔하게 저장
def jsondump(j, fname):
    with open(fname, "w", encoding="UTF8") as f:
        json.dump(j, f, ensure_ascii=False)

def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config, writer):
    # train
    model.train()

    loss_total = 0
    acc_total = 0

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps * step_size

    tqdm_data_loader = tqdm(data_loader, miniters=print_freq, desc=header)
    for i, data_dict in enumerate(tqdm_data_loader):
        text = data_dict['fact']
        for j, t in enumerate(text):    # 특수문자, 숫자 제거
            t = re.sub(r"[^\uAC00-\uD7A3\s]", "", t)
            text[j] = t
        label = data_dict['laws_service']
        one_hot_label = data_dict['one_hot_label'].to(device)

        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=20, return_tensors="pt").to(device)

        if epoch > 0 or not config['warm_up']:
            alpha = config['alpha']
        else:
            alpha = config['alpha'] * min(1., i / len(data_loader))

        loss, prediction = model(text_input, [], one_hot_label, alpha=alpha)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prediction = F.softmax(prediction, dim=-1)
        prediction[prediction>=0.5]=1.
        prediction[prediction<0.5]=0.
        prediction = prediction.cpu()
        one_hot_label = one_hot_label.cpu()
        acc = ((prediction == one_hot_label).sum() / one_hot_label.numel()).item()

        tqdm_data_loader.set_description(f'loss={loss.item():.4f}, ACC={acc:.4f}, lr={optimizer.param_groups[0]["lr"]:.6f}')
        loss_total += loss
        acc_total += acc
        
        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
            scheduler.step(i // step_size)
    
    writer.add_scalar("Train/Loss", loss_total/len(tqdm_data_loader), epoch)
    writer.add_scalar("Train/ACC", acc_total/len(tqdm_data_loader), epoch)


@torch.no_grad()
def evaluate(model, data_loader, tokenizer, device, epoch, writer):
    # test
    model.eval()

    preds=[]
    answers=[]

    for data_dict in data_loader:
        text = data_dict['fact']
        for j, t in enumerate(text):    # 특수문자, 숫자 제거
            t = re.sub(r"[^\uAC00-\uD7A3\s]", "", t)
            text[j] = t
        one_hot_label = data_dict['one_hot_label'].to(device)

        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=20, return_tensors="pt").to(device)
        prediction = model(text_input, [], one_hot_label, train=False)
        prediction = F.softmax(prediction, dim=-1).cpu()
        _, index = torch.topk(prediction, k=1, dim=-1)
        prediction = F.one_hot(index, num_classes=177)
        preds.append(prediction.squeeze().cpu())
        answers.append(one_hot_label.cpu())

    preds=torch.cat(preds,dim=0)
    answers=torch.cat(answers,dim=0)
    print('F1 score[macro]:',f1_score(answers, preds, average='macro'))
    print('AUROC:',roc_auc_score(answers, preds, average='micro'))
    print('Accuracy:',((answers == preds).sum() / answers.numel()).item())
    score = f1_score(answers, preds, average='micro')

    writer.add_scalar("Eval/F1 score", f1_score(answers, preds, average='macro'), epoch)
    writer.add_scalar("Eval/F1 AUROC", roc_auc_score(answers, preds, average='micro'), epoch)
    writer.add_scalar("Eval/Accuracy", ((answers == preds).sum() / answers.numel()).item(), epoch)

    return score


def train_sentiment_analysis(args, config):

    writer = SummaryWriter(args.output_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


    print('aa', args.evaluate)
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset ####
    print("Creating dataset")
    dataset = load_dataset("lawcompany/KLAID")
    dataset = dataset['train']

    _, dataset_test = train_test_split(dataset, test_size=0.1, shuffle=True, random_state=232)
    print("Making train dataset")
    dataset_train = KLAID_Dataset(dataset, mode="train")
    print("Making eval dataset")
    dataset_test = KLAID_Dataset(dataset_test, mode="eval")

    train_loader = DataLoader(dataset_train, batch_size=config['batch_size_train'], pin_memory=True, drop_last=True, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=config['batch_size_test'], pin_memory=True, drop_last=False, shuffle=True)

    tokenizer = BertTokenizer.from_pretrained(args.text_encoder, truncation_side="left")

    #### Model ####
    print("Creating model")
    model = ALBEF_n_class(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer, num_category=config['num_category'])

    if args.checkpoint:
        print('LOADING PRETRAINED MODEL..')
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']

        # reshape positional embedding to accomodate for image resolution change
        #pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.visual_encoder)
        #state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped

        # if not args.evaluate:
        #     if config['distill']:
        #         m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],
        #                                                      model.visual_encoder_m)
        #         state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped

        for key in list(state_dict.keys()): #to match the key name when the pretrained model used BertForMaskedLM(which contains BertModel), but fine-tuning model used BertModel
            if 'bert' in key:
                new_key = key.replace('bert.', '')
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

        msg = model.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % args.checkpoint)
        print(msg)

    model = model.to(device)
    model_without_ddp = model

    arg_opt = config['optimizer']
    optimizer = optim.AdamW(model.parameters(), lr=arg_opt['lr'], weight_decay=arg_opt['weight_decay'])

    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']
    best = 0
    best_epoch = 0

    start_time = time.time()

    for epoch in range(0, max_epoch):
        print(">>>>>> EPOCH : ", epoch)
        if not args.evaluate:
            print('TRAIN')
            train_stats = train(model, train_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler,
                                config, writer)

        print('VALIDATION')
        val_stats = evaluate(model, test_loader, tokenizer, device, epoch, writer)
        # print('TEST')
        # test_stats = evaluate(model, test_loader, tokenizer, device)

        if not args.evaluate:
            if val_stats > best:
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }
                torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))
                best = val_stats
                best_epoch = epoch
        if args.evaluate:   break
        lr_scheduler.step(epoch + warmup_steps + 1)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def test_classification(model, tokenizer, device, data_loader):
    # test
    model.to(device)
    model.eval()

    header = 'Test '
    print_freq = 50
    step_size = 100

    predict_res = []

    tqdm_data_loader = tqdm(data_loader, miniters=print_freq, desc=header)
    for i, data_dict in enumerate(tqdm_data_loader) :
        text = data_dict['fact']
        for j, t in enumerate(text):    # 특수문자, 숫자 제거
            t = re.sub(r"[^\uAC00-\uD7A3\s]", "", t)
            text[j] = t
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=150, return_tensors="pt").to(device)
        
        prediction = model(text_input, [], one_hot_label=False, train=False)
        prediction = F.softmax(prediction, dim=-1).cpu()
        prediction_idx = torch.argmax(prediction, axis=1)
        data_dict['laws_service_id'] = prediction_idx
        predict_res.append(prediction_idx)
    predict_res = torch.cat(predict_res, dim=0)
    predict_res = predict_res.tolist()
    return predict_res

def test_sentiment_analysis(args, config):
    print('aa', args.evaluate)
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset ####
    print("Creating dataset")
    dataset = jsonlload(args.input_data)
    dataset_test = KLAID_Dataset(dataset, mode="test")
    test_loader = DataLoader(dataset_test, batch_size=config['batch_size_test'], pin_memory=True, drop_last=False, shuffle=False)
    
    tokenizer = BertTokenizer.from_pretrained(args.text_encoder, truncation_side="left")

    #### Model ####
    print("Creating model")
    model = ALBEF_n_class(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer, num_category=177)

    if args.checkpoint:
        print('LOADING PRETRAINED MODEL..')
        checkpoint = torch.load(args.checkpoint, map_location='cpu')

        state_dict = checkpoint['model']

        # reshape positional embedding to accomodate for image resolution change
        # pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.visual_encoder)

        # state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped

        # if config['distill']:
        #     m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],
        #                                                     model.visual_encoder_m)
        #     m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],
        #                                                     model.visual_encoder_m)

        #     state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped

        for key in list(state_dict.keys()): #to match the key name when the pretrained model used BertForMaskedLM(which contains BertModel), but fine-tuning model used BertModel
            if 'bert' in key:
                new_key = key.replace('bert.', '')
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

        msg = model.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % args.checkpoint)

    model = model.to(device)

    start_time = time.time()
    
    pred_data = test_classification(model, tokenizer, device, test_loader)

    jsondump(pred_data, args.output_json)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_train", default=True, action="store_true")
    parser.add_argument("--do_test", default=False, action="store_true")
    parser.add_argument('--input_data', default='E:\\STUDY\\KAIST_BISPL\\ALBEF_2textencoder\\KLAID_dev_data.json')
    parser.add_argument('--output_dir', default='E:\\STUDY\\KAIST_BISPL\\ALBEF_2textencoder/output/KLAID_2_2/')
    parser.add_argument('--checkpoint', default='E:\\STUDY\\KAIST_BISPL\\ALBEF_2textencoder\\KLAID_2\\checkpoint_29.pth')
    # parser.add_argument('--checkpoint', default='E:\\STUDY\\KAIST_BISPL\\ALBEF_2textencoder\\output\\KLAID_1\\checkpoint_best.pth')
    parser.add_argument('--output_json', default='E:\\STUDY\\KAIST_BISPL\\ALBEF_2textencoder\\output_json\\KLAID_2_2_output.json')
    parser.add_argument('--text_encoder', default='kykim/bert-kor-base')
    parser.add_argument('--evaluate', default=False)    #"True" for validation&test only
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, type=bool)
    args = parser.parse_args()

    cls_config = {
        'image_res': 256,
        'batch_size_train': 32,
        'batch_size_test': 32,
        'alpha': 0.4,
        'distill':True,
        'warm_up':False,
        'bert_config': 'E:\\STUDY\\KAIST_BISPL\\ALBEF_2textencoder\\config_kor_bert.json',
        'schedular': {'sched': 'cosine', 'lr': 2e-5, 'epochs': 60, 'min_lr': 1e-6,
                      'decay_rate': 1, 'warmup_lr': 1e-5, 'warmup_epochs': 1, 'cooldown_epochs': 0},
        'optimizer': {'opt': 'adamW', 'lr': 2e-5, 'weight_decay': 0.02},
        'num_category' : 177
    }

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.do_train:
        train_sentiment_analysis(args, cls_config)
    elif args.do_test :
        test_sentiment_analysis(args, cls_config)