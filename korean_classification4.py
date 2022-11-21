import argparse
from tqdm import tqdm
from pathlib import Path
import utils
import torch
import numpy as np
import random
from classification_albef import ALBEF
from classification_albef_predict3class import ALBEF3class
import time
import os
from vit import interpolate_pos_embed
import torch.backends.cudnn as cudnn
from transformers import BertTokenizer, BertTokenizerFast, BertModel, BertForMaskedLM
import datetime
from torchvision import transforms
import torchvision.transforms as T
import torch.nn.functional as F
from dataset import NIKL_Dataset_2, NIKL_Dataset_3
from torch.utils.data import DataLoader
import torch.optim as optim
from scheduler import create_scheduler
from sklearn.metrics import f1_score,roc_auc_score
from utils import *
import copy
from torch.utils.tensorboard import SummaryWriter

# json 개체를 파일이름으로 깔끔하게 저장
def jsondump(j, fname):
    with open(fname, "w", encoding="UTF8") as f:
        json.dump(j, f, ensure_ascii=False)

def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config, case, writer):
    # train
    model.train()

    loss_total = 0
    acc_total = 0

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps * step_size

    property_pair, prompt_sentence = choose_prompt(case)

    tqdm_data_loader = tqdm(data_loader, miniters=print_freq, desc=header)
    # for i, (text, label, entity_one_hot_label, polarity_one_hot_label) in enumerate(tqdm_data_loader):
    for i, data_dict in enumerate(tqdm_data_loader):
        text = data_dict['sentence_form']
        label = data_dict['annotation']
        one_hot_label = data_dict['one_hot_label'].to(device)

        label = make_prompt(label, property_pair, prompt_sentence, case)

        text_input = tokenizer(text, padding='longest', return_tensors="pt").to(device)
        label_input = tokenizer(label, padding='longest', return_tensors="pt").to(device)

        if epoch > 0 or not config['warm_up']:
            alpha = config['alpha']
        else:
            alpha = config['alpha'] * min(1., i / len(data_loader))

        if case == "entity" : 
            loss, prediction = model(text_input, label_input, one_hot_label, alpha=alpha)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            prediction = torch.sigmoid(prediction)
            prediction[prediction>=0.5]=1.
            prediction[prediction<0.5]=0.
            acc = ((prediction == one_hot_label).sum() / one_hot_label.numel()).item()

            tqdm_data_loader.set_description(f'loss={loss.item():.4f}, ACC={acc:.4f}, lr={optimizer.param_groups[0]["lr"]:.6f}')
            loss_total += loss.item()
            acc_total += acc
        else :
            loss = model(text_input, [], real_one_hot_label, alpha=alpha)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tqdm_data_loader.set_description(f'loss={loss.item():.4f}, lr={optimizer.param_groups[0]["lr"]:.6f}')
            loss_total += loss
        
        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
            scheduler.step(i // step_size)
    
    if case == "entity" : 
        writer.add_scalar("Train/Loss(entity)", loss_total/len(tqdm_data_loader), epoch)
        writer.add_scalar("Train/ACC(entity)", acc_total/len(tqdm_data_loader), epoch)
    else : 
        writer.add_scalar("Train/Loss(polarity)", loss_total/len(tqdm_data_loader), epoch)



@torch.no_grad()
def evaluate(model, data_loader, tokenizer, device, epoch, case, writer):
    # test
    model.eval()

    preds=[]
    answers=[]

    property_pair, prompt_sentence = choose_prompt(case)

    # for text, label, entity_one_hot_label, polarity_one_hot_label in data_loader:
    for data_dict in data_loader:
        text = data_dict['sentence_form']
        label = data_dict['annotation']
        one_hot_label = data_dict['one_hot_label'].to(device)

        # text, label, one_hot_label = data_sampling(text, label, property_pair, prompt_sentence, case, sampling_num=args.num_sampling)
        label = make_prompt(label, property_pair, prompt_sentence, case)

        text_input = tokenizer(text, padding='longest', return_tensors="pt").to(device)
        label_input = tokenizer(label, padding='longest',return_tensors="pt").to(device)
        if case == "entity" : 
            prediction = model(text_input, label_input, [], train=False)
            prediction = torch.sigmoid(prediction)
            prediction[prediction>=0.5]=1.
            prediction[prediction<0.5]=0.
            preds.append(prediction.cpu())
            answers.append(one_hot_label.cpu())
        else : 
            prediction = model(text_input, [], real_one_hot_label, train=False)
            prediction = F.softmax(prediction, dim=-1).cpu()
            preds.append(np.round(prediction.cpu()))
            answers.append(real_one_hot_label.cpu())

    preds=torch.cat(preds,dim=0)
    answers=torch.cat(answers,dim=0)
    print('F1 score[micro]:',f1_score(answers, preds, average='micro'))
    print('AUROC:',roc_auc_score(answers, preds, average='micro'))
    print('Accuracy:',((answers == preds).sum() / answers.numel()).item())
    score = f1_score(answers, preds, average='micro')

    if case == "entity" : 
        writer.add_scalar("Eval/(entity)F1 score", f1_score(answers, preds, average='micro'), epoch)
        writer.add_scalar("Eval/(entity)F1 AUROC", roc_auc_score(answers, preds, average='micro'), epoch)
        writer.add_scalar("Eval/(entity)Accuracy", ((answers == preds).sum() / answers.numel()).item(), epoch)
    else : 
        writer.add_scalar("Eval/(polarity)F1 score", f1_score(answers, preds, average='micro'), epoch)
        writer.add_scalar("Eval/(polarity)F1 AUROC", roc_auc_score(answers, preds, average='micro'), epoch)
        writer.add_scalar("Eval/(polarity)Accuracy", ((answers == preds).sum() / answers.numel()).item(), epoch)

    return score

def test_classification(ce_model, pc_model, tokenizer, device, data_loader, original_data):
    entity_property_pair = [
                    '제품 전체 일반', '제품 전체 가격', '제품 전체 디자인', '제품 전체 품질', '제품 전체 편의성', '제품 전체 인지도',
                    '본품 일반', '본품 디자인', '본품 품질', '본품 편의성', '본품 다양성','본품 가격','본품 인지도',
                    '패키지 구성품 일반', '패키지 구성품 가격', '패키지 구성품 디자인', '패키지 구성품 품질', '패키지 구성품 편의성', '패키지 구성품 다양성',
                    '브랜드 일반', '브랜드 가격', '브랜드 디자인', '브랜드 품질', '브랜드 인지도',
                                    ]
    entity_TF = ["True", "False"]
    original_entity_property_pair = [
                '제품 전체#일반', '제품 전체#가격', '제품 전체#디자인', '제품 전체#품질', '제품 전체#편의성', '제품 전체#인지도',
                '본품#일반', '본품#디자인', '본품#품질', '본품#편의성', '본품#다양성','본품#가격','본품#인지도',
                '패키지/구성품#일반', '패키지/구성품#가격', '패키지/구성품#디자인', '패키지/구성품#품질', '패키지/구성품#편의성', '패키지/구성품#다양성',
                '브랜드#일반', '브랜드#가격', '브랜드#디자인', '브랜드#품질', '브랜드#인지도',
                                ]

    polarity_id_to_name = ['positive', 'negative', 'neutral']
    prompt_sentence = "이 제품은 {}에 대한 평가이다."

    # test
    pc_model.to(device)
    ce_model.to(device)
    pc_model.eval()
    ce_model.eval()

    header = 'Text '
    print_freq = 50
    step_size = 100

    tqdm_data_loader = tqdm(data_loader, miniters=print_freq, desc=header)
    # for sentence in data[0]:
    for i, data_dict in enumerate(tqdm_data_loader) :
        original_text = original_data[i]['sentence_form']
        text = data_dict['sentence_form'][0]
        text_input = tokenizer(text, padding='longest', return_tensors="pt").to(device)
        for j, pair in enumerate(entity_property_pair) : 
            prompt_pair = prompt_sentence.format(pair)
            pair_input = tokenizer(prompt_pair, padding='longest', return_tensors="pt").to(device)

            ce_prediction = ce_model(text_input, label=pair_input, one_hot_label=False, train=False)
            ce_prediction = F.softmax(ce_prediction, dim=-1).cpu()
            if ce_prediction[0][0] >= 0.6 : 
                # ce_prediction_idx = torch.argmax(ce_prediction, axis=1)
                # T_F = entity_TF[ce_prediction_idx]
                ce_result = original_entity_property_pair[j]
                pc_prediction = pc_model(text_input, label=False, one_hot_label=False, train=False)
                pc_prediction = F.softmax(pc_prediction, dim=-1).cpu()
                pc_prediction_idx = torch.argmax(pc_prediction, axis=1)
                pc_result = polarity_id_to_name[pc_prediction_idx]
                original_data[i]['annotation'].append([ce_result, pc_result])
                print(original_text, ce_result, pc_result)
            else :
                pass
    return original_data


def train_sentiment_analysis(args, config):

    writer = SummaryWriter(args.output_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    if args.case == "entity" : 
        config['num_category'] = 2
    else :          # polarity
        config['num_category'] = 3

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
    # dataset_train = NIKL_Dataset_2(args.train_data_path, sampling_num=args.num_sampling, case=args.case)
    dataset_train = NIKL_Dataset_3(args.train_data_path, sampling_num=args.num_sampling, case=args.case)
    dataset_test = NIKL_Dataset_3(args.dev_data_path, sampling_num=args.num_sampling, case=args.case)

    train_loader = DataLoader(dataset_train, batch_size=config['batch_size_train'], pin_memory=True, drop_last=True, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=config['batch_size_test'], pin_memory=True, drop_last=False, shuffle=True)

    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    #### Model ####
    print("Creating model")
    if args.case == 'entity' : 
        model = ALBEF(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer, num_category=config['num_category'])
    else :
        model = ALBEF3class(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer, num_category=config['num_category'])

    if args.checkpoint:
        print('LOADING PRETRAINED MODEL..')
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']

        # reshape positional embedding to accomodate for image resolution change
        pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.visual_encoder)
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped

        if not args.evaluate:
            if config['distill']:
                m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],
                                                             model.visual_encoder_m)
                state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped

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
                                config, args.case, writer)

        print('VALIDATION')
        val_stats = evaluate(model, test_loader, tokenizer, device, epoch, args.case, writer)
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
    dataset_test = NIKL_Dataset_2(args.test_data_path, case=args.case, test=True)
    # test_loader = DataLoader(dataset_test, batch_size=config['batch_size_test'], pin_memory=True, drop_last=False, shuffle=False)
    test_loader = DataLoader(dataset_test, batch_size=1, pin_memory=True, drop_last=False, shuffle=False)
    
    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    original_test_data = jsonlload("D:\\Data\\NIKL_ABSA_2022_COMPETITION_v1.0\\nikluge-sa-2022-test.jsonl")

    #### Model ####
    print("Creating model")
    ce_model = ALBEF(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer, num_category=2)
    pc_model = ALBEF3class(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer, num_category=3)

    if args.checkpoint:
        print('LOADING PRETRAINED MODEL..')
        ce_checkpoint = torch.load(args.CE_checkpoint, map_location='cpu')
        pc_checkpoint = torch.load(args.PC_checkpoint, map_location='cpu')

        ce_state_dict = ce_checkpoint['model']
        pc_state_dict = pc_checkpoint['model']

        # reshape positional embedding to accomodate for image resolution change
        ce_pos_embed_reshaped = interpolate_pos_embed(ce_state_dict['visual_encoder.pos_embed'], ce_model.visual_encoder)
        pc_pos_embed_reshaped = interpolate_pos_embed(pc_state_dict['visual_encoder.pos_embed'], pc_model.visual_encoder)

        ce_state_dict['visual_encoder.pos_embed'] = ce_pos_embed_reshaped
        pc_state_dict['visual_encoder.pos_embed'] = pc_pos_embed_reshaped

        if config['distill']:
            ce_m_pos_embed_reshaped = interpolate_pos_embed(ce_state_dict['visual_encoder_m.pos_embed'],
                                                            ce_model.visual_encoder_m)
            pc_m_pos_embed_reshaped = interpolate_pos_embed(ce_state_dict['visual_encoder_m.pos_embed'],
                                                            ce_model.visual_encoder_m)

            ce_state_dict['visual_encoder_m.pos_embed'] = ce_m_pos_embed_reshaped
            pc_state_dict['visual_encoder_m.pos_embed'] = pc_m_pos_embed_reshaped

        for key in list(ce_state_dict.keys()): #to match the key name when the pretrained model used BertForMaskedLM(which contains BertModel), but fine-tuning model used BertModel
            if 'bert' in key:
                new_key = key.replace('bert.', '')
                ce_state_dict[new_key] = ce_state_dict[key]
                del ce_state_dict[key]

        for key in list(pc_state_dict.keys()): #to match the key name when the pretrained model used BertForMaskedLM(which contains BertModel), but fine-tuning model used BertModel
            if 'bert' in key:
                new_key = key.replace('bert.', '')
                pc_state_dict[new_key] = pc_state_dict[key]
                del pc_state_dict[key]

        ce_msg = ce_model.load_state_dict(ce_state_dict, strict=False)
        pc_msg = pc_model.load_state_dict(pc_state_dict, strict=False)
        print('load checkpoint from %s' % args.CE_checkpoint)
        print('load checkpoint from %s' % args.PC_checkpoint)

    ce_model = ce_model.to(device)
    pc_model = pc_model.to(device)

    start_time = time.time()
    
    pred_data = test_classification(ce_model, pc_model, tokenizer, device, test_loader, original_test_data)

    jsondump(pred_data, 'E:\\STUDY\\KAIST_BISPL\\ALBEF_2textencoder\\output_json\\prompt_pred_data_5.json')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_train", default=True, action="store_true")
    parser.add_argument("--do_eval", default=True, action="store_true")
    parser.add_argument("--do_test", default=False, action="store_true")
    parser.add_argument('--output_dir', default='E:\\STUDY\\KAIST_BISPL\\ALBEF_2textencoder/output/kor_CE4/')
    parser.add_argument('--checkpoint', default='E:\\STUDY\\KAIST_BISPL\\ALBEF_2textencoder\\Pretrain_kor_bert3\\checkpoint_23.pth')
    parser.add_argument('--case', default="entity")   # entity, polarity
    parser.add_argument('--CE_checkpoint', default='E:\\STUDY\\KAIST_BISPL\\ALBEF_2textencoder\\output\\kor_CE4\\checkpoint_best.pth')
    parser.add_argument('--PC_checkpoint', default='E:\\STUDY\\KAIST_BISPL\\ALBEF_2textencoder\\output\\kor_PC3\\checkpoint_best.pth')
    parser.add_argument('--text_encoder', default='kykim/bert-kor-base')
    parser.add_argument('--num_sampling', default=4, help='number of sampling data, True=n/2, False=n/2)') 
    parser.add_argument('--evaluate', default=False)    #"True" for validation&test only
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, type=bool)
    parser.add_argument('--train_data_path', default="D:\\Data\\NIKL_ABSA_2022_COMPETITION_v1.0\\prepocess_train.json") 
    parser.add_argument('--dev_data_path', default="D:\\Data\\NIKL_ABSA_2022_COMPETITION_v1.0\\prepocess_dev.json") 
    parser.add_argument('--test_data_path', default="D:\\Data\\NIKL_ABSA_2022_COMPETITION_v1.0\\prepocess_test.json") 
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
        'num_category' : 24
    }

    # num_sampling = 24 lr = 2e-5
    # num_sampling = 8 lr = 2e-5
    # num_sampling = 4 lr = 2e-4
    # num_sampling = 2 lr = 1e-4

    # weight decay 하는 정도를 바꿀 수 있나?    
    # # num_category : 24 (entity) , 3 (polarity)


    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.do_train:
        train_sentiment_analysis(args, cls_config)
    elif args.do_test:
        test_sentiment_analysis(args, cls_config)