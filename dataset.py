from torch.utils.data import Dataset
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import transforms
import os
import json
import csv
from utils import *
from konlpy.tag import Okt


class MIMIC_CXRDataset(Dataset):
    def __init__(self, data_path, transform=None, data_length=None):
        self.transform = transform
        self.data_dir = os.path.dirname(data_path)
        self.data = [json.loads(l) for l in open(data_path)]
        if data_length is not None: self.data=self.data[:data_length]
        self.ltoi = ['','Pneumothorax', 'Pleural Other', 'No Finding', 'Cardiomegaly', 'Atelectasis',
                     'Consolidation', 'Support Devices', 'Edema', 'Pneumonia', 'Pleural Effusion',
                     'Enlarged Cardiomediastinum', 'Fracture', 'Lung Opacity', 'Lung Lesion']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        info=self.data[index]
        img_path=info['img'].replace('/home/mimic-cxr/dataset/image_preprocessing','.')
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        raw_labels=[w.strip().replace('\'', '') for w in info['label'].split(',')]
        labels=torch.zeros(len(self.ltoi))
        for l in raw_labels:    labels[self.ltoi.index(l)]=1

        return img, info['text'], labels


class Dreaddit_Dataset(Dataset):
    def __init__(self, data_path, transform=None, data_length=None):
        self.transform = transform
        self.data_dir = os.path.dirname(data_path)
        self.data = pd.read_csv(data_path)
        self.prompt = "The user has a {}"
        self.subreddit = ['domesticviolence','survivorsofabuse','anxiety','stress',
                          'almosthomeless','assistance','food_pantry','homeless','ptsd','relationships']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        info = self.data.iloc[index]
        text = info['text']
        label = info['subreddit']
        prompt_label = self.prompt.format(label)

        one_hot_label =torch.zeros(len(self.subreddit))
        one_hot_label[self.subreddit.index(label)]=1

        return text, prompt_label, one_hot_label

class NIKL_Dataset_before(Dataset):
    def __init__(self, data_path, case='polarity'):
        self.case = case
        self.data_dir = os.path.dirname(data_path)
        self.data = jsonlload(data_path)
        self.prompt_1 = "이 제품은 {}에 대한 평가이다."
        self.prompt_2 = "{} 평가를 받았다."
        self.entity_property_pair = [
                '제품 전체#일반', '제품 전체#가격', '제품 전체#디자인', '제품 전체#품질', '제품 전체#편의성', '제품 전체#인지도',
                '본품#일반', '본품#디자인', '본품#품질', '본품#편의성', '본품#다양성','본품#가격','본품#인지도',
                '패키지/구성품#일반', '패키지/구성품#가격', '패키지/구성품#디자인', '패키지/구성품#품질', '패키지/구성품#편의성', '패키지/구성품#다양성',
                '브랜드#일반', '브랜드#가격', '브랜드#디자인', '브랜드#품질', '브랜드#인지도',
                                ]
        self.polarity_id_to_name_eng = ['positive', 'negative', 'neutral']
        self.polarity_id_to_name = ['긍정적인', '부정적인', '중립적인']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        info = self.data[index]
        text = info['sentence_form']
        annotation = info['annotation']

        if self.case == 'entity' : 
            prompt_label = self.prompt_1.format(annotation[0][0]) # entitiy

        else : # polarity
            prompt_label = self.prompt_2.format(annotation[0][2]) # polarity
        
        #annotation_word = annotation[0][1]

        entity_one_hot_label = torch.zeros(len(self.entity_property_pair))
        entity_one_hot_label[self.entity_property_pair.index(annotation[0][0])]=1

        polarity_one_hot_label = torch.zeros(len(self.polarity_id_to_name))
        polarity_one_hot_label[self.polarity_id_to_name_eng.index(annotation[0][2])]=1

        return text, prompt_label, entity_one_hot_label, polarity_one_hot_label#, annotation_word




class NIKL_Dataset(Dataset):
    def __init__(self, data_path, case='entity'):
        self.case = case
        self.data_dir = os.path.dirname(data_path)
        self.data = jsonlload(data_path)
        self.prompt_1 = "이 제품은 {}에 대한 평가이다."
        self.prompt_2 = "{} 평가를 받았다."
        self.entity_property_pair = [
                '제품 전체#일반', '제품 전체#가격', '제품 전체#디자인', '제품 전체#품질', '제품 전체#편의성', '제품 전체#인지도',
                '본품#일반', '본품#디자인', '본품#품질', '본품#편의성', '본품#다양성','본품#가격','본품#인지도',
                '패키지/구성품#일반', '패키지/구성품#가격', '패키지/구성품#디자인', '패키지/구성품#품질', '패키지/구성품#편의성', '패키지/구성품#다양성',
                '브랜드#일반', '브랜드#가격', '브랜드#디자인', '브랜드#품질', '브랜드#인지도',
                                ]
        self.label_id_to_name = ['True', 'False']

        self.polarity_id_to_name_eng = ['positive', 'negative', 'neutral']
        self.polarity_id_to_name = ['긍정적인', '부정적인', '중립적인']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        property_data_dict = {
            'sentence_form': [],
            'annotation': [],
            'one_hot_label': []
        }

        entity_one_hot_label = torch.zeros(len(self.entity_property_pair))
        polarity_one_hot_label = torch.zeros(len(self.polarity_id_to_name))

        info = self.data[index]
        text = info['sentence_form']
        annotation = info['annotation']

        property_data_dict['sentence_form'].append(text)

        if self.case == 'entity' : 
            property_data_dict['annotation'].append(self.prompt_1.format(annotation[0][0]))
            entity_one_hot_label[self.entity_property_pair.index(annotation[0][0])] = 1
            one_hot_label = entity_one_hot_label

        else : # polarity
            kor_property = self.polarity_id_to_name[self.polarity_id_to_name_eng.index(annotation[0][2])]
            property_data_dict['annotation'].append(self.prompt_2.format(kor_property))
            polarity_one_hot_label[self.polarity_id_to_name_eng.index(annotation[0][2])] = 1
            one_hot_label = polarity_one_hot_label

        property_data_dict['one_hot_label'].append(one_hot_label)
        return property_data_dict

#######################################################################

class NIKL_Dataset_2(Dataset):
    def __init__(self, data_path, case='entity', sampling_num=4, test=False):
        self.case = case
        self.data_dir = os.path.dirname(data_path)
        self.data = jsonlload(data_path)[0]
        self.prompt_1 = "이 제품은 {}에 대한 평가이다."
        self.prompt_2 = "{} 평가를 받았다."
        self.entity_property_pair = [
                '제품 전체 일반', '제품 전체 가격', '제품 전체 디자인', '제품 전체 품질', '제품 전체 편의성', '제품 전체 인지도',
                '본품 일반', '본품 디자인', '본품 품질', '본품 편의성', '본품 다양성','본품 가격','본품 인지도',
                '패키지 구성품 일반', '패키지 구성품 가격', '패키지 구성품 디자인', '패키지 구성품 품질', '패키지 구성품 편의성', '패키지 구성품 다양성',
                '브랜드 일반', '브랜드 가격', '브랜드 디자인', '브랜드 품질', '브랜드 인지도',
                                ]
        self.label_id_to_name = ['True', 'False']

        self.polarity_id_to_name_eng = ['positive', 'negative', 'neutral']
        self.polarity_id_to_name = ['긍정적인', '부정적인', '중립적인']

        self.test = test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        if self.test == False :
            property_data_dict = {
                'sentence_form': [],
                'annotation': [],
                'one_hot_label': []
            }
        else :  # test dataset일 때는 one_hot_label 이 없다.
            property_data_dict = {
                'sentence_form': [],
                'annotation': []
            }

        entity_one_hot_label = torch.zeros(len(self.entity_property_pair))
        polarity_one_hot_label = torch.zeros(len(self.polarity_id_to_name))

        info = self.data[index]
        text = info['sentence_form']
        annotation = info['annotation']

        property_data_dict['sentence_form'].append(text)

        if self.test == False : 
            if self.case == 'entity' : 
                # property_data_dict['annotation'].append(self.prompt_1.format(annotation[0]))
                property_data_dict['annotation'].append(annotation[0])
                entity_one_hot_label[self.entity_property_pair.index(annotation[0])] = 1
                one_hot_label = entity_one_hot_label

            else : # polarity
                kor_property = self.polarity_id_to_name[self.polarity_id_to_name_eng.index(annotation[1])]
                # property_data_dict['annotation'].append(self.prompt_2.format(kor_property))
                property_data_dict['annotation'].append(kor_property)
                polarity_one_hot_label[self.polarity_id_to_name_eng.index(annotation[1])] = 1
                one_hot_label = polarity_one_hot_label
            
            property_data_dict['one_hot_label'].append(one_hot_label)

        else : # test dataset일 때는 annotation 이 없다.
            pass

        return property_data_dict

#######################################################################

class NIKL_Dataset_3(Dataset):
    def __init__(self, data_path, case='entity', sampling_num=4, test=False):
        self.case = case
        self.data_dir = os.path.dirname(data_path)
        self.data = jsonlload(data_path)[0]
        self.prompt_1 = "이 제품은 {}에 대한 평가이다."
        self.prompt_2 = "{} 평가를 받았다."
        self.entity_property_pair = [
                '제품 전체 일반', '제품 전체 가격', '제품 전체 디자인', '제품 전체 품질', '제품 전체 편의성', '제품 전체 인지도',
                '본품 일반', '본품 디자인', '본품 품질', '본품 편의성', '본품 다양성','본품 가격','본품 인지도',
                '패키지 구성품 일반', '패키지 구성품 가격', '패키지 구성품 디자인', '패키지 구성품 품질', '패키지 구성품 편의성', '패키지 구성품 다양성',
                '브랜드 일반', '브랜드 가격', '브랜드 디자인', '브랜드 품질', '브랜드 인지도',
                                ]
        self.label_id_to_name = ['True', 'False']

        self.polarity_id_to_name_eng = ['positive', 'negative', 'neutral']
        self.polarity_id_to_name = ['긍정적인', '부정적인', '중립적인']

        self.test = test

        data_dict = {
            'sentence_form' : 0,
            'annotation' : [],
            'one_hot_label' : []
        }
        data_list = []
        
        entity_one_hot_label = torch.zeros(len(self.entity_property_pair))

        if case == "entity" : 
            for i, info in enumerate(self.data) :
                for j in range(int(sampling_num/2)) : # True data
                    dict = copy.deepcopy(data_dict)
                    dict['sentence_form'] = info['sentence_form']
                    dict['annotation'] = info['annotation'][0]
                    entity_one_hot_label = torch.zeros(len(self.label_id_to_name))
                    entity_one_hot_label[0] = 1.
                    dict['one_hot_label'] = entity_one_hot_label
                    data_list.append(dict)
                # Making False data
                
                property_pair_copy = copy.deepcopy(self.entity_property_pair)
                property_pair_copy.remove(info['annotation'][0])
                false_label_idx = random.randint(low=len(property_pair_copy), size=int(sampling_num/2)) 
                for false_idx in false_label_idx :
                    dict = copy.deepcopy(data_dict)
                    dict['sentence_form'] = info['sentence_form']
                    dict['annotation'] = property_pair_copy[false_idx]
                    entity_one_hot_label = torch.zeros(len(self.label_id_to_name))
                    entity_one_hot_label[1] = 1.
                    dict['one_hot_label'] = entity_one_hot_label
                    data_list.append(dict)
        else : 
            for i, info in enumerate(self.data) :
                dict = copy.deepcopy(data_dict)
                dict['sentence_form'] = info['sentence_form']
                dict['annotation'] = info['annotation'][1]
                polarity_one_hot_label = torch.zeros(len(self.polarity_id_to_name))
                polarity_one_hot_label[self.polarity_id_to_name_eng.index(info['annotation'][1])] = 1
                dict['one_hot_label'] = polarity_one_hot_label
                data_list.append(dict)
        self.data_sampling = data_list

    def __len__(self):
        return len(self.data_sampling)

    def __getitem__(self, index):

        if self.test == False :
            property_data_dict = {
                'sentence_form': [],
                'annotation': [],
                'one_hot_label': []
            }
        else :  # test dataset일 때는 one_hot_label 이 없다.
            property_data_dict = {
                'sentence_form': [],
                'annotation': []
            }

        entity_one_hot_label = torch.zeros(len(self.entity_property_pair))
        polarity_one_hot_label = torch.zeros(len(self.polarity_id_to_name))

        # import ipdb; ipdb.set_trace()
        info = self.data_sampling[index]
        # text = info['sentence_form']
        # annotation = info['annotation']
        # one_hot_label = info['one_hot_label']

        # property_data_dict['sentence_form'].append(text)
        # property_data_dict['annotation'].append(annotation)

        # if self.test == False : 
        #     if self.case == 'entity' : 
        #         # property_data_dict['annotation'].append(self.prompt_1.format(annotation[0]))
        #         property_data_dict['annotation'].append(annotation[0])
        #         entity_one_hot_label[self.entity_property_pair.index(annotation[0])] = 1
        #         one_hot_label = entity_one_hot_label

        #     else : # polarity
        #         kor_property = self.polarity_id_to_name[self.polarity_id_to_name_eng.index(annotation[1])]
        #         # property_data_dict['annotation'].append(self.prompt_2.format(kor_property))
        #         property_data_dict['annotation'].append(kor_property)
        #         polarity_one_hot_label[self.polarity_id_to_name_eng.index(annotation[1])] = 1
        #         one_hot_label = polarity_one_hot_label
            
        #     property_data_dict['one_hot_label'].append(one_hot_label)

        # else : # test dataset일 때는 annotation 이 없다.
        #     pass

        # return property_data_dict
        return info
