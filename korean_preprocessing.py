import json
import re
from konlpy.tag import Okt
import ipdb
import copy
from collections import OrderedDict


# jsonl 파일 읽어서 list에 저장
def jsonlload(fname, encoding="utf-8"):
    json_list = []
    with open(fname, encoding=encoding) as f:
        for line in f.readlines():
            json_list.append(json.loads(line))
    return json_list

#전처리 함수 만들기
def preprocessing(review, okt, remove_stopwords = False, stop_words =[]):
  #함수인자설명
  # review: 전처리할 텍스트
  # okt: okt객체를 반복적으로 생성하지 않고 미리 생성 후 인자로 받음
  # remove_stopword: 불용어를 제거할지 여부 선택. 기본값 False
  # stop_words: 불용어 사전은 사용자가 직접 입력, 기본값 빈 리스트

  # 1. 한글 및 공백 제외한 문자 모두 제거
  review_text = re.sub('[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]','',review)
  
  #2. okt 객체를 활용하여 형태소 단어로 나눔
  word_review = okt.morphs(review_text, stem=True)

  if remove_stopwords:
    #3. 불용어 제거(선택)
    word_review = [token for token in word_review if not token in stop_words]
  word_review = ' '.join(word_review)
  return word_review

# 전체 텍스트 전처리
stop_words = ['은','는','이','가','하','아','것','들','의','있','되','수','보','주','등','한','서','에']
okt = Okt()

def make_json_file (data, save_file_name, test=False) :
  data_dict = {
            'sentence_form' : 0,
            'annotation' : []
        }

  data_list = []
  ipdb.set_trace()
  for review in data:
    form = review['sentence_form']
    form = preprocessing(form,okt,remove_stopwords=True,stop_words= stop_words)
    if test == True : 
      data_test_dict = {
            'id' : 0,
            'sentence_form' : 0,
            'annotation' : []
        }
      data_test_dict['id'] = review['id']
      data_test_dict['sentence_form'] = form
      print(json.dumps(data_test_dict, ensure_ascii=False))
      data_list.append(data_test_dict)
    else :
      for annot in review['annotation'] :
        dict = copy.deepcopy(data_dict)
        entity = annot[0]
        polarity = annot[2]
        dict['sentence_form'] = form
        dict['annotation'].append(re.sub('[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]',' ',entity))
        dict['annotation'].append(polarity)

        print(json.dumps(dict, ensure_ascii=False))
        data_list.append(dict)

  # with open("./train_data_json.json", 'w', encoding="utf-8") as make_file :
  with open(save_file_name, 'w', encoding="utf-8") as make_file :
    json.dump(data_list, make_file, ensure_ascii=False)


train_data_path = "D:\\Data\\NIKL_ABSA_2022_COMPETITION_v1.0\\nikluge-sa-2022-train.jsonl"
dev_data_path = "D:\\Data\\NIKL_ABSA_2022_COMPETITION_v1.0\\nikluge-sa-2022-dev.jsonl"
test_data_path = "D:\\Data\\NIKL_ABSA_2022_COMPETITION_v1.0\\nikluge-sa-2022-test.jsonl"

train_data = jsonlload(train_data_path)
dev_data = jsonlload(dev_data_path)
test_data = jsonlload(test_data_path)

make_json_file (train_data, "D:\\Data\\NIKL_ABSA_2022_COMPETITION_v1.0\prepocess_train.json")
make_json_file (dev_data, "D:\\Data\\NIKL_ABSA_2022_COMPETITION_v1.0\prepocess_dev.json")
make_json_file (test_data, "D:\\Data\\NIKL_ABSA_2022_COMPETITION_v1.0\prepocess_test.json", test=True)