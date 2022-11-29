import argparse
import json


def cal_precesion(pred, answer, correct):
    if sum(pred):
        return sum(correct) / sum(pred)
    else:
        return False


def cal_recall(pred, answer, correct):
    if sum(answer):
        return sum(correct) / sum(answer)
    else:
        return False


def cal_f1_score(pred, answer):
    correct = [1 if p == 1 and a == 1 else 0 for p, a in zip(pred, answer)]
    recall = cal_recall(pred, answer, correct)
    precesion = cal_precesion(pred, answer, correct)
    if recall and precesion:
        return 2 * (recall * precesion) / (recall + precesion)
    else:
        return 0


def macro_f1(pred, answer):
    f1_list = list()
    max_label_num = max(max(pred), max(answer))
    for i in range(max_label_num + 1):
        onehot_pred = [1 if p == i else 0 for p in pred]
        onehot_answer = [1 if a == i else 0 for a in answer]

        if sum(onehot_pred) or sum(onehot_answer):
            f1_score = cal_f1_score(onehot_pred, onehot_answer)
            f1_list.append(f1_score)
    return sum(f1_list) / len(f1_list)


def micro_avg(pred, answer):
    correct = [1 if p == a else 0 for p, a in zip(pred, answer)]
    return sum(correct) / len(pred)


def evaluate(pred, answer, eval_function=macro_f1):
    return eval_function(pred, answer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction_file', help='predction file')
    parser.add_argument('--answer_file', help='answer File')
    args = parser.parse_args()

    with open(args.prediction_file) as prediction_file:
        prediction_data = json.load(prediction_file)
    with open(args.answer_file) as answer_file:
        answer_data = json.load(answer_file)
    print(json.dumps(evaluate(prediction_data, answer_data, eval_function=macro_f1)))