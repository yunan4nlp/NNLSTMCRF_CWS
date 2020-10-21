from data.Metric import *

def get_ent(words):
    start = 0
    ents = []
    for w in words:
        ent_str = '[' + str(start) + ',' + str(start + len(w) - 1) + ']'
        ents.append(ent_str)
        start += len(w)
    return ents


def segprf(onebatch, outputs):
    predict_num = 0
    correct_num = 0
    gold_num = 0
    assert len(onebatch) == len(outputs)
    for idx, inst in enumerate(onebatch):

        check(inst.words, outputs[idx])

        gold_set = set(get_ent(inst.words))
        predict_set = set(get_ent(outputs[idx]))

        predict_num += len(predict_set)
        gold_num += len(gold_set)

        correct_num += len(predict_set & gold_set)
    return correct_num, predict_num, gold_num

def check(words1, words2):
    str1 = ''
    for w in words1:
        str1 += w
    str2 = ''
    for w in words2:
        str2 += w
    assert str1 == str2

def evaluate(gold_file, predict_file):
    metric = Metric()
    g_inf = open(gold_file, mode='r', encoding='utf8')
    p_inf = open(predict_file, mode='r', encoding='utf8')

    predict_num = 0
    correct_num = 0
    gold_num = 0

    for g_line, p_line in zip(g_inf.readlines(), p_inf.readlines()):
        g_words = g_line.strip().split(" ")
        p_words = p_line.strip().split(" ")
        check(g_words, p_words)

        gold_set = set(get_ent(g_words))
        predict_set = set(get_ent(p_words))

        predict_num += len(predict_set)
        gold_num += len(gold_set)
        correct_num += len(predict_set & gold_set)

    metric.correct_label_count = correct_num
    metric.predicated_label_count = predict_num
    metric.overall_label_count = gold_num

    g_inf.close()
    p_inf.close()

    return metric