import json
import re


def parse_pred_ans(filename):
    total, correct = 0, 0
    with open(filename, "r", encoding="utf-8") as fr:
        for line in fr:
            jo = json.loads(line.strip())
            correct += jo["pred"] == jo["answer"]
            total += 1
    print('num_q %d correct %d ratio %.4f' % (total, correct, float(correct / total)))

def parse_rethink_pred_ans(filename1, filename2):
    total, correct = 0, 0
    idx_list = []
    with open(filename1, "r", encoding="utf-8") as fr:
        for line in fr:
            # jo = json.loads(line.strip())["rk_steps"][-1]
            if len(json.loads(line.strip())["rk_steps"]) < 1:
                idx_list.append(json.loads(line.strip())["idx"])
                total += 1
                continue
            else:
                jo = json.loads(line.strip())["rk_steps"][-1]
                correct += jo["pred"] == jo["answer"]
                total += 1
    with open(filename2, "r", encoding="utf-8") as fo:
        for line in fo:
            jo = json.loads(line.strip())
            if jo['idx'] in idx_list:
                correct += jo["pred"] == jo["answer"]
    print('num_q %d correct %d ratio %.4f' % (total, correct, float(correct / total)))

def parse_skip_ans(filename):
    total, correct = 0, 0
    with open(filename, "r", encoding="utf-8") as fr:
        for line in fr:
            jo = json.loads(line.strip())["rk_steps"][-1]
            correct += jo["pred"] == jo["answer"]
            total += 1
    print('num_q %d correct %d ratio %.4f' % (total, correct, float(correct / total)))
