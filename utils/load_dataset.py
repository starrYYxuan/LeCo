import json
import re

def load_dataset(test_set, test_idx_set):
    if 'aqua' in test_set.lower():
        aqua_test = []
        with open(test_set, 'r') as rf2:
            for line in rf2:
                d = json.loads(line)
                aqua_test.append(d)

        aqua_test_qs = []
        aqua_test_ans = []
        for item in aqua_test:
            optn_str = ''
            for ele in item['options']:
                optn_str += ele
                optn_str += ', '
            optn_str = optn_str[:-2]
            qs = item['question'] + '\n' + 'Options: ' + optn_str + '\n'
            aqua_test_qs.append(qs)
            aqua_test_ans.append(item['correct'])


        system_prompt = {
            "role": "system",
            "content": "Follow the given examples and answer the question."
                       "You must choose one from the five options A/B/C/D/E, vague answers are not allowed."
                       "The last sentence must be: The answer is \\box{}. The choice of final answer is filled into \\boxed{}."
                       "Describe your thought process step-by-step using Step 1, Step 2, ... etc."
        }

        return aqua_test_qs, aqua_test_ans, system_prompt

    if 'csqa' in test_set.lower():
        csqa_test = []
        with open(test_set, 'r') as rf1:
            for line in rf1:
                d = json.loads(line)
                csqa_test.append(d)

        csqa_test_qs = []
        csqa_test_ans = []
        for line in csqa_test:
            q = line["question"]["stem"].strip() + '\n' + 'Answer Choices\n'
            s = ''
            for item in line["question"]["choices"]:
                choice_lt = item["label"].lower()
                choice_tx = item["text"] + '\n'
                s += f"({choice_lt}) {choice_tx}"
            q = q + s
            a = line["answerKey"].lower().strip()
            csqa_test_ans.append(a)
            csqa_test_qs.append(q)

        system_prompt = {
            "role": "system",
            "content": "Follow the given examples and answer the question."
                       "You must choose one from the five options a/b/c/d/e, vague answers are not allowed."
                       "The last sentence must be: The answer is \\box{}. The choice of final answer is filled into \\boxed{}."
                       "Describe your thought process step-by-step using Step 1, Step 2, ... etc."
        }
        return csqa_test_qs, csqa_test_ans, system_prompt

    if 'date' in test_set.lower():
        date_test_qs = []
        date_test_ans = []
        with open(test_set, 'r') as rf2:
            json_data = json.load(rf2)
            for line in json_data["examples"]:
                q = line["input"].strip()
                a = line["target"].strip()
                date_test_qs.append(q)
                date_test_ans.append(a)

        system_prompt = {
            "role": "system",
            "content": "Follow the given examples and answer the question."
                       "You must choose one from the five options A/B/C/D/E, vague answers are not allowed."
                       "The last sentence must be: The answer is \\box{}. The choice of final answer is filled into \\boxed{}."
                       "Describe your thought process step-by-step using Step 1, Step 2, ... etc."
        }

        return date_test_qs, date_test_ans, system_prompt

    if 'gsm' in test_set.lower():
        gsm8k_test = []
        with open(test_set, 'r') as rf:
            for line in rf:
                d = json.loads(line)
                gsm8k_test.append(d)

        gsm8k_test_qs = []
        gsm8k_test_ans = []
        for item in gsm8k_test:
            gsm8k_test_qs.append(item['question'])
            ans_num = float(re.search(r"#### ([-\d,]+)", item['answer']).group(1).replace(",", ""))
            gsm8k_test_ans.append(ans_num)

        system_prompt = {
            "role": "system",
            "content": "Follow the given examples and answer the question."
                       "The last sentence must be: The answer is \\box{}. The number of final answer is filled into \\boxed{}."
                       "Describe your thought process step-by-step using Step 1, Step 2, ... etc."
        }

        return gsm8k_test_qs, gsm8k_test_ans, system_prompt

    if 'strategyqa' in test_set.lower():
        sqa_idx = []
        with open(test_idx_set, 'r') as rf1:
            for line in rf1:
                d = int(line)
                sqa_idx.append(d)

        sqa_test_qs = []
        sqa_test_ans = []
        with open(test_set, 'r') as f1:
            json_data = json.load(f1)
            for idx in sqa_idx:
                q = json_data["examples"][idx]["input"]
                a = json_data["examples"][idx]["target"][:3].replace('.', '').lower()
                sqa_test_ans.append(a)
                sqa_test_qs.append(q)

        system_prompt = {
            "role": "system",
            "content": "Follow the given examples and answer the question. "
                       "You must give a clear answer of 'yes' or 'no'. Vague answers are not allowed."
                       "The last sentence must be: The answer is \\box{}. The judgement string('yes'/'no') of final answer is filled into \\boxed{}."
                       "Describe your thought process step-by-step using Step 1, Step 2, ... etc."
        }

        return sqa_test_qs, sqa_test_ans, system_prompt
