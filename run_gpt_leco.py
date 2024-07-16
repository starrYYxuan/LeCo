import re
import openai
from tqdm import tqdm
import os
import argparse
from utils.visualization import visualization, error_cases_visualization

from utils.ans_process import *
from utils.divergence import *
from utils.load_dataset import *
from utils.process_pred_str import *

os.environ['http_proxy'] = 'http://w84317177:%40Wh138870710@proxysg.huawei.com:8080'
os.environ['https_proxy'] = 'http://w84317177:%40Wh138870710@proxysg.huawei.com:8080'
os.environ['no_proxy'] = '10.213.96.114'


def majorElement(num_list):
    dict = {}
    num_list.reverse()
    for i in range(len(num_list)):
      if i == len(num_list)-1:
        dict[num_list[i]] = dict.get(num_list[i], 0)+1.5
      else:
        dict[num_list[i]] = dict.get(num_list[i], 0)+1.0
    max_v = 1.0
    major_num = []
    # print(dict)
    for k,v in dict.items():
      if v > max_v:
        major_num = k
        max_v = v
    return major_num


def find_error_step(step_info):
    ans_text = " \n".join([step["sentence"] for step in step_info])
    conf_sorted_steps = sorted(step_info, key=lambda x: sum(x["confidence"]))
    res = []
    for step in conf_sorted_steps:
        if "think step by step" in step["sentence"] or "the answer is" in step["sentence"].lower():
            continue
        else:
            res.append(step)
    return ans_text, res


def extract_step_info(logprob_tokens, logprobs, divergence_type, ans_solution, test_set):
    step_token_prob = []
    # total confidence
    step_confidence = []
    # record sentence words
    curr_sent = ""
    # 遍历字典列表，计算每个 '\n\n' 或 '\n' 之间的句子的 softmax(values) 平均值
    for d, t in zip(logprobs, logprob_tokens):
        _sum = np.sum(np.exp(list(d.values())))
        for k, v in d.items():
            d[k] = np.exp(v) / _sum
        try:
            assert t in d
        except AssertionError:
            continue

        if "\n" in t:
            if len(step_token_prob) > 0:
                step_avg_prob = np.mean(step_token_prob)
                if curr_sent.startswith("Step"):
                    transition_prob = np.mean(step_token_prob[3:4])
                else:
                    transition_prob = np.mean(step_token_prob[:1])
                step_divergence = kl_divergence(step_token_prob) if divergence_type.lower() == "kl" else js_divergence(
                    step_token_prob)
                curr_sent += t.strip()
                step_confidence.append({
                    "id": len(step_confidence),
                    "sentence": curr_sent,
                    "confidence": [step_avg_prob, transition_prob, -step_divergence]
                })

                curr_sent = ""
                step_token_prob = []
        else:
            step_token_prob.append(d[t])
            curr_sent += t

    if len(step_token_prob) > 0 and len(curr_sent.strip()) > 0:
        step_avg_prob = np.mean(step_token_prob)
        if curr_sent.startswith("Step"):
            transition_prob = np.mean(step_token_prob[3:8])
        else:
            transition_prob = np.mean(step_token_prob[:5])
        step_divergence = kl_divergence(step_token_prob) if divergence_type.lower() == "kl" else js_divergence(
            step_token_prob)
        step_confidence.append({
            "id": len(step_confidence),
            "sentence": curr_sent,
            "confidence": [step_avg_prob, transition_prob, -step_divergence]
        })
    # extract predicted result
    pred_num = extract_math_answer(ans_solution, test_set)

    return step_confidence, pred_num


def extract_token_logprobs(content):
    top_logits_dict = {}
    top_logits = []
    top_logits_token = []
    for item in content:
        for log in item.top_logprobs:
            top_logits_dict.update({log.token: log.logprob})
        sorted_top_logits_dict = dict(sorted(top_logits_dict.items(), key=lambda x: x[1], reverse=True))
        top_logits.append(sorted_top_logits_dict)
        top_logits_dict = {}
        top_logits_token.append(item.token)

    return top_logits_token, top_logits


def token_efficient_reasoning(seed):
    # load datasets
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("--test_set", type=str, default="datasets/gsm8k.jsonl")
    arg_parse.add_argument("--prompts", type=str, default="D:\deepseek-aideepseek-math-7b-rl\LeCo-main\prompts\gsm8k\complex_step.txt")
    arg_parse.add_argument("--engine", type=str, default="gpt-3.5-turbo-1106")
    arg_parse.add_argument("--test_idx_set", type=str, default=None)
    arg_parse.add_argument("--temp", type=float, default=0.)
    arg_parse.add_argument("--output_dir", type=str, default="D:\deepseek-aideepseek-math-7b-rl\LeCo-main\exp_res")
    arg_parse.add_argument("--divergence_type", type=str, default='kl')
    arg_parse.add_argument("--api_key", type=str, default='sk-fRHfIhmhcHiLS0sGCd7d9c5c752b4aFcAfB972E51c73F2D7')
    args = arg_parse.parse_args()

    test_qs, test_ans, _ = load_dataset(args.test_set, args.test_idx_set)
    test_qs, test_ans = test_qs[:5], test_ans[:5]

    # load prompts
    prompt_complex = open(args.prompts, "r", encoding="utf-8").read()

    from tenacity import (
        retry,
        wait_chain,
        wait_fixed
    )

    @retry(wait=wait_chain(*[wait_fixed(3) for i in range(3)] +
                            [wait_fixed(5) for i in range(2)] +
                            [wait_fixed(10)]))
    def completion_with_backoff(**kwargs):
        return openai.ChatCompletion.create(**kwargs)

    api_key = args.api_key
    openai.api_key = api_key
    engine, temp, output_dir, diver = args.engine, args.temp, args.output_dir, args.divergence_type
    print(f"Engine: {engine}\tTemp: {str(temp)}\tSeed: {seed}\t")

    input_token_num = 0
    output_token_num = 0
    system_prompt = {
        "role": "system",
        "content": "Follow the given examples and answer the question."
                   "The last sentence must be: The answer is \\box{}. The number of final answer is filled into \\boxed{}."
                   "Describe your thought process step-by-step using Step 1, Step 2, ... etc."
    }

    origin_file = f"{output_dir}/major_gsm8k.ori.{engine}.temp{str(temp)}.{str(seed)}.{diver}.txt"
    rethink_file = f"{output_dir}/major_gsm8k.rk.{engine}.temp{str(temp)}.{str(seed)}.{diver}.txt"


    with (open(origin_file, 'a', encoding="utf-8") as fo, open(rethink_file, 'a', encoding="utf-8") as fr):
        for idx, (ques, gold_ans) in tqdm(enumerate(zip(test_qs, test_ans)),
                                          total=len(test_qs)):
            ans_num = gold_ans
            ans_candidates = []
            # print(f"Question {str(idx)} - Gold Ans: {str(ans_num)}")
            prompt_q = prompt_complex + f'\n\nQuestion: {ques}\nAnswer: '
            response = openai.chat.completions.create(
                model=engine,
                messages=[
                    system_prompt,
                    {"role": "user", "content": prompt_q},
                ],
                max_tokens=512,
                temperature=temp,
                logprobs=True,
                top_logprobs=5,
                seed=seed
            )
            logprob_tokens, logprobs = extract_token_logprobs(response.choices[0].logprobs.content)
            ans_solution = response.choices[0].message.content
            step_info, pred_num = extract_step_info(logprob_tokens, logprobs, diver,ans_solution,args.test_set)
            # print(f"Ori Pred Ans: {str(pred_num)}")
            original_ans, potential_error_step = find_error_step(step_info)
            ans_candidates.append(pred_num)

            current_solution = [step["sentence"] for step in step_info]

            used_out_tokens = response.usage.completion_tokens
            used_in_tokens = response.usage.prompt_tokens
            output_token_num += used_out_tokens
            input_token_num += used_in_tokens

            fo.write(json.dumps({"idx": str(idx), "question": ques, "step_info": step_info, "pred": pred_num,
                                 "answer": ans_num, "used_in_tokens": used_in_tokens,
                                 "used_out_tokens": used_out_tokens}, ensure_ascii=False) + "\n")

            # rethink stage
            iteration_num = 0
            prev_error_step = 0
            prev_ans = float("nan")
            rethink_steps = []
            # messages = [system_prompt, {"role": "user", "content": prompt_q},
            #             {"role": "assistant", "content": response.choices[0].message.content}]

            while prev_ans != pred_num and iteration_num < 3:

                prev_ans = pred_num
                min_step_num = 100
                for i in range(min(len(potential_error_step), 2)):
                    step_num = potential_error_step[i]["id"]
                    # if potential_error_step[i]["sentence"].startswith("Step"):
                    #     _step_num = int(re.findall(r"Step ?(\d+):", potential_error_step[i])[0])
                    #     assert step_num == _step_num
                    min_step_num = min(min_step_num, step_num)
                min_step_num += prev_error_step
                prompt_r = prompt_q
                if min_step_num < len(current_solution):
                    rethink_step_materials = " \n".join([current_solution[i] for i in range(min_step_num)])
                    re_material = (f"{rethink_step_materials}\nStep{min_step_num}: ")
                    # role_dict["content"] = re_material
                    # messages.append(role_dict)
                    prompt_r = prompt_q + re_material

                messages = [system_prompt, {"role": "user", "content": prompt_r}]

                rethink_response = openai.chat.completions.create(
                    model=engine,
                    messages=messages,
                    max_tokens=512,
                    temperature=temp,
                    logprobs=True,
                    top_logprobs=5,
                    seed=seed
                )

                # role_dict = {"role": "assistant", "content": f"Answer: {rethink_step_materials}\nStep{min_step_num}: " + rethink_response.choices[0].message.content}
                # messages.append(role_dict)
                used_out_tokens = rethink_response.usage.completion_tokens
                used_in_tokens = rethink_response.usage.prompt_tokens
                output_token_num += used_out_tokens
                input_token_num += used_in_tokens
                logprob_tokens, logprobs = extract_token_logprobs(rethink_response.choices[0].logprobs.content)
                ans_solution = rethink_response.choices[0].message.content
                step_info, pred_num = extract_step_info(logprob_tokens, logprobs, diver,ans_solution,args.test_set)
                ans_candidates.append(pred_num)

                step_info[0]["sentence"] = f"Step{min_step_num}: {step_info[0]['sentence']}"
                # print(f"Rethink round {str(iteration_num)} - Ans: {str(pred_num)}")
                curr_ans, potential_error_step = find_error_step(step_info)

                current_solution = current_solution[:min_step_num]
                current_solution += [s["sentence"] for s in step_info]


                # rethink_steps.append({"rk_idx": f"{str(idx)}_iter{str(iteration_num)}", "step_info": step_info,
                #                       "pred": pred_num, "answer": ans_num, "used_in_tokens": used_in_tokens,
                #                       "used_out_tokens": used_out_tokens})

                # majority vote
                if prev_ans == pred_num or iteration_num == 2:
                    pred_num_fw = majorElement(ans_candidates)
                    # print(f"Total round {str(iteration_num)} - Final Ans: {str(pred_num)}")
                else:
                    pred_num_fw = pred_num

                rethink_steps.append({"rk_idx": f"{str(idx)}_iter{str(iteration_num)}", "step_info": step_info,
                                      "pred": pred_num_fw, "answer": ans_num, "ans_candidates":ans_candidates,"used_in_tokens": used_in_tokens,
                                      "used_out_tokens": used_out_tokens})

                iteration_num += 1
                prev_error_step = min_step_num

            fr.write(json.dumps({"idx": str(idx), "rk_steps": rethink_steps}, ensure_ascii=False) + "\n")

    print("-----The totally consumed input token number is: ", input_token_num)

    print("-----The totally consumed output token number is: ", output_token_num)

    print('-----Results of original reasoning process:')
    # original answer
    parse_pred_ans(origin_file)

    print('-----Results of rethink reasoning process:')
    # rethink answer
    parse_rethink_pred_ans(rethink_file,origin_file)

    overall_file = f"{output_dir}/major_gsm8k.overall.{engine}.temp{str(temp)}.{str(seed)}.{diver}.txt"
    visualization(origin_file, rethink_file, overall_file)
    error_analysis_file = f"{output_dir}/major_gsm8k.ea.{engine}.temp{str(temp)}.{str(seed)}.{diver}.txt"
    error_cases_visualization(origin_file, rethink_file, error_analysis_file)


if __name__ == "__main__":
    seed_list = [111]
    for seed in seed_list:
        print('Start reasoning *********************:')
        token_efficient_reasoning(seed=seed)
        print('End reasoning =======================:')
