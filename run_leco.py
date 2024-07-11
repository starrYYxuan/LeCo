from tqdm import tqdm
import argparse

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig

from utils.visualization import visualization, error_cases_visualization

from utils.ans_process import *
from utils.divergence import *
from utils.load_dataset import *
from utils.process_pred_str import *

def majorElement(num_list):
    dict = {}
    num_list.reverse()
    for i in range(len(num_list)):
        if i == len(num_list) - 1:
            dict[num_list[i]] = dict.get(num_list[i], 0) + 1.5
        else:
            dict[num_list[i]] = dict.get(num_list[i], 0) + 1.0
    max_v = 1.0
    major_num = []
    # print(dict)
    for k, v in dict.items():
        if v > max_v:
            major_num = k
            max_v = v
    return major_num


def find_error_step(step_info):
    ans_text = " \n".join([step["sentence"] for step in step_info])
    conf_sorted_steps = sorted(step_info, key=lambda x: sum(x["confidence"]))
    res = []
    for step in conf_sorted_steps:
        if "step by step" in step["sentence"] or "the answer is" in step["sentence"].lower():
            continue
        else:
            res.append(step)
    return ans_text, res


def extract_step_info(logprob_tokens, logprobs, divergence_type, ans_solution, tokenizer, test_set):
    step_token_prob = []
    step_confidence = []
    curr_sent = ""

    for i in range(len(logprobs)):
        if logprob_tokens[i] == 185:
            if len(step_token_prob) > 0:
                step_avg_prob = np.mean(step_token_prob)
                if curr_sent.startswith("Step"):
                    transition_prob = np.mean(step_token_prob[3:8])
                else:
                    transition_prob = np.mean(step_token_prob[:5])
                step_divergence = kl_divergence(step_token_prob) if divergence_type.lower() == "kl" else js_divergence(
                    step_token_prob)
                curr_sent += tokenizer.decode(logprob_tokens[i].item(), skip_special_tokens=True)
                curr_sent += ' '
                step_confidence.append({
                    "id": len(step_confidence),
                    "sentence": curr_sent,
                    "confidence": [step_avg_prob, transition_prob, -step_divergence]
                })

                curr_sent = ""
                step_token_prob = []
        else:
            step_token_prob.append(logprobs[i])
            curr_sent += tokenizer.decode(logprob_tokens[i].item(), skip_special_tokens=True)

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


def extract_token_logprobs(logits):
    processed_logits = []
    for logit in logits:
        processed_logits.append(logit[0].item())
    return processed_logits


def token_efficient_reasoning():
    # load datasets
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("--test_set", type=str, default=None, required=True)
    arg_parse.add_argument("--test_idx_set", type=str, default=None)
    arg_parse.add_argument("--prompts", type=str, default=None, required=True)
    arg_parse.add_argument("--engine", type=str, default="deepseek")
    arg_parse.add_argument("--temp", type=float, default=0.)
    arg_parse.add_argument("--output_dir", type=str, default=None, required=True)
    arg_parse.add_argument("--ckpt_dir", type=str, , default=None, required=True)

    arg_parse.add_argument("--divergence_type", type=str, default='KL')

    args = arg_parse.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load data
    test_qs, test_ans, _ = load_dataset(args.test_set, args.test_idx_set)

    test_qs, test_ans = test_qs[:30], test_ans[:30]

    # load prompts
    prompt_complex = open(args.prompts, "r", encoding="utf-8").read()

    engine, temp, output_dir, diver = args.engine, args.temp, args.output_dir, args.divergence_type

    origin_file = f"{output_dir}/ori.{engine}.temp{str(temp)}.{diver}.txt"
    rethink_file = f"{output_dir}/rk.{engine}.temp{str(temp)}.{diver}.txt"

    model_path = args.ckpt_dir
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=device,
                                                 attn_implementation="flash_attention_2")
    model.generation_config = GenerationConfig.from_pretrained(model_path)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    torch.set_grad_enabled(False)
    model.eval()

    input_token_num = 0
    output_token_num = 0

    with open(origin_file, 'w', encoding="utf-8") as fo, open(rethink_file, 'w', encoding="utf-8") as fr:
        for idx, (ques, gold_ans) in tqdm(enumerate(zip(test_qs, test_ans)),
                                          total=len(test_qs)):
            # ans_num = float(re.search(r"#### (-?\d+)", gold_ans).group(1))
            ans_num = gold_ans
            ans_candidates = []

            prompt_q = prompt_complex + f'\n\nQuestion: {ques}\n' + 'Please reason step by step, and put your final answer within \\boxed{}.\nAnswer:'

            original_prompt = tokenizer(prompt_q, return_tensors="pt")

            generation_config = dict(
                do_sample=False,
                num_beams=1,
                max_new_tokens=512
            )

            with torch.no_grad():
                inputs = original_prompt
                inputs.to("cuda")

                generation_output = model.generate(
                    **inputs,
                    output_hidden_states=True,
                    output_scores=True,
                    return_dict_in_generate=True,
                    **generation_config
                )

                logprob_token_ids = generation_output.sequences[0][len(inputs["input_ids"][0]):]

                logits = generation_output.scores
                soft_logits = []
                top_idx = []
                for item in logits:
                    mid_logits, m_idx = torch.sort(item[0], descending=True)  # descending为False，升序，为True，降序
                    mid_logits = mid_logits[:5]
                    soft_logits.append(torch.nn.functional.softmax(mid_logits, dim=-1))

                logprobs = extract_token_logprobs(soft_logits)

                assert len(logprob_token_ids) == len(logprobs)
                ans_solution = tokenizer.decode(logprob_token_ids, skip_special_tokens=True)

                step_info, pred_num = extract_step_info(logprob_token_ids, logprobs, diver, ans_solution, tokenizer, args.test_set)

                original_ans, potential_error_step = find_error_step(step_info)
                ans_candidates.append(pred_num)

                current_solution = [step["sentence"] for step in step_info]

                used_out_tokens = len(generation_output.scores)
                used_in_tokens = len(inputs["input_ids"][0])
                output_token_num += used_out_tokens
                input_token_num += used_in_tokens

                fo.write(json.dumps({"idx": str(idx), "question": ques, "step_info": step_info, "pred": pred_num,
                                     "answer": ans_num, "used_in_tokens": used_in_tokens,
                                     "used_out_tokens": used_out_tokens}, ensure_ascii=False) + "\n")

                # rethink stage
                iteration_num = 0
                prev_error_step = 0
                prev_ans = 'Null'
                rethink_steps = []

                while prev_ans != pred_num and iteration_num < 3:

                    prev_ans = pred_num
                    min_step_num = 100
                    for i in range(min(len(potential_error_step), 2)):
                        step_num = potential_error_step[i]["id"]
                        min_step_num = min(min_step_num, step_num)
                    min_step_num += prev_error_step
                    # print('min_step_num:-------',min_step_num)

                    # If min_step_num is the last step, generate a new solution instead.
                    if min_step_num < len(current_solution):
                        rethink_step_materials = " \n".join([current_solution[i] for i in range(min_step_num)])
                        re_material = (f"{rethink_step_materials}")
                        prompt_r = prompt_complex + f'\n\nQuestion: {ques}\n' + "Answer: " + re_material + '\nPlease reason step by step, and put your final answer within \\boxed{}.'

                    rethink_inputs = tokenizer(prompt_r, padding=True, return_tensors="pt")

                    rethink_inputs.to("cuda")
                    rethink_output = model.generate(
                        # **{key: tensor.to(model.device) for key, tensor in rethink_input.items()},
                        input_ids=rethink_inputs["input_ids"].to("cuda"),
                        attention_mask=rethink_inputs['attention_mask'].to("cuda"),
                        # eos_token_id=tokenizer.eos_token_id,
                        # pad_token_id=tokenizer.pad_token_id,
                        output_hidden_states=True,
                        output_scores=True,
                        return_dict_in_generate=True,
                        **generation_config
                    )

                    used_out_tokens = len(rethink_output.scores)
                    used_in_tokens = len(rethink_inputs["input_ids"][0])
                    output_token_num += used_out_tokens
                    input_token_num += used_in_tokens

                    logprob_token_ids = rethink_output.sequences[0][len(rethink_inputs["input_ids"][0]):]
                    logits = rethink_output.scores
                    soft_logits = []
                    for item in logits:
                        mid_logits, m_idx = torch.sort(item[0], descending=True)  # descending为False，升序，为True，降序
                        mid_logits = mid_logits[:5]
                        soft_logits.append(torch.nn.functional.softmax(mid_logits, dim=-1))
                    logprobs = extract_token_logprobs(soft_logits)
                    ans_solution = tokenizer.decode(logprob_token_ids, skip_special_tokens=True)
                    ans_solution = ans_solution.split('Question', 1)[0]
                    step_info, pred_num = extract_step_info(logprob_token_ids, logprobs, diver, ans_solution, tokenizer, args.test_set)

                    curr_ans, potential_error_step = find_error_step(step_info)
                    ans_candidates.append(pred_num)

                    current_solution = current_solution[:min_step_num]
                    current_solution += [s["sentence"] for s in step_info]

                    # majority vote
                    if prev_ans == pred_num or iteration_num == 2:
                        pred_num_fw = majorElement(ans_candidates)
                    else:
                        pred_num_fw = pred_num

                    rethink_steps.append({"rk_idx": f"{str(idx)}_iter{str(iteration_num)}", "step_info": step_info,
                                          "pred": pred_num_fw, "answer": ans_num, "ans_candidates": ans_candidates,
                                          "used_in_tokens": used_in_tokens,
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
    parse_rethink_pred_ans(rethink_file, origin_file)

    overall_file = f"{output_dir}/overall.{engine}.temp{str(temp)}.{diver}.txt"
    visualization(origin_file, rethink_file, overall_file)
    error_analysis_file = f"{output_dir}/ea.{engine}.temp{str(temp)}.{diver}.txt"
    error_cases_visualization(origin_file, rethink_file, error_analysis_file)


if __name__ == "__main__":
    print('Start reasoning *********************:')
    token_efficient_reasoning()
    print('End reasoning =======================:')
