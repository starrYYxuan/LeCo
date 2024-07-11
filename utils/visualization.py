import json


def visualization(ori_file, rthk_file, output_file):
    with open(output_file, "w", encoding="utf-8") as fw:
        ori_lines = open(ori_file, "r", encoding="utf-8")
        rthk_lines = open(rthk_file, "r", encoding="utf-8")
        for idx, (ori_line, rthk_line) in enumerate(zip(ori_lines, rthk_lines)):
            ori_jo = json.loads(ori_line.strip())
            rthk_jo = json.loads(rthk_line.strip())
            fw.write(f"Question:\n{ori_jo['question']}\nGold Answer: {ori_jo['answer']}\nOriginal Answer:\n")
            for step in ori_jo["step_info"]:
                fw.write(f"{step['sentence']}\t\t({sum(step['confidence'])}){step['confidence']}\n")
            for i, rk_steps in enumerate(rthk_jo["rk_steps"]):
                fw.write(f"Rethink Answer {str(i)}:\n")
                for rk_step in rk_steps["step_info"]:
                    fw.write(f"{rk_step['sentence']}\t\t({sum(rk_step['confidence'])}){rk_step['confidence']}\n")
            fw.write("\n\n")




def error_cases_visualization(ori_file, rthk_file, output_file):
    R2W = 0  # Right 2 Wrong Ratio
    W2W = 0  # Wrong 2 Wrong Ratio
    W2R = 0  # Wrong 2 Right Ratio
    TW = 0  # Total Wrong instances
    with open(output_file, "w", encoding="utf-8") as fw:
        ori_lines = open(ori_file, "r", encoding="utf-8")
        rthk_lines = open(rthk_file, "r", encoding="utf-8")
        for idx, (ori_line, rthk_line) in enumerate(zip(ori_lines, rthk_lines)):
            ori_jo = json.loads(ori_line.strip())
            rthk_jo = json.loads(rthk_line.strip())
            if rthk_jo['rk_steps'] == [] :
                continue
            else:
                if ori_jo['pred'] != ori_jo['answer'] or rthk_jo['rk_steps'][-1]['pred'] != rthk_jo['rk_steps'][-1]['answer']:
                    TW += 1
                    if ori_jo['pred'] != ori_jo['answer'] and rthk_jo['rk_steps'][-1]['pred'] != rthk_jo['rk_steps'][-1]['answer']:
                        W2W += 1
                        fw.write("Both original answer and rethink answer are not correct.\n")
                        fw.write(f"Question:\n{ori_jo['question']}\nGold Answer: {ori_jo['answer']}\nOriginal Answer:\n")
                        for step in ori_jo["step_info"]:
                            fw.write(f"{step['sentence']}\t\t({sum(step['confidence'])}){step['confidence']}\n")
                        for i, rk_steps in enumerate(rthk_jo["rk_steps"]):
                            fw.write(f"Rethink Answer {str(i)}:\n")
                            for rk_step in rk_steps["step_info"]:
                                fw.write(f"{rk_step['sentence']}\t\t({sum(rk_step['confidence'])}){rk_step['confidence']}\n")
                        fw.write("\n\n")
                    if ori_jo['pred'] == ori_jo['answer'] and rthk_jo['rk_steps'][-1]['pred'] != rthk_jo['rk_steps'][-1]['answer']:
                        R2W += 1
                        fw.write("Original answer is correct, while rethink answer is not correct.\n")
                        fw.write(f"Question:\n{ori_jo['question']}\nGold Answer: {ori_jo['answer']}\nOriginal Answer:\n")
                        for step in ori_jo["step_info"]:
                            fw.write(f"{step['sentence']}\t\t({sum(step['confidence'])}){step['confidence']}\n")
                        for i, rk_steps in enumerate(rthk_jo["rk_steps"]):
                            fw.write(f"Rethink Answer {str(i)}:\n")
                            for rk_step in rk_steps["step_info"]:
                                fw.write(f"{rk_step['sentence']}\t\t({sum(rk_step['confidence'])}){rk_step['confidence']}\n")
                        fw.write("\n\n")
                    if ori_jo['pred'] != ori_jo['answer'] and rthk_jo['rk_steps'][-1]['pred'] == rthk_jo['rk_steps'][-1]['answer']:
                        W2R += 1
                        fw.write("Original answer is not correct, while rethink answer is correct.\n")
                        fw.write(f"Question:\n{ori_jo['question']}\nGold Answer: {ori_jo['answer']}\nOriginal Answer:\n")
                        for step in ori_jo["step_info"]:
                            fw.write(f"{step['sentence']}\t\t({sum(step['confidence'])}){step['confidence']}\n")
                        for i, rk_steps in enumerate(rthk_jo["rk_steps"]):
                            fw.write(f"Rethink Answer {str(i)}:\n")
                            for rk_step in rk_steps["step_info"]:
                                fw.write(f"{rk_step['sentence']}\t\t({sum(rk_step['confidence'])}){rk_step['confidence']}\n")
                        fw.write("\n\n")

    print("W2W: There are %d wrong original cases which is not modified, and the ratio is %f.\n"%(W2W,W2W/TW))
    print("W2R: There are %d wrong original cases which is modified to be right, and the ratio is %f.\n" % (W2R, W2R/TW))
    print("R2W: There are %d right original cases which is modified to be wrong, and the ratio is %f." % (R2W, R2W/TW))
