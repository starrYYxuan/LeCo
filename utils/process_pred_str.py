import re
def extract_math_answer(pred_str,test_set):
    try:
        if ('boxed' in pred_str):
            ans = pred_str.split('boxed')[-1]
        elif ('the answer is ' in pred_str):
            ans = pred_str.split('the answer is ')[-1].strip()
        elif 'The answer is ' in pred_str:
            ans = pred_str.split('The answer is ')[-1].strip()
        else:
            ans = pred_str

        if 'aqua' in test_set.lower():
            pattern = r'\D\)'
            pred = re.findall(pattern, ans)

            if (len(pred) >= 1):
                pred = pred[-1][:-1]
            else:
                pred = 'Null'

        if 'csqa' in test_set.lower():
            pattern = r'\(\D\)'
            pred = re.findall(pattern, ans)

            if (len(pred) >= 1):
                pred = pred[-1][1:-1]
            else:
                pred = 'Null'

        if 'date' in test_set.lower():
            pattern = r'\(\D\)'
            pred = re.findall(pattern, ans)

            if (len(pred) >= 1):
                pred = pred[-1]
            else:
                pred = 'Null'

        if 'gsm8k' in test_set.lower():
            pattern = r'-?\d*[\.,]?\d+'
            pred = re.findall(pattern, ans)

            if (len(pred) >= 1):
                pred = float(pred[-1].replace(',', ''))
            else:
                pred = float('nan')

        if 'strategyqa' in test_set.lower():
            pattern = r'\(\D+\)'
            pred = re.findall(pattern, ans)

            if (len(pred) >= 1):
                # print(pred_str)
                pred = pred[-1][1:-1]
                pred = pred.lower()
            else:
                pred = 'Null'


    except Exception:
        print(f"Cannot parse the resulting num in predicted solution {pred_str}.\n")
        pred = float("nan")

    return pred
