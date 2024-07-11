import numpy as np
import scipy.stats


def kl_divergence(current_sentence):
    step_len = len(current_sentence)
    # print(step_len)
    # kl_uniform = np.random.uniform(0,1, size=(step_len))
    kl_uniform = []
    for i in range(step_len):
        kl_uniform.append(float(1 / step_len))
    # print(kl_uniform)
    step_kl = np.log(np.power(scipy.stats.entropy(current_sentence, kl_uniform), 1/3)+1)
    # step_kl = scipy.stats.entropy(current_sentence, kl_uniform)

    return step_kl

def js_divergence(current_sentence):
    step_len = len(current_sentence)
    # print(step_len)
    # kl_uniform = np.random.uniform(0,1, size=(step_len))
    kl_uniform = []
    for i in range(step_len):
        kl_uniform.append(float(1/step_len))
    # print(kl_uniform)
    M = (np.asarray(kl_uniform) + np.asarray(current_sentence))/2
    # M = (kl_uniform + current_sentence)/2
    # step_kl = np.log(np.power(0.5 * scipy.stats.entropy(current_sentence, M, base=2) + 0.5 * scipy.stats.entropy(kl_uniform,M, base=2),1/9)+1)
    step_kl = 0.5 * scipy.stats.entropy(current_sentence, M, base=2) + 0.5 * scipy.stats.entropy(kl_uniform, M, base=2)
    return step_kl
