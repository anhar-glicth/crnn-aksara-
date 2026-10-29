import numpy as np
from collections import defaultdict

def levenshtein_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]

def greedy_decoder(log_probs, idx2char, input_lengths=None):
    best_paths = log_probs.argmax(dim=2).permute(1, 0)
    decoded_strings = []
    for i, path in enumerate(best_paths):
        if input_lengths is not None:
            seq = path[:input_lengths[i]].tolist()
        else:
            seq = path.tolist()
        chars = []
        prev = 0
        for idx in seq:
            if idx != 0 and idx != prev:
                chars.append(idx2char.get(idx, ""))
            prev = idx
        decoded_strings.append("".join(chars))
    return decoded_strings

def ctc_prefix_beam_search(log_probs_matrix, idx2char, beam_width=10):
    beams = {("", False): 0.0}
    
    for t in range(len(log_probs_matrix)):
        step_probs = log_probs_matrix[t]
        new_beams = defaultdict(lambda: -float('inf'))
        
        for (prefix, ended_in_blank), log_p in beams.items():
            for c_idx, lp in enumerate(step_probs):
                total_lp = log_p + lp
                if c_idx == 0:
                    curr_p = new_beams[(prefix, True)]
                    new_beams[(prefix, True)] = np.logaddexp(curr_p, total_lp)
                else:
                    char = idx2char.get(c_idx, "")
                    if len(prefix) > 0 and prefix[-1] == char and not ended_in_blank:
                        curr_p = new_beams[(prefix, False)]
                        new_beams[(prefix, False)] = np.logaddexp(curr_p, total_lp)
                    else:
                        new_prefix = prefix + char
                        curr_p = new_beams[(new_prefix, False)]
                        new_beams[(new_prefix, False)] = np.logaddexp(curr_p, total_lp)
                        
        sorted_beams = sorted(new_beams.items(), key=lambda x: x[1], reverse=True)[:beam_width]
        beams = dict(sorted_beams)
        
    best_prefix, _ = max(beams.keys(), key=lambda k: beams[k])
    return best_prefix
