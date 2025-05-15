def compute_acc(seed_eval_results_df):
    predictions = seed_eval_results_df["parsed"]
    answers = seed_eval_results_df["answer"]
    matched = predictions == answers
    acc = matched.sum() / matched.shape[0]
    return acc
