from nnsight import LanguageModel
import pandas as pd
import torch as t
import argparse
import os
from generate_acts import load_model, tracer_kwargs


def compute_logprobs(model, dataset, remote=True):

    df = pd.read_csv(f'datasets/{dataset}.csv')

    all_logprobs = []
    # for each statement, get the logprob of the statement
    for statement in df['statement'].tolist():
        # tokenized outside the trace: the runner.batched_input this used to read is 0.2-era
        # API, and the tokenizer call is cheap next to the forward pass
        tokens = t.tensor(model.tokenizer(statement).input_ids[1:])
        with model.trace(statement, remote=remote, **tracer_kwargs):
            logprobs = model.lm_head.output.log_softmax(dim=-1)
            summed = logprobs[0, t.arange(len(tokens)), tokens].sum().save()
        all_logprobs.append(summed.item())
    
    df['logprob'] = all_logprobs

    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute logprobs for statements in a dataset")
    parser.add_argument("--model", default="llama-2-13b")
    parser.add_argument("--dataset", default="care_harm")
    parser.add_argument("--device", default="remote")
    args = parser.parse_args()

    model = load_model(args.model, args.device)

    remote = args.device == 'remote'

    df = compute_logprobs(model, args.dataset, remote=remote)

    out_path = f'experimental_outputs/logprobs/{args.dataset}.csv'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path)