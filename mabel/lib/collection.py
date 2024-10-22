import math
from dataclasses import dataclass, field
from os import makedirs
from typing import List, Optional, Tuple, Union

import torch as pt
from datasets import DatasetDict
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import AutoModelForCausalLM, PreTrainedModel

from lib.collapse import Statistics
from lib.model import ModelArguments, strip_model, strip_model_bert


@dataclass
class CollectArguments:
    do_collect: bool = field(
        default=False,
        metadata={"help": ("Whether to collect statistics.")},
    )
    batch_size: Optional[int] = field(
        default=1,
        metadata={"help": ("Batch size of collection.")},
    )
    device: Optional[str] = field(
        default="cpu",
        metadata={"help": ("Device to run the collection on.")},
    )
    model_ckpt_idx: Optional[int] = field(
        default=None,
        metadata={"help": ("Which checkpoint (by list index) to load for the model.")},
    )
    save_every: Optional[int] = field(
        default=1024,
        metadata={"help": ("How often to save stats.")},
    )
    single: bool = field(
        default=False,
        metadata={"help": ("Save only once.")},
    )
    stage: Optional[str] = field(
        default="means",
        metadata={"help": ("Which measure to collect.")},
    )
    stats_dir: Optional[str] = field(
        default="stats",
        metadata={"help": ("Where to save statistics.")},
    )
    verbose: bool = field(
        default=False,
        metadata={"help": ("Verbose logging and information.")},
    )


def truncate_and_pad(batch: List[Tensor]) -> Tuple[Tensor, Tensor]:
    """Construct a uniform batch of sequences.
    batch: list of original sequences.
    """
    assert len(batch) > 0
    if len(batch) == 1:
        return None, batch[0].unsqueeze(0)

    masks = [[1] * len(seq) for seq in batch]
    longest = max([sum(seq) for seq in masks])

    masks = pt.tensor([m + [0] * (longest - len(m)) for m in masks])
    batch = pad_sequence(batch, batch_first=True).clone().detach()

    return masks, batch


def process_batch(
    model: PreTrainedModel,
    batch: List[Tensor],
    stats_device: Union[str, pt.device] = "cpu",
) -> Tuple[Tensor, Tensor]:
    """Construct a uniform batch of sequences.
    model: CausalLM to to make token predictions on sequences.
    batch: list of original sequences.
    stats_device: which device (cpu/gpu) on which to infer.
    """
    masks, batch = truncate_and_pad(batch)
    output = model(
        batch.to(model.device),
        attention_mask=masks if masks is None else masks.to(model.device),
        output_hidden_states=False,
    )

    import pdb; pdb.set_trace()
    embeds = output.logits
    if masks is not None:
        embeds = pt.unsqueeze(masks.to(embeds.device), -1) * embeds

    # offset by one for the next word prediction
    Y = batch[:, 1:].to(stats_device)
    X = embeds[:, :-1].to(stats_device)
    if masks is not None:
        idx = masks[:, 1:].bool()
        Y, X = Y[idx], X[idx]

    return X.squeeze(), Y.squeeze()


def collect_embeddings(
    args: CollectArguments,
    # model_args: ModelArguments,
    model: AutoModelForCausalLM,
    datasets: DatasetDict,
):
    """Collection function for any stage (means/vars/decs).
    args: Collection arguments supplied from top-level script.
    model_args: Model/training arguments supplied from top-level script.
    model: CausalLM architecture and weights.
    datasets: Processed dataset.
    """
    pt.set_grad_enabled(False)

    means_dir = f"{args.stats_dir}/means@{args.model_ckpt_idx}"
    vars_dir = f"{args.stats_dir}/vars@{args.model_ckpt_idx}"
    decs_dir = f"{args.stats_dir}/decs@{args.model_ckpt_idx}"

    makedirs(args.stats_dir, exist_ok=True)
    makedirs(means_dir, exist_ok=True)
    makedirs(vars_dir, exist_ok=True)
    makedirs(decs_dir, exist_ok=True)

    # model_name = model_args.model_name_or_path.split("/")[-1]
    # short_name = model_name.replace("TinyStories-", "TS")
    model_name = 'mabel-bert-base-uncased'
    short_name = 'mabel-bert-base-uncased'
    means_path = f"{means_dir}/{short_name}@{args.model_ckpt_idx}-means.pt"
    vars_path = f"{vars_dir}/{short_name}@{args.model_ckpt_idx}-vars.pt"
    decs_path = f"{decs_dir}/{short_name}@{args.model_ckpt_idx}-decs.pt"

    import pdb; pdb.set_trace()
    # data = datasets["validation" if args.stage == "decs" else "train"]
    data = datasets
    N_seqs = len(data)
    # extract input_ids
    # extract = lambda i: pt.tensor(data[i]["input_ids"], dtype=pt.int32)
    extract = lambda i: pt.tensor(data[i][2], dtype=pt.int32)
    N_batches = int(math.ceil(N_seqs / args.batch_size))

    import pdb; pdb.set_trace()
    if model_name == 'mabel-bert-base-uncased' or model_name == 'bert-base-uncased':
        C, D, model, W = strip_model_bert(model, args.device)
    else:
        C, D, model, W = strip_model(model, args.device)
    stats = Statistics(C, D, args.device)

    N_seen = 0
    if args.stage == "means":  # first pass
        N_seen = stats.load_totals(means_path)
    elif args.stage == "vars":  # second pass
        stats.load_totals(means_path)
        N_seen = stats.load_var_sums(vars_path)
    elif args.stage == "decs":
        stats.load_totals(means_path)
        N_seen = stats.load_decs(decs_path)

    N_batches_seen = int(math.ceil(N_seen / args.batch_size))
    if N_seen > 0:
        print(f"skipping {N_seen} sequences ({N_batches_seen} batches) already seen...")

    for b_idx in tqdm(range(N_batches_seen, N_batches), ncols=79):
        start = b_idx * args.batch_size
        end = min(start + args.batch_size, N_seqs)
        batch = [extract(i) for i in range(start, end)]
        import pdb; pdb.set_trace()
        X, Y = process_batch(model, batch, args.device)
        
        if args.stage == "means":  # first pass
            stats.collect_means(X, Y, len(batch))
        elif args.stage == "vars":  # second pass
            stats.collect_vars(X, Y, len(batch))
        elif args.stage == "decs":  # third pass
            stats.collect_decs(X, Y, W, len(batch))

        if (b_idx + 1) % (args.save_every // args.batch_size) != 0:
            continue  # don't save on most iterations

        if args.stage == "means":
            stats.save_totals(means_path, args.verbose)
        elif args.stage == "vars":
            stats.save_var_sums(vars_path, args.verbose)
        elif args.stage == "decs":
            stats.save_decs(decs_path, args.verbose)

        if args.single:
            break

    if args.stage == "means":
        stats.save_totals(means_path, args.verbose)
    elif args.stage == "vars":
        stats.save_var_sums(vars_path, args.verbose)
    elif args.stage == "decs":
        stats.save_decs(decs_path, args.verbose)
