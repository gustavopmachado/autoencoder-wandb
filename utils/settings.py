"""Methods for setting the training process"""

# pylint: disable=consider-iterating-dictionary

import os

import torch
from torch.distributed import ReduceOp, all_reduce, init_process_group

__all__ = ["ddp"]


def ddp():
    """Setup GPU Data Distributed Parallelisation"""
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    rank = torch.tensor(int(os.environ["LOCAL_RANK"]),
                        requires_grad=False).to(int(os.environ["LOCAL_RANK"]))
    all_reduce(rank, op=ReduceOp.MIN)
    if "MAIN_RANK" not in os.environ.keys():
        os.environ["MAIN_RANK"] = str(rank.item())
