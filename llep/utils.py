"""
Copyright: Salesforce AI Research

"""
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch
import numpy as np
import math
from typing import Callable, Optional, Union

import torch
import torch
import torch.distributed as dist

import os

from torch import nn
from torch.nn import functional as F


from transformers.utils import logging

logger = logging.get_logger(__name__)



DIST_ATTN_GROUP = None
EP_GROUP = None

MOE_WEIGHT_TRANSFER_GROUP = None

def set_moe_weight_transfer_group(group):
    global MOE_WEIGHT_TRANSFER_GROUP
    MOE_WEIGHT_TRANSFER_GROUP = group

def get_moe_weight_transfer_group():
    return MOE_WEIGHT_TRANSFER_GROUP

def set_ep_group(group):
    global EP_GROUP
    EP_GROUP = group

def get_ep_group():
    return EP_GROUP

def get_ep_rank():
    return dist.get_rank(group=EP_GROUP)


def set_dist_attn_group(group):
    global DIST_ATTN_GROUP
    DIST_ATTN_GROUP = group


def get_dist_attn_group():
    return DIST_ATTN_GROUP


def get_dist_attn_rank():
    return dist.get_rank(group=DIST_ATTN_GROUP)




class AllToAllAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, send, input_splits, output_splits, group):
        # Save for backward
        ctx.group = group
        # Save splits (as Python lists for ease)
        ctx.input_splits = list(input_splits) if input_splits is not None else None
        ctx.output_splits = list(output_splits) if output_splits is not None else None

        # Allocate recv (rows sum to sum(output_splits))
        recv_rows = sum(ctx.output_splits) if ctx.output_splits is not None else send.size(0)
        recv = send.new_empty(recv_rows, *send.shape[1:])

        # Forward A2A
        dist.all_to_all_single(
            output=recv,
            input=send.contiguous(),
            output_split_sizes=ctx.output_splits,
            input_split_sizes=ctx.input_splits,
            group=group,
        )
        return recv

    @staticmethod
    def backward(ctx, grad_recv):
        # Backward is the inverse A2A: swap split sizes
        grad_send_rows = sum(ctx.input_splits) if ctx.input_splits is not None else grad_recv.size(0)
        grad_send = grad_recv.new_empty(grad_send_rows, *grad_recv.shape[1:])

        dist.all_to_all_single(
            output=grad_send,
            input=grad_recv.contiguous(),
            # SWAP splits:
            output_split_sizes=ctx.input_splits,
            input_split_sizes=ctx.output_splits,
            group=ctx.group,
        )
        # Gradients for splits/group are None
        return grad_send, None, None, None


def a2a_autograd(send, input_splits, output_splits, group):
    return AllToAllAutograd.apply(send, input_splits, output_splits, group)


# Global variable for merged A2A inputs
# Set via environment variable: MERGE_INPUTS_FOR_A2A=1
# Concatenates hidden_states, routing_weights, router_indices into one tensor for single A2A
MERGE_INPUTS_FOR_A2A = None


def get_merge_inputs_for_a2a():
    """Get whether to merge inputs for A2A from environment variable."""
    global MERGE_INPUTS_FOR_A2A
    
    if MERGE_INPUTS_FOR_A2A is None:
        env_val = os.environ.get("MERGE_INPUTS_FOR_A2A", "0")
        MERGE_INPUTS_FOR_A2A = env_val == "1"
    
    return MERGE_INPUTS_FOR_A2A


# Global variable for adaptive LPT routing threshold
# Set via environment variable: MOE_ADAPTIVE_LPT_ROUTING_RATIO=1.3
# If set and valid float, enables adaptive path selection
# If not set or invalid, always uses LPT path
MOE_ADAPTIVE_LPT_ROUTING_RATIO = None


def get_adaptive_lpt_threshold():
    """Get adaptive LPT threshold from environment variable."""
    global MOE_ADAPTIVE_LPT_ROUTING_RATIO
    
    if MOE_ADAPTIVE_LPT_ROUTING_RATIO is None:
        env_val = os.environ.get("MOE_ADAPTIVE_LPT_ROUTING_RATIO", "")
        if env_val:
            try:
                MOE_ADAPTIVE_LPT_ROUTING_RATIO = float(env_val)
            except ValueError:
                MOE_ADAPTIVE_LPT_ROUTING_RATIO = -1.0  # Invalid, disable adaptive
        else:
            MOE_ADAPTIVE_LPT_ROUTING_RATIO = -1.0  # Not set, disable adaptive
    
    return MOE_ADAPTIVE_LPT_ROUTING_RATIO



def compute_gpu_imbalance_ratio(global_expert_counts, ep_size, num_local_experts):
    """
    Compute GPU load imbalance ratio under default expert assignment.
    
    Returns max_load / mean_load:
    - 1.0 = perfectly balanced
    - >1.0 = imbalanced (higher = more imbalanced)
    """
    # Reshape to (ep_size, num_local_experts) and sum to get per-GPU load
    # This assumes num_experts = ep_size * num_local_experts
    gpu_loads = global_expert_counts.view(ep_size, num_local_experts).sum(dim=1).float()
    
    mean_load = gpu_loads.mean()
    max_load = gpu_loads.max()
    
    if mean_load == 0:
        return 1.0
    
    return (max_load / mean_load).item()
