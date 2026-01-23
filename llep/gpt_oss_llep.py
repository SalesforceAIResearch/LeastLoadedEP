"""
Copyright: Salesforce AI Research
"""

import os
import torch
import torch.distributed as dist
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from transformers.utils import logging

from .utils import (
    a2a_autograd,
    get_merge_inputs_for_a2a,
    get_adaptive_lpt_threshold,
    compute_gpu_imbalance_ratio,
    get_moe_weight_transfer_group,
)

logger = logging.get_logger(__name__)

# Environment variable to enable batched isend/irecv (more efficient, avoids NCCL warnings)
# Set BATCH_ISEND_IRECV=1 to enable, BATCH_ISEND_IRECV=0 (default) to use individual isend/irecv
BATCH_ISEND_IRECV = os.environ.get("BATCH_ISEND_IRECV", "1") == "1"

# Environment variable to enable autograd for weight transfer (supports backward pass)
# Set LLEP_W_TRANSFER_AUTOGRAD=1 to enable differentiable weight transfer
# Set LLEP_W_TRANSFER_AUTOGRAD=0 (default) to use original non-differentiable transfer
LLEP_W_TRANSFER_AUTOGRAD = os.environ.get("LLEP_W_TRANSFER_AUTOGRAD", "1") == "1"

# Barrier before weight transfer to prevent cross-layer P2P deadlock during checkpointing
# Set SYNC_BEFORE_WEIGHT_TRANSFER=1 to enable (adds latency but prevents deadlock)
SYNC_BEFORE_WEIGHT_TRANSFER = os.environ.get("SYNC_BEFORE_WEIGHT_TRANSFER", "1") == "1"

# Barrier before A2A in forward pass to prevent cross-layer A2A deadlock during checkpointing
# When gradient checkpointing is enabled, the forward pass is recomputed during backward.
# With LPT, different ranks have different autograd graph structures, causing them to
# recompute different layers' A2A at the same time, leading to deadlock.
# Set SYNC_A2A_FWD=1 to add barriers before A2A operations in forward pass.
# NOTE: Don't use it
SYNC_A2A_FWD = os.environ.get("SYNC_A2A_FWD", "0") == "1"

@dataclass
class WeightTransferPlan:
    """Describes a weight transfer between two GPUs."""
    expert_id: int           # Global expert ID
    src_rank: int            # Rank that owns the weight (native)
    dst_rank: int            # Rank that will receive the weight (helper)
    token_start: int         # Start token index (global) for dst_rank to process
    token_end: int           # End token index (global) for dst_rank to process


@dataclass 
class LLEPLptPlan:
    """Complete LPT plan with weight transfer info for LL EP."""
    # lpt_plan: expert_id -> [(gpu_id, token_start, token_end), ...]
    lpt_plan: Dict[int, List[Tuple[int, int, int]]]
    
    # Weight transfers needed (globally consistent across all ranks)
    weight_transfers: List[WeightTransferPlan]
    
    # gpu_loads[gpu_id] = total tokens assigned to that GPU
    gpu_loads: torch.Tensor
    
    # For this rank: which expert weights to send and to whom
    # List of (expert_id, dst_rank)
    weights_to_send: List[Tuple[int, int]]
    
    # For this rank: which expert weights to receive and from whom
    # List of (expert_id, src_rank)
    weights_to_receive: List[Tuple[int, int]]


def compute_llep_lpt_plan(
    global_expert_counts: torch.Tensor,  # (num_experts,) global token counts
    ep_size: int,
    ep_rank: int,
    num_local_experts: int,
    max_tokens_factor: float = 1.1,
    min_tokens_per_gemm: int = 1024,
) -> LLEPLptPlan:
    """
    Compute LPT plan for LLEP with weight spilling.
    
    Key insight: When selecting a helper GPU, we must account for the
    "effective load" which includes both:
    - assigned_load: tokens already assigned (from earlier LPT iterations)
    - pending_native_load: native experts not yet processed in LPT order
    
    This prevents spilling to a GPU that will be heavily loaded by its own natives.
    
    Algorithm:
    1. Pre-compute native load per GPU (total tokens for native experts)
    2. Sort experts by token count (largest first) - LPT ordering
    3. For each expert:
       a. Remove from pending native load
       b. Compute effective load = assigned + pending_native
       c. Assign to native GPU if capacity allows
       d. If overflow, spill to least-loaded GPU (by effective load) + weight transfer
    4. Build weight transfer plan (globally consistent)
    
    Args:
        global_expert_counts: (num_experts,) tensor with global token counts per expert
        ep_size: number of EP ranks
        ep_rank: current rank (for extracting send/receive lists)
        num_local_experts: number of experts native to each GPU
        max_tokens_factor: max_tokens_per_gpu = factor * (total_tokens / ep_size)
        min_tokens_per_gemm: minimum tokens per GEMM to avoid overhead
    
    Returns:
        LLEPLptPlan with routing plan and weight transfer info
    """
    num_experts = global_expert_counts.size(0)
    device = global_expert_counts.device
    
    total_tokens = global_expert_counts.sum().item()
    balanced_tokens = total_tokens // ep_size if ep_size > 0 else total_tokens
    max_tokens_per_gpu = int(max_tokens_factor * balanced_tokens) if balanced_tokens > 0 else total_tokens
    
    max_tokens_per_gpu = max(max_tokens_per_gpu, 1)
    
    native_load_per_gpu = [0] * ep_size
    for expert_id in range(num_experts):
        native_gpu = expert_id // num_local_experts
        native_load_per_gpu[native_gpu] += global_expert_counts[expert_id].item()
    
    pending_native_load = list(native_load_per_gpu)
    
    assigned_load = [0] * ep_size
    
    expert_counts_list = [(e, int(global_expert_counts[e].item())) for e in range(num_experts)]
    expert_counts_sorted = sorted(expert_counts_list, key=lambda x: -x[1])
    
    lpt_plan: Dict[int, List[Tuple[int, int, int]]] = {}
    
    # Track weight transfers
    weight_transfers: List[WeightTransferPlan] = []
    
    for expert_id, expert_tokens in expert_counts_sorted:
        if expert_tokens == 0:
            continue
        
        native_gpu = expert_id // num_local_experts
        
        # === This expert is now being processed - remove from pending ===
        pending_native_load[native_gpu] -= expert_tokens
        
        # === Helper to compute effective load ===
        def get_effective_load(gpu_id):
            return assigned_load[gpu_id] + pending_native_load[gpu_id]
        
        # Capacity for native GPU considering effective load
        native_current_effective = get_effective_load(native_gpu)
        native_available = max_tokens_per_gpu - native_current_effective
        
        assignments = []
        
        if native_available >= expert_tokens:
            # Case 1: Native GPU can handle all tokens
            assignments.append((native_gpu, 0, expert_tokens))
            assigned_load[native_gpu] += expert_tokens
            
        elif native_available > 0:
            # Case 2: Native GPU takes what it can, spill rest to helper(s)
            native_chunk = min(native_available, expert_tokens)
            assignments.append((native_gpu, 0, native_chunk))
            assigned_load[native_gpu] += native_chunk
            
            remaining = expert_tokens - native_chunk
            token_offset = native_chunk
            
            while remaining > 0:
                # Find least-loaded GPU by EFFECTIVE load (excluding native if already assigned)
                other_gpus = []
                for g in range(ep_size):
                    if g == native_gpu:
                        continue
                    eff_load = get_effective_load(g)
                    available = max_tokens_per_gpu - eff_load
                    other_gpus.append((g, eff_load, available))
                
                # Sort by effective load (ascending = least loaded first)
                other_gpus_sorted = sorted(other_gpus, key=lambda x: x[1])
                
                if not other_gpus_sorted:
                    # No other GPUs available, force remaining to native (over capacity)
                    old_end = assignments[0][2]
                    assignments[0] = (native_gpu, 0, old_end + remaining)
                    assigned_load[native_gpu] += remaining
                    break
                
                # Try to find a helper with capacity
                assigned_this_round = False
                for helper_gpu, helper_eff_load, helper_available in other_gpus_sorted:
                    if helper_available <= 0:
                        continue
                    
                    chunk = min(remaining, helper_available)
                    
                    # Skip if chunk too small (unless it's all that's left)
                    if chunk < min_tokens_per_gemm and remaining > chunk:
                        continue
                    
                    assignments.append((helper_gpu, token_offset, token_offset + chunk))
                    assigned_load[helper_gpu] += chunk
                    
                    weight_transfers.append(WeightTransferPlan(
                        expert_id=expert_id,
                        src_rank=native_gpu,
                        dst_rank=helper_gpu,
                        token_start=token_offset,
                        token_end=token_offset + chunk,
                    ))
                    
                    token_offset += chunk
                    remaining -= chunk
                    assigned_this_round = True
                    break  # Re-evaluate for next chunk if any remains
                
                if not assigned_this_round:
                    # All helpers at capacity, force to least loaded helper
                    helper_gpu = other_gpus_sorted[0][0]
                    assignments.append((helper_gpu, token_offset, token_offset + remaining))
                    assigned_load[helper_gpu] += remaining
                    
                    weight_transfers.append(WeightTransferPlan(
                        expert_id=expert_id,
                        src_rank=native_gpu,
                        dst_rank=helper_gpu,
                        token_start=token_offset,
                        token_end=token_offset + remaining,
                    ))
                    remaining = 0
        
        else:
            # Case 3: Native GPU is at/over capacity, must spill EVERYTHING
            other_gpus = []
            for g in range(ep_size):
                if g == native_gpu:
                    continue
                eff_load = get_effective_load(g)
                available = max_tokens_per_gpu - eff_load
                other_gpus.append((g, eff_load, available))
            
            other_gpus_sorted = sorted(other_gpus, key=lambda x: x[1])
            
            remaining = expert_tokens
            token_offset = 0
            
            for helper_gpu, helper_eff_load, helper_available in other_gpus_sorted:
                if remaining <= 0:
                    break
                
                if helper_available <= 0:
                    continue
                
                chunk = min(remaining, helper_available)
                if chunk < min_tokens_per_gemm and remaining > chunk:
                    continue
                
                assignments.append((helper_gpu, token_offset, token_offset + chunk))
                assigned_load[helper_gpu] += chunk
                
                weight_transfers.append(WeightTransferPlan(
                    expert_id=expert_id,
                    src_rank=native_gpu,
                    dst_rank=helper_gpu,
                    token_start=token_offset,
                    token_end=token_offset + chunk,
                ))
                
                token_offset += chunk
                remaining -= chunk
            
            # If still remaining (all GPUs at capacity), force to least loaded
            if remaining > 0:
                if other_gpus_sorted:
                    helper_gpu = other_gpus_sorted[0][0]
                    assignments.append((helper_gpu, token_offset, token_offset + remaining))
                    assigned_load[helper_gpu] += remaining
                    
                    weight_transfers.append(WeightTransferPlan(
                        expert_id=expert_id,
                        src_rank=native_gpu,
                        dst_rank=helper_gpu,
                        token_start=token_offset,
                        token_end=token_offset + remaining,
                    ))
                else:
                    # Edge case: only one GPU, assign everything to native
                    assignments.append((native_gpu, 0, expert_tokens))
                    assigned_load[native_gpu] += expert_tokens
        
        lpt_plan[expert_id] = assignments
    
    weights_to_send: List[Tuple[int, int]] = []
    weights_to_receive: List[Tuple[int, int]] = []
    
    for wt in weight_transfers:
        if wt.src_rank == ep_rank:
            weights_to_send.append((wt.expert_id, wt.dst_rank))
        if wt.dst_rank == ep_rank:
            weights_to_receive.append((wt.expert_id, wt.src_rank))
    
    gpu_loads_tensor = torch.tensor(assigned_load, dtype=torch.int64, device=device)
    
    return LLEPLptPlan(
        lpt_plan=lpt_plan,
        weight_transfers=weight_transfers,
        gpu_loads=gpu_loads_tensor,
        weights_to_send=weights_to_send,
        weights_to_receive=weights_to_receive,
    )


# =============================================================================
# Weight Transfer Communication
# =============================================================================

def transfer_expert_weights(
    ep_rank: int,
    ep_group,
    lpt_plan_result: LLEPLptPlan,
    # Local expert weights (only native experts)
    local_gate_up_proj: torch.Tensor,      # (num_local_experts, hidden_size, 2*intermediate)
    local_gate_up_proj_bias: torch.Tensor, # (num_local_experts, 2*intermediate)
    local_down_proj: torch.Tensor,         # (num_local_experts, intermediate, hidden_size)
    local_down_proj_bias: torch.Tensor,    # (num_local_experts, hidden_size)
    num_local_experts: int,
    return_handles: bool = False,
):
    """
    Execute weight transfers based on LPT plan using point-to-point communication.
    
    Uses async isend/irecv for potentially overlapping transfers with computation.
    When BATCH_ISEND_IRECV=1 environment variable is set, uses batched P2P operations
    for better NCCL efficiency and to avoid 2-rank communicator warnings.
    
    Args:
        ep_rank: current EP rank
        ep_group: EP process group
        lpt_plan_result: computed LPT plan with weight transfer info
        local_gate_up_proj: native gate_up weights (num_local_experts, H, 2*I)
        local_gate_up_proj_bias: native gate_up biases (num_local_experts, 2*I)
        local_down_proj: native down weights (num_local_experts, I, H)
        local_down_proj_bias: native down biases (num_local_experts, H)
        num_local_experts: number of native experts per GPU
        return_handles: if True, return handles without waiting (for async overlap)
    
    Returns:
        If return_handles=False:
            Four dicts mapping global_expert_id -> weight tensor for received (foreign) experts:
            (recv_gate_up_proj, recv_gate_up_proj_bias, recv_down_proj, recv_down_proj_bias)
        If return_handles=True:
            (recv_gate_up_proj, recv_gate_up_proj_bias, recv_down_proj, recv_down_proj_bias, handles)
            where handles is a list of async request handles to wait on
    """
    device = local_gate_up_proj.device
    dtype = local_gate_up_proj.dtype
    
    # Get weight transfer group (separate NCCL group for overlapping with inputs a2a)
    weight_transfer_group = get_moe_weight_transfer_group()
    if weight_transfer_group is None:
        logger.warning_once(
            "gpt_oss_llep.transfer_expert_weights MOE weight transfer group not set, falling back to ep_group. "
            "This may prevent overlapping weight transfer with inputs a2a."
        )
        weight_transfer_group = ep_group
    
    # Get weight shapes from local experts
    _, hidden_size, gate_up_out_dim = local_gate_up_proj.shape
    _, intermediate_dim, out_hidden_size = local_down_proj.shape
    
    weights_to_send = lpt_plan_result.weights_to_send
    weights_to_receive = lpt_plan_result.weights_to_receive
    
    # Prepare receive buffers
    recv_gate_up_proj: Dict[int, torch.Tensor] = {}
    recv_gate_up_proj_bias: Dict[int, torch.Tensor] = {}
    recv_down_proj: Dict[int, torch.Tensor] = {}
    recv_down_proj_bias: Dict[int, torch.Tensor] = {}
    
    for expert_id, src_rank in weights_to_receive:
        recv_gate_up_proj[expert_id] = torch.empty(
            (hidden_size, gate_up_out_dim), dtype=dtype, device=device
        )
        recv_gate_up_proj_bias[expert_id] = torch.empty(
            (gate_up_out_dim,), dtype=dtype, device=device
        )
        recv_down_proj[expert_id] = torch.empty(
            (intermediate_dim, out_hidden_size), dtype=dtype, device=device
        )
        recv_down_proj_bias[expert_id] = torch.empty(
            (out_hidden_size,), dtype=dtype, device=device
        )
    
    DEBUG_WEIGHT_TRANSFER = os.environ.get("DEBUG_WEIGHT_TRANSFER", "0") == "1"
    if DEBUG_WEIGHT_TRANSFER:
        import time as _time
        _ts = lambda: f"[{_time.strftime('%H:%M:%S')}.{int((_time.time() % 1) * 1000):03d}]"
        print(f"{_ts()} Rank {ep_rank}: [WT] to_send={weights_to_send} to_recv={weights_to_receive} BATCH={BATCH_ISEND_IRECV}", flush=True)
    
    if BATCH_ISEND_IRECV:
        # Use batched P2P operations for better NCCL efficiency
        # This avoids creating many 2-rank communicators and eliminates warnings
        p2p_ops = []
        
        # Add all recv operations first
        for expert_id, src_rank in weights_to_receive:
            p2p_ops.append(dist.P2POp(dist.irecv, recv_gate_up_proj[expert_id], src_rank, group=weight_transfer_group))
            p2p_ops.append(dist.P2POp(dist.irecv, recv_gate_up_proj_bias[expert_id], src_rank, group=weight_transfer_group))
            p2p_ops.append(dist.P2POp(dist.irecv, recv_down_proj[expert_id], src_rank, group=weight_transfer_group))
            p2p_ops.append(dist.P2POp(dist.irecv, recv_down_proj_bias[expert_id], src_rank, group=weight_transfer_group))
        
        # Add all send operations
        for expert_id, dst_rank in weights_to_send:
            local_expert_idx = expert_id % num_local_experts
            p2p_ops.append(dist.P2POp(dist.isend, local_gate_up_proj[local_expert_idx].contiguous(), dst_rank, group=weight_transfer_group))
            p2p_ops.append(dist.P2POp(dist.isend, local_gate_up_proj_bias[local_expert_idx].contiguous(), dst_rank, group=weight_transfer_group))
            p2p_ops.append(dist.P2POp(dist.isend, local_down_proj[local_expert_idx].contiguous(), dst_rank, group=weight_transfer_group))
            p2p_ops.append(dist.P2POp(dist.isend, local_down_proj_bias[local_expert_idx].contiguous(), dst_rank, group=weight_transfer_group))
        
        # Execute all P2P ops as a batch
        if DEBUG_WEIGHT_TRANSFER:
            print(f"{_ts()} Rank {ep_rank}: [WT] Posting {len(p2p_ops)} P2P ops...", flush=True)
        if p2p_ops:
            reqs = dist.batch_isend_irecv(p2p_ops)
            if DEBUG_WEIGHT_TRANSFER:
                print(f"{_ts()} Rank {ep_rank}: [WT] P2P ops posted, return_handles={return_handles}", flush=True)
            if return_handles:
                return recv_gate_up_proj, recv_gate_up_proj_bias, recv_down_proj, recv_down_proj_bias, reqs
            for req in reqs:
                req.wait()
            if DEBUG_WEIGHT_TRANSFER:
                print(f"{_ts()} Rank {ep_rank}: [WT] All P2P ops completed!", flush=True)
        elif return_handles:
            return recv_gate_up_proj, recv_gate_up_proj_bias, recv_down_proj, recv_down_proj_bias, []
    else:
        # Original unbatched implementation with individual isend/irecv calls
        # Issue all irecv first (post receives before sends for MPI efficiency)
        if DEBUG_WEIGHT_TRANSFER:
            print(f"{_ts()} Rank {ep_rank}: [WT] Unbatched path: posting {len(weights_to_receive)} recvs...", flush=True)
        recv_handles = []
        
        for expert_id, src_rank in weights_to_receive:
            # Use unique tags: expert_id * 4 + offset for each weight type
            h1 = dist.irecv(
                recv_gate_up_proj[expert_id], 
                src=src_rank, 
                group=weight_transfer_group,
                tag=expert_id * 4 + 0
            )
            recv_handles.append(h1)
            
            h2 = dist.irecv(
                recv_gate_up_proj_bias[expert_id],
                src=src_rank,
                group=weight_transfer_group,
                tag=expert_id * 4 + 1
            )
            recv_handles.append(h2)
            
            h3 = dist.irecv(
                recv_down_proj[expert_id],
                src=src_rank,
                group=weight_transfer_group,
                tag=expert_id * 4 + 2
            )
            recv_handles.append(h3)
            
            h4 = dist.irecv(
                recv_down_proj_bias[expert_id],
                src=src_rank,
                group=weight_transfer_group,
                tag=expert_id * 4 + 3
            )
            recv_handles.append(h4)
        
        # Issue all isend
        send_handles = []
        
        for expert_id, dst_rank in weights_to_send:
            # Local index of this expert (global_expert_id % num_local_experts)
            local_expert_idx = expert_id % num_local_experts
            
            h1 = dist.isend(
                local_gate_up_proj[local_expert_idx].contiguous(),
                dst=dst_rank,
                group=weight_transfer_group,
                tag=expert_id * 4 + 0
            )
            send_handles.append(h1)
            
            h2 = dist.isend(
                local_gate_up_proj_bias[local_expert_idx].contiguous(),
                dst=dst_rank,
                group=weight_transfer_group,
                tag=expert_id * 4 + 1
            )
            send_handles.append(h2)
            
            h3 = dist.isend(
                local_down_proj[local_expert_idx].contiguous(),
                dst=dst_rank,
                group=weight_transfer_group,
                tag=expert_id * 4 + 2
            )
            send_handles.append(h3)
            
            h4 = dist.isend(
                local_down_proj_bias[local_expert_idx].contiguous(),
                dst=dst_rank,
                group=weight_transfer_group,
                tag=expert_id * 4 + 3
            )
            send_handles.append(h4)
        
        # Wait for all transfers to complete (or return handles for async)
        if return_handles:
            return recv_gate_up_proj, recv_gate_up_proj_bias, recv_down_proj, recv_down_proj_bias, recv_handles + send_handles
        for h in recv_handles + send_handles:
            h.wait()
    
    return recv_gate_up_proj, recv_gate_up_proj_bias, recv_down_proj, recv_down_proj_bias


class WeightTransferAutograd(torch.autograd.Function):
    """
    Differentiable P2P weight transfer for LLEP.
    
    Forward: Sends local expert weights to helper GPUs, receives foreign expert weights.
    Backward: Sends foreign weight gradients back to owners, receives gradients for sent weights.
    
    Returns stacked tensors instead of dicts for autograd compatibility.
    Also returns a mapping tensor: foreign_expert_id_mapping[global_expert_id] = stacked_index
    """
    
    @staticmethod
    def forward(
        ctx,
        # Local weights (may be sent to other GPUs)
        local_gate_up_proj: torch.Tensor,      # (num_local_experts, H, 2*I)
        local_gate_up_proj_bias: torch.Tensor, # (num_local_experts, 2*I)
        local_down_proj: torch.Tensor,         # (num_local_experts, I, H)
        local_down_proj_bias: torch.Tensor,    # (num_local_experts, H)
        # Metadata tensors (for autograd to track)
        weights_to_send_tensor: torch.Tensor,    # (num_send, 2) - [[expert_id, dst_rank], ...]
        weights_to_receive_tensor: torch.Tensor, # (num_recv, 2) - [[expert_id, src_rank], ...]
        num_local_experts_tensor: torch.Tensor,  # scalar tensor
        num_experts_tensor: torch.Tensor,        # scalar tensor for mapping size
        # Non-tensor args
        ep_group,
        return_handles,
    ):
        """
        Forward pass: transfer weights via P2P.
        
        Returns:
            recv_gate_up_stacked: (num_recv, H, 2*I)
            recv_gate_up_bias_stacked: (num_recv, 2*I)
            recv_down_stacked: (num_recv, I, H)
            recv_down_bias_stacked: (num_recv, H)
            foreign_expert_id_mapping: (num_experts,) tensor where mapping[expert_id] = stacked_index or -1
        """
        device = local_gate_up_proj.device
        dtype = local_gate_up_proj.dtype
        
        # Get weight transfer group (separate NCCL group for overlapping with inputs a2a)
        weight_transfer_group = get_moe_weight_transfer_group()
        if weight_transfer_group is None:
            logger.warning_once(
                "gpt_oss_llep.WeightTransferAutograd MOE weight transfer group not set, falling back to ep_group. "
                "This may prevent overlapping weight transfer with inputs a2a."
            )
            weight_transfer_group = ep_group
        
        # Extract metadata
        weights_to_send = weights_to_send_tensor.tolist()  # List of [expert_id, dst_rank]
        weights_to_receive = weights_to_receive_tensor.tolist()  # List of [expert_id, src_rank]
        num_local_experts = int(num_local_experts_tensor.item())
        num_experts = int(num_experts_tensor.item())
        
        # Get weight shapes
        _, hidden_size, gate_up_out_dim = local_gate_up_proj.shape
        _, intermediate_dim, out_hidden_size = local_down_proj.shape
        
        num_recv = len(weights_to_receive)
        
        # Create mapping: global_expert_id -> stacked_index (or -1 if not received)
        foreign_expert_id_mapping = torch.full((num_experts,), -1, dtype=torch.long, device=device)
        recv_expert_ids = []
        for stacked_idx, (expert_id, src_rank) in enumerate(weights_to_receive):
            foreign_expert_id_mapping[expert_id] = stacked_idx
            recv_expert_ids.append(expert_id)
        
        # Allocate stacked receive buffers
        recv_gate_up_stacked = torch.empty(num_recv, hidden_size, gate_up_out_dim, dtype=dtype, device=device)
        recv_gate_up_bias_stacked = torch.empty(num_recv, gate_up_out_dim, dtype=dtype, device=device)
        recv_down_stacked = torch.empty(num_recv, intermediate_dim, out_hidden_size, dtype=dtype, device=device)
        recv_down_bias_stacked = torch.empty(num_recv, out_hidden_size, dtype=dtype, device=device)
        
        weight_transfer_handles = []
        
        # NOTE: Do NOT put a barrier here on weight_transfer_group!
        # The weight transfer group is separate from the EP group used for A2A.
        # If we barrier here, ranks doing A2A on EP group won't reach this barrier,
        # causing deadlock. The P2P operations should be self-synchronizing
        # (sender waits for receiver and vice versa).
        
        # Perform P2P transfer using batched or unbatched based on env
        if BATCH_ISEND_IRECV:
            p2p_ops = []
            
            # Recv operations
            for stacked_idx, (expert_id, src_rank) in enumerate(weights_to_receive):
                p2p_ops.append(dist.P2POp(dist.irecv, recv_gate_up_stacked[stacked_idx], src_rank, group=weight_transfer_group))
                p2p_ops.append(dist.P2POp(dist.irecv, recv_gate_up_bias_stacked[stacked_idx], src_rank, group=weight_transfer_group))
                p2p_ops.append(dist.P2POp(dist.irecv, recv_down_stacked[stacked_idx], src_rank, group=weight_transfer_group))
                p2p_ops.append(dist.P2POp(dist.irecv, recv_down_bias_stacked[stacked_idx], src_rank, group=weight_transfer_group))
            
            # Send operations
            for expert_id, dst_rank in weights_to_send:
                local_idx = expert_id % num_local_experts
                p2p_ops.append(dist.P2POp(dist.isend, local_gate_up_proj[local_idx].contiguous(), dst_rank, group=weight_transfer_group))
                p2p_ops.append(dist.P2POp(dist.isend, local_gate_up_proj_bias[local_idx].contiguous(), dst_rank, group=weight_transfer_group))
                p2p_ops.append(dist.P2POp(dist.isend, local_down_proj[local_idx].contiguous(), dst_rank, group=weight_transfer_group))
                p2p_ops.append(dist.P2POp(dist.isend, local_down_proj_bias[local_idx].contiguous(), dst_rank, group=weight_transfer_group))
            
            if p2p_ops:
                reqs = dist.batch_isend_irecv(p2p_ops)
                weight_transfer_handles = reqs
                if not return_handles:
                    for req in reqs:
                        req.wait()
        else:
            # Unbatched implementation with tags
            recv_handles = []
            for stacked_idx, (expert_id, src_rank) in enumerate(weights_to_receive):
                recv_handles.append(dist.irecv(recv_gate_up_stacked[stacked_idx], src=src_rank, group=weight_transfer_group, tag=expert_id * 4 + 0))
                recv_handles.append(dist.irecv(recv_gate_up_bias_stacked[stacked_idx], src=src_rank, group=weight_transfer_group, tag=expert_id * 4 + 1))
                recv_handles.append(dist.irecv(recv_down_stacked[stacked_idx], src=src_rank, group=weight_transfer_group, tag=expert_id * 4 + 2))
                recv_handles.append(dist.irecv(recv_down_bias_stacked[stacked_idx], src=src_rank, group=weight_transfer_group, tag=expert_id * 4 + 3))
            
            send_handles = []
            for expert_id, dst_rank in weights_to_send:
                local_idx = expert_id % num_local_experts
                send_handles.append(dist.isend(local_gate_up_proj[local_idx].contiguous(), dst=dst_rank, group=weight_transfer_group, tag=expert_id * 4 + 0))
                send_handles.append(dist.isend(local_gate_up_proj_bias[local_idx].contiguous(), dst=dst_rank, group=weight_transfer_group, tag=expert_id * 4 + 1))
                send_handles.append(dist.isend(local_down_proj[local_idx].contiguous(), dst=dst_rank, group=weight_transfer_group, tag=expert_id * 4 + 2))
                send_handles.append(dist.isend(local_down_proj_bias[local_idx].contiguous(), dst=dst_rank, group=weight_transfer_group, tag=expert_id * 4 + 3))
            
            weight_transfer_handles = recv_handles + send_handles
            if not return_handles:
                for h in weight_transfer_handles:
                    h.wait()
        
        # Save context for backward
        ctx.weights_to_send = weights_to_send
        ctx.weights_to_receive = weights_to_receive
        ctx.num_local_experts = num_local_experts
        ctx.weight_transfer_group = weight_transfer_group
        ctx.local_weight_shapes = {
            'gate_up': (hidden_size, gate_up_out_dim),
            'gate_up_bias': (gate_up_out_dim,),
            'down': (intermediate_dim, out_hidden_size),
            'down_bias': (out_hidden_size,),
        }
        ctx.dtype = dtype
        ctx.device = device
        ctx.recv_expert_ids = recv_expert_ids
        
        # CRITICAL: Create a gradient anchor to ensure backward is called even if
        gradient_anchor = local_gate_up_proj.sum() * 0.0
        
        if return_handles:
            return (recv_gate_up_stacked, recv_gate_up_bias_stacked,
                    recv_down_stacked, recv_down_bias_stacked,
                    foreign_expert_id_mapping, weight_transfer_handles, gradient_anchor)
        return (recv_gate_up_stacked, recv_gate_up_bias_stacked,
                recv_down_stacked, recv_down_bias_stacked,
                foreign_expert_id_mapping, gradient_anchor)
    
    @staticmethod
    def backward(ctx, grad_recv_gate_up, grad_recv_gate_up_bias,
                 grad_recv_down, grad_recv_down_bias, grad_mapping, 
                 handles_or_anchor=None, grad_anchor=None):
        """
        Backward pass: P2P gradient transfer (reverse of forward).
        
        With the gradient_anchor fix, ALL ranks enter this backward at the same layer
        simultaneously, so P2P pairing works correctly.
        
        Flow:
        1. Send gradients for foreign weights back to their owners (reverse of recv in forward)
        2. Receive gradients for weights we sent to others (reverse of send in forward)
        3. Return accumulated gradients for local weights
        """
        weights_to_send = ctx.weights_to_send  # What we SENT in forward (recv grad from there)
        weights_to_receive = ctx.weights_to_receive  # What we RECEIVED in forward (send grad to there)
        num_local_experts = ctx.num_local_experts
        weight_transfer_group = ctx.weight_transfer_group
        shapes = ctx.local_weight_shapes
        dtype = ctx.dtype
        device = ctx.device
        
        hidden_size, gate_up_out_dim = shapes['gate_up']
        intermediate_dim, out_hidden_size = shapes['down']
        
        # Initialize gradient accumulators for local weights
        grad_local_gate_up = torch.zeros(num_local_experts, hidden_size, gate_up_out_dim, dtype=dtype, device=device)
        grad_local_gate_up_bias = torch.zeros(num_local_experts, gate_up_out_dim, dtype=dtype, device=device)
        grad_local_down = torch.zeros(num_local_experts, intermediate_dim, out_hidden_size, dtype=dtype, device=device)
        grad_local_down_bias = torch.zeros(num_local_experts, out_hidden_size, dtype=dtype, device=device)
        
        # Prepare recv buffers for gradients we expect (for weights we SENT in forward)
        recv_grad_buffers = {}  # (expert_id, src_rank) -> (gate_up, gate_up_bias, down, down_bias)
        for expert_id, dst_rank in weights_to_send:
            # We sent weights to dst_rank, now recv gradients from dst_rank
            recv_grad_buffers[(expert_id, dst_rank)] = (
                torch.empty(hidden_size, gate_up_out_dim, dtype=dtype, device=device),
                torch.empty(gate_up_out_dim, dtype=dtype, device=device),
                torch.empty(intermediate_dim, out_hidden_size, dtype=dtype, device=device),
                torch.empty(out_hidden_size, dtype=dtype, device=device),
            )
        
        # Perform P2P using batched or unbatched based on env
        if BATCH_ISEND_IRECV:
            p2p_ops = []
            
            # Recv operations (for gradients of weights we sent)
            for (expert_id, src_rank), bufs in recv_grad_buffers.items():
                p2p_ops.append(dist.P2POp(dist.irecv, bufs[0], src_rank, group=weight_transfer_group))
                p2p_ops.append(dist.P2POp(dist.irecv, bufs[1], src_rank, group=weight_transfer_group))
                p2p_ops.append(dist.P2POp(dist.irecv, bufs[2], src_rank, group=weight_transfer_group))
                p2p_ops.append(dist.P2POp(dist.irecv, bufs[3], src_rank, group=weight_transfer_group))
            
            # Send operations (gradients for weights we received)
            for stacked_idx, (expert_id, src_rank) in enumerate(weights_to_receive):
                # We received weights from src_rank, now send gradients back to src_rank
                p2p_ops.append(dist.P2POp(dist.isend, grad_recv_gate_up[stacked_idx].contiguous(), src_rank, group=weight_transfer_group))
                p2p_ops.append(dist.P2POp(dist.isend, grad_recv_gate_up_bias[stacked_idx].contiguous(), src_rank, group=weight_transfer_group))
                p2p_ops.append(dist.P2POp(dist.isend, grad_recv_down[stacked_idx].contiguous(), src_rank, group=weight_transfer_group))
                p2p_ops.append(dist.P2POp(dist.isend, grad_recv_down_bias[stacked_idx].contiguous(), src_rank, group=weight_transfer_group))
            
            if p2p_ops:
                reqs = dist.batch_isend_irecv(p2p_ops)
                for req in reqs:
                    req.wait()
        else:
            # Unbatched implementation with tags
            all_handles = []
            
            # Recv operations
            for (expert_id, src_rank), bufs in recv_grad_buffers.items():
                all_handles.append(dist.irecv(bufs[0], src=src_rank, group=weight_transfer_group, tag=expert_id * 4 + 0))
                all_handles.append(dist.irecv(bufs[1], src=src_rank, group=weight_transfer_group, tag=expert_id * 4 + 1))
                all_handles.append(dist.irecv(bufs[2], src=src_rank, group=weight_transfer_group, tag=expert_id * 4 + 2))
                all_handles.append(dist.irecv(bufs[3], src=src_rank, group=weight_transfer_group, tag=expert_id * 4 + 3))
            
            # Send operations
            for stacked_idx, (expert_id, src_rank) in enumerate(weights_to_receive):
                all_handles.append(dist.isend(grad_recv_gate_up[stacked_idx].contiguous(), dst=src_rank, group=weight_transfer_group, tag=expert_id * 4 + 0))
                all_handles.append(dist.isend(grad_recv_gate_up_bias[stacked_idx].contiguous(), dst=src_rank, group=weight_transfer_group, tag=expert_id * 4 + 1))
                all_handles.append(dist.isend(grad_recv_down[stacked_idx].contiguous(), dst=src_rank, group=weight_transfer_group, tag=expert_id * 4 + 2))
                all_handles.append(dist.isend(grad_recv_down_bias[stacked_idx].contiguous(), dst=src_rank, group=weight_transfer_group, tag=expert_id * 4 + 3))
            
            for h in all_handles:
                h.wait()
        
        # Accumulate received gradients into local weight gradients
        for (expert_id, _), (grad_gu, grad_gub, grad_d, grad_db) in recv_grad_buffers.items():
            local_idx = expert_id % num_local_experts
            grad_local_gate_up[local_idx] += grad_gu
            grad_local_gate_up_bias[local_idx] += grad_gub
            grad_local_down[local_idx] += grad_d
            grad_local_down_bias[local_idx] += grad_db
        
        # Return gradients: 4 for local weights, None for metadata tensors and non-tensor args
        # Must match number of forward inputs:
        #   1-4: local_gate_up/bias, local_down/bias
        #   5-8: weights_to_send/receive_tensor, num_local/num_experts_tensor (None - metadata)
        #   9: ep_group (None - non-tensor)
        #   10: return_handles (None - non-tensor)
        return (grad_local_gate_up, grad_local_gate_up_bias,
                grad_local_down, grad_local_down_bias,
                None, None, None, None,  # metadata tensors
                None, None)              # ep_group, return_handles


def transfer_expert_weights_autograd(
    ep_rank: int,
    ep_group,
    lpt_plan_result: LLEPLptPlan,
    local_gate_up_proj: torch.Tensor,
    local_gate_up_proj_bias: torch.Tensor,
    local_down_proj: torch.Tensor,
    local_down_proj_bias: torch.Tensor,
    num_local_experts: int,
    num_experts: int,
    return_handles: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Differentiable weight transfer wrapper using WeightTransferAutograd.
    
    Returns stacked tensors instead of dicts, plus a mapping tensor and gradient_anchor.
    
    Args:
        ep_rank: current EP rank
        ep_group: EP process group
        lpt_plan_result: computed LPT plan with weight transfer info
        local_gate_up_proj: native gate_up weights (num_local_experts, H, 2*I)
        local_gate_up_proj_bias: native gate_up biases (num_local_experts, 2*I)
        local_down_proj: native down weights (num_local_experts, I, H)
        local_down_proj_bias: native down biases (num_local_experts, H)
        num_local_experts: number of native experts per GPU
        num_experts: total number of global experts
    
    Returns:
        foreign_gate_up_stacked: (num_recv, H, 2*I)
        foreign_gate_up_bias_stacked: (num_recv, 2*I)
        foreign_down_stacked: (num_recv, I, H)
        foreign_down_bias_stacked: (num_recv, H)
        foreign_expert_id_mapping: (num_experts,) tensor where mapping[expert_id] = stacked_index or -1
        [weight_transfer_handles]: optional, if return_handles=True
        gradient_anchor: scalar tensor that MUST be added to output to ensure backward for all ranks
    """
    device = local_gate_up_proj.device
    
    weights_to_send = lpt_plan_result.weights_to_send
    weights_to_receive = lpt_plan_result.weights_to_receive
    
    # Convert metadata to tensors
    if weights_to_send:
        weights_to_send_tensor = torch.tensor(weights_to_send, dtype=torch.long, device=device)
    else:
        weights_to_send_tensor = torch.empty(0, 2, dtype=torch.long, device=device)
    
    if weights_to_receive:
        weights_to_receive_tensor = torch.tensor(weights_to_receive, dtype=torch.long, device=device)
    else:
        weights_to_receive_tensor = torch.empty(0, 2, dtype=torch.long, device=device)
    
    num_local_experts_tensor = torch.tensor(num_local_experts, dtype=torch.long, device=device)
    num_experts_tensor = torch.tensor(num_experts, dtype=torch.long, device=device)
    
    return WeightTransferAutograd.apply(
        local_gate_up_proj,
        local_gate_up_proj_bias,
        local_down_proj,
        local_down_proj_bias,
        weights_to_send_tensor,
        weights_to_receive_tensor,
        num_local_experts_tensor,
        num_experts_tensor,
        ep_group,
        return_handles,
    )


# =============================================================================
# Token Routing (A2A)
# =============================================================================


def llep_lpt_plan_to_compute_ranks(
    compute_ranks: torch.Tensor,
    lpt_plan: Dict[int, List[Tuple[int, int, int]]],  # expert_id -> [(gpu_id, start, end), ...]
    ep_size: int,
    ep_rank: int,
    num_tokens: int,
    top_k: int,
    num_experts: int,
    num_local_experts: int,
    flat_indices: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    sort_by_expert_perm: torch.Tensor,
    local_expert_counts: torch.Tensor,  # (num_experts,) local counts
    all_expert_counts: List[torch.Tensor],  # list of (num_experts,) tensors, one per rank
    device: torch.device,
    **kwargs,
):
    """
    Optimized version using numpy for CPU operations.
    
    Key optimizations:
    1. Uses numpy arrays instead of Python lists for send_matrix (faster indexing)
    2. Uses numpy cumsum for global offsets (replaces nested Python loops)
    3. Single GPU-CPU sync for local_expert_offsets
    4. Only iterates over LPT experts for compute_rank assignment (not all experts)
    """
    import numpy as np
    
    # === CPU Phase: Compute send_matrix using numpy ===
    # Stack all expert counts into numpy array: (ep_size, num_experts)
    all_counts_np = np.stack([ec.cpu().numpy() for ec in all_expert_counts])
    
    # Compute cumulative counts for global offsets: (ep_size + 1, num_experts)
    # cum_counts_np[r, e] = total tokens for expert e from ranks 0..r-1
    cum_counts_np = np.zeros((ep_size + 1, num_experts), dtype=np.int64)
    cum_counts_np[1:] = np.cumsum(all_counts_np, axis=0)
    
    # Global offset for this rank per expert
    global_offsets_np = cum_counts_np[ep_rank]  # (num_experts,)
    
    # Initialize send_matrix as numpy array
    send_matrix_np = np.zeros((ep_size, ep_size), dtype=np.int64)
    
    # Identify experts with/without LPT plan
    lpt_expert_set = set(lpt_plan.keys())
    
    # Default routing for experts without LPT plan
    for expert_id in range(num_experts):
        if expert_id not in lpt_expert_set:
            default_owner = expert_id // num_local_experts
            send_matrix_np[:, default_owner] += all_counts_np[:, expert_id]
    
    # LPT routing for experts with plan
    for expert_id, assignments in lpt_plan.items():
        for src_rank in range(ep_size):
            src_start = cum_counts_np[src_rank, expert_id]
            src_end = cum_counts_np[src_rank + 1, expert_id]
            if src_start == src_end:
                continue
            
            for dst_gpu, dst_start, dst_end in assignments:
                overlap_start = max(src_start, dst_start)
                overlap_end = min(src_end, dst_end)
                if overlap_start < overlap_end:
                    send_matrix_np[src_rank, dst_gpu] += overlap_end - overlap_start
    
    # Convert send_matrix to tensor (must match non-LPT path which returns tensor)
    send_matrix = torch.from_numpy(send_matrix_np).to(device)
    
    # === GPU Phase: Compute compute_ranks ===
    # Compute local expert boundaries on GPU, then sync once to CPU
    local_expert_offsets = torch.zeros(num_experts + 1, dtype=torch.int64, device=device)
    local_expert_offsets[1:] = local_expert_counts.cumsum(0)
    local_expert_offsets_np = local_expert_offsets.cpu().numpy()  # Single sync
    
    # Batch collect assignment info (only for LPT experts)
    gpu_to_positions = {g: [] for g in range(ep_size)}
    multi_gpu_assignments = []
    
    for expert_id, assignments in lpt_plan.items():
        local_start = local_expert_offsets_np[expert_id]
        local_end = local_expert_offsets_np[expert_id + 1]
        local_count = local_end - local_start
        
        if local_count == 0:
            continue
        
        original_positions = sort_by_expert_perm[local_start:local_end]
        
        if len(assignments) == 1:
            gpu_id, _, _ = assignments[0]
            gpu_to_positions[gpu_id].append(original_positions)
        else:
            my_global_offset = global_offsets_np[expert_id]
            local_indices = torch.arange(local_count, device=device)
            global_positions = my_global_offset + local_indices
            multi_gpu_assignments.append((original_positions, global_positions, assignments))
    
    # Batch apply compute_rank assignments
    for gpu_id, positions_list in gpu_to_positions.items():
        if len(positions_list) == 0:
            continue
        elif len(positions_list) == 1:
            compute_ranks[positions_list[0]] = gpu_id
        else:
            all_positions = torch.cat(positions_list)
            compute_ranks[all_positions] = gpu_id
    
    # Multi-GPU assignments
    for original_positions, global_positions, assignments in multi_gpu_assignments:
        for gpu_id, start, end in assignments:
            mask = (global_positions >= start) & (global_positions < end)
            if mask.any():
                compute_ranks[original_positions[mask]] = gpu_id
    
    return compute_ranks, send_matrix


def assign_compute_rank_llep(
    router_indices: torch.Tensor,  # (num_tokens, top_k) global expert ids
    lpt_plan: Dict[int, List[Tuple[int, int, int]]],  # expert_id -> [(gpu_id, start, end), ...]
    ep_size: int,
    ep_rank: int,
    num_local_experts: int,
    local_expert_counts: torch.Tensor,  # (num_experts,) local counts
    all_expert_counts: List[torch.Tensor],  # list of (num_experts,) tensors, one per rank
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[List[int]]]:
    """
    Assign compute_rank for each token-copy based on LLEP LPT plan.
    
    Uses global sequential assignment: tokens are assigned based on their
    global position across all ranks.
    
    Returns:
        sorted_compute_ranks: (num_tokens * top_k,) sorted by compute_rank
        sorted_indices: permutation to get sorted order
        undo_sorted_indices: permutation to restore original order
        sorted_router_indices: (num_tokens * top_k,) expert ids sorted by compute_rank
        send_matrix: list of lists, send_matrix[src][dst] = tokens src sends to dst
    """
    device = router_indices.device
    num_tokens, top_k = router_indices.shape
    num_experts = local_expert_counts.size(0)
    
    # Flatten to (num_tokens * top_k,)
    flat_indices = router_indices.view(-1)
    total_tokens_flat = flat_indices.size(0)
    
    # Default: owner rank = expert_id // num_local_experts
    owner_ranks = flat_indices // num_local_experts
    compute_ranks = owner_ranks.clone()
    
    # # Initialize send_matrix on CPU for fast scalar operations
    # send_matrix = [[0] * ep_size for _ in range(ep_size)]
    
    # # Precompute expert counts on CPU
    # all_expert_counts_cpu = [ec.cpu() for ec in all_expert_counts]
    
    if not lpt_plan:
        # No LPT plan - use default routing
        all_counts_tensor = torch.stack(all_expert_counts)
        # Reshape to (ep_size, ep_size, num_local_experts) where dim1 is dst_gpu
        send_matrix = all_counts_tensor.view(ep_size, ep_size, num_local_experts).sum(dim=2)
        # send_matrix[src, dst] = tokens src sends to dst
        
        sorted_compute_ranks, sorted_indices = compute_ranks.sort(stable=False)
        undo_sorted_indices = sorted_indices.argsort()
        sorted_router_indices = flat_indices[sorted_indices]
        return sorted_compute_ranks, sorted_indices, undo_sorted_indices, sorted_router_indices, send_matrix
    
    # Sort local tokens by expert id
    sorted_expert_ids, sort_by_expert_perm = flat_indices.sort(stable=False)
    
    # Compute compute_ranks using the lpt_plan
    
    compute_ranks, send_matrix = llep_lpt_plan_to_compute_ranks(
        compute_ranks=compute_ranks,
        lpt_plan=lpt_plan,
        ep_size=ep_size,
        ep_rank=ep_rank,
        num_tokens=num_tokens,
        top_k=top_k,
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        local_expert_counts=local_expert_counts,
        flat_indices=flat_indices,
        sorted_expert_ids=sorted_expert_ids,
        sort_by_expert_perm=sort_by_expert_perm,
        all_expert_counts=all_expert_counts,
        device=device,    
    )
    
    # Sort by compute_rank for A2A
    sorted_compute_ranks, sorted_indices = compute_ranks.sort(stable=False)
    undo_sorted_indices = sorted_indices.argsort()
    sorted_router_indices = flat_indices[sorted_indices]
    
    # Convert send_matrix to tensor for consistent return type
    if not isinstance(send_matrix, torch.Tensor):
        send_matrix = torch.tensor(send_matrix, dtype=torch.int64, device=device)
    
    return sorted_compute_ranks, sorted_indices, undo_sorted_indices, sorted_router_indices, send_matrix


def llep_forward_a2a_inputs(
    ep_rank: int,
    ep_size: int,
    num_local_experts: int,
    hidden_states: torch.Tensor,  # (num_tokens, hidden_size)
    router_indices: torch.Tensor,  # (num_tokens, top_k)
    routing_weights: torch.Tensor,  # (num_tokens, num_experts)
    lpt_plan: Dict[int, List[Tuple[int, int, int]]],
    ep_group,
) -> Tuple:
    """
    Perform A2A routing of inputs based on LLEP LPT plan.
    
    Returns:
        Tuple containing:
        - num_tokens, top_k
        - input_split_sizes, output_split_sizes
        - a2a_sorted_hidden_states_topk
        - a2a_sorted_routing_weight_topk
        - a2a_sorted_router_indices (global expert ids)
        - a2a_sorted_router_indices_packed (indices into packed weights)
        - experts_needed (unique global expert ids this rank computes)
        - undo_sorted_indices
    """
    device = hidden_states.device
    dtype = hidden_states.dtype
    
    num_experts = routing_weights.shape[1]
    num_tokens = hidden_states.shape[0]
    top_k = router_indices.shape[1]
    router_indices = router_indices.to(torch.int32)
    
    # Gather expert counts from all ranks
    local_expert_counts = torch.bincount(
        router_indices.view(-1).to(torch.int64),
        minlength=num_experts
    ).to(torch.int64)
    
    all_expert_counts = [torch.zeros_like(local_expert_counts) for _ in range(ep_size)]
    dist.all_gather(all_expert_counts, local_expert_counts, group=ep_group)
    
    # Assign compute_rank per token based on LPT plan
    (
        sorted_compute_ranks,
        sorted_indices,
        undo_sorted_indices,
        sorted_router_indices,
        send_matrix,
    ) = assign_compute_rank_llep(
        router_indices,
        lpt_plan,
        ep_size,
        ep_rank,
        num_local_experts,
        local_expert_counts,
        all_expert_counts,
    )
    
    # Compute split sizes from send_matrix (tensor)
    input_split_sizes = send_matrix[ep_rank].to(torch.int64)
    output_split_sizes = send_matrix[:, ep_rank].tolist()
    
    # Prepare data for A2A
    routing_weight_topk = routing_weights.gather(1, router_indices.to(torch.int64))
    hidden_states_topk = hidden_states.unsqueeze(1).repeat(1, top_k, 1)
    
    routing_weight_topk = routing_weight_topk.view(-1)
    hidden_states_topk = hidden_states_topk.view(-1, hidden_states.shape[-1])
    
    sorted_routing_weight_topk = routing_weight_topk[sorted_indices]
    sorted_hidden_states_topk = hidden_states_topk[sorted_indices]
    
    # Validate split sizes before A2A
    total_tokens_to_send = sorted_hidden_states_topk.shape[0]
    input_split_sum = input_split_sizes.sum().item() if isinstance(input_split_sizes, torch.Tensor) else sum(input_split_sizes)
    if input_split_sum != total_tokens_to_send:
        raise ValueError(
            f"A2A input_split_sizes mismatch! sum(input_split_sizes)={input_split_sum}, "
            f"but tensor has {total_tokens_to_send} tokens. "
            f"input_split_sizes={input_split_sizes.tolist() if isinstance(input_split_sizes, torch.Tensor) else input_split_sizes}, "
            f"num_tokens={num_tokens}, top_k={top_k}, ep_rank={ep_rank}"
        )
    
    # Synchronize before A2A to prevent cross-layer deadlock during checkpointing recomputation
    # This is needed because LPT creates different autograd graph structures across ranks,
    # causing them to recompute different layers' A2A at the same time during backward.
    if SYNC_A2A_FWD:
        dist.barrier(group=ep_group)
    
    # A2A
    if get_merge_inputs_for_a2a():
        # Check dtype compatibility
        if sorted_hidden_states_topk.dtype != sorted_routing_weight_topk.dtype:
            raise ValueError(
                f"MERGE_INPUTS_FOR_A2A requires same dtype. "
                f"Got hidden: {sorted_hidden_states_topk.dtype}, routing: {sorted_routing_weight_topk.dtype}"
            )
        
        hidden_size = sorted_hidden_states_topk.shape[1]
        
        merged_input = torch.cat([
            sorted_hidden_states_topk,
            sorted_routing_weight_topk.unsqueeze(1),
            sorted_router_indices.to(dtype).unsqueeze(1),
        ], dim=1)
        
        merged_output = a2a_autograd(
            merged_input, input_split_sizes, output_split_sizes, ep_group
        )
        
        a2a_sorted_hidden_states_topk = merged_output[:, :hidden_size]
        a2a_sorted_routing_weight_topk = merged_output[:, hidden_size]
        a2a_sorted_router_indices = merged_output[:, hidden_size + 1].to(torch.int32)
    else:
        a2a_sorted_hidden_states_topk = a2a_autograd(
            sorted_hidden_states_topk, input_split_sizes, output_split_sizes, ep_group
        )
        a2a_sorted_routing_weight_topk = a2a_autograd(
            sorted_routing_weight_topk, input_split_sizes, output_split_sizes, ep_group
        )
        a2a_sorted_router_indices = a2a_autograd(
            sorted_router_indices, input_split_sizes, output_split_sizes, ep_group
        ).to(torch.int32)
    
    # Build packed expert set
    if a2a_sorted_router_indices.numel() > 0:
        experts_needed, inverse_indices = torch.unique(
            a2a_sorted_router_indices,
            sorted=True,
            return_inverse=True
        )
        a2a_sorted_router_indices_packed = inverse_indices
    else:
        experts_needed = torch.tensor([], dtype=torch.int32, device=device)
        a2a_sorted_router_indices_packed = torch.tensor([], dtype=torch.int32, device=device)
    
    return (
        num_tokens,
        top_k,
        input_split_sizes,
        output_split_sizes,
        a2a_sorted_hidden_states_topk,
        a2a_sorted_routing_weight_topk,
        a2a_sorted_router_indices,
        a2a_sorted_router_indices_packed,
        experts_needed,
        undo_sorted_indices,
    )


def llep_forward_a2a_outputs(
    ep_rank: int,
    ep_size: int,
    input_split_sizes,
    output_split_sizes,
    num_tokens: int,
    top_k: int,
    a2a_proj_out: torch.Tensor,
    undo_sorted_indices: torch.Tensor,
    ep_group,
) -> torch.Tensor:
    """
    Undo A2A routing - route output tensors back to origin ranks.
    """
    # Synchronize before A2A to prevent cross-layer deadlock during checkpointing recomputation
    if SYNC_A2A_FWD:
        dist.barrier(group=ep_group)
    
    # A2A outputs back (swap split sizes)
    a2a_weighted_proj_out = a2a_autograd(
        a2a_proj_out, output_split_sizes, input_split_sizes, ep_group
    )
    
    # Unsort to original order
    unsorted_weighted_proj_out = a2a_weighted_proj_out[undo_sorted_indices]
    
    # Reshape and sum over top_k
    unsorted_weighted_proj_out = unsorted_weighted_proj_out.view(num_tokens, top_k, -1)
    unsorted_weighted_proj_out = unsorted_weighted_proj_out.sum(dim=1)
    
    return unsorted_weighted_proj_out


def llep_ffn_forward(
    hidden_states: torch.Tensor,           # (num_tokens, hidden_size)
    router_indices_packed: torch.Tensor,   # (num_tokens,) packed indices (0, 1, 2, ...)
    experts_needed: torch.Tensor,          # (num_unique_experts,) global expert IDs
    ep_rank: int,
    num_local_experts: int,
    # Native weights
    local_gate_up_proj: torch.Tensor,      # (num_local_experts, hidden_size, 2*intermediate)
    local_gate_up_proj_bias: torch.Tensor, # (num_local_experts, 2*intermediate)
    local_down_proj: torch.Tensor,         # (num_local_experts, intermediate, hidden_size)
    local_down_proj_bias: torch.Tensor,    # (num_local_experts, hidden_size)
    # Foreign weights - supports both Dict interface and stacked tensor interface
    # Dict interface (original):
    foreign_gate_up_proj: Optional[Dict[int, torch.Tensor]] = None,      # global_expert_id -> (hidden_size, 2*intermediate)
    foreign_gate_up_proj_bias: Optional[Dict[int, torch.Tensor]] = None, # global_expert_id -> (2*intermediate,)
    foreign_down_proj: Optional[Dict[int, torch.Tensor]] = None,         # global_expert_id -> (intermediate, hidden_size)
    foreign_down_proj_bias: Optional[Dict[int, torch.Tensor]] = None,    # global_expert_id -> (hidden_size,)
    # Stacked tensor interface (for autograd):
    foreign_gate_up_stacked: Optional[torch.Tensor] = None,      # (num_foreign, hidden_size, 2*intermediate)
    foreign_gate_up_bias_stacked: Optional[torch.Tensor] = None, # (num_foreign, 2*intermediate)
    foreign_down_stacked: Optional[torch.Tensor] = None,         # (num_foreign, intermediate, hidden_size)
    foreign_down_bias_stacked: Optional[torch.Tensor] = None,    # (num_foreign, hidden_size)
    foreign_expert_id_mapping: Optional[torch.Tensor] = None,    # (num_experts,) mapping[expert_id] = stacked_index or -1
    # Activation params
    limit: float = 7.0,
    alpha: float = 1.702,
) -> torch.Tensor:
    """
    FFN forward that directly accesses native or foreign weights without packing.
    
    This avoids extra memory allocation and copies by directly indexing into
    native weights (for native experts) or foreign weight dicts (for spilled experts).
    
    The key insight is that inputs for foreign experts are never mixed with inputs
    for native experts in the same GEMM - each expert's tokens are processed separately.
    
    Supports two interfaces for foreign weights:
    1. Dict interface (original): foreign_gate_up_proj[global_expert_id] -> tensor
    2. Stacked tensor interface (for autograd): foreign_gate_up_stacked[mapping[expert_id]] -> tensor
    
    Args:
        hidden_states: Input tokens (num_tokens, hidden_size)
        router_indices_packed: Packed expert indices (0, 1, 2, ...) pointing into experts_needed
        experts_needed: Global expert IDs that this rank will compute, in sorted order
        ep_rank: Current EP rank
        num_local_experts: Number of native experts per GPU
        local_gate_up_proj: Native gate_up weights
        local_gate_up_proj_bias: Native gate_up biases
        local_down_proj: Native down weights
        local_down_proj_bias: Native down biases
        foreign_gate_up_proj: Dict of received foreign gate_up weights (original interface)
        foreign_gate_up_proj_bias: Dict of received foreign gate_up biases (original interface)
        foreign_down_proj: Dict of received foreign down weights (original interface)
        foreign_down_proj_bias: Dict of received foreign down biases (original interface)
        foreign_gate_up_stacked: Stacked foreign gate_up weights (autograd interface)
        foreign_gate_up_bias_stacked: Stacked foreign gate_up biases (autograd interface)
        foreign_down_stacked: Stacked foreign down weights (autograd interface)
        foreign_down_bias_stacked: Stacked foreign down biases (autograd interface)
        foreign_expert_id_mapping: Mapping from global_expert_id to stacked index
        limit: Activation clamp limit
        alpha: SiLU alpha parameter
    
    Returns:
        Output tensor (num_tokens, hidden_size)
    """
    # Determine which interface to use
    use_stacked_interface = foreign_expert_id_mapping is not None
    num_tokens, hidden_size = hidden_states.shape
    device = hidden_states.device
    dtype = hidden_states.dtype
    
    if num_tokens == 0:
        return torch.empty(0, hidden_size, device=device, dtype=dtype)
    
    num_packed_experts = experts_needed.numel()
    
    # Native expert range for this rank
    native_expert_start = ep_rank * num_local_experts
    native_expert_end = native_expert_start + num_local_experts
    
    # Convert experts_needed to list for fast lookup
    experts_needed_list = experts_needed.tolist()
    
    # Step 1: Sort tokens by packed expert index (groups tokens per expert)
    sorted_packed_indices, sort_perm = router_indices_packed.sort(stable=False)
    inverse_perm = sort_perm.argsort()
    
    # Sorted input (contiguous memory access per expert)
    x_sorted = hidden_states[sort_perm]
    
    # Step 2: Compute segment boundaries
    expert_counts = torch.bincount(sorted_packed_indices.to(torch.int64), minlength=num_packed_experts)
    expert_offsets = torch.zeros(num_packed_experts + 1, dtype=torch.int64, device=device)
    expert_offsets[1:] = expert_counts.cumsum(0)
    
    # Step 3: Collect active expert slices
    # Each slice is (packed_idx, global_expert_id, start, end)
    slices = []
    for packed_idx in range(num_packed_experts):
        start = expert_offsets[packed_idx].item()
        end = expert_offsets[packed_idx + 1].item()
        if start < end:
            global_expert_id = experts_needed_list[packed_idx]
            slices.append((packed_idx, global_expert_id, start, end))
    
    # Step 4: Process each expert's tokens
    # Directly access native or foreign weights based on global_expert_id
    out_sorted = torch.empty(num_tokens, hidden_size, device=device, dtype=dtype)
    
    for packed_idx, global_expert_id, start, end in slices:
        x_slice = x_sorted[start:end]
        
        # Determine if this is a native or foreign expert
        if native_expert_start <= global_expert_id < native_expert_end:
            # Native expert - index into local weights
            local_idx = global_expert_id - native_expert_start
            gate_up_w = local_gate_up_proj[local_idx]
            gate_up_b = local_gate_up_proj_bias[local_idx]
            down_w = local_down_proj[local_idx]
            down_b = local_down_proj_bias[local_idx]
        else:
            # Foreign expert - get from dict or stacked tensor
            if use_stacked_interface:
                # Use stacked tensor interface (autograd-compatible)
                stacked_idx = foreign_expert_id_mapping[global_expert_id].item()
                gate_up_w = foreign_gate_up_stacked[stacked_idx]
                gate_up_b = foreign_gate_up_bias_stacked[stacked_idx]
                down_w = foreign_down_stacked[stacked_idx]
                down_b = foreign_down_bias_stacked[stacked_idx]
            else:
                # Use dict interface (original)
                gate_up_w = foreign_gate_up_proj[global_expert_id]
                gate_up_b = foreign_gate_up_proj_bias[global_expert_id]
                down_w = foreign_down_proj[global_expert_id]
                down_b = foreign_down_proj_bias[global_expert_id]
        
        # gate_up projection: x @ W + b
        gate_up = torch.addmm(gate_up_b, x_slice, gate_up_w)
        
        # Split and apply activation (gpt-oss specific)
        gate, up = gate_up[..., ::2], gate_up[..., 1::2]
        gate = gate.clamp(min=None, max=limit)
        up = up.clamp(min=-limit, max=limit)
        glu = gate * torch.sigmoid(gate * alpha)
        down_input = (up + 1) * glu
        
        # down projection
        out_sorted[start:end] = torch.addmm(down_b, down_input, down_w)
    
    # Step 5: Restore original order
    out = out_sorted[inverse_perm]
    
    return out


def gptoss_llep_forward(
    hidden_states: torch.Tensor,          # (batch_size, seq_len, hidden_size)
    router_indices: torch.Tensor,         # (batch_size * seq_len, top_k) global expert ids
    routing_weights: torch.Tensor,        # (batch_size * seq_len, num_experts)
    local_gate_up_proj: torch.Tensor,     # (num_local_experts, hidden_size, 2*intermediate)
    local_gate_up_bias: torch.Tensor,     # (num_local_experts, 2*intermediate)
    local_down_proj: torch.Tensor,        # (num_local_experts, intermediate, hidden_size)
    local_down_bias: torch.Tensor,        # (num_local_experts, hidden_size)
    num_experts: int,                     # Total global experts
    num_local_experts: int,               # Experts per GPU
    ep_group,                             # EP process group
    limit: float = 18.0,                  # Clamp limit for activation
    alpha: float = 0.5,                   # SiLU alpha
    max_tokens_factor: float = 1.1,       # Capacity factor
    min_tokens_per_gemm: int = 1024,      # Min tokens per GEMM
) -> torch.Tensor:
    """
    LLEP MoE forward with LPT load balancing and weight spilling.
    
    This is the main entrypoint for expert parallelism where each GPU
    only holds its native expert weights. When a GPU is overloaded, both tokens
    AND weights are transferred to helper GPUs via point-to-point communication.
    
    With adaptive path selection (if MOE_ADAPTIVE_LPT_ROUTING_RATIO is set):
    - If GPU imbalance ratio < threshold: use fast balanced path (no LPT overhead)
    - If GPU imbalance ratio >= threshold: use LPT path for load balancing
    
    Forward-only implementation (no backward pass support for now).
    
    Args:
        hidden_states: Input tensor (batch_size, seq_len, hidden_size)
        router_indices: Router decisions (batch_size * seq_len, top_k) as global expert ids
        routing_weights: Router weights (batch_size * seq_len, num_experts)
        local_gate_up_proj: Native gate_up weights (num_local_experts, H, 2*I)
        local_gate_up_bias: Native gate_up biases (num_local_experts, 2*I)
        local_down_proj: Native down weights (num_local_experts, I, H)
        local_down_bias: Native down biases (num_local_experts, H)
        num_experts: Total number of experts globally
        num_local_experts: Number of experts native to each GPU
        ep_group: Expert parallelism process group
        limit: Activation clamp limit (for gpt-oss specific activation)
        alpha: SiLU alpha parameter
        max_tokens_factor: max_tokens_per_gpu = factor * balanced_tokens
        min_tokens_per_gemm: Minimum tokens per GEMM operation
    
    Environment Variables:
        MOE_ADAPTIVE_LPT_ROUTING_RATIO: If set to a float (e.g., "1.3"), enables
            adaptive path selection. When GPU load imbalance ratio (max/mean) is
            below this threshold, the fast balanced path is used instead of LPT.
            This improves performance when load is already balanced.
    
    Returns:
        Output tensor (batch_size, seq_len, hidden_size)
    """
    
    # Override from env if set
    try:
        max_tokens_factor = float(os.environ.get("EP_MAX_TOKENS_FACTOR", str(max_tokens_factor)))
    except:
        pass
    try:
        min_tokens_per_gemm = int(os.environ.get("EP_MIN_TOKENS_PER_GEMM", str(min_tokens_per_gemm)))
    except:
        pass
        
    assert ep_group is not None, "EP group required for EP"
    
    ep_rank = dist.get_rank(group=ep_group)
    ep_size = dist.get_world_size(group=ep_group)
    
    logger.warning_once(
        f"Using gpt_oss_llep.gptoss_llep_forward "
        f"{ep_rank=} {ep_size=} {max_tokens_factor=} {min_tokens_per_gemm=} "
    )
    
    device = hidden_states.device
    dtype = hidden_states.dtype
    
    batch_size = hidden_states.shape[0]
    hidden_size = hidden_states.shape[-1]
    hidden_states = hidden_states.reshape(-1, hidden_size)  # (num_tokens, hidden_size)
    
    local_expert_counts = torch.bincount(
        router_indices.view(-1).to(torch.int64),
        minlength=num_experts
    ).to(torch.int64)
    
    global_expert_counts = local_expert_counts.clone()
    dist.all_reduce(global_expert_counts, op=dist.ReduceOp.SUM, group=ep_group)
    
    adaptive_threshold = get_adaptive_lpt_threshold()
    use_lpt = True
    
    if adaptive_threshold > 0:
        imbalance_ratio = compute_gpu_imbalance_ratio(
            global_expert_counts, ep_size, num_local_experts
        )
        use_lpt = imbalance_ratio >= adaptive_threshold
        logger.warning_once(
            f"LLEP adaptive threshold: {adaptive_threshold:.2f}, "
            f"use_lpt={use_lpt}"
        )
    
    if use_lpt:
        # Full LPT path with weight spilling
        lpt_plan_result = compute_llep_lpt_plan(
            global_expert_counts,
            ep_size,
            ep_rank,
            num_local_experts,
            max_tokens_factor=max_tokens_factor,
            min_tokens_per_gemm=min_tokens_per_gemm,
        )
        
        # Log info about weight transfers
        if lpt_plan_result.weight_transfers:
            num_transfers = len(lpt_plan_result.weight_transfers)
            num_unique_experts = len(set(wt.expert_id for wt in lpt_plan_result.weight_transfers))
    else:
        # Balanced path - skip LPT, use default routing (no weight transfers)
        lpt_plan_result = LLEPLptPlan(
            lpt_plan={},  # Empty plan triggers default routing in assign_compute_rank_llep
            weight_transfers=[],
            gpu_loads=torch.zeros(ep_size, dtype=torch.int64, device=device),
            weights_to_send=[],
            weights_to_receive=[],
        )
        
    # CRITICAL: Barrier before weight transfer to prevent cross-layer P2P deadlock.
    if SYNC_BEFORE_WEIGHT_TRANSFER:
        # ! this is required for backward ?
        weight_transfer_group_for_sync = get_moe_weight_transfer_group()
        if weight_transfer_group_for_sync is None:
            weight_transfer_group_for_sync = ep_group
        dist.barrier(group=weight_transfer_group_for_sync)
    
    weight_transfer_handles = []
    gradient_anchor = None  # Will be set if using collective autograd

    if LLEP_W_TRANSFER_AUTOGRAD:
        # Use P2P autograd with gradient anchor (ensures all ranks enter backward)
        (
            foreign_gate_up_stacked,
            foreign_gate_up_bias_stacked,
            foreign_down_stacked,
            foreign_down_bias_stacked,
            foreign_expert_id_mapping,
            weight_transfer_handles,
            gradient_anchor,  # CRITICAL: Must be added to output to ensure backward for all ranks
        ) = transfer_expert_weights_autograd(
            ep_rank,
            ep_group,
            lpt_plan_result,
            local_gate_up_proj,
            local_gate_up_bias,
            local_down_proj,
            local_down_bias,
            num_local_experts,
            num_experts,
            return_handles=True,
        )
        foreign_gate_up_proj = None
        foreign_gate_up_bias = None
        foreign_down_proj = None
        foreign_down_bias = None
    else:
        # Use original P2P weight transfer (dict interface)
        # Start weight transfer asynchronously to overlap with A2A input routing
        (
            foreign_gate_up_proj,
            foreign_gate_up_bias,
            foreign_down_proj,
            foreign_down_bias,
            weight_transfer_handles,
        ) = transfer_expert_weights(
            ep_rank,
            ep_group,
            lpt_plan_result,
            local_gate_up_proj,
            local_gate_up_bias,
            local_down_proj,
            local_down_bias,
            num_local_experts,
            return_handles=True,  # Async mode for overlapping with A2A
        )
        foreign_gate_up_stacked = None
        foreign_gate_up_bias_stacked = None
        foreign_down_stacked = None
        foreign_down_bias_stacked = None
        foreign_expert_id_mapping = None
    
    (
        num_tokens,
        top_k,
        input_split_sizes,
        output_split_sizes,
        a2a_sorted_hidden_states_topk,
        a2a_sorted_routing_weight_topk,
        a2a_sorted_router_indices,
        a2a_sorted_router_indices_packed,
        experts_needed,
        undo_sorted_indices,
    ) = llep_forward_a2a_inputs(
        ep_rank,
        ep_size,
        num_local_experts,
        hidden_states,
        router_indices,
        routing_weights,
        lpt_plan_result.lpt_plan,
        ep_group,
    )
    

    # Wait for weight transfer to complete (only for non-autograd path)
    for h in weight_transfer_handles:
        h.wait()
    
    if a2a_sorted_hidden_states_topk.numel() > 0:
        a2a_proj_out = llep_ffn_forward(
            a2a_sorted_hidden_states_topk,
            a2a_sorted_router_indices_packed,
            experts_needed,
            ep_rank,
            num_local_experts,
            # Native weights
            local_gate_up_proj,
            local_gate_up_bias,
            local_down_proj,
            local_down_bias,
            # Foreign weights - dict interface (original)
            foreign_gate_up_proj=foreign_gate_up_proj,
            foreign_gate_up_proj_bias=foreign_gate_up_bias,
            foreign_down_proj=foreign_down_proj,
            foreign_down_proj_bias=foreign_down_bias,
            # Foreign weights - stacked tensor interface (autograd)
            foreign_gate_up_stacked=foreign_gate_up_stacked,
            foreign_gate_up_bias_stacked=foreign_gate_up_bias_stacked,
            foreign_down_stacked=foreign_down_stacked,
            foreign_down_bias_stacked=foreign_down_bias_stacked,
            foreign_expert_id_mapping=foreign_expert_id_mapping,
            # Activation params
            limit=limit,
            alpha=alpha,
        )
    else:
        # Empty tensor case - still touch weights for graph recording
        # Touch native weights
        weight_touch = (local_gate_up_proj.sum() * 0) + \
            (local_gate_up_bias.sum() * 0) + \
            (local_down_proj.sum() * 0) + \
            (local_down_bias.sum() * 0)
        # Touch foreign weights if any
        if LLEP_W_TRANSFER_AUTOGRAD:
            # Stacked tensor interface
            if foreign_gate_up_stacked is not None and foreign_gate_up_stacked.numel() > 0:
                weight_touch = weight_touch + (foreign_gate_up_stacked.sum() * 0)
                weight_touch = weight_touch + (foreign_gate_up_bias_stacked.sum() * 0)
                weight_touch = weight_touch + (foreign_down_stacked.sum() * 0)
                weight_touch = weight_touch + (foreign_down_bias_stacked.sum() * 0)
        else:
            # Dict interface
            for expert_id in foreign_gate_up_proj:
                weight_touch = weight_touch + (foreign_gate_up_proj[expert_id].sum() * 0)
                weight_touch = weight_touch + (foreign_gate_up_bias[expert_id].sum() * 0)
                weight_touch = weight_touch + (foreign_down_proj[expert_id].sum() * 0)
                weight_touch = weight_touch + (foreign_down_bias[expert_id].sum() * 0)
        a2a_proj_out = a2a_sorted_hidden_states_topk + weight_touch
    
    # Apply routing weights
    weighted_a2a_proj_out = a2a_proj_out * a2a_sorted_routing_weight_topk.unsqueeze(-1)
    
    # A2A inverse
    ep_out = llep_forward_a2a_outputs(
        ep_rank,
        ep_size,
        input_split_sizes,
        output_split_sizes,
        num_tokens,
        top_k,
        weighted_a2a_proj_out,
        undo_sorted_indices,
        ep_group,
    )
    
    ep_out = ep_out.view(batch_size, -1, hidden_size)
    
    # Add gradient anchor for autograd    
    if gradient_anchor is not None:
        # gradient_anchor is scalar (), ep_out is (B, S, H) - need to reshape for broadcast
        ep_out = ep_out + gradient_anchor.view(1, 1, 1)  # Shape (1,1,1) broadcasts to (B,S,H)
    
    # delete stuff
    del foreign_gate_up_proj
    del foreign_gate_up_bias
    del foreign_down_proj
    del foreign_down_bias
    del foreign_gate_up_stacked
    del foreign_gate_up_bias_stacked
    del foreign_down_stacked
    del foreign_down_bias_stacked
    del foreign_expert_id_mapping
    
    return ep_out
