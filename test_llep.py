
import os
import argparse
import torch
import torch.distributed as dist
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict


@dataclass
class MockExpertsConfig:
    """Mock config for GptOssExperts-like module."""
    hidden_size: int = 512
    intermediate_size: int = 1024
    num_experts: int = 16
    top_k: int = 2
    limit: float = 7.0
    alpha: float = 1.702


class MockGptOssExperts(nn.Module):
    """Mock GptOssExperts module for testing."""
    def __init__(self, config: MockExpertsConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.limit = config.limit
        self.alpha = config.alpha
        
        self.gate_up_proj = nn.Parameter(
            torch.randn(config.num_experts, config.hidden_size, config.intermediate_size * 2) * 0.02
        )
        self.gate_up_proj_bias = nn.Parameter(
            torch.zeros(config.num_experts, config.intermediate_size * 2)
        )
        self.down_proj = nn.Parameter(
            torch.randn(config.num_experts, config.intermediate_size, config.hidden_size) * 0.02
        )
        self.down_proj_bias = nn.Parameter(
            torch.zeros(config.num_experts, config.hidden_size)
        )
    
    def get_local_weights(self, ep_rank: int, ep_size: int):
        """Slice weights to get local expert weights for this rank."""
        num_local_experts = self.num_experts // ep_size
        start_idx = ep_rank * num_local_experts
        end_idx = start_idx + num_local_experts
        
        return (
            self.gate_up_proj[start_idx:end_idx],
            self.gate_up_proj_bias[start_idx:end_idx],
            self.down_proj[start_idx:end_idx],
            self.down_proj_bias[start_idx:end_idx],
        )


def create_balanced_routing(
    num_tokens: int, 
    num_experts: int, 
    top_k: int,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create balanced routing where tokens are evenly distributed across experts."""
    router_indices = torch.zeros(num_tokens, top_k, dtype=torch.int64, device=device)
    for k in range(top_k):
        router_indices[:, k] = (torch.arange(num_tokens, device=device) + k) % num_experts
    
    routing_weights = torch.zeros(num_tokens, num_experts, device=device, dtype=dtype)
    for i in range(num_tokens):
        for k in range(top_k):
            expert_id = router_indices[i, k].item()
            routing_weights[i, expert_id] = 1.0 / top_k
    
    return router_indices, routing_weights


def create_imbalanced_routing(
    num_tokens: int, 
    num_experts: int, 
    top_k: int,
    device: torch.device,
    hot_expert_ratio: float = 0.7,
    num_hot_experts: int = 2,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create imbalanced routing where most tokens go to a few hot experts."""
    hot_experts = list(range(num_hot_experts))
    cold_experts = list(range(num_hot_experts, num_experts))
    
    router_indices = torch.zeros(num_tokens, top_k, dtype=torch.int64, device=device)
    routing_weights = torch.zeros(num_tokens, num_experts, device=device, dtype=dtype)
    
    num_hot_tokens = int(num_tokens * hot_expert_ratio)
    
    for i in range(num_tokens):
        if i < num_hot_tokens:
            for k in range(top_k):
                expert_id = hot_experts[k % len(hot_experts)]
                router_indices[i, k] = expert_id
                routing_weights[i, expert_id] = 1.0 / top_k
        else:
            for k in range(top_k):
                expert_id = cold_experts[(i + k) % len(cold_experts)]
                router_indices[i, k] = expert_id
                routing_weights[i, expert_id] = 1.0 / top_k
    
    return router_indices, routing_weights


def get_max_peak_memory_across_ranks(device) -> float:
    """Get max peak memory across all ranks in MB."""
    local_peak = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    peak_tensor = torch.tensor([local_peak], device=device)
    dist.all_reduce(peak_tensor, op=dist.ReduceOp.MAX)
    return peak_tensor.item()


def setup_distributed():
    """Initialize distributed environment."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    
    return local_rank, dist.get_world_size()


def setup_ep_group():
    """Setup EP group (all ranks in world)."""
    from llep.utils import set_ep_group, set_moe_weight_transfer_group
    
    ep_group = dist.new_group(ranks=list(range(dist.get_world_size())))
    set_ep_group(ep_group)
    
    if os.environ.get("MOE_WEIGHT_TRANSFER_SEPARATE_NCCL", "1") == "1":
        weight_transfer_group = dist.new_group(ranks=list(range(dist.get_world_size())))
        set_moe_weight_transfer_group(weight_transfer_group)
    
    return ep_group


def parse_imbalance_configs(config_str: str, num_local_experts: int) -> List[Tuple[str, Optional[float], Optional[int]]]:
    """
    Parse imbalance configs string.
    Format: "balanced,70:2,90:1,35:local"
    Returns list of (name, hot_ratio, num_hot_experts) tuples.
    For balanced, hot_ratio and num_hot_experts are None.
    """
    configs = []
    for item in config_str.split(","):
        item = item.strip()
        if item.lower() == "balanced":
            configs.append(("balanced", None, None))
        else:
            parts = item.split(":")
            if len(parts) != 2:
                raise ValueError(f"Invalid config: {item}. Expected 'ratio:num_hot' or 'balanced'")
            ratio = float(parts[0]) / 100.0
            if parts[1].lower() == "local":
                num_hot = num_local_experts
            else:
                num_hot = int(parts[1])
            name = f"imbal_{int(ratio*100)}pct_{num_hot}hot"
            configs.append((name, ratio, num_hot))
    return configs


def run_single_config(
    config_name: str,
    router_indices: torch.Tensor,
    routing_weights: torch.Tensor,
    hidden_states: torch.Tensor,
    local_gate_up_proj: torch.Tensor,
    local_gate_up_proj_bias: torch.Tensor,
    local_down_proj: torch.Tensor,
    local_down_proj_bias: torch.Tensor,
    num_experts: int,
    num_local_experts: int,
    ep_group,
    model,
    max_tokens_factor: float,
    min_tokens_per_gemm: int,
    num_warmup: int,
    num_runs: int,
    device,
) -> Dict:
    """Run timing for a single routing config."""
    from llep.gpt_oss_llep import gptoss_llep_forward
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = gptoss_llep_forward(
                hidden_states.clone(),
                router_indices.clone(),
                routing_weights.clone(),
                local_gate_up_proj,
                local_gate_up_proj_bias,
                local_down_proj,
                local_down_proj_bias,
                num_experts,
                num_local_experts,
                ep_group,
                limit=model.limit,
                alpha=model.alpha,
                max_tokens_factor=max_tokens_factor,
                min_tokens_per_gemm=min_tokens_per_gemm,
            )
    torch.cuda.synchronize()
    dist.barrier()
    
    # Reset memory stats before timed runs
    torch.cuda.reset_peak_memory_stats(device)
    
    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            dist.barrier()
            start_event.record()
            _ = gptoss_llep_forward(
                hidden_states.clone(),
                router_indices.clone(),
                routing_weights.clone(),
                local_gate_up_proj,
                local_gate_up_proj_bias,
                local_down_proj,
                local_down_proj_bias,
                num_experts,
                num_local_experts,
                ep_group,
                limit=model.limit,
                alpha=model.alpha,
                max_tokens_factor=max_tokens_factor,
                min_tokens_per_gemm=min_tokens_per_gemm,
            )
            end_event.record()
            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))
    
    peak_memory_mb = get_max_peak_memory_across_ranks(device)
    
    avg_time = sum(times) / len(times)
    std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
    
    return {
        "name": config_name,
        "avg_ms": avg_time,
        "std_ms": std_time,
        "min_ms": min(times),
        "max_ms": max(times),
        "peak_memory_mb": peak_memory_mb,
    }


def run_llep_timing_test(
    num_tokens: int = 1024,
    hidden_size: int = 512,
    intermediate_size: int = None,
    num_experts: int = 32,
    top_k: int = 4,
    num_warmup: int = 3,
    num_runs: int = 20,
    max_tokens_factor: float = 1.1,
    min_tokens_per_gemm: int = 1024,
    seed: int = 42,
    imbalance_configs: str = "balanced,70:2,90:1",
):
    """Run timing test for gptoss_llep_forward with multiple routing configs."""
    rank, world_size = setup_distributed()
    ep_group = setup_ep_group()
    device = torch.device(f"cuda:{rank}")
    dtype = torch.bfloat16
    
    # Set intermediate_size to hidden_size if not specified
    if intermediate_size is None:
        intermediate_size = hidden_size
    
    # Set seed for reproducibility
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed(seed + rank)
    
    # Create model
    config = MockExpertsConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
    )
    model = MockGptOssExperts(config).to(device).to(dtype)
    
    num_local_experts = num_experts // world_size
    
    # Get local weights
    local_gate_up_proj, local_gate_up_proj_bias, local_down_proj, local_down_proj_bias = \
        model.get_local_weights(rank, world_size)
    
    # Create hidden states (batch_size=1, seq_len=num_tokens)
    hidden_states = torch.randn(1, num_tokens, hidden_size, device=device, dtype=dtype)
    
    # Parse imbalance configs
    configs = parse_imbalance_configs(imbalance_configs, num_local_experts)
    
    if rank == 0:
        print(f"\n{'='*70}")
        print(f"LLEP Forward Timing Test")
        print(f"{'='*70}")
        print(f"num_tokens={num_tokens}")
        print(f"hidden_size={hidden_size}, intermediate_size={intermediate_size}")
        print(f"num_experts={num_experts}, top_k={top_k}, ep_size={world_size}")
        print(f"num_local_experts={num_local_experts}")
        print(f"num_warmup={num_warmup}, num_runs={num_runs}")
        print(f"max_tokens_factor={max_tokens_factor}, min_tokens_per_gemm={min_tokens_per_gemm}")
        print(f"imbalance_configs: {[c[0] for c in configs]}")
        print(f"{'='*70}\n")
    
    results = []
    
    for config_name, hot_ratio, num_hot in configs:
        # Create routing
        if hot_ratio is None:  # balanced
            router_indices, routing_weights = create_balanced_routing(
                num_tokens, num_experts, top_k, device, dtype=dtype
            )
        else:
            router_indices, routing_weights = create_imbalanced_routing(
                num_tokens, num_experts, top_k, device,
                hot_expert_ratio=hot_ratio,
                num_hot_experts=num_hot,
                dtype=dtype,
            )
        
        result = run_single_config(
            config_name,
            router_indices,
            routing_weights,
            hidden_states,
            local_gate_up_proj,
            local_gate_up_proj_bias,
            local_down_proj,
            local_down_proj_bias,
            num_experts,
            num_local_experts,
            ep_group,
            model,
            max_tokens_factor,
            min_tokens_per_gemm,
            num_warmup,
            num_runs,
            device,
        )
        results.append(result)
        
        if rank == 0:
            print(f"[{config_name}] Avg: {result['avg_ms']:.3f} ± {result['std_ms']:.3f} ms | "
                  f"Peak Memory: {result['peak_memory_mb']:.1f} MB")
    
    # Print summary table
    if rank == 0:
        print(f"\n{'='*70}")
        print(f"Summary Table")
        print(f"{'='*70}")
        print(f"{'Config':<25} {'Avg (ms)':<12} {'Std (ms)':<12} {'Peak Mem (MB)':<15}")
        print(f"{'-'*70}")
        for r in results:
            print(f"{r['name']:<25} {r['avg_ms']:<12.3f} {r['std_ms']:<12.3f} {r['peak_memory_mb']:<15.1f}")
        print(f"{'='*70}\n")
    
    dist.barrier()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="LLEP Forward Timing Test")
    parser.add_argument("--num_tokens", type=int, default=1024, help="Number of tokens per rank")
    parser.add_argument("--hidden_size", type=int, default=512, help="Hidden size")
    parser.add_argument("--intermediate_size", type=int, default=None, help="Intermediate size (default: hidden_size)")
    parser.add_argument("--num_experts", type=int, default=16, help="Number of experts")
    parser.add_argument("--top_k", type=int, default=2, help="Top-k routing")
    parser.add_argument("--max_tokens_factor", type=float, default=2.0, help="Max tokens factor for LPT")
    parser.add_argument("--min_tokens_per_gemm", type=int, default=256, help="Min tokens per GEMM")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_warmup", type=int, default=3, help="Number of warmup iterations")
    parser.add_argument("--num_runs", type=int, default=10, help="Number of timed runs")
    parser.add_argument(
        "--imbalance_configs",
        type=str,
        default="balanced,70:2,90:1,35:local",
        help="Comma-separated imbalance configs as ratio:num_hot_experts (e.g., 70:2,80:2,90:1,99:1). "
             "Use 'balanced' for balanced routing. Use 'local' for num_hot_experts to use num_local_experts."
    )
    args = parser.parse_args()
    
    run_llep_timing_test(
        num_tokens=args.num_tokens,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_experts=args.num_experts,
        top_k=args.top_k,
        num_warmup=args.num_warmup,
        num_runs=args.num_runs,
        max_tokens_factor=args.max_tokens_factor,
        min_tokens_per_gemm=args.min_tokens_per_gemm,
        seed=args.seed,
        imbalance_configs=args.imbalance_configs,
    )


if __name__ == "__main__":
    main()
