import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import Dict, List, Tuple
import torch

@dataclass
class WeightTransferPlan:
    expert_id: int
    src_rank: int
    dst_rank: int
    token_start: int
    token_end: int


@dataclass
class LLEPLptPlan:
    lpt_plan: Dict[int, List[Tuple[int, int, int]]]
    weight_transfers: List[WeightTransferPlan]
    gpu_loads: torch.Tensor


def compute_gpu_imbalance_ratio(global_expert_counts: torch.Tensor, ep_size: int, num_local_experts: int) -> float:
    """
    GPU-level imbalance ratio: max(gpu_load) / mean(gpu_load)
    """
    gpu_loads = global_expert_counts.view(ep_size, num_local_experts).sum(dim=1).float()
    mean_load = gpu_loads.mean()
    max_load = gpu_loads.max()
    if mean_load.item() == 0:
        return 1.0
    return (max_load / mean_load).item()


def compute_expert_imbalance_ratio(global_expert_counts: torch.Tensor, ignore_zeros: bool = False) -> float:
    """
    Expert-level imbalance ratio: max(v) / mean(v)

    Note:
    - The paper pseudocode uses max(v) / mean(v) on the expert load vector v.
    - If many experts have zero load, mean(v) can be small and inflate this ratio.
    """
    v = global_expert_counts.float()
    if ignore_zeros:
        v = v[v > 0]
    if v.numel() == 0:
        return 1.0
    mean_v = v.mean()
    if mean_v.item() == 0:
        return 1.0
    return (v.max() / mean_v).item()


def compute_llep_lpt_plan(
    global_expert_counts: torch.Tensor,
    ep_size: int,
    num_local_experts: int,
    max_tokens_factor: float = 1.1,
    min_tokens_per_gemm: int = 512,
) -> LLEPLptPlan:
    """
    LLA/LLAS-style plan construction.

    Mapping to your pseudocode:
    - alpha == max_tokens_factor
    - m_alpha = alpha * (sum(v) / P)
    - pending load g_p starts as native loads g_n; for each expert, subtract e from native pending.
    - available on gpu o is (m_alpha - g_a[o] - g_p[o])
    - LLAS: pick least effective-load GPU among other GPUs; respect min_tokens_per_gemm skip rule,
      else force assign to least-loaded (even if it exceeds capacity).
    """
    num_experts = global_expert_counts.size(0)
    total_tokens = int(global_expert_counts.sum().item())
    alpha = float(max_tokens_factor)

    # Paper: m_alpha = alpha * (1/P) * sum(v)
    m_alpha = alpha * (total_tokens / ep_size) if ep_size > 0 else float(total_tokens)
    max_tokens_per_gpu = max(int(np.ceil(m_alpha)), 1)

    # Native load per GPU: g_n
    native_load_per_gpu = [0] * ep_size
    for expert_id in range(num_experts):
        native_gpu = expert_id // num_local_experts
        native_load_per_gpu[native_gpu] += int(global_expert_counts[expert_id].item())

    # g_p (pending) and g_a (assigned)
    pending_native_load = list(native_load_per_gpu)  # g_p
    assigned_load = [0] * ep_size                    # g_a

    # Sort experts by load, decreasing: hat(v)
    expert_counts_list = [(e, int(global_expert_counts[e].item())) for e in range(num_experts)]
    expert_counts_sorted = sorted(expert_counts_list, key=lambda x: -x[1])

    lpt_plan: Dict[int, List[Tuple[int, int, int]]] = {}
    weight_transfers: List[WeightTransferPlan] = []

    def effective_load(gpu_id: int) -> int:
        # g_a + g_p
        return assigned_load[gpu_id] + pending_native_load[gpu_id]

    def capacity_remaining(gpu_id: int) -> int:
        # m_alpha - g_a - g_p
        return max_tokens_per_gpu - effective_load(gpu_id)

    for expert_id, expert_tokens in expert_counts_sorted:
        if expert_tokens <= 0:
            continue

        native_gpu = expert_id // num_local_experts

        # g_p[native] -= e
        pending_native_load[native_gpu] -= expert_tokens

        # na = m_alpha - g_a[native] - g_p[native]
        native_available = capacity_remaining(native_gpu)

        assignments: List[Tuple[int, int, int]] = []

        # -----------------------
        # Case 1: native can take all
        # -----------------------
        if native_available >= expert_tokens:
            assignments.append((native_gpu, 0, expert_tokens))
            assigned_load[native_gpu] += expert_tokens

        # -----------------------
        # Case 2: native takes some, spill rest via LLAS
        # -----------------------
        elif native_available > 0:
            native_chunk = min(native_available, expert_tokens)
            assignments.append((native_gpu, 0, native_chunk))
            assigned_load[native_gpu] += native_chunk

            remaining = expert_tokens - native_chunk
            token_offset = native_chunk

            while remaining > 0:
                # other GPUs sorted by effective load (g_a + g_p)
                other_gpus = []
                for g in range(ep_size):
                    if g == native_gpu:
                        continue
                    other_gpus.append((g, effective_load(g), capacity_remaining(g)))
                other_gpus_sorted = sorted(other_gpus, key=lambda x: x[1])

                if not other_gpus_sorted:
                    # Degenerate fallback: keep on native
                    old_end = assignments[0][2]
                    assignments[0] = (native_gpu, 0, old_end + remaining)
                    assigned_load[native_gpu] += remaining
                    break

                assigned_this_round = False
                for helper_gpu, _, helper_cap in other_gpus_sorted:
                    if helper_cap <= 0:
                        continue

                    chunk = min(remaining, helper_cap)

                    # LLAS skip rule: if chunk < m and r > chunk => skip
                    if chunk < min_tokens_per_gemm and remaining > chunk:
                        continue

                    assignments.append((helper_gpu, token_offset, token_offset + chunk))
                    assigned_load[helper_gpu] += chunk
                    weight_transfers.append(
                        WeightTransferPlan(expert_id, native_gpu, helper_gpu, token_offset, token_offset + chunk)
                    )

                    token_offset += chunk
                    remaining -= chunk
                    assigned_this_round = True
                    break

                if not assigned_this_round:
                    # Force assign the least effective-load GPU (can exceed cap)
                    helper_gpu = other_gpus_sorted[0][0]
                    assignments.append((helper_gpu, token_offset, token_offset + remaining))
                    assigned_load[helper_gpu] += remaining
                    weight_transfers.append(
                        WeightTransferPlan(expert_id, native_gpu, helper_gpu, token_offset, token_offset + remaining)
                    )
                    token_offset += remaining
                    remaining = 0

        # -----------------------
        # Case 3: native has no available, spill all via LLAS
        # -----------------------
        else:
            remaining = expert_tokens
            token_offset = 0

            other_gpus = []
            for g in range(ep_size):
                if g == native_gpu:
                    continue
                other_gpus.append((g, effective_load(g), capacity_remaining(g)))
            other_gpus_sorted = sorted(other_gpus, key=lambda x: x[1])

            while remaining > 0:
                if not other_gpus_sorted:
                    # Degenerate fallback: keep on native
                    assignments.append((native_gpu, 0, expert_tokens))
                    assigned_load[native_gpu] += expert_tokens
                    break

                assigned_this_round = False
                for helper_gpu, _, helper_cap in other_gpus_sorted:
                    if helper_cap <= 0:
                        continue

                    chunk = min(remaining, helper_cap)

                    if chunk < min_tokens_per_gemm and remaining > chunk:
                        continue

                    assignments.append((helper_gpu, token_offset, token_offset + chunk))
                    assigned_load[helper_gpu] += chunk
                    weight_transfers.append(
                        WeightTransferPlan(expert_id, native_gpu, helper_gpu, token_offset, token_offset + chunk)
                    )

                    token_offset += chunk
                    remaining -= chunk
                    assigned_this_round = True
                    break

                if not assigned_this_round:
                    helper_gpu = other_gpus_sorted[0][0]
                    assignments.append((helper_gpu, token_offset, token_offset + remaining))
                    assigned_load[helper_gpu] += remaining
                    weight_transfers.append(
                        WeightTransferPlan(expert_id, native_gpu, helper_gpu, token_offset, token_offset + remaining)
                    )
                    token_offset += remaining
                    remaining = 0

        lpt_plan[expert_id] = assignments

    return LLEPLptPlan(lpt_plan=lpt_plan, weight_transfers=weight_transfers, gpu_loads=torch.tensor(assigned_load))


# ============================================================================
# ANIMATION TAB FUNCTIONS
# ============================================================================

EXPERT_COLORS = ['#3b82f6', '#8b5cf6', '#ec4899', '#14b8a6', '#f97316', '#84cc16', '#06b6d4', '#f43f5e']


def get_effective_load_anim(assigned: List[int], pending: List[int], gpu_id: int) -> int:
    return assigned[gpu_id] + pending[gpu_id]


def generate_animation_steps(
    expert_loads: List[int],
    alpha: float,
    num_gpus: int,
    local_experts_per_gpu: int,
    min_tokens_per_gemm: int,
) -> List[dict]:
    """
    Step-by-step LLA/LLAS animation.

    This follows the same logic as your pseudocode:
    - pending starts as native loads
    - for each expert in sorted order: pending[native] -= e
    - na = m_alpha - assigned[native] - pending[native]
    - case 1/2/3 and LLAS spill with skip rule and force-assign fallback
    """
    total_experts = num_gpus * local_experts_per_gpu
    loads = [int(x) for x in expert_loads[:total_experts]]

    steps: List[dict] = []

    sorted_indices = sorted(range(total_experts), key=lambda i: loads[i], reverse=True)
    sorted_loads = [loads[i] for i in sorted_indices]

    total_load = int(sum(sorted_loads))
    m_alpha = float(alpha) * (total_load / num_gpus) if num_gpus > 0 else float(total_load)
    max_per_gpu = float(m_alpha)

    native_loads = [0] * num_gpus
    for i in range(total_experts):
        native_loads[i // local_experts_per_gpu] += loads[i]

    state = {
        "sorted_indices": sorted_indices,
        "sorted_loads": sorted_loads,
        "total_load": total_load,
        "max_per_gpu": max_per_gpu,
        "min_tokens_per_gemm": int(min_tokens_per_gemm),
        "g_pending": list(native_loads),
        "g_assigned": [0] * num_gpus,
        "assignments": {},
        "current_expert_idx": -1,
        "phase": "init",
        "message": f"Sorted experts by load. Total={total_load}, m_alpha={max_per_gpu:.2f} (α={alpha:.2f}, m={min_tokens_per_gemm})",
        "case_type": None,
        "highlight_gpu": None,
        "spill_flows": [],
        "spill_targets": [],
    }
    steps.append(dict(state))

    def cap_remaining(g_assigned: List[int], g_pending: List[int], gpu_id: int) -> float:
        return max_per_gpu - float(get_effective_load_anim(g_assigned, g_pending, gpu_id))

    for i in range(total_experts):
        expert_load = int(state["sorted_loads"][i])
        original_idx = int(state["sorted_indices"][i])
        native_gpu = original_idx // local_experts_per_gpu

        # g_p[native] -= e
        new_pending = list(state["g_pending"])
        new_pending[native_gpu] -= expert_load

        na = cap_remaining(state["g_assigned"], new_pending, native_gpu)

        state = dict(state)
        state["g_pending"] = new_pending
        state["current_expert_idx"] = i
        state["highlight_gpu"] = native_gpu
        state["phase"] = "evaluate"
        state["message"] = f"Expert E{original_idx} (load={expert_load}) native=GPU{native_gpu}. na={max(0.0, na):.2f}"
        state["spill_flows"] = []
        state["spill_targets"] = []
        state["case_type"] = None
        steps.append(dict(state))

        new_assigned = list(state["g_assigned"])
        assignments = []
        spill_flows = []
        spill_targets = []

        # Case 1
        if na >= expert_load:
            assignments.append({"gpu": native_gpu, "start": 0, "end": expert_load})
            new_assigned[native_gpu] += expert_load
            state["case_type"] = 1
            state["message"] = f"Case 1: native GPU{native_gpu} takes all {expert_load}"

        # Case 2
        elif na > 0:
            native_chunk = int(np.floor(na))
            native_chunk = max(0, min(native_chunk, expert_load))

            assignments.append({"gpu": native_gpu, "start": 0, "end": native_chunk})
            new_assigned[native_gpu] += native_chunk

            remaining = expert_load - native_chunk
            token_offset = native_chunk

            while remaining > 0:
                helper_gpus = []
                for g in range(num_gpus):
                    if g == native_gpu:
                        continue
                    eff_load = float(get_effective_load_anim(new_assigned, new_pending, g))
                    avail = cap_remaining(new_assigned, new_pending, g)
                    helper_gpus.append({"gpu": g, "eff_load": eff_load, "avail": avail})
                helper_gpus.sort(key=lambda x: x["eff_load"])

                if not helper_gpus:
                    # Degenerate fallback: keep on native
                    assignments[-1]["end"] += remaining
                    new_assigned[native_gpu] += remaining
                    remaining = 0
                    break

                assigned_flag = False
                for helper in helper_gpus:
                    if helper["avail"] <= 0:
                        continue

                    c = int(min(remaining, np.floor(helper["avail"])))
                    if c <= 0:
                        continue

                    if c < min_tokens_per_gemm and remaining > c:
                        continue

                    assignments.append({"gpu": helper["gpu"], "start": token_offset, "end": token_offset + c})
                    spill_flows.append({"from": native_gpu, "to": helper["gpu"], "amount": c})
                    spill_targets.append(helper["gpu"])
                    new_assigned[helper["gpu"]] += c
                    token_offset += c
                    remaining -= c
                    assigned_flag = True
                    break

                if not assigned_flag:
                    # Force assign to least effective-load helper (may exceed capacity)
                    helper = helper_gpus[0]
                    c = remaining
                    assignments.append({"gpu": helper["gpu"], "start": token_offset, "end": token_offset + c})
                    spill_flows.append({"from": native_gpu, "to": helper["gpu"], "amount": c})
                    spill_targets.append(helper["gpu"])
                    new_assigned[helper["gpu"]] += c
                    token_offset += c
                    remaining = 0

            state["case_type"] = 2
            spill_target_str = ", ".join([f"GPU{g}" for g in sorted(set(spill_targets))]) if spill_targets else "none"
            state["message"] = f"Case 2: native GPU{native_gpu} takes {native_chunk}, spill {expert_load - native_chunk} -> {spill_target_str}"

        # Case 3
        else:
            remaining = expert_load
            token_offset = 0

            while remaining > 0:
                helper_gpus = []
                for g in range(num_gpus):
                    if g == native_gpu:
                        continue
                    eff_load = float(get_effective_load_anim(new_assigned, new_pending, g))
                    avail = cap_remaining(new_assigned, new_pending, g)
                    helper_gpus.append({"gpu": g, "eff_load": eff_load, "avail": avail})
                helper_gpus.sort(key=lambda x: x["eff_load"])

                if not helper_gpus:
                    # Degenerate fallback: keep on native
                    assignments.append({"gpu": native_gpu, "start": 0, "end": expert_load})
                    new_assigned[native_gpu] += expert_load
                    remaining = 0
                    break

                assigned_flag = False
                for helper in helper_gpus:
                    if helper["avail"] <= 0:
                        continue

                    c = int(min(remaining, np.floor(helper["avail"])))
                    if c <= 0:
                        continue

                    if c < min_tokens_per_gemm and remaining > c:
                        continue

                    assignments.append({"gpu": helper["gpu"], "start": token_offset, "end": token_offset + c})
                    spill_flows.append({"from": native_gpu, "to": helper["gpu"], "amount": c})
                    spill_targets.append(helper["gpu"])
                    new_assigned[helper["gpu"]] += c
                    token_offset += c
                    remaining -= c
                    assigned_flag = True
                    break

                if not assigned_flag:
                    helper = helper_gpus[0]
                    c = remaining
                    assignments.append({"gpu": helper["gpu"], "start": token_offset, "end": token_offset + c})
                    spill_flows.append({"from": native_gpu, "to": helper["gpu"], "amount": c})
                    spill_targets.append(helper["gpu"])
                    new_assigned[helper["gpu"]] += c
                    token_offset += c
                    remaining = 0

            state["case_type"] = 3
            spill_target_str = ", ".join([f"GPU{g}" for g in sorted(set(spill_targets))]) if spill_targets else "none"
            state["message"] = f"Case 3: native GPU{native_gpu} full; spill all {expert_load} -> {spill_target_str}"

        state["g_assigned"] = new_assigned
        state["assignments"] = dict(state["assignments"])
        state["assignments"][i] = assignments
        state["spill_flows"] = spill_flows
        state["spill_targets"] = sorted(list(set(spill_targets)))
        state["phase"] = "assign"
        steps.append(dict(state))

    case_counts = {1: 0, 2: 0, 3: 0}
    for s in steps:
        if s.get("case_type") in case_counts:
            case_counts[int(s["case_type"])] += 1

    state["phase"] = "complete"
    state["message"] = f"Complete. Case1={case_counts[1]}, Case2={case_counts[2]}, Case3={case_counts[3]}"
    state["current_expert_idx"] = -1
    state["highlight_gpu"] = None
    state["spill_flows"] = []
    state["spill_targets"] = []
    steps.append(dict(state))

    return steps


def create_gpu_topology_chart(state: dict, num_gpus: int) -> go.Figure:
    """
    GPU topology with spill arrows and overflow indication.
    """
    fig = go.Figure()

    if num_gpus <= 4:
        gpu_positions = [(i % 2, 1 - i // 2) for i in range(num_gpus)]
    else:
        cols = 4
        gpu_positions = [(i % cols, -(i // cols)) for i in range(num_gpus)]

    max_load = float(state["max_per_gpu"])
    assigned = state["g_assigned"]

    for gpu_id in range(num_gpus):
        x, y = gpu_positions[gpu_id]
        a = float(assigned[gpu_id])

        fill_pct = (a / max_load) if max_load > 0 else 0.0
        fill_pct_clamped = min(fill_pct, 1.0)

        is_highlighted = gpu_id == state.get("highlight_gpu")
        is_spill_target = gpu_id in state.get("spill_targets", [])

        overflow = (a - max_load) if max_load > 0 and a > max_load else 0.0

        if is_highlighted:
            box_color = "#facc15"
        elif is_spill_target:
            box_color = "#f97316"
        elif overflow > 0:
            box_color = "#ef4444"
        else:
            box_color = "#4b5563"

        fig.add_shape(
            type="rect", x0=x - 0.3, y0=y - 0.15, x1=x + 0.3, y1=y + 0.15,
            fillcolor="#1f2937", line=dict(color=box_color, width=3)
        )

        bar_width = 0.5 * fill_pct_clamped
        bar_color = "#ef4444" if fill_pct >= 1 else "#3b82f6"
        fig.add_shape(
            type="rect", x0=x - 0.25, y0=y - 0.08, x1=x - 0.25 + bar_width, y1=y - 0.02,
            fillcolor=bar_color, line=dict(width=0)
        )

        fig.add_annotation(
            x=x, y=y + 0.05, text=f"<b>GPU {gpu_id}</b>",
            showarrow=False, font=dict(color="white", size=12)
        )

        text = f"{a:.0f} / {max_load:.0f}"
        if overflow > 0:
            text = f"{a:.0f} / {max_load:.0f} (+{overflow:.0f})"
        fig.add_annotation(
            x=x, y=y - 0.05, text=text,
            showarrow=False, font=dict(color="white", size=10)
        )

        if is_highlighted:
            fig.add_annotation(x=x, y=y - 0.22, text="NATIVE", showarrow=False, font=dict(color="#facc15", size=9))
        elif is_spill_target:
            fig.add_annotation(x=x, y=y - 0.22, text="HELPER", showarrow=False, font=dict(color="#f97316", size=9))
        elif overflow > 0:
            fig.add_annotation(x=x, y=y - 0.22, text="OVER", showarrow=False, font=dict(color="#ef4444", size=9))

    for flow in state.get("spill_flows", []):
        from_pos = gpu_positions[flow["from"]]
        to_pos = gpu_positions[flow["to"]]

        fig.add_annotation(
            x=to_pos[0], y=to_pos[1],
            ax=from_pos[0], ay=from_pos[1],
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True,
            arrowhead=2, arrowsize=1.5, arrowwidth=3,
            arrowcolor="#f97316"
        )

        mid_x = (from_pos[0] + to_pos[0]) / 2
        mid_y = (from_pos[1] + to_pos[1]) / 2
        fig.add_annotation(
            x=mid_x, y=mid_y + 0.1,
            text=f"<b>{flow['amount']}</b>",
            showarrow=False,
            font=dict(color="#f97316", size=12),
            bgcolor="#1f2937"
        )

    y_min = min(p[1] for p in gpu_positions) - 0.4
    y_max = max(p[1] for p in gpu_positions) + 0.4
    x_min = min(p[0] for p in gpu_positions) - 0.5
    x_max = max(p[0] for p in gpu_positions) + 0.5

    fig.update_layout(
        xaxis=dict(range=[x_min, x_max], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[y_min, y_max], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x"),
        plot_bgcolor="#1f2937",
        paper_bgcolor="#1f2937",
        margin=dict(l=10, r=10, t=10, b=10),
        height=280
    )
    return fig


def create_load_bars_chart(state: dict, num_gpus: int) -> go.Figure:
    """
    GPU load bar chart with capacity marker, showing overflow if it occurs.
    """
    max_load = float(state["max_per_gpu"])
    gpus = [f"GPU {i}" for i in range(num_gpus)]
    assigned = [float(x) for x in state["g_assigned"]]

    colors = []
    for i in range(num_gpus):
        if i == state.get("highlight_gpu"):
            colors.append("#facc15")
        elif i in state.get("spill_targets", []):
            colors.append("#f97316")
        elif assigned[i] > max_load:
            colors.append("#ef4444")
        else:
            colors.append("#3b82f6")

    x_max = max(max_load * 1.1, (max(assigned) * 1.1 if assigned else 1.0), 1.0)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=gpus, x=assigned, orientation="h",
        marker_color=colors,
        text=[f"{a:.0f}/{max_load:.0f}" for a in assigned],
        textposition="inside",
        textfont=dict(color="white")
    ))
    fig.add_vline(x=max_load, line_dash="dash", line_color="#ef4444", line_width=2)

    fig.update_layout(
        xaxis=dict(title="Tokens", range=[0, x_max]),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="#1f2937",
        paper_bgcolor="#1f2937",
        font=dict(color="white"),
        margin=dict(l=10, r=10, t=10, b=30),
        height=max(160, num_gpus * 40),
        showlegend=False
    )
    return fig


# ============================================================================
# STATISTICS TAB FUNCTIONS
# ============================================================================

def generate_loads(n_experts: int, n_tokens: int, k: int, skew: float) -> np.ndarray:
    alpha = 10.0 * ((1.0 - skew) ** 2) + 0.05
    probs = np.random.dirichlet(np.ones(n_experts) * alpha)
    return np.random.multinomial(n_tokens * k, probs)


def plot_gpu_load(data: List[dict], title: str, ep_world_size: int, gpu_color_map: dict) -> go.Figure:
    fig = go.Figure()
    df = pd.DataFrame(data)
    if df.empty:
        return fig

    df_grouped = df.groupby(["GPU", "Owner", "Type"])["Tokens"].sum().reset_index()
    type_order = {"Native": 0, "Spill": 1}
    df_grouped["TypeOrder"] = df_grouped["Type"].map(type_order)
    df_grouped = df_grouped.sort_values(by=["GPU", "TypeOrder"]).reset_index(drop=True)

    for _, row in df_grouped.iterrows():
        gpu_id = int(row["GPU"])
        owner_id = int(row["Owner"])
        val = float(row["Tokens"])
        is_spill = row["Type"] == "Spill"

        fig.add_trace(go.Bar(
            name=f"Exp from GPU {owner_id}",
            x=[f"GPU {gpu_id}"],
            y=[val],
            marker_color=gpu_color_map[owner_id],
            marker_pattern_shape="/" if is_spill else "",
            marker_line_color="black",
            marker_line_width=0.5,
            showlegend=False,
            hoverinfo="text",
            hovertext=f"Processing work for native owner GPU {owner_id}<br>Tokens: {val:.0f}<br>{'SPILL' if is_spill else 'NATIVE'}"
        ))

    fig.update_layout(
        barmode="stack",
        title=title,
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig


def plot_expert_distribution(data: List[dict], title: str, gpu_color_map: dict) -> go.Figure:
    df = pd.DataFrame(data)
    if df.empty:
        return go.Figure()

    fig = go.Figure()
    df_grouped = df.groupby(["Expert", "GPU", "Type"])["Tokens"].sum().reset_index()
    type_order = {"Native": 0, "Spill": 1}
    df_grouped["TypeOrder"] = df_grouped["Type"].map(type_order)
    df_grouped = df_grouped.sort_values(by=["Expert", "TypeOrder"]).reset_index(drop=True)

    for _, row in df_grouped.iterrows():
        expert = int(row["Expert"])
        gpu = int(row["GPU"])
        val = float(row["Tokens"])
        is_spill = row["Type"] == "Spill"

        fig.add_trace(go.Bar(
            name=f"GPU {gpu}",
            x=[f"E{expert}"],
            y=[val],
            marker_color=gpu_color_map[gpu],
            marker_pattern_shape="/" if is_spill else "",
            marker_line_color="black",
            marker_line_width=0.5,
            showlegend=False,
            hoverinfo="text",
            hovertext=f"Processed by GPU {gpu}<br>Tokens: {val:.0f}<br>{'SPILL' if is_spill else 'NATIVE'}"
        ))

    fig.update_layout(
        barmode="stack",
        title=title,
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    fig.update_xaxes(type="category")
    return fig


# ============================================================================
# MAIN STREAMLIT APP
# ============================================================================

st.set_page_config(layout="wide", page_title="LLEP Simulator & Visualizer")

st.title("Least-Loaded Expert Parallelism (LLEP)")
st.markdown("Compare **Standard EP** against the **LLEP (LLA/LLAS)** plan and visualize step-by-step spilling.")
st.caption("""
[Xuan-Phi Nguyen](https://scholar.google.com/citations?user=HN8VxX4AAAAJ&hl=en) · 
[Shrey Pandit](https://scholar.google.com/citations?user=a-dG59sAAAAJ&hl=en) · 
[Austin Xu](https://scholar.google.com/citations?user=OUw3iQgAAAAJ&hl=en) · 
[Caiming Xiong](https://scholar.google.com/citations?user=vaSdahkAAAAJ&hl=en) · 
[Shafiq Joty](https://scholar.google.com/citations?user=hR249csAAAAJ&hl=en)  
**Salesforce AI Research** · xnguyen@salesforce.com
""", unsafe_allow_html=True)

tab_stats, tab_anim = st.tabs(["Statistics & Comparison", "Step-by-Step Animation"])

# ============================================================================
# TAB 1: STATISTICS & COMPARISON
# ============================================================================
with tab_stats:
    cfg_col, out_col = st.columns([0.36, 0.64], gap="large")

    with cfg_col:
        st.subheader("Statistics Config")

        num_experts = st.selectbox("Num Experts", [32, 64, 128, 256], index=0, key="stats_experts")
        ep_world_size = st.selectbox("World Size (GPUs)", [4, 8, 16, 32], index=1, key="stats_gpus")
        experts_per_gpu = num_experts // ep_world_size

        st.markdown("#### Traffic Config")
        total_tokens = st.selectbox("Batch Tokens", [4096, 8192, 16384, 32768, 65536, 131072], index=3, key="stats_tokens")
        top_k = st.slider("Top K", 1, num_experts // 2, min(4, num_experts // 2), key="stats_topk")
        imbalance = st.slider("Skew (Imbalance)", 0.0, 0.99, 0.6, key="stats_skew", help="Higher = more hotspots")

        st.markdown("#### LLEP / LLA Config")
        alpha_capacity = st.slider(
            "α (capacity factor)",
            1.0, 2.0, 1.1, 0.05,
            key="stats_alpha",
            help="m_alpha = α * (sum(v)/P). Lower α -> more spilling."
        )
        min_tokens_per_gemm = st.slider(
            "Min tokens per GEMM (m)",
            1, 4096, 512, 32,
            key="stats_min_gemm",
            help="If a candidate chunk c < m and remaining r > c, we skip that GPU (LLAS rule)."
        )

        st.markdown("#### Activation Threshold (λ)")
        imbalance_metric = st.radio(
            "Imbalance metric used for λ check",
            ["Expert-level (paper)", "GPU-level (practical)"],
            index=0,
            key="stats_metric",
            help="Paper pseudocode uses max(v)/mean(v) on expert loads v. Your earlier code used GPU aggregation."
        )
        ignore_zeros = st.checkbox(
            "Ignore zero-load experts for expert-level mean",
            value=True,
            key="stats_ignore_zeros",
            help="Prevents max(v)/mean(v) from exploding when many experts are unused."
        )
        imbalance_threshold = st.slider(
            "λ (threshold)",
            1.0, 10.0, 1.3, 0.1,
            key="stats_lambda",
            help="If ratio < λ, we use standard EP. Else, compute LLA/LLAS plan."
        )

        regen = st.button("Regenerate Traffic", key="stats_regen")

    # Generate Synthetic Data (scoped to this tab, no sidebar bleed)
    config_key = (num_experts, total_tokens, top_k, imbalance, "stats")
    if ("stats_config_key" not in st.session_state) or (st.session_state["stats_config_key"] != config_key) or regen:
        st.session_state["stats_config_key"] = config_key
        st.session_state["stats_expert_loads"] = generate_loads(num_experts, total_tokens, top_k, imbalance)

    expert_loads = st.session_state["stats_expert_loads"]
    expert_loads_tensor = torch.tensor(expert_loads, dtype=torch.int64)

    # Standard EP
    ep_gpu_loads = [0] * ep_world_size
    ep_expert_assignment = []
    for e_id, count in enumerate(expert_loads):
        if int(count) == 0:
            continue
        owner_gpu = e_id // experts_per_gpu
        ep_gpu_loads[owner_gpu] += int(count)
        ep_expert_assignment.append({
            "Expert": int(e_id),
            "GPU": int(owner_gpu),
            "Tokens": int(count),
            "Type": "Native",
            "Owner": int(owner_gpu),
        })

    # Ratios
    ratio_expert = compute_expert_imbalance_ratio(expert_loads_tensor, ignore_zeros=bool(ignore_zeros))
    ratio_gpu = compute_gpu_imbalance_ratio(expert_loads_tensor, ep_world_size, experts_per_gpu)

    if imbalance_metric == "Expert-level (paper)":
        imbalance_ratio = ratio_expert
    else:
        imbalance_ratio = ratio_gpu

    use_lpt = imbalance_ratio >= float(imbalance_threshold)

    # LLEP (LLA/LLAS) plan
    if use_lpt:
        llep_result = compute_llep_lpt_plan(
            expert_loads_tensor,
            ep_world_size,
            experts_per_gpu,
            max_tokens_factor=float(alpha_capacity),
            min_tokens_per_gemm=int(min_tokens_per_gemm),
        )
        llep_expert_assignment = []
        for e_id, assigns in llep_result.lpt_plan.items():
            native_owner = int(e_id) // experts_per_gpu
            for (assigned_gpu, start_t, end_t) in assigns:
                count = int(end_t - start_t)
                if count <= 0:
                    continue
                is_spill = (int(assigned_gpu) != int(native_owner))
                llep_expert_assignment.append({
                    "Expert": int(e_id),
                    "GPU": int(assigned_gpu),
                    "Tokens": count,
                    "Type": "Spill" if is_spill else "Native",
                    "Owner": int(native_owner),
                })
    else:
        llep_result = LLEPLptPlan(lpt_plan={}, weight_transfers=[], gpu_loads=torch.tensor(ep_gpu_loads))
        llep_expert_assignment = ep_expert_assignment.copy()

    colors = px.colors.qualitative.Plotly
    gpu_color_map = {i: colors[i % len(colors)] for i in range(ep_world_size)}

    with out_col:
        st.subheader("Status")
        st.write(
            pd.DataFrame([{
                "expert_ratio max/mean": f"{ratio_expert:.2f}x",
                "gpu_ratio max/mean": f"{ratio_gpu:.2f}x",
                "metric_used": imbalance_metric,
                "λ": float(imbalance_threshold),
                "activated": bool(use_lpt),
                "α": float(alpha_capacity),
                "m": int(min_tokens_per_gemm),
            }])
        )

        if not use_lpt:
            st.warning(
                f"LLA skipped: ratio {imbalance_ratio:.2f}x < λ {imbalance_threshold:.2f}. Using standard EP."
            )

        st.markdown("---")

        # GPU Load Comparison
        st.subheader("1. GPU Load Comparison")
        c_load1, c_load2 = st.columns(2)

        with c_load1:
            st.markdown("##### Standard EP")
            st.caption("Each GPU processes its native experts only.")
            st.plotly_chart(plot_gpu_load(ep_expert_assignment, "", ep_world_size, gpu_color_map), use_container_width=True, key="ep_gpu_load")

        with c_load2:
            st.markdown("##### LLEP / LLA (Solid=Native, Hatched=Spill)" if use_lpt else "##### LLEP (standard EP fallback)")
            st.caption("Overloaded GPUs spill to least-loaded helpers, following LLAS rules." if use_lpt else "Imbalance below λ, so no spilling.")
            st.plotly_chart(plot_gpu_load(llep_expert_assignment, "", ep_world_size, gpu_color_map), use_container_width=True, key="llep_gpu_load")

        # Expert Assignment
        st.subheader("2. Experts' GPU Assignment")
        c_exp1, c_exp2 = st.columns(2)

        with c_exp1:
            st.markdown("##### Standard EP (Fixed)")
            st.caption("Each expert is assigned to exactly one GPU.")
            st.plotly_chart(plot_expert_distribution(ep_expert_assignment, "", gpu_color_map), use_container_width=True, key="ep_expert_dist")

        with c_exp2:
            st.markdown("##### LLEP (Split across GPUs)" if use_lpt else "##### LLEP (standard EP fallback)")
            st.caption("Experts may be split across GPUs when spilling is needed." if use_lpt else "Same as standard EP.")
            st.plotly_chart(plot_expert_distribution(llep_expert_assignment, "", gpu_color_map), use_container_width=True, key="llep_expert_dist")

        legend_html = " &nbsp; ".join(
            f"<span style='display:inline-block;width:14px;height:14px;background-color:{gpu_color_map[i]};border:1px solid black;vertical-align:middle;'></span> GPU {i}"
            for i in range(ep_world_size)
        )
        st.markdown(f"**Legend:** {legend_html}", unsafe_allow_html=True)

        with st.expander("Show Plan Details"):
            st.write("Weight Transfers Needed:", len(llep_result.weight_transfers))
            if len(llep_result.weight_transfers) > 0:
                st.dataframe([vars(x) for x in llep_result.weight_transfers])


# ============================================================================
# TAB 2: STEP-BY-STEP ANIMATION
# ============================================================================
with tab_anim:
    anim_num_gpus = 4
    anim_local_experts = 2
    anim_total_experts = anim_num_gpus * anim_local_experts

    # Initialize widget-backed state once
    if "anim_alpha" not in st.session_state:
        st.session_state["anim_alpha"] = 1.0
    if "anim_min_gemm" not in st.session_state:
        st.session_state["anim_min_gemm"] = 1
    if "anim_step" not in st.session_state:
        st.session_state["anim_step"] = 0
    for idx in range(anim_total_experts):
        key = f"anim_load_{idx}"
        if key not in st.session_state:
            default = [150, 50, 20, 20, 100, 40, 40, 20][idx]
            st.session_state[key] = int(default)

    PRESETS = {
        "No Spill (high α)": {"alpha": 1.5, "loads": [50, 50, 50, 50, 50, 50, 50, 50]},
        "Some Spills": {"alpha": 1.0, "loads": [150, 50, 20, 20, 100, 40, 40, 20]},
        "Many Spills (low α)": {"alpha": 0.8, "loads": [150, 50, 20, 20, 100, 40, 40, 20]},
        "Extreme Imbalance": {"alpha": 0.6, "loads": [200, 10, 10, 10, 180, 10, 10, 10]},
    }

    # Define callback to apply preset BEFORE widgets are instantiated
    def apply_preset_callback():
        preset_name = st.session_state.get("anim_preset", "Some Spills")
        if preset_name in PRESETS:
            st.session_state["anim_alpha"] = float(PRESETS[preset_name]["alpha"])
            st.session_state["anim_min_gemm"] = st.session_state.get("anim_min_gemm", 1)
            for idx, v in enumerate(PRESETS[preset_name]["loads"]):
                st.session_state[f"anim_load_{idx}"] = int(v)
            st.session_state["anim_step"] = 0

    cfg_col, out_col = st.columns([0.32, 0.68], gap="large")

    with cfg_col:
        st.subheader("Animation Config")
        st.caption("LLA + LLAS with α capacity and min-tokens-per-GEMM (m).")

        preset = st.selectbox("Preset", list(PRESETS.keys()), key="anim_preset")
        st.button("Apply Preset", key="anim_apply_preset", on_click=apply_preset_callback)

        st.markdown("#### Parameters")
        st.slider(
            "α (capacity factor)",
            0.5, 1.5,
            step=0.05,
            key="anim_alpha"
        )
        st.slider(
            "m (min tokens per GEMM)",
            1, 512,
            step=1,
            key="anim_min_gemm",
            help="LLAS rule: if candidate chunk c < m and remaining r > c, skip that GPU; else may force-assign."
        )

        st.markdown("#### Expert Loads")
        st.caption("E{i} → GPU{i//2}")
        load_cols = st.columns(2)
        for gpu_idx in range(anim_num_gpus):
            with load_cols[gpu_idx % 2]:
                st.markdown(f"**GPU {gpu_idx}**")
                for local_idx in range(anim_local_experts):
                    idx = gpu_idx * anim_local_experts + local_idx
                    st.number_input(
                        f"E{idx}",
                        min_value=0,
                        max_value=500,
                        value=int(st.session_state[f"anim_load_{idx}"]),
                        step=1,
                        key=f"anim_load_{idx}"
                    )

        loads_now = [int(st.session_state[f"anim_load_{i}"]) for i in range(anim_total_experts)]
        alpha_now = float(st.session_state["anim_alpha"])
        m_now = int(st.session_state["anim_min_gemm"])

        total_now = sum(loads_now)
        m_alpha_now = alpha_now * (total_now / anim_num_gpus) if anim_num_gpus > 0 else float(total_now)

        st.info(f"α={alpha_now:.2f}, m={m_now}, Total={total_now}, m_α={m_alpha_now:.2f}")

        if st.button("Reset Animation Step", key="anim_reset_step"):
            st.session_state["anim_step"] = 0
            st.rerun()

    # Build steps from current widget values (so changes are visible immediately)
    anim_steps = generate_animation_steps(
        expert_loads=[int(st.session_state[f"anim_load_{i}"]) for i in range(anim_total_experts)],
        alpha=float(st.session_state["anim_alpha"]),
        num_gpus=anim_num_gpus,
        local_experts_per_gpu=anim_local_experts,
        min_tokens_per_gemm=int(st.session_state["anim_min_gemm"]),
    )

    current_step = int(st.session_state["anim_step"])
    current_step = max(0, min(current_step, len(anim_steps) - 1))
    st.session_state["anim_step"] = current_step
    state = anim_steps[current_step]

    with out_col:
        st.subheader("Step-by-Step Animation")

        # Controls
        ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4, ctrl_col5 = st.columns([1, 1, 1, 1, 4])
        with ctrl_col1:
            if st.button("Reset", key="anim_reset"):
                st.session_state["anim_step"] = 0
                st.rerun()
        with ctrl_col2:
            if st.button("Prev", key="anim_prev") and current_step > 0:
                st.session_state["anim_step"] -= 1
                st.rerun()
        with ctrl_col3:
            if st.button("Next", key="anim_next") and current_step < len(anim_steps) - 1:
                st.session_state["anim_step"] += 1
                st.rerun()
        with ctrl_col4:
            if st.button("End", key="anim_end"):
                st.session_state["anim_step"] = len(anim_steps) - 1
                st.rerun()

        st.progress(current_step / max(len(anim_steps) - 1, 1), text=f"Step {current_step + 1} / {len(anim_steps)}")

        case_type = state.get("case_type")
        if case_type in (1, 2, 3):
            label = "Case 1" if case_type == 1 else "Case 2" if case_type == 2 else "Case 3"
            st.write(f"**{label}** — {state['message']}")
        else:
            st.info(state["message"])

        viz_col1, viz_col2, viz_col3 = st.columns([1.3, 1.2, 1.5])

        with viz_col1:
            st.markdown("##### Experts (sorted by load)")
            exp_cols = st.columns(2)

            for idx in range(anim_total_experts):
                if idx >= len(state["sorted_loads"]):
                    continue
                load = int(state["sorted_loads"][idx])
                original_idx = int(state["sorted_indices"][idx])
                is_processed = idx in state.get("assignments", {})
                is_current = idx == int(state["current_expert_idx"])

                color = EXPERT_COLORS[original_idx % len(EXPERT_COLORS)]
                opacity = "0.4" if is_processed else "1"
                border = "3px solid #facc15" if is_current else "1px solid #4b5563"

                with exp_cols[idx % 2]:
                    st.markdown(
                        f"""<div style="background-color: {color}22; border: {border}; border-radius: 6px;
                        padding: 6px; margin: 2px 0; opacity: {opacity};">
                        <span style="color: #9ca3af; font-size: 10px;">E{original_idx} -> GPU{original_idx // anim_local_experts}</span>
                        <span style="color: {color}; font-size: 16px; font-weight: bold; float: right;">{load}</span>
                        </div>""",
                        unsafe_allow_html=True
                    )

        with viz_col2:
            st.markdown("##### GPU Topology")
            st.plotly_chart(create_gpu_topology_chart(state, anim_num_gpus), use_container_width=True, key="anim_topology")
            st.caption("Helpers exclude the native GPU. Overflow is possible via force-assign in LLAS.")

        with viz_col3:
            st.markdown("##### GPU Loads")
            st.plotly_chart(create_load_bars_chart(state, anim_num_gpus), use_container_width=True, key="anim_loads")

            st.markdown("##### Assignment Map")
            st.caption("Format: (GPU, start, end)")
            if state.get("assignments"):
                rows = []
                for idx, assigns in state["assignments"].items():
                    original_idx = int(state["sorted_indices"][idx])
                    native_gpu = original_idx // anim_local_experts
                    has_spill = any(int(a["gpu"]) != int(native_gpu) for a in assigns)

                    assign_str = " ".join([f"(G{int(a['gpu'])},{int(a['start'])},{int(a['end'])})" for a in assigns])

                    rows.append({
                        "Expert": f"E{original_idx}",
                        "Load": int(state["sorted_loads"][idx]),
                        "Assignments": assign_str,
                        "Spilled?": "Yes" if has_spill else "No",
                    })

                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, hide_index=True, height=220)
            else:
                st.caption("No assignments yet")
