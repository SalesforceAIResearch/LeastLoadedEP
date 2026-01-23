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


def compute_llep_lpt_plan(
    global_expert_counts: torch.Tensor,
    ep_size: int,
    num_local_experts: int,
    max_tokens_factor: float = 1.1,
    min_tokens_per_gemm: int = 512, # lowered for viz demo
) -> LLEPLptPlan:
    
    num_experts = global_expert_counts.size(0)
    total_tokens = global_expert_counts.sum().item()
    balanced_tokens = total_tokens // ep_size if ep_size > 0 else total_tokens
    max_tokens_per_gpu = int(max_tokens_factor * balanced_tokens) if balanced_tokens > 0 else total_tokens
    max_tokens_per_gpu = max(max_tokens_per_gpu, 1)
    
    # 1. Native load per GPU
    native_load_per_gpu = [0] * ep_size
    for expert_id in range(num_experts):
        native_gpu = expert_id // num_local_experts
        native_load_per_gpu[native_gpu] += global_expert_counts[expert_id].item()
    
    pending_native_load = list(native_load_per_gpu)
    assigned_load = [0] * ep_size
    
    # Sort LPT
    expert_counts_list = [(e, int(global_expert_counts[e].item())) for e in range(num_experts)]
    expert_counts_sorted = sorted(expert_counts_list, key=lambda x: -x[1])
    
    lpt_plan = {}
    weight_transfers = []
    
    def get_effective_load(gpu_id):
        return assigned_load[gpu_id] + pending_native_load[gpu_id]
    
    for expert_id, expert_tokens in expert_counts_sorted:
        if expert_tokens == 0:
            continue
        
        native_gpu = expert_id // num_local_experts
        pending_native_load[native_gpu] -= expert_tokens
        
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
                # Find least-loaded GPU by EFFECTIVE load (excluding native)
                other_gpus = []
                for g in range(ep_size):
                    if g == native_gpu:
                        continue
                    eff_load = get_effective_load(g)
                    available = max_tokens_per_gpu - eff_load
                    other_gpus.append((g, eff_load, available))
                
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
                    
                    weight_transfers.append(WeightTransferPlan(expert_id, native_gpu, helper_gpu, token_offset, token_offset + chunk))
                    
                    token_offset += chunk
                    remaining -= chunk
                    assigned_this_round = True
                    break  # Re-evaluate for next chunk if any remains
                
                if not assigned_this_round:
                    # All helpers at capacity, force to least loaded helper
                    helper_gpu = other_gpus_sorted[0][0]
                    assignments.append((helper_gpu, token_offset, token_offset + remaining))
                    assigned_load[helper_gpu] += remaining
                    
                    weight_transfers.append(WeightTransferPlan(expert_id, native_gpu, helper_gpu, token_offset, token_offset + remaining))
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
                
                weight_transfers.append(WeightTransferPlan(expert_id, native_gpu, helper_gpu, token_offset, token_offset + chunk))
                
                token_offset += chunk
                remaining -= chunk
            
            # If still remaining (all GPUs at capacity), force to least loaded
            if remaining > 0:
                if other_gpus_sorted:
                    helper_gpu = other_gpus_sorted[0][0]
                    assignments.append((helper_gpu, token_offset, token_offset + remaining))
                    assigned_load[helper_gpu] += remaining
                    
                    weight_transfers.append(WeightTransferPlan(expert_id, native_gpu, helper_gpu, token_offset, token_offset + remaining))
                else:
                    # Edge case: only one GPU, assign everything to native
                    assignments.append((native_gpu, 0, expert_tokens))
                    assigned_load[native_gpu] += expert_tokens
        
        lpt_plan[expert_id] = assignments

    return LLEPLptPlan(lpt_plan, weight_transfers, torch.tensor(assigned_load))


# ! Streamlit

st.set_page_config(layout="wide", page_title="LLEP Simulator")

st.title("Least-Loaded Expert Parallelism (LLEP)")
st.markdown("""
Compare **Standard EP** (where stragglers slow everyone down) against your new **LLEP Algorithm** (which spills excess tokens to idle GPUs).
""")
st.markdown("""
**Authors:** [Xuan-Phi Nguyen](https://scholar.google.com/citations?user=HN8VxX4AAAAJ&hl=en), [Shrey Pandit](https://scholar.google.com/citations?user=a-dG59sAAAAJ&hl=en), [Austin Xu](https://scholar.google.com/citations?user=OUw3iQgAAAAJ&hl=en), [Caiming Xiong](https://scholar.google.com/citations?user=vaSdahkAAAAJ&hl=en), [Shafiq Joty](https://scholar.google.com/citations?user=hR249csAAAAJ&hl=en)  
**Salesforce AI Research**
**Contact:** xnguyen@salesforce.com
""")

# --- Sidebar ---
st.sidebar.header("Cluster Config")
num_experts = st.sidebar.selectbox("Num Experts", [32, 64, 128, 256], index=0)
ep_world_size = st.sidebar.selectbox("World Size (GPUs)", [4, 8, 16, 32], index=1)
experts_per_gpu = num_experts // ep_world_size

st.sidebar.header("Traffic Config")
total_tokens = st.sidebar.selectbox("Batch Tokens", [4096, 8192, 16384, 32768, 65536, 131072], index=3)
top_k = st.sidebar.slider("Top K", 1, num_experts // 2, min(4, num_experts // 2))

st.sidebar.header("LLEP Config")
max_load_factor = st.sidebar.slider("Max Load Factor", 1.0, 2.0, 1.1, 0.1, help="How much over-balance is allowed before spilling?")
imbalance_threshold = st.sidebar.slider("Imbalance Threshold", 1.0, 2.0, 1.3, 0.1, help="LPT only activates if max/mean load ratio exceeds this")
imbalance = st.sidebar.slider("Skew (Imbalance)", 0.0, 0.99, 0.6, help="Higher = More hotspots")

# Generate Synthetic Data
def generate_loads(n_experts, n_tokens, k, skew):
    # Dirichlet alpha: high = uniform, low = sparse/skewed
    # Quadratic scaling for smooth transition: alpha = 10*(1-skew)^2 + 0.05
    # skew=0 -> alpha=10 (very uniform), skew=1 -> alpha=0.05 (very sparse)
    alpha = 10.0 * ((1.0 - skew) ** 2) + 0.05
    probs = np.random.dirichlet(np.ones(n_experts) * alpha)
    return np.random.multinomial(n_tokens * k, probs)

# Auto-regenerate traffic when configs change
config_key = (num_experts, total_tokens, top_k, imbalance)
if 'config_key' not in st.session_state or st.session_state['config_key'] != config_key:
    st.session_state['config_key'] = config_key
    st.session_state['expert_loads_cache'] = generate_loads(num_experts, total_tokens, top_k, imbalance)

# Button to regenerate manually
if st.sidebar.button("Regenerate Traffic"):
    st.session_state['expert_loads_cache'] = generate_loads(num_experts, total_tokens, top_k, imbalance)

expert_loads = st.session_state['expert_loads_cache']
expert_loads_tensor = torch.tensor(expert_loads, dtype=torch.int64)

# Standard EP Simulation ---
ep_gpu_loads = [0] * ep_world_size
ep_expert_assignment = [] # (Expert, GPU, Count, Type)

for e_id, count in enumerate(expert_loads):
    if count == 0: continue
    owner_gpu = e_id // experts_per_gpu
    ep_gpu_loads[owner_gpu] += count
    ep_expert_assignment.append({
        "Expert": e_id, "GPU": owner_gpu, "Tokens": count, 
        "Type": "Native", "Owner": owner_gpu
    })

# LLEP Simulation ---
# Check imbalance ratio to decide whether to use LPT
imbalance_ratio = compute_gpu_imbalance_ratio(expert_loads_tensor, ep_world_size, experts_per_gpu)
use_lpt = imbalance_ratio >= imbalance_threshold

if use_lpt:
    llep_result = compute_llep_lpt_plan(
        expert_loads_tensor, ep_world_size, experts_per_gpu, max_tokens_factor=max_load_factor
    )
    llep_expert_assignment = []
    for e_id, assignments in llep_result.lpt_plan.items():
        native_owner = e_id // experts_per_gpu
        for (assigned_gpu, start_t, end_t) in assignments:
            count = end_t - start_t
            if count > 0:
                is_spill = (assigned_gpu != native_owner)
                llep_expert_assignment.append({
                    "Expert": e_id,
                    "GPU": assigned_gpu,
                    "Tokens": count,
                    "Type": "Spill" if is_spill else "Native",
                    "Owner": native_owner
                })
else:
    # Fall back to standard EP (no spilling)
    llep_result = LLEPLptPlan(
        lpt_plan={},
        weight_transfers=[],
        gpu_loads=torch.tensor(ep_gpu_loads)
    )
    llep_expert_assignment = ep_expert_assignment.copy()




# Define a consistent color map for GPUs so we can track "Ownership"
colors = px.colors.qualitative.Plotly
gpu_color_map = {i: colors[i % len(colors)] for i in range(ep_world_size)}

def plot_gpu_load(data, title):
    """
    Custom Plotly Graph Object chart to handle patterns.
    data: list of dicts with keys [GPU, Tokens, Owner, Type]
    """
    fig = go.Figure()
    
    # We aggregate data by (GPU, Owner, Type) to create stacked bars
    df = pd.DataFrame(data)
    if df.empty: return fig
    
    # Group to get segments
    df_grouped = df.groupby(["GPU", "Owner", "Type"])["Tokens"].sum().reset_index()
    
    # Sort so Native comes before Spill (Native at bottom, Spill on top)
    type_order = {"Native": 0, "Spill": 1}
    df_grouped["TypeOrder"] = df_grouped["Type"].map(type_order)
    df_grouped = df_grouped.sort_values(by=["GPU", "TypeOrder"]).reset_index(drop=True)
    
    for _, row in df_grouped.iterrows():
        gpu_id = row['GPU']
        owner_id = row['Owner']
        val = row['Tokens']
        is_spill = row['Type'] == 'Spill'
        
        fig.add_trace(go.Bar(
            name=f"Exp from GPU {owner_id}",
            x=[f"GPU {gpu_id}"],
            y=[val],
            marker_color=gpu_color_map[owner_id],
            marker_pattern_shape='/' if is_spill else '',
            marker_line_color='black',
            marker_line_width=0.5,
            showlegend=False, # Too many legend items otherwise
            hoverinfo="text",
            hovertext=f"Processing Work for GPU {owner_id}<br>Tokens: {val}<br>{'SPILLED' if is_spill else 'NATIVE'}"
        ))

    fig.update_layout(barmode='stack', title=title, height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def plot_expert_distribution(data, title):
    df = pd.DataFrame(data)
    if df.empty: return go.Figure()
    
    fig = go.Figure()
    
    # Group by Expert, GPU, Type and sort so Native comes before Spill
    df_grouped = df.groupby(["Expert", "GPU", "Type"])["Tokens"].sum().reset_index()
    type_order = {"Native": 0, "Spill": 1}
    df_grouped["TypeOrder"] = df_grouped["Type"].map(type_order)
    df_grouped = df_grouped.sort_values(by=["Expert", "TypeOrder"]).reset_index(drop=True)
    
    for _, row in df_grouped.iterrows():
        expert = row['Expert']
        gpu = row['GPU']
        val = row['Tokens']
        is_spill = row['Type'] == 'Spill'
        
        fig.add_trace(go.Bar(
            name=f"GPU {gpu}",
            x=[f"E{expert}"],
            y=[val],
            marker_color=gpu_color_map[gpu],
            marker_pattern_shape='/' if is_spill else '',
            marker_line_color='black',
            marker_line_width=0.5,
            showlegend=False,
            hoverinfo="text",
            hovertext=f"Processed by GPU {gpu}<br>Tokens: {val}<br>{'SPILLED' if is_spill else 'NATIVE'}"
        ))
        
    fig.update_layout(barmode='stack', title=title, height=300, margin=dict(l=20, r=20, t=40, b=20))
    fig.update_xaxes(type='category')
    return fig



# ! Streamlit UI

# --- Metrics ---
ep_max = max(ep_gpu_loads)
llep_max = llep_result.gpu_loads.max().item()
speedup = (ep_max - llep_max) / ep_max if ep_max > 0 else 0

# c1, c2, c3, c4 = st.columns(4)
# c1.metric("Standard EP Max Load", f"{ep_max} toks")
# c2.metric("LLEP Max Load", f"{llep_max} toks")
# c3.metric("LLEP Improvement", f"{speedup:.1%}", delta_color="normal")
# c4.metric("Imbalance Ratio", f"{imbalance_ratio:.2f}x")

# Show indicator when LPT is skipped
if not use_lpt:
    st.warning(f"⚠️ **LPT Skipped**: Imbalance ratio ({imbalance_ratio:.2f}x) is below threshold ({imbalance_threshold:.1f}x). LLEP uses standard EP assignment.")

st.markdown("---")

# --- Row 1: GPU Load ---
st.subheader("1. GPU Load Comparison")
c_load1, c_load2 = st.columns(2)
with c_load1:
    st.markdown("##### Standard EP")
    st.caption("Each GPU or EP rank processes its assigned experts exclusively.")
    st.plotly_chart(plot_gpu_load(ep_expert_assignment, ""), use_container_width=True, key="ep_gpu_load")
with c_load2:
    if use_lpt:
        st.markdown("##### LLEP (Solid=Native, Hatched=Spill)")
        st.caption("Overloaded GPUs spill excess load to under-utilized GPUs.")
    else:
        st.markdown("##### LLEP ⚠️ (Using Standard EP)")
        st.caption("Imbalance below threshold — no spilling needed.")
    st.plotly_chart(plot_gpu_load(llep_expert_assignment, ""), use_container_width=True, key="llep_gpu_load")

# --- Row 2: Expert Assignment ---
st.subheader("2. Experts' GPU Assignment")
c_exp1, c_exp2 = st.columns(2)
with c_exp1:
    st.markdown("##### Standard EP (Fixed)")
    st.caption("Each expert is assigned to exactly one GPU.")
    st.plotly_chart(plot_expert_distribution(ep_expert_assignment, ""), use_container_width=True, key="ep_expert_dist")
with c_exp2:
    if use_lpt:
        st.markdown("##### LLEP (Split across GPUs)")
        st.caption("If an expert is overloaded, its load may be split across multiple GPUs.")
    else:
        st.markdown("##### LLEP ⚠️ (Using Standard EP)")
        st.caption("Imbalance below threshold — same as standard EP.")
    st.plotly_chart(plot_expert_distribution(llep_expert_assignment, ""), use_container_width=True, key="llep_expert_dist")

# --- GPU Color Legend (below charts) ---
legend_html = " &nbsp; ".join(
    f"<span style='display:inline-block;width:14px;height:14px;background-color:{gpu_color_map[i]};border:1px solid black;vertical-align:middle;'></span> GPU {i}"
    for i in range(ep_world_size)
)
st.markdown(f"**Legend:** {legend_html}", unsafe_allow_html=True)

# --- Debug Data ---
with st.expander("Show Plan Details"):
    st.write("Weight Transfers Needed:", len(llep_result.weight_transfers))
    if len(llep_result.weight_transfers) > 0:
        st.dataframe([vars(x) for x in llep_result.weight_transfers])