import torch
import flashinfer
from  flashinfer.cascade import *
import math
import torch.nn as nn
import random
num_qo_heads = 32
num_kv_heads = 16
head_dim = 128
max_num_pages = 150
page_size = 256
inf = 32768
batch_size = 3
recompute_ratio = 0.5
max_chunk_num = 20
max_chunk_length = 1024
import math
torch.manual_seed(42)
random.seed(42)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
    
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed

def create_lower_triangular_matrix(batch_size, num_heads, context_size):
    # 创建一个下三角掩码矩阵，形状为 (context_size, context_size)
    lower_triangular_mask = torch.triu(torch.full((context_size , context_size ), -65504.0),diagonal=1)
    return lower_triangular_mask.to('cuda')
def create_mask_from_position_for_standard(q_position:torch.tensor, kv_position:torch.tensor, critical_pos:list):
    # 创建一个掩码矩阵，形状为 (len(q_position), len(kv_position))
    q_position = q_position.unsqueeze(1)
    kv_position = kv_position.unsqueeze(0)
    mask = q_position >= kv_position
    mask[:,critical_pos] = False
    mask = torch.where(mask, torch.tensor(0), torch.tensor(-65504))
    return mask

def create_mask_from_position_for_flashinfer(q_position:list, kv_position:list, critical_pos:list):
    q_position = q_position.unsqueeze(1)
    kv_position = kv_position.unsqueeze(0)
    mask = q_position >= kv_position
    mask[:,critical_pos] = False
    return mask

def create_mask_for_batch(sub_q_positions:list, sub_kv_positions:list, sub_critical_poses:list):
    mask = []
    for i in range(len(sub_q_positions)):
        tmp_mask = create_mask_from_position_for_flashinfer(sub_q_positions[i], sub_kv_positions[i], sub_critical_poses[i] if len(sub_critical_poses) != 0 else []).flatten()
        mask.append(tmp_mask)
    return torch.cat(mask)

def caculate_qkvo(query: torch.Tensor, # batch_size, num_heads, context_size, head_dim
                  key: torch.Tensor, # batch_size, num_kv_heads, context_size, head_dim
                  value: torch.Tensor, # batch_size, num_kv_heads, context_size, head_dim
                  sub_q_positions: list,
                  sub_kv_positions: list,
                  sub_critical_poses: list): 
    attn_outputs = []
    q_start = 0
    k_start = 0
    for i, q_pos in enumerate(sub_q_positions):
        q_len = len(q_pos)
        q_end = q_start + q_len
        
        k_pos = sub_kv_positions[i]
        k_len = len(k_pos)
        k_end = k_start + k_len
        q = query[:, :, q_start:q_end, :]
        k = key[:, :, k_start:k_end, :]
        v = value[:, :, k_start:k_end, :]
        batch_size, num_heads, context_size, head_dim = q.shape
        _, num_kv_heads, kv_len, _ = k.shape
        if num_kv_heads != num_heads:
            k = k[:, :, None, :, :].expand(1, num_kv_heads, num_heads // num_kv_heads, kv_len, head_dim)
            k = k.reshape(1, num_heads, kv_len, head_dim)
            v = v[:, :, None, :, :].expand(1, num_kv_heads, num_heads // num_kv_heads, kv_len, head_dim)
            v = v.reshape(1, num_heads, kv_len, head_dim)
        q_start = q_end
        k_start = k_end
        mask = create_mask_from_position_for_flashinfer(q_pos, k_pos, sub_critical_poses[i] if len(sub_critical_poses) != 0 else [])
        v = v.contiguous()
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=False,
        )
        attn_outputs.append(attn_output)

    return torch.cat(attn_outputs,dim=2)

def get_random_indices(n, num):
    if num > n:
        raise ValueError("num cannot be greater than n.")
    # 生成一个随机排列的 tensor，长度为 n
    indices = torch.randperm(n)[:num]
    return indices

def get_sorted_unique_indices(n, recompute_ratio):
    # 计算需要选择的下标数量
    num_indices = int(n * recompute_ratio)
    
    # 生成范围为 0 到 n-1 的所有下标
    indices = list(range(1, n))
    
    # 随机选择 num_indices 个不重复的下标
    selected_indices = random.sample(indices, num_indices)
    
    # 对选中的下标进行排序
    selected_indices.sort()
    
    return selected_indices

rotary_emb = RotaryEmbedding(head_dim, device='cuda:0')
position = torch.tensor([], dtype=torch.int32, device="cuda:0")
critical_position = torch.tensor([], dtype=torch.int32, device="cuda:0")
q_position = torch.tensor([], dtype=torch.int32, device="cuda:0")
critical_pos = torch.tensor([], dtype=torch.int32, device="cuda:0")
qo_indptr = torch.tensor([0], dtype=torch.int32, device="cuda:0")
paged_kv_indptr = torch.tensor([0], dtype=torch.int32, device="cuda:0")
attached_kv_indptr = torch.tensor([0], dtype=torch.int32, device="cuda:0")
sub_q_positions = []
sub_kv_positions = []
sub_critical_poses = []
sub_critical_positions = []
paged_kv_last_page_len = []
attached_kv_last_page_len = []
paged_num = 0
attached_page_num = 0

for mini_batch in range(batch_size):

    last_position = 0
    sub_position = torch.tensor([], dtype=torch.int32, device="cuda:0")
    chunk_num = random.randint(1, max_chunk_num)
    for chunk_index in range(chunk_num):
        # chunk_length = page_size
        chunk_length = random.randint(max_chunk_length // 2, max_chunk_length)
        sub_critical_position = get_sorted_unique_indices(chunk_length, recompute_ratio)
        sub_critical_position = last_position + torch.tensor(sub_critical_position, dtype=torch.int32, device="cuda:0")
        
        sub_position = torch.cat((sub_position, last_position + torch.arange(chunk_length, device="cuda:0").to(torch.int32)))
        last_position = sub_position[-1].item() + 1
        padding_chunk_length = (page_size - chunk_length % page_size if chunk_length % page_size != 0 else 0)
        sub_position = torch.cat((sub_position, inf * torch.ones(padding_chunk_length, device="cuda:0").to(torch.int32)))

    prefill_length = random.randint(max_chunk_length // 4, max_chunk_length // 2)
    # prefill_length = page_size
    
    sub_q_position = torch.arange(last_position,last_position+prefill_length, device="cuda:0").to(torch.int32)
    sub_position = torch.cat((sub_position, torch.arange(last_position,last_position+prefill_length, device="cuda:0").to(torch.int32)))
    sub_q_position = torch.cat((sub_critical_position, sub_q_position))
    q_position = torch.cat((q_position, sub_q_position))
    sub_critical_pos = torch.nonzero(torch.isin(sub_position, sub_critical_position)).squeeze()
    sub_critical_poses.append(sub_critical_pos.clone())
    sub_critical_pos += len(position)
    critical_position = torch.cat((critical_position, sub_critical_position))
    position = torch.cat((position, sub_position))
    critical_pos = torch.cat((critical_pos, sub_critical_pos))

    sub_q_positions.append(sub_q_position)
    sub_kv_positions.append(sub_position)
    sub_critical_positions.append(sub_critical_position)
    sub_paged_num = math.ceil(len(sub_position) / page_size)
    paged_num += sub_paged_num
    qo_indptr = torch.cat((qo_indptr, torch.tensor([qo_indptr[-1]+len(sub_q_position)], dtype=torch.int32, device="cuda:0")))
    paged_kv_indptr = torch.cat((paged_kv_indptr, torch.tensor([paged_kv_indptr[-1]+sub_paged_num], dtype=torch.int32, device="cuda:0")))
    paged_kv_last_page_len.append(len(sub_position) % page_size if len(sub_position) % page_size != 0 else page_size)
    

    cri_length = len(sub_critical_pos)
    sub_attached_page_num = math.ceil(sub_critical_pos.shape[0] / page_size)
    attached_page_num += sub_attached_page_num
    attached_kv_last_page_len.append(cri_length % page_size if cri_length % page_size != 0 else page_size)
    attached_kv_indptr = torch.cat((attached_kv_indptr, torch.tensor([attached_kv_indptr[-1]+sub_attached_page_num], dtype=torch.int32, device="cuda:0")))


attached_paged_kv_indices = torch.arange(max_num_pages-attached_page_num,max_num_pages, device="cuda:0").to(dtype=torch.int32)
kv_position=torch.tensor(position).to('cuda').to(torch.int32)
attached_position = torch.tensor(critical_position).to('cuda').to(torch.int32)
paged_kv_indices = get_random_indices(max_num_pages-attached_page_num, paged_num).to(torch.int32).to('cuda:0')
paged_kv_last_page_len = torch.tensor(paged_kv_last_page_len, dtype=torch.int32, device="cuda:0")
attached_kv_last_page_len = torch.tensor(attached_kv_last_page_len, dtype=torch.int32, device="cuda:0")
# k = torch.randn(batch_size,  num_kv_heads, len(position), head_dim).half().to('cuda:0')
# v = torch.randn(batch_size,  num_kv_heads, len(position), head_dim).half().to('cuda:0')
kv_cache_at_layer = torch.randn(
1, max_num_pages, 2, page_size, num_kv_heads, head_dim, dtype=torch.float16, device="cuda:0"
)
num_q = len(q_position)
q = torch.randn(1, num_q, num_qo_heads, head_dim).half().to('cuda:0')

# critical_kv_cache = kv_cache_at_layer.transpose(1, 2).reshape(1, 2, max_num_pages * page_size, num_kv_heads, head_dim)[:,:,critical_position]
# kv_cache_at_layer[:, attached_paged_kv_indices][0][0][:,:len(critical_position)].copy_(critical_kv_cache[0])

for i,c_pos in enumerate(critical_pos):
    attached_index = i // page_size
    attached_sub_index = i % page_size
    page_index = c_pos // page_size
    page_sub_index = c_pos % page_size
    kv_cache_at_layer[0,attached_paged_kv_indices[attached_index],:,attached_sub_index,:] = kv_cache_at_layer[0,paged_kv_indices[page_index],:,page_sub_index,:]
k = []
v = []
paged_num = 0
for i, k_pos in enumerate(sub_kv_positions):
    sub_paged_num = math.ceil(len(k_pos) / page_size)
    last_page_len = paged_kv_last_page_len[i]
    k.append(kv_cache_at_layer[0,paged_kv_indices[paged_num:paged_num+sub_paged_num-1],0,:,:,:].reshape(-1, num_kv_heads, head_dim))
    v.append(kv_cache_at_layer[0,paged_kv_indices[paged_num:paged_num+sub_paged_num-1],1,:,:,:].reshape(-1, num_kv_heads, head_dim))
    k.append(kv_cache_at_layer[0,paged_kv_indices[paged_num+sub_paged_num-1],0,:,:,:].reshape(-1, num_kv_heads, head_dim)[:last_page_len])
    v.append(kv_cache_at_layer[0,paged_kv_indices[paged_num+sub_paged_num-1],1,:,:,:].reshape(-1, num_kv_heads, head_dim)[:last_page_len])
    paged_num += sub_paged_num
k = torch.cat(k)
v = torch.cat(v)
assert len(k) == len(position)

attached_k = []
attached_v = []
attached_paged_num = 0
for i, sub_critical_position in enumerate(sub_critical_positions):
    sub_paged_num = math.ceil(len(sub_critical_position) / page_size)
    last_page_len = attached_kv_last_page_len[i]
    attached_k.append(kv_cache_at_layer[0,attached_paged_kv_indices[attached_paged_num:attached_paged_num+sub_paged_num-1],0,:,:,:].reshape(-1, num_kv_heads, head_dim))
    attached_v.append(kv_cache_at_layer[0,attached_paged_kv_indices[attached_paged_num:attached_paged_num+sub_paged_num-1],1,:,:,:].reshape(-1, num_kv_heads, head_dim))
    attached_k.append(kv_cache_at_layer[0,attached_paged_kv_indices[attached_paged_num+sub_paged_num-1],0,:,:,:].reshape(-1, num_kv_heads, head_dim)[:last_page_len])
    attached_v.append(kv_cache_at_layer[0,attached_paged_kv_indices[attached_paged_num+sub_paged_num-1],1,:,:,:].reshape(-1, num_kv_heads, head_dim)[:last_page_len])
    attached_paged_num += sub_paged_num
attached_k = torch.cat(attached_k)
attached_v= torch.cat(attached_v)
assert len(attached_k) == len(attached_position)
    





# cos, sin = rotary_emb(q.transpose(1, 2), torch.arange(len(q_position)).to(torch.int32).unsqueeze(0).to('cuda:0'))
cos, sin = rotary_emb(q.transpose(1, 2), q_position.unsqueeze(0).to('cuda:0'))
rope_q = apply_rotary_pos_emb(q.transpose(1, 2), cos, sin)
# cos, sin = rotary_emb(q.transpose(1, 2), torch.arange(len(kv_position)).to(torch.int32).unsqueeze(0).to('cuda:0'))
cos, sin = rotary_emb(k.unsqueeze(0).transpose(1,2), kv_position.unsqueeze(0).to('cuda:0'))
rope_k = apply_rotary_pos_emb(k.unsqueeze(0).transpose(1,2), cos, sin)
# standard_output = caculate_qkvo(q.transpose(1, 2), k.unsqueeze(0).transpose(1,2), v.unsqueeze(0).transpose(1,2), sub_q_positions, sub_kv_positions, sub_critical_poses, )

standard_output = caculate_qkvo(rope_q, rope_k, v.unsqueeze(0).transpose(1,2), sub_q_positions, sub_kv_positions, sub_critical_poses, )
standard_output = standard_output[0].transpose(0, 1).contiguous()



cos, sin = rotary_emb(attached_k.unsqueeze(0).transpose(1,2), attached_position.unsqueeze(0).to('cuda:0'))
rope_attached_k = apply_rotary_pos_emb(attached_k.unsqueeze(0).transpose(1,2), cos, sin)
# standard_output = caculate_qkvo(q.transpose(1, 2), k.unsqueeze(0).transpose(1,2), v.unsqueeze(0).transpose(1,2), sub_q_positions, sub_kv_positions, sub_critical_poses, )

standard_attached_output = caculate_qkvo(rope_q, rope_attached_k, attached_v.unsqueeze(0).transpose(1,2), sub_q_positions, sub_critical_positions, sub_critical_poses = [] )
standard_attached_output = standard_attached_output[0].transpose(0, 1).contiguous()
# print('standard_output')
# print(f'{standard_output}')
# flashinfer 
workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
    workspace_buffer, "NHD"
)

    # create mask

mask=create_mask_for_batch(sub_q_positions, sub_kv_positions, sub_critical_poses,)
mask=mask.flatten()

prefill_wrapper.plan(
    qo_indptr,
    paged_kv_indptr,
    paged_kv_indices,
    paged_kv_last_page_len,
    torch.tensor(batch_size,dtype=torch.int32).to('cuda:0'),
    torch.tensor(kv_position.shape[0], dtype=torch.int32).to('cuda:0'),
    num_qo_heads,
    num_kv_heads,
    head_dim,
    page_size,
    custom_mask=mask,
    pos_encoding_mode="ROPE_LLAMA",
)


flashinfer_output, lse = prefill_wrapper.run(q[0], kv_cache_at_layer[0], return_lse=True, q_position=q_position, kv_position=kv_position)

# flashinfer_output,lse = prefill_wrapper.run(q[0], kv_cache_at_layer[0], return_lse=True, q_position=q_position, kv_position=kv_position)
# print('flashinfer_output')
# print(f'{flashinfer_output}')
print(torch.allclose(flashinfer_output, standard_output, rtol=1e-2, atol=1e-2))
abs_difference = torch.abs(flashinfer_output-standard_output)
max_abs_difference = torch.max(abs_difference)
print(f'only rope, no attached page merge, max difference: {max_abs_difference}')
# attached_kv_indptr = torch.tensor([0,len(attached_paged_kv_indices)], dtype=torch.int32, device="cuda:0")
# attached_paged_kv_last_page_len=torch.tensor([len(critical_position) % page_size if len(critical_position) % page_size != 0 else page_size], dtype=torch.int32, device="cuda:0")
# attached_mask = create_mask_from_position_for_flashinfer(q_position, critical_position, [])
# attached_mask = attached_mask.flatten()
attached_mask=create_mask_for_batch(sub_q_positions, sub_critical_positions, [])
attached_mask=attached_mask.flatten()
prefill_wrapper.plan(
    qo_indptr,
    attached_kv_indptr,
    attached_paged_kv_indices,
    attached_kv_last_page_len,
    torch.tensor(batch_size,dtype=torch.int32).to('cuda:0'),
    torch.tensor(attached_position.shape[0], dtype=torch.int32).to('cuda:0'),
    num_qo_heads,
    num_kv_heads,
    head_dim,
    page_size,
    custom_mask=attached_mask,
    pos_encoding_mode="ROPE_LLAMA",
)
attached_flashinfer_output,attached_lse = prefill_wrapper.run(q[0], kv_cache_at_layer[0], return_lse=True, q_position=q_position, kv_position=critical_position)
abs_difference = torch.abs(standard_attached_output-attached_flashinfer_output)
max_abs_difference = torch.max(abs_difference)
print(f'attached: {max_abs_difference}')
entire_output,_ = merge_state(flashinfer_output, lse, attached_flashinfer_output, attached_lse)
cos, sin = rotary_emb(k.unsqueeze(0).transpose(1,2), kv_position.unsqueeze(0).to('cuda:0'))
rope_k = apply_rotary_pos_emb(k.unsqueeze(0).transpose(1,2), cos, sin)
merge_standard_output = caculate_qkvo(rope_q, rope_k, v.unsqueeze(0).transpose(1,2), sub_q_positions, sub_kv_positions, [])
merge_standard_output = merge_standard_output[0].transpose(0, 1).contiguous()
abs_difference = torch.abs(merge_standard_output-entire_output)
max_abs_difference = torch.max(abs_difference)
print(f'rope, position max difference: {max_abs_difference}')
print(torch.allclose(entire_output, merge_standard_output, rtol=1e-2, atol=1e-2))
print(f"prompt length: {len(position)}")
print('')