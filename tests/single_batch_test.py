import torch
import flashinfer
from  flashinfer.cascade import *
import math
import torch.nn as nn
num_qo_heads = 32
num_kv_heads = 8
head_dim = 128
max_num_pages = 12
page_size = 8
inf = 32768
batch_size = 1
torch.manual_seed(42)

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

def caculate_qkvo(query: torch.Tensor, # batch_size, num_heads, context_size, head_dim
                  key: torch.Tensor, # batch_size, num_kv_heads, context_size, head_dim
                  value: torch.Tensor, # batch_size, num_kv_heads, context_size, head_dim
                  q_position: list,
                  kv_position: list,
                  critical_pos: list,): 
    batch_size, num_heads, context_size, head_dim = query.shape
    _, num_kv_heads, kv_len, _ = key.shape
    if num_kv_heads != num_heads:
        key = key[:, :, None, :, :].expand(batch_size, num_kv_heads, num_heads // num_kv_heads, kv_len, head_dim)
        key = key.reshape(batch_size, num_heads, kv_len, head_dim)
        value = value[:, :, None, :, :].expand(batch_size, num_kv_heads, num_heads // num_kv_heads, kv_len, head_dim)
        value = value.reshape(batch_size, num_heads, kv_len, head_dim)
    key = key.transpose(-1, -2)
    attn_weights = torch.matmul(query, key)
    mask = create_mask_from_position_for_standard(q_position, kv_position, critical_pos)
    attn_weights += mask #  batch_size, num_heads, context_size, context_size
    attn_weights /= math.sqrt(head_dim)
    attn_weights = nn.functional.softmax(attn_weights, dim = -1, dtype = query.dtype)
    # print(f'attn_weights: {attn_weights}')
    attn_outputs = torch.matmul(attn_weights, value)
    return attn_outputs



critical_position = torch.tensor([6,9,12,14], dtype=torch.int32, device="cuda:0")
q_position = torch.tensor([6,9,12,14,15,16,17,18], dtype=torch.int32, device="cuda:0")
position=[0,1,2,3,4,inf,inf,inf,5,6,7,8,9,10,11,12,13,14,inf,inf,inf,inf,inf,inf,15,16,17,18,inf,inf,inf,inf]
attached_paged_kv_indices = torch.tensor([10], dtype=torch.int32, device="cuda:0")
kv_position=torch.tensor(position).to('cuda')
paged_kv_indices = torch.tensor([8,0,3,4], dtype=torch.int32, device="cuda:0")
paged_kv_last_page_len=torch.tensor([len(position) % page_size if len(position) % page_size != 0 else page_size], dtype=torch.int32, device="cuda:0")
# k = torch.randn(batch_size,  num_kv_heads, len(position), head_dim).half().to('cuda:0')
# v = torch.randn(batch_size,  num_kv_heads, len(position), head_dim).half().to('cuda:0')
kv_cache_at_layer = torch.randn(
    1, max_num_pages, 2, page_size, num_kv_heads, head_dim, dtype=torch.float16, device="cuda:0"
)
# critical_kv_cache = kv_cache_at_layer.transpose(1, 2).reshape(1, 2, max_num_pages * page_size, num_kv_heads, head_dim)[:,:,critical_position]
# kv_cache_at_layer[:, attached_paged_kv_indices][0][0][:,:len(critical_position)].copy_(critical_kv_cache[0])

kv_cache_at_layer[0,10,:,0,:] = kv_cache_at_layer[0,0,:,1,:]
kv_cache_at_layer[0,10,:,1,:] = kv_cache_at_layer[0,0,:,4,:]
kv_cache_at_layer[0,10,:,2,:] = kv_cache_at_layer[0,0,:,7,:]
kv_cache_at_layer[0,10,:,3,:] = kv_cache_at_layer[0,3,:,1,:]


k = kv_cache_at_layer[0,paged_kv_indices,0,:,:,:].reshape(-1, num_kv_heads, head_dim)[:len(position)]
v = kv_cache_at_layer[0,paged_kv_indices,1,:,:,:].reshape(-1, num_kv_heads, head_dim)[:len(position)]

critical_pos = []
q_pos = []

num_q = len(q_position)
q = torch.randn(batch_size, num_q, num_qo_heads, head_dim).half().to('cuda:0')
for pos in critical_position:
    subscript = position.index(pos)
    critical_pos.append(subscript)

for pos in q_position:
    subscript = position.index(pos)
    q_pos.append(subscript)

standard_output = caculate_qkvo(q.transpose(1,2), k.unsqueeze(0).transpose(1,2), v.unsqueeze(0).transpose(1,2), q_position, kv_position, critical_pos)
standard_output = standard_output[0].transpose(0, 1).contiguous()
print('standard_output')
print(f'{standard_output}')


# flashinfer 
workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
    workspace_buffer, "NHD"
)

# create mask

mask=create_mask_from_position_for_flashinfer(q_position, kv_position, critical_pos)
mask=mask.flatten()
qo_indptr = torch.tensor([0,len(q_position)], dtype=torch.int32, device="cuda:0")
paged_kv_indptr = torch.tensor([0,len(paged_kv_indices)], dtype=torch.int32, device="cuda:0")
prefill_wrapper.plan(
    qo_indptr,
    paged_kv_indptr,
    paged_kv_indices,
    paged_kv_last_page_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    page_size,
    custom_mask=mask
)



flashinfer_output,lse = prefill_wrapper.run(q[0], kv_cache_at_layer[0], return_lse=True)
print('flashinfer_output')
print(f'{flashinfer_output}')
print(torch.allclose(flashinfer_output, standard_output, rtol=1e-2, atol=1e-2))
abs_difference = torch.abs(flashinfer_output-standard_output)
max_abs_difference = torch.max(abs_difference)
print(f'max difference: {max_abs_difference}')


# attached kvcache


attached_kv_indptr = torch.tensor([0,len(attached_paged_kv_indices)], dtype=torch.int32, device="cuda:0")
attached_paged_kv_last_page_len=torch.tensor([len(critical_position) % page_size if len(critical_position) % page_size != 0 else page_size], dtype=torch.int32, device="cuda:0")
attached_mask = create_mask_from_position_for_flashinfer(q_position, critical_position, [])
attached_mask = attached_mask.flatten()
prefill_wrapper.plan(
    qo_indptr,
    attached_kv_indptr,
    attached_paged_kv_indices,
    attached_paged_kv_last_page_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    page_size,
    custom_mask=attached_mask
)

attached_flashinfer_output,attached_lse = prefill_wrapper.run(q[0], kv_cache_at_layer[0], return_lse=True)

entire_output,_ = merge_state(flashinfer_output, lse, attached_flashinfer_output, attached_lse)

merge_standard_output = caculate_qkvo(q.transpose(1,2), k.unsqueeze(0).transpose(1,2), v.unsqueeze(0).transpose(1,2), q_position, kv_position, critical_pos=[])
merge_standard_output = merge_standard_output[0].transpose(0, 1).contiguous()
print(torch.allclose(entire_output, merge_standard_output, rtol=1e-3, atol=1e-3))