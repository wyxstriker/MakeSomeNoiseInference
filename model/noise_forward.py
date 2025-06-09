# from model.utils import *
from model.kv_cache import initialize_past_key_values
import torch

@torch.no_grad()
def noise_forward(input_ids, model, tokenizer, max_new_tokens, temperature=0.0, do_sample=False):
    accept_length_list = []
    step = 0
    start_len = input_ids.size(1)
    cur_seq_len = start_len - 1
    verify_input_ids = input_ids[:, :-1]
    input_ids = input_ids[:, -1:]
    
    (
        past_key_values,
        past_key_values_data,
        current_length_data,
    ) = initialize_past_key_values(model)
        
    model(input_ids=verify_input_ids, past_key_values=past_key_values)
    
    K = 10
    deep = 5
    pad_len = 6
    tree_template = [[K], [K, K], [K, K, K], [K, K, K, K], [K, K, K, K, K], [0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2], [3], [0, 3], [4], [0, 4], [2, 0], [0, 5], [0, 0, 1], [5], [0, 6], [6], [0, 7], [0, 1, 0], [1, 1], [7], [0, 8], [0, 0, 2], [3, 0], [0, 9], [8], [9], [1, 0, 0], [0, 2, 0], [1, 2], [0, 0, 3], [4, 0], [2, 1], [0, 0, 4], [0, 0, 5], [0, 0, 0, 0], [0, 1, 1], [0, 0, 6], [0, 3, 0], [5, 0], [1, 3], [0, 0, 7], [0, 0, 8], [0, 0, 9], [6, 0], [0, 4, 0], [1, 4], [7, 0], [0, 1, 2], [2, 0, 0], [3, 1], [2, 2], [8, 0], [0, 5, 0], [1, 5], [1, 0, 1], [0, 2, 1], [9, 0], [0, 6, 0], [0, 0, 0, 1], [1, 6], [0, 7, 0]]
    topK_candi = torch.ones([deep, K+1], dtype=torch.long, device=torch.device('cuda:0')) * -1
    
    def pad_topk(candi_list, pool, k=K):
        candi_list = torch.where(candi_list!=-1, candi_list, pool[:, torch.randint(pool.size(1), candi_list.size())])[0]
        return candi_list
    
    def prepare_data(template, input_ids):
        candi_attention_ids = torch.zeros([input_ids.size(0), len(template)+1, len(template)+1], dtype=torch.long)
        candi_positon_ids = torch.zeros([input_ids.size(0), len(template)+1], dtype=torch.long)
        # init input_ids
        candi_attention_ids[:, :, 0] = 1
        candi_positon_ids[:, 0] = 0
        search_path = []
        template_index = []
        # pad
        for candi_i in range(len(template)):
            tree_deep = len(template[candi_i])
            tree_index = template[candi_i][-1]

            template_index.append((tree_deep-1)*(K+1) + tree_index)
            
            candi_positon_ids[:, candi_i+1] = tree_deep
            cur_path = [0]
            candi_attention_ids[:, candi_i+1, candi_i+1] = 1
            for candi_j in range(candi_i):
                if template[candi_j] == template[candi_i][:len(template[candi_j])]:
                    candi_attention_ids[:, candi_i+1, candi_j+1] = 1
                    cur_path.append(candi_j+1)
            cur_path.append(candi_i+1)
            for sub_i in range(1, len(cur_path)):
                if cur_path[:sub_i] in search_path:
                    search_path.remove(cur_path[:sub_i])
            search_path.append(cur_path)
        search_path = torch.tensor([path + [0] * (pad_len-len(path)) for path in search_path], dtype=torch.long, device=torch.device('cuda:0'))
        template_index = torch.tensor(template_index, dtype=torch.long, device=torch.device('cuda:0'))
        
        return candi_attention_ids.cuda(), candi_positon_ids.cuda(), search_path, template_index

    def prepare_data_input_ids(input_ids, candi_list, template_index):
        input_ids =  torch.cat([input_ids, candi_list.view(-1)[template_index].view(input_ids.size(1), -1)], dim=-1)
        return input_ids
    
    eos_token_id = torch.tensor(tokenizer.eos_token_id, dtype=torch.long, device=torch.device('cuda:0'))
        
    topK_candi = pad_topk(topK_candi, verify_input_ids)
    
    attention_mask, position_ids, search_path, template_index = prepare_data(tree_template, input_ids)

    for i in range(max_new_tokens):
        topK_candi = pad_topk(topK_candi, verify_input_ids)

        input_ids = prepare_data_input_ids(input_ids, topK_candi, template_index)
        
        merge_attention_mask = torch.cat([torch.ones([input_ids.size(0), len(tree_template)+1, verify_input_ids.size(1)], dtype=torch.long, device=torch.device('cuda:0')), attention_mask], dim=-1)
        merge_positon_ids = position_ids + verify_input_ids.size(1)

        outputs = model(input_ids=input_ids, attention_mask=merge_attention_mask.unsqueeze(1), position_ids=merge_positon_ids, past_key_values=past_key_values)

        model_res = torch.argmax(outputs.logits, dim=-1)
        all_input_path = input_ids[0][search_path]
        all_output_path = model_res[0][search_path]
            
        reward = torch.cumprod(all_input_path[:, 1:].eq(all_output_path[:, :-1]), dim=-1).sum(dim=-1)
        best_reward = reward.max()
        accept_len = 1 + best_reward

        # select best path
        best_path_index = torch.argmax(reward, dim=-1).to(torch.long)
        index_path = search_path[:, :accept_len]
        index_path = index_path[best_path_index]
        best_path_input = torch.index_select(input_ids, index=index_path, dim=1)
        
        # KV Cache
        tgt = past_key_values_data[..., verify_input_ids.size(1)+index_path, :]
        dst = past_key_values_data[..., verify_input_ids.size(1) : verify_input_ids.size(1) + tgt.shape[-2], :]
        dst.copy_(tgt, non_blocking=True)
        current_length_data.fill_(verify_input_ids.size(1) + tgt.shape[-2])
        
        # update input ids
        verify_input_ids = torch.cat([verify_input_ids, best_path_input], dim=-1)
        
        if (best_path_input == eos_token_id).any() or verify_input_ids.size(1) - start_len >= max_new_tokens:
            break

        topK_candi = torch.ones([deep, K], dtype=torch.long, device=torch.device('cuda:0')) * -1
        topK_path = torch.topk(outputs.logits[:, search_path[best_path_index][accept_len:]], k=K, dim=-1)[1][0]
        topK_candi[:topK_path.size(0), :K] = topK_path
        
        input_ids = model_res[:, search_path[best_path_index][accept_len-1]].unsqueeze(-1).cuda()
        
        def get_retrieval(pool_input_ids, ngram_size=3, num_pred_tokens=5):
            windows = pool_input_ids[:, :-ngram_size].unfold(dimension=1, size=ngram_size, step=1)
            ngram_tensor = pool_input_ids[:, -ngram_size:]
            
            matches = torch.cumprod((windows == ngram_tensor).flip(-1), dim=-1).sum(-1)
            match_indices = torch.argmax(matches, dim=-1)[0]
            
            start_idx = match_indices + ngram_size
            end_idx = start_idx + num_pred_tokens
            return pool_input_ids[0][start_idx:end_idx].tolist()
        
        retrieval_list = get_retrieval(torch.cat([verify_input_ids.cuda(), input_ids.cuda()], dim=-1))
        retrieval_list = torch.tensor(retrieval_list + [-1] * (pad_len-1-len(retrieval_list)), dtype=torch.long, device=torch.device('cuda:0')).view(-1, 1)
        
        topK_candi = torch.cat([topK_candi, retrieval_list], dim=-1)
        step += 1

        accept_length_list.append(accept_len.item())
        
                
    return verify_input_ids, verify_input_ids.size(1) - start_len, step, accept_length_list