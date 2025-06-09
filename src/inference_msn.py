from transformers import  AutoTokenizer, GenerationConfig
import torch
from argparse import ArgumentParser
import sys
sys.path.append("./")
from model.modeling_llama_kv import LlamaForCausalLM
from model.noise_forward import noise_forward

if __name__ == "__main__":
    # paser
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    args = parser.parse_args()

    # prepare model
    model_path = args.model_path
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # prepare data
    case_text = "Give me some advices about how to write an academic paper?"
    input_ids = tokenizer.apply_chat_template([{"role": "user",
                                                "content": case_text}], add_generation_prompt=True, return_tensors="pt")
    
    # jacobi decoding
    spec_res_ids, new_tokens, forward_steps, accpet_list = noise_forward(input_ids.cuda(), model, tokenizer, args.max_new_tokens)
    
    # auto-regressive decoding if needed
    # gold_res_ids = model.generate(input_ids.cuda(), generation_config=GenerationConfig(max_new_tokens=args.max_new_tokens, 
    #                                                                                    do_sample=False, 
    #                                                                                    eos_token_id=tokenizer.eos_token_id, 
    #                                                                                    pad_token_id=tokenizer.pad_token_id))
    
    print("msn output")
    print(tokenizer.decode(spec_res_ids[0]))
    print("#MTA")
    print(new_tokens/forward_steps)
    print("Accepted Length List")
    print(accpet_list)