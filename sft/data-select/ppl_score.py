#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Distributed batch PPL + TokenCount computation for JSONL
where each line is:
{
  "dataset": "...",
  "id": "...",
  "messages": [
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "..."},
      ...
  ]
}
Prompt = 所有 user-侧 content 的串联
Response = 与之配对的 **最后一个** assistant content
"""
import argparse, json
from pathlib import Path
import torch, torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM

def parse_args():
    parser = argparse.ArgumentParser(
        description="Distributed PPL/TokenCount for JSONL(conversation style)"
    )
    parser.add_argument("--model_name",  type=str,  required=True)
    parser.add_argument("--input_path",  type=Path, required=True)
    parser.add_argument("--output_prefix", type=str, required=True)
    parser.add_argument("--batch_size",   type=int, default=8)
    parser.add_argument("--max_length",   type=int, default=1024)
    return parser.parse_args()

def init_distributed():
    dist.init_process_group(backend="nccl")
    return dist.get_rank(), dist.get_world_size()

# ---------- 与原脚本保持一致的 compute_metrics() 略 ---------- #

def compute_metrics(prompts, responses, model, tokenizer, device, max_len):
    ppls_batch, tokencounts_batch = [], []
    for prompt, resp in zip(prompts, responses):
        enc = tokenizer(
            [prompt + resp], padding=True, truncation=True,
            max_length=max_len, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            logits = model(**enc).logits          # [1, L, V]
        log_probs = torch.log_softmax(logits, -1)

        p_len = len(tokenizer(prompt, add_special_tokens=False).input_ids)
        labels = enc.input_ids.clone()
        labels[:, :p_len] = -100
        seq_logp = log_probs.gather(2, enc.input_ids.unsqueeze(-1)).squeeze(-1)
        mask = (enc.attention_mask == 1) & (labels != -100)
        sum_logp, n_tok = (seq_logp * mask).sum(), mask.sum()
        ppl = torch.exp(-sum_logp / n_tok).item()
        ppls_batch.append([ppl])           # 仍以 list 形式保持兼容
        tokencounts_batch.append([n_tok.item()])
    return ppls_batch, tokencounts_batch

# -------------- 主流程 ---------------- #

if __name__ == "__main__":
    args      = parse_args()
    rank, ws  = init_distributed()
    device    = f"cuda:{rank}"

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.add_special_tokens({'pad_token': '[PAD]'})
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    if len(tok) != model.config.vocab_size:
        model.resize_token_embeddings(len(tok))
    model.to(device).eval()

    out_path = Path(f"{args.output_prefix}_rank{rank}.jsonl")

    buf_prompts, buf_resps, buf_entries = [], [], []
    with args.input_path.open("r", encoding="utf-8") as fin, \
         out_path.open("w", encoding="utf-8") as fout:

        for idx, raw in enumerate(fin):
            if idx % ws != rank:   # 分片
                continue
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                continue

            # === NEW 1 ===  解析 messages ------------------------------------
            msgs = entry.get("messages", [])
            user_parts  = [m["content"] for m in msgs if m["role"] == "user"]
            asst_parts  = [m["content"] for m in msgs if m["role"] == "assistant"]
            if not user_parts or not asst_parts:
                continue
            prompt, resp = "\n".join(user_parts), asst_parts[-1]
            # -----------------------------------------------------------------

            buf_prompts.append(prompt)
            buf_resps.append(resp)
            buf_entries.append(entry)

            if len(buf_prompts) >= args.batch_size:
                ppls, toks = compute_metrics(buf_prompts, buf_resps,
                                              model, tok, device, args.max_length)
                # === NEW 2 ===  写回单条 response 情况 -----------------------
                for e, ppl, tc in zip(buf_entries, ppls, toks):
                    e["PPL"]        = ppl           # [ppl_val]
                    e["TokenCount"] = tc            # [tok_cnt]
                    fout.write(json.dumps(e, ensure_ascii=False) + "\n")
                buf_prompts.clear(); buf_resps.clear(); buf_entries.clear()

        # 处理残留
        if buf_prompts:
            ppls, toks = compute_metrics(buf_prompts, buf_resps,
                                          model, tok, device, args.max_length)
            for e, ppl, tc in zip(buf_entries, ppls, toks):
                e["PPL"] = ppl; e["TokenCount"] = tc
                fout.write(json.dumps(e, ensure_ascii=False) + "\n")

    dist.barrier(); dist.destroy_process_group()
    if rank == 0:
        print(f"All done -> {args.output_prefix}_rank*.jsonl")
