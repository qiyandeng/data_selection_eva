#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在会话式 JSONL 文件上计算 NLL（负对数似然）。

输入文件的每一行都是下列格式的 JSON 对象::
    {
        "dataset": "...",
        "id": "...",
        "messages": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."},
            ...
        ]
    }

对每段对话，我们将所有 user 消息拼接成 prompt，
response 取最后一条 assistant 消息。NLL 在给定 prompt
的条件下，对 response 的 token 计算。
脚本支持通过 ``torch.distributed`` 进行分布式执行。
"""

import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="针对会话式 JSONL 的分布式 NLL 计算"
    )
    parser.add_argument("--model_name", type=str, required=True,
                        help="因果语言模型的名称或路径")
    parser.add_argument("--input_path", type=Path, required=True,
                        help="每行都是一次对话的 JSONL 文件")
    parser.add_argument("--output_prefix", type=str, required=True,
                        help="输出 JSONL 文件的前缀")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="每次前向计算的样本数")
    parser.add_argument("--max_length", type=int, default=1024,
                        help="分词时的最大序列长度")
    return parser.parse_args()


def init_distributed():
    """初始化分布式后端，返回 rank 与 world_size。"""
    dist.init_process_group(backend="nccl")
    return dist.get_rank(), dist.get_world_size()


# ---------------------------------------------------------------------------
# NLL 计算函数
# ---------------------------------------------------------------------------

def compute_nll(prompts, responses, model, tokenizer, device, max_len):
    """批量计算样本的平均 NLL。

    参数
    ------
    prompts: List[str]
        每个样本的 user 消息串联结果。
    responses: List[str]
        每个样本对应的 assistant 回复（仅最后一条）。
    model: transformers.PreTrainedModel
        用于评分的因果语言模型。
    tokenizer: transformers.PreTrainedTokenizer
        与 ``model`` 对应的分词器。
    device: torch.device 或 ``str``
        ``model`` 所在的设备。
    max_len: int
        分词时用于截断/填充的最大长度。

    返回
    ------
    nlls_batch: List[List[float]]
        每个样本的平均 NLL，外层 list 用于与其他脚本兼容。
    tokencounts_batch: List[List[int]]
        每个样本的回复 token 数，同样使用 list 包裹。
    """
    nlls_batch, tokencounts_batch = [], []

    for prompt, resp in zip(prompts, responses):
        # 对完整序列 ``prompt + resp`` 进行分词，得到 input_ids 与 attention_mask。
        enc = tokenizer(
            [prompt + resp],
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(device)

        # 只需 logits，为提升效率关闭梯度计算。
        with torch.no_grad():
            logits = model(**enc).logits  # 形状: [1, seq_len, vocab_size]

        # 在词表维度上将 logits 转为对数概率。
        log_probs = torch.log_softmax(logits, dim=-1)

        # 计算 prompt 的 token 数，之后计算 response NLL 时将其忽略。
        p_len = len(tokenizer(prompt, add_special_tokens=False).input_ids)

        # 使用 ``labels`` 并设定忽略索引 (-100)，仅在 response 部分计算损失。
        labels = enc.input_ids.clone()
        labels[:, :p_len] = -100

        # 取出每个位置正确下一个 token 的对数概率。
        seq_logp = log_probs.gather(2, enc.input_ids.unsqueeze(-1)).squeeze(-1)

        # 掩码 ``mask`` 在 response 区域为 1，其余为 0，确保 prompt 不计入得分。
        mask = (enc.attention_mask == 1) & (labels != -100)

        # 累加 response token 的对数概率并统计数量。
        sum_logp = (seq_logp * mask).sum()
        n_tok = mask.sum()

        # 将总对数概率除以 token 数得到平均 NLL。
        nll = (-sum_logp / n_tok).item()
        nlls_batch.append([nll])
        tokencounts_batch.append([n_tok.item()])

    return nlls_batch, tokencounts_batch


# ---------------------------------------------------------------------------
# 主程序入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    # 初始化分布式环境，每个进程处理数据的一部分。
    rank, world_size = init_distributed()
    device = f"cuda:{rank}"

    # 加载分词器和模型。若缺少 pad token，则添加以保证 `pad_sequence` 正常工作。
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    if len(tokenizer) != model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    model.to(device).eval()

    # 每个进程都会写出自己的结果文件 ``_rank{rank}.jsonl``。
    out_path = Path(f"{args.output_prefix}_rank{rank}.jsonl")

    buf_prompts, buf_resps, buf_entries = [], [], []

    with args.input_path.open("r", encoding="utf-8") as fin, \
         out_path.open("w", encoding="utf-8") as fout:

        for idx, raw in enumerate(fin):
            # 按行号对数据进行分片，不属于本进程的行会被跳过。
            if idx % world_size != rank:
                continue

            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                # 格式不正确的行会被忽略。
                continue

            # --------------------- 解析对话消息 -----------------
            msgs = entry.get("messages", [])
            user_parts = [m["content"] for m in msgs if m.get("role") == "user"]
            asst_parts = [m["content"] for m in msgs if m.get("role") == "assistant"]
            if not user_parts or not asst_parts:
                # 任一角色缺失则跳过该样本。
                continue

            prompt = "\n".join(user_parts)  # 拼接 user 消息
            resp = asst_parts[-1]            # 取最后一条 assistant 消息
            # -------------------------------------------------------------------

            buf_prompts.append(prompt)
            buf_resps.append(resp)
            buf_entries.append(entry)

            if len(buf_prompts) >= args.batch_size:
                nlls, toks = compute_nll(
                    buf_prompts, buf_resps, model, tokenizer, device, args.max_length
                )
                for e, nll_val, tc in zip(buf_entries, nlls, toks):
                    e["NLL"] = nll_val           # [nll_per_token]
                    e["TokenCount"] = tc         # [token_count]
                    fout.write(json.dumps(e, ensure_ascii=False) + "\n")
                buf_prompts.clear(); buf_resps.clear(); buf_entries.clear()

        # 处理缓冲区中剩余的样本。
        if buf_prompts:
            nlls, toks = compute_nll(
                buf_prompts, buf_resps, model, tokenizer, device, args.max_length
            )
            for e, nll_val, tc in zip(buf_entries, nlls, toks):
                e["NLL"] = nll_val
                e["TokenCount"] = tc
                fout.write(json.dumps(e, ensure_ascii=False) + "\n")

    # 等待所有进程结束后再退出。
    dist.barrier()
    dist.destroy_process_group()

    if rank == 0:
        print(f"All done -> {args.output_prefix}_rank*.jsonl")