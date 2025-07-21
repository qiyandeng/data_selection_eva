import warnings

# 忽略特定的 FutureWarning
warnings.filterwarnings(
    "ignore",
    message="`torch.utils._pytree._register_pytree_node` is deprecated. Please use `torch.utils._pytree.register_pytree_node` instead.",
    category=FutureWarning
)

import json
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import argparse

# 步骤 1: 读取并解析 JSONL 文件，返回完整的JSON对象和组合文本
def read_jsonl(file_path):
    """
    读取 JSONL 文件，将每一行解析为 JSON 对象，
    如果存在 "messages" 字段则将其中每条记录的 content 连接起来；
    否则按照默认规则（instruction, input, output）组合文本，
    返回 JSON 对象列表和对应的文本列表。
    """
    data_objects = []  # 存储解析后的 JSON 对象
    texts = []         # 存储组合后的文本
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                json_obj = json.loads(line)
            except json.JSONDecodeError:
                # 如果某一行不是有效的 JSON，跳过该行
                continue
            # 判断是否存在 "messages" 字段
            if "messages" in json_obj:
                combined_text = " ".join(msg.get("content", "") for msg in json_obj.get("messages", []))
            else:
                # 默认方式：合并 instruction, input 和 output 字段
                instruction = json_obj.get("instruction", "")
                input_data = json_obj.get("input", "")
                output = json_obj.get("output", "")
                combined_text = f"{instruction} {input_data} {output}"
                
            data_objects.append(json_obj)
            texts.append(combined_text)
    return data_objects, texts

# 步骤 2: 加载自定义词向量
def load_embeddings(embedding_file):
    word_vectors = KeyedVectors.load_word2vec_format(embedding_file, binary=False, unicode_errors='ignore')
    return word_vectors

# 步骤 3: 定义线性映射层
class EmbeddingMapper(nn.Module):
    def __init__(self, input_dim=300, output_dim=768):
        super(EmbeddingMapper, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


# 步骤 5: 保存修改后的模型
def save_model(model, tokenizer, save_path):
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"修改后的BERT模型已保存到 {save_path}")

# 步骤 6: 获取句子向量
def get_sentence_embeddings(texts, tokenizer, model, device, batch_size=16):
    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="获取句子向量"):
            batch_texts = texts[i:i+batch_size]
            encoded_input = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,  # 明确设置最大长度为512
                return_tensors='pt'
            ).to(device)
            
            # 检查序列长度
            seq_lengths = (encoded_input['input_ids'] != tokenizer.pad_token_id).sum(dim=1)
            print(f"Batch {i//batch_size +1}: Max sequence length = {seq_lengths.max().item()}")
            
            outputs = model(**encoded_input)
            # 使用 [CLS] token 的输出作为句子向量
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)
    all_embeddings = np.vstack(all_embeddings)
    return all_embeddings

# 步骤 7: 计算梯度余弦（在此上下文中，我们使用余弦相似度）
def compute_gc_scores_gpu(data_embeddings, target_embeddings, device):
    # 转换为PyTorch张量并迁移到GPU
    data_tensor = torch.tensor(data_embeddings, dtype=torch.float32).to(device)
    target_tensor = torch.tensor(target_embeddings, dtype=torch.float32).to(device)

    # 归一化张量
    data_norm = data_tensor.norm(dim=1, keepdim=True)
    target_norm = target_tensor.norm(dim=1, keepdim=True)

    # 计算余弦相似度
    similarity_matrix = torch.mm(data_tensor, target_tensor.t()) / (data_norm * target_norm.t())

    # 将相似度矩阵转移回CPU并转换为NumPy数组
    similarity_matrix = similarity_matrix.cpu().numpy()

    # 对每条 data 的相似度求和
    gc_scores = similarity_matrix.sum(axis=1)
    return gc_scores

# 步骤 8: 筛选前 n 名数据，并保持原始JSON结构
def select_top_n(data_objects, gc_scores, n=10):
    df = pd.DataFrame({
        'index': range(len(data_objects)),
        'gc_score': gc_scores
    })
    df_sorted = df.sort_values(by='gc_score', ascending=False)
    top_n_indices = df_sorted.head(n)['index'].tolist()
    top_n_data = [data_objects[idx] for idx in top_n_indices]
    return top_n_data

def main(args):
    # 通过命令行参数传入各个文件及模型路径
    data_jsonl_file = args.data_jsonl_file
    target_jsonl_file = args.target_jsonl_file
    bert_model_path = args.bert_model_path
    modified_bert_model_path = args.modified_bert_model_path
    top_n = args.top_n
    output_file = args.output

    # 步骤 0: 设备选择（强制使用 cuda:1）
    gpu_id = 1
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")


    # 步骤 1: 读取文本数据
    print("读取 JSONL 文件...")
    data_objects, data_texts = read_jsonl(data_jsonl_file)
    target_objects, target_texts = read_jsonl(target_jsonl_file)
    print(f"{data_jsonl_file} 提取了 {len(data_texts)} 条文本数据。")
    print(f"{target_jsonl_file} 提取了 {len(target_texts)} 条文本数据。")

    # 步骤 2: 加载自定义词向量
    print("加载自定义词向量...")
    word_vectors = load_embeddings(embedding_file)
    print(f"词向量库大小: {len(word_vectors.key_to_index)}")

    # 步骤 3: 加载 BERT 模型和分词器
    print("加载 BERT 模型和分词器...")
    tokenizer = BertTokenizer.from_pretrained(bert_model_path)
    model = BertModel.from_pretrained(bert_model_path).to(device)
    model.eval()
    print("BERT 模型加载完成。")

    # 步骤 4: 定义并初始化线性映射层
    print("初始化线性映射层...")
    embedding_mapper = EmbeddingMapper().to(device)
    # 注意：这里可根据需要训练映射层，本示例直接使用未训练的映射层

    # 步骤 6: 保存修改后的 BERT 模型
    print("保存修改后的 BERT 模型...")
    save_model(model, tokenizer, modified_bert_model_path)

    # 步骤 7: 获取句子向量
    print(f"获取 {data_jsonl_file} 的句子向量...")
    data_embeddings = get_sentence_embeddings(data_texts, tokenizer, model, device)
    print(f"{data_jsonl_file} 的句子向量形状: {data_embeddings.shape}")

    print(f"获取 {target_jsonl_file} 的句子向量...")
    target_embeddings = get_sentence_embeddings(target_texts, tokenizer, model, device)
    print(f"{target_jsonl_file} 的句子向量形状: {target_embeddings.shape}")

    # 步骤 8: 计算梯度余弦（余弦相似度）分数
    print("计算梯度余弦（GC）分数...")
    gc_scores = compute_gc_scores_gpu(data_embeddings, target_embeddings, device)
    print("GC 分数计算完成。")

    # 步骤 9: 筛选前 n 名数据
    print(f"筛选出 {data_jsonl_file} 中排名前 {top_n} 的数据...")
    top_n_data = select_top_n(data_objects, gc_scores, n=top_n)
    print("筛选完成。")

    # 步骤 10: 保存筛选结果，保持原始 JSON 结构
    # output_file = f'top_{top_n}_data.jsonl'
    print(f"保存筛选结果到 {output_file} ...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in top_n_data:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')
    print(f"前 {top_n} 名的数据已保存到 {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="基于 BERT 模型与自定义词向量进行数据筛选")
    
    # 添加命令行参数，使得路径等均可由命令行指定
    parser.add_argument('--data_jsonl_file', type=str,
                        default='/mnt/petrelfs/zhangchi/wjc/RDS/all_data.jsonl',
                        help="原始数据的 JSONL 文件路径")
    parser.add_argument('--target_jsonl_file', type=str,
                        default='/mnt/petrelfs/zhangchi/wjc/RDS/target_data.jsonl',
                        help="目标数据的 JSONL 文件路径")
    parser.add_argument('--bert_model_path', type=str,
                        default='bert-base-uncased',
                        help="BERT 模型或本地模型的路径")
    parser.add_argument('--modified_bert_model_path', type=str,
                        default='bert-base-uncased-custom',
                        help="修改后的 BERT 模型保存路径")
    parser.add_argument('--top_n', type=int,
                        default=15000,
                        help="选取前 n 名数据 (默认 15000)")
    parser.add_argument('--output', type=str,
                        default="/fs-computility/llmit_d/shared/zhangchi/wjc/RDS/data_select/output.jsonl",
                        help="输出")
    
    args = parser.parse_args()
    main(args)
