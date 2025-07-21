import math
import json
from collections import Counter
import re
import os

class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        """
        BM25算法的构造器
        :param docs: 分词后的文档列表，每个文档是一个包含词汇的列表
        :param k1: BM25算法中的调节参数k1
        :param b: BM25算法中的调节参数b
        """
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.doc_len = [len(doc) for doc in docs]  # 计算每个文档的长度
        self.avgdl = sum(self.doc_len) / len(docs)  # 计算所有文档的平均长度
        self.doc_freqs = []  # 存储每个文档的词频
        self.idf = {}  # 存储每个词的逆文档频率
        self.initialize()

    def initialize(self):
        """
        初始化方法，计算所有词的逆文档频率
        """
        df = {}  # 用于存储每个词在多少不同文档中出现
        for doc in self.docs:
            # 为每个文档创建一个词频统计
            self.doc_freqs.append(Counter(doc))
            # 更新df值
            for word in set(doc):
                df[word] = df.get(word, 0) + 1
        # 计算每个词的IDF值
        for word, freq in df.items():
            self.idf[word] = math.log((len(self.docs) - freq + 0.5) / (freq + 0.5) + 1)

    def score(self, doc, query):
        """
        计算文档与查询的BM25得分
        :param doc: 文档的索引
        :param query: 查询词列表
        :return: 该文档与查询的相关性得分
        """
        score = 0.0
        for word in query:
            if word in self.doc_freqs[doc]:
                freq = self.doc_freqs[doc][word]  # 词在文档中的频率
                # 应用BM25计算公式
                score += (self.idf[word] * freq * (self.k1 + 1)) / (
                        freq + self.k1 * (1 - self.b + self.b * self.doc_len[doc] / self.avgdl))
        return score

def tokenize(text):
    """
    使用正则表达式进行分词，去除标点符号。
    :param text: 输入文本
    :return: 分词后的词列表
    """
    # 去除标点符号
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower().split()

def load_documents(file_path):
    """
    读取 documents.jsonl 文件，并返回所有文档的合并字段列表。
    
    处理逻辑：
    1. 如果 JSON 行中存在 "messages" 字段，则将其中各消息的 "content" 内容按顺序拼接。
    2. 如果没有 "messages" 字段，则按 "instruction"、"input"、"output" 字段合并文本。
    3. 对合并后的文本进行分词处理，并存入列表中。
    
    :param file_path: documents.jsonl 文件路径
    :return: 输出字段列表，每个元素是一个词列表
    """
    docs = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        for line in file:
            try:
                data = json.loads(line)
                # 如果存在 "messages" 字段，则将 messages 中每条记录的 content 连接起来
                if "messages" in data:
                    combined_text = " ".join(msg.get("content", "") for msg in data.get("messages", []))
                else:
                    # 否则按默认方式合并 instruction, input 和 output 字段
                    instruction = data.get("instruction", "")
                    input_field = data.get("input", "")
                    output = data.get("output", "")
                    combined_text = f"{instruction} {input_field} {output}"
                # 去除左右空白
                combined_text = combined_text.strip()
                # 对合并后的文本进行分词处理（此处需保证 tokenize 函数已定义）
                docs.append(tokenize(combined_text))
            except json.JSONDecodeError:
                # 如果某一行不是有效的 JSON，跳过
                continue
    return docs

def load_combined_query(file_path):
    """
    读取queries.jsonl文件，并将所有查询内容合并为一个整体查询
    :param file_path: queries.jsonl文件路径
    :return: 整体查询的词列表
    """
    combined_query = ""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        for line in file:
            try:
                data = json.loads(line)
                # 将 'instruction', 'input', 'output' 字段合并作为查询内容
                instruction = data.get("instruction", "")
                input_field = data.get("input", "")
                output = data.get("output", "")
                # 合并所有相关字段
                combined_query += f" {instruction} {input_field} {output}"
            except json.JSONDecodeError:
                # 如果某一行不是有效的JSON，跳过
                continue
    # 分词处理
    query_terms = tokenize(combined_query)
    return query_terms

def get_top_n_documents(file_path, top_indices):
    """
    根据文档索引获取对应的JSON对象
    :param file_path: documents.jsonl文件路径
    :param top_indices: 需要获取的文档索引列表
    :return: 包含前n个文档的列表
    """
    top_documents = []
    indices_set = set(top_indices)  # 转换为集合以提高查找效率
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        for i, line in enumerate(file):
            if i in indices_set:
                try:
                    data = json.loads(line)
                    top_documents.append(data)
                except json.JSONDecodeError:
                    # 如果某一行不是有效的JSON，跳过
                    continue
    return top_documents

def search_jsonl(documents_file, queries_file, n, output_file):
    """
    在documents.jsonl文件中针对queries.jsonl文件中的所有查询进行BM25搜索，返回前n个相关性得分最高的文档并保存到新文件
    :param documents_file: documents.jsonl文件路径
    :param queries_file: queries.jsonl文件路径
    :param n: 返回排名前n的文档
    :param output_file: 保存搜索结果的文件路径
    :return: None
    """
    # 检查输入文件是否存在
    if not os.path.isfile(documents_file):
        print(f"Error: Documents file '{documents_file}' does not exist.")
        return
    if not os.path.isfile(queries_file):
        print(f"Error: Queries file '{queries_file}' does not exist.")
        return

    # 加载被查询文档
    print("Loading documents...")
    docs = load_documents(documents_file)
    print(f"Loaded {len(docs)} documents.")

    # 初始化BM25模型
    print("Initializing BM25 model...")
    bm25 = BM25(docs)
    print("BM25 model initialized.")

    # 加载并合并查询
    print("Loading and combining queries...")
    combined_query_terms = load_combined_query(queries_file)
    print(f"Combined query has {len(combined_query_terms)} terms.")

    if not combined_query_terms:
        print("Error: Combined query has no terms.")
        return

    # 计算每个文档与查询的得分
    print("Scoring documents...")
    scores = [(i, bm25.score(i, combined_query_terms)) for i in range(len(docs))]

    # 按照得分降序排列，取前n个
    print(f"Selecting top {n} documents...")
    top_n_scores = sorted(scores, key=lambda x: x[1], reverse=True)[:n]

    # 获取前n个文档的索引
    top_n_indices = [idx for idx, score in top_n_scores]

    # 获取前n个文档的内容
    print("Retrieving top documents...")
    top_n_documents = get_top_n_documents(documents_file, top_n_indices)

    # 将前n个文档分别保存为新的JSONL文件，每个文档单独一行
    print(f"Saving top {n} documents to '{output_file}'...")
    with open(output_file, 'w', encoding='utf-8') as out_file:
        for doc in top_n_documents:
            json.dump(doc, out_file, ensure_ascii=False)
            out_file.write('\n')

    print(f"Top {n} results have been saved to '{output_file}'.")

# 示例用法
import argparse

if __name__ == "__main__":
    # 创建命令行解析器
    parser = argparse.ArgumentParser(description="Search documents based on queries")
    
    # 添加命令行参数
    parser.add_argument('--n', type=int, default=2707, help="Number of documents to return (default is 2707)")
    parser.add_argument('--queries_file', type=str, default="/mnt/petrelfs/zhangchi/wjc/bm25/all_data/target_data.jsonl", help="Path to the queries file")
    parser.add_argument('--documents_file', type=str, default="/mnt/petrelfs/zhangchi/wjc/bm25/all_data/all_data.jsonl", help="Path to the documents file")
    parser.add_argument('--output_file', type=str, default="/mnt/petrelfs/zhangchi/wjc/bm25/train_data_0.01.jsonl", help="Path to the output file")

    # 解析命令行参数
    args = parser.parse_args()

    # 调用search_jsonl函数
    search_jsonl(args.documents_file, args.queries_file, args.n, args.output_file)
