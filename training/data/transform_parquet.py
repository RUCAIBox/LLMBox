import pandas as pd

# 文件路径列表
parquet_files = [
    '/data/LLMBox/training/data/train_sft-00000-of-00003-a3ecf92756993583.parquet',
    '/data/LLMBox/training/data/train_sft-00001-of-00003-0a1804bcb6ae68c6.parquet',
    '/data/LLMBox/training/data/train_sft-00002-of-00003-ee46ed25cfae92c6.parquet'
]

# 读取Parquet文件并将它们合并成一个DataFrame
df_list = [pd.read_parquet(file) for file in parquet_files]
df_combined = pd.concat(df_list, ignore_index=True)

# 指定jsonl文件的路径
jsonl_file = '/data/LLMBox/training/data/ultrachat.jsonl'

# 将合并后的DataFrame保存为JSONL文件
df_combined.to_json(jsonl_file, orient='records', lines=True)

print(f"JSONL file saved as {jsonl_file}")