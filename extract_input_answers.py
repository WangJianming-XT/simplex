import json

# # 输入文件路径（JSONL格式：每行一个JSON对象）
# input_file = "f:\\work\\Hyper-RAG-main2\\datasets\\legal\\legal.jsonl"
# # 输出文件路径
# output_file = "f:\\work\\Hyper-RAG-main2\\datasets\\legal\\legal2.jsonl"

# 输入文件路径（JSONL格式：每行一个JSON对象）
input_file = "f:\\work\\Hyper-RAG-main2\\datasets\\legal\\legal未格式化.jsonl"
# 输出文件路径
output_file = "f:\\work\\Hyper-RAG-main2\\datasets\\legal\\legal.jsonl"

try:
    data = []
    
    # 读取输入文件（JSONL格式：按行读取并解析每个JSON对象）
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    
    # 提取input和answers字段并写入JSON数组格式
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('[\n')
        for i, item in enumerate(data):
            extracted_item = {
                "input": item.get("input", ""),
                "answers": item.get("answers", [])
            }
            json.dump(extracted_item, f, ensure_ascii=False)
            if i < len(data) - 1:
                f.write(',\n')
            else:
                f.write('\n')
        f.write(']')
    
    print(f"成功提取数据并保存到 {output_file}")
    print(f"处理了 {len(data)} 条记录")
    
except Exception as e:
    print(f"处理过程中出现错误: {e}")
