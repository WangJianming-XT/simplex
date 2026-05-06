import json
import argparse
import os
from pathlib import Path


def validate_input_directory(input_directory):
    """验证输入目录是否存在且包含JSONL文件"""
    in_dir = Path(input_directory)
    if not in_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_directory}")
    if not in_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_directory}")
    
    # 检查是否有JSONL文件
    jsonl_files = list(in_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise ValueError(f"No JSONL files found in input directory: {input_directory}")
    
    return in_dir, jsonl_files

def extract_contexts_from_jsonl(file_path):
    """从JSONL文件中提取上下文"""
    contexts = []
    try:
        with open(file_path, "r", encoding="utf-8") as infile:
            for line_number, line in enumerate(infile, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    json_obj = json.loads(line)
                    context = json_obj.get("context")
                    if context and isinstance(context, str) and len(context.strip()) > 0:
                        # 清理上下文，确保适合复形RAG处理
                        cleaned_context = context.strip()
                        contexts.append(cleaned_context)
                except json.JSONDecodeError as e:
                    print(f"⚠️  JSON decoding error in file {file_path.name} at line {line_number}: {e}")
                except Exception as e:
                    print(f"⚠️  Error processing line {line_number} in {file_path.name}: {e}")
    except Exception as e:
        print(f"❌ An error occurred while processing file {file_path.name}: {e}")
    
    return contexts

def extract_unique_contexts(input_directory, output_directory, data_name):
    """提取唯一上下文，专为复形RAG系统准备"""
    try:
        # 验证输入目录
        in_dir, jsonl_files = validate_input_directory(input_directory)
        
        # 创建输出目录
        out_dir = Path(output_directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Found {len(jsonl_files)} JSONL files to process.")

        all_contexts = []
        total_extracted = 0
        
        for file_path in jsonl_files:
            print(f"Processing file: {file_path.name}")
            contexts = extract_contexts_from_jsonl(file_path)
            all_contexts.extend(contexts)
            total_extracted += len(contexts)
            print(f"  Extracted {len(contexts)} contexts from {file_path.name}")

        # 去重，确保每个上下文都是唯一的
        unique_contexts = list(set(all_contexts))
        print(f"\nTotal contexts extracted: {total_extracted}")
        print(f"Unique contexts after deduplication: {len(unique_contexts)}")

        if not unique_contexts:
            print("❌ No valid contexts found. Please check your JSONL files.")
            return

        # 保存结果，格式适合复形RAG系统使用
        output_path = out_dir / f"{data_name}_unique_contexts.json"
        try:
            with open(output_path, "w", encoding="utf-8") as outfile:
                json.dump(unique_contexts, outfile, ensure_ascii=False, indent=4)
            print(f"✅ Unique contexts have been saved to: {output_path}")
            print(f"✅ Output file size: {os.path.getsize(output_path)} bytes")
        except Exception as e:
            print(f"❌ An error occurred while saving to the file {output_path}: {e}")
            raise

        print("\n🎉 All files have been processed successfully!")
        print(f"📊 Summary: {len(jsonl_files)} JSONL files processed, {len(unique_contexts)} unique contexts extracted")

    except Exception as e:
        print(f"❌ Fatal error: {e}")
        raise


if __name__ == "__main__":
    data_name = 'legal'
    parser = argparse.ArgumentParser(
        description="Extract unique contexts from JSONL files for Simplicial Complex RAG"
    )
    parser.add_argument(
        "-i", "--input_dir", 
        type=str, 
        default=f"datasets/{data_name}",
        help="Directory containing JSONL files"
    )
    parser.add_argument(
        "-o", "--output_dir", 
        type=str, 
        default=f"caches/{data_name}/contexts",
        help="Directory to save unique contexts"
    )

    args = parser.parse_args()

    try:
        extract_unique_contexts(args.input_dir, args.output_dir, data_name)
    except Exception as e:
        print(f"❌ Program failed: {e}")
        exit(1)
