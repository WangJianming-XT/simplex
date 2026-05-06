# -*- coding: utf-8 -*-
"""
API测试文件
用于测试与jeniya.top API的连接
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import numpy as np
from hyperrag.llm import openai_embedding, openai_complete_if_cache

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://jeniya.top/v1")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")

EMB_BASE_URL = os.environ.get("EMB_BASE_URL", LLM_BASE_URL)
EMB_API_KEY = os.environ.get("EMB_API_KEY", LLM_API_KEY)
EMB_MODEL = os.environ.get("EMB_MODEL", "text-embedding-3-small")
EMB_DIM = int(os.environ.get("EMB_DIM", "1536"))

async def test_embedding_api():
    """测试嵌入API"""
    print("=" * 50)
    print("测试嵌入API...")
    print("=" * 50)
    
    try:
        result = await openai_embedding(
            ["你好，世界！"],
            model=EMB_MODEL,
            api_key=EMB_API_KEY,
            base_url=EMB_BASE_URL,
        )
        print(f"✓ 嵌入API调用成功")
        print(f"  返回形状: {result.shape}")
        print(f"  前5个值: {result[0][:5]}")
        return True
    except Exception as e:
        print(f"✗ 嵌入API调用失败: {e}")
        return False

async def test_llm_api():
    """测试LLM API"""
    print("\n" + "=" * 50)
    print("测试LLM API...")
    print("=" * 50)
    
    try:
        result = await openai_complete_if_cache(
            LLM_MODEL,
            "请简要介绍一下自己，只用一句话。",
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
        )
        print(f"✓ LLM API调用成功")
        print(f"  返回结果: {result}")
        return True
    except Exception as e:
        print(f"✗ LLM API调用失败: {e}")
        return False

async def test_rate_limit_handling():
    """测试限流处理"""
    print("\n" + "=" * 50)
    print("测试限流处理（连续3次调用）...")
    print("=" * 50)
    
    for i in range(3):
        try:
            result = await openai_complete_if_cache(
                LLM_MODEL,
                f"请回答数字{i}。",
                api_key=LLM_API_KEY,
                base_url=LLM_BASE_URL,
            )
            print(f"  第{i+1}次调用成功: {result[:50]}...")
            await asyncio.sleep(1)
        except Exception as e:
            print(f"  第{i+1}次调用失败: {e}")
            if "429" in str(e):
                print("  检测到429错误（请求过多），建议降低调用频率")

async def main():
    print("\n" + "=" * 60)
    print("开始API测试")
    print("=" * 60 + "\n")
    
    emb_success = await test_embedding_api()
    llm_success = await test_llm_api()
    
    if emb_success and llm_success:
        print("\n" + "=" * 60)
        print("✓ 所有API测试通过！")
        print("=" * 60 + "\n")
    else:
        print("\n" + "=" * 60)
        print("✗ 部分API测试失败，请检查配置和网络")
        print("=" * 60 + "\n")
    
    await test_rate_limit_handling()

if __name__ == "__main__":
    asyncio.run(main())