# -*- coding: utf-8 -*-
# 配置模板文件 - 复制此文件为 my_config.py 并填入实际的API配置

import os

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://your-api-endpoint/v1")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "your-api-key")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")

EMB_BASE_URL = os.environ.get("EMB_BASE_URL", LLM_BASE_URL)
EMB_API_KEY = os.environ.get("EMB_API_KEY", LLM_API_KEY)
EMB_MODEL = os.environ.get("EMB_MODEL", "text-embedding-3-small")
EMB_DIM = int(os.environ.get("EMB_DIM", "1536"))
