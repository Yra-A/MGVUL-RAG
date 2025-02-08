import os
from pathlib import Path
import common.constant as constant
from tokenizers import Tokenizer
from transformers import AutoTokenizer

# 例如加载 Qwen2.5-Coder 的 tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-32B-Instruct")

