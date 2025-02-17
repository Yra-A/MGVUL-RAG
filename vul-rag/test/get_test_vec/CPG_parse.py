import os
import pandas as pd
import common.constant as constant
from pathlib import Path
import subprocess
from common.tool.progress import print_progress
import json

def to_code_file(source_code, save_dir):
    save_path = save_dir / "code.c"
    with open(save_path, 'w', encoding='utf-8') as file:
        file.write(source_code)

def joern_parse(code_dir): 
    print("开始转换 code.c 到 CPGs")

    input = str(code_dir / 'code.c')

    output = str(Path(code_dir) / f"cpg.bin")
    os.makedirs(code_dir, exist_ok=True)

    joern_path = constant.joern_path
    os.chdir(joern_path)
    
    os.environ['input'] = input
    os.environ['output'] = output

    print("input: {}\noutput: {}".format(input, output))

    process = subprocess.Popen('sh joern-parse $input --out $output',    
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    shell=True, close_fds=True)
    output = process.communicate()