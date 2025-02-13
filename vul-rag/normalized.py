import re
import codecs
import os
import pandas as pd
from pathlib import Path
import common.constant as constant
from common.tool.progress import print_progress
import json

# Clean Gadget
# Author https://github.com/johnb110/VDPython:
# For each gadget, replaces all user variables with "VAR#" and user functions with "FUN#"
# Removes content from string and character literals keywords up to C11 and C++17; immutable set

from typing import List

keywords = frozenset({'__asm', '__builtin', '__cdecl', '__declspec', '__except', '__export', '__far16', '__far32',
                      '__fastcall', '__finally', '__import', '__inline', '__int16', '__int32', '__int64', '__int8',
                      '__leave', '__optlink', '__packed', '__pascal', '__stdcall', '__system', '__thread', '__try',
                      '__unaligned', '_asm', '_Builtin', '_Cdecl', '_declspec', '_except', '_Export', '_Far16',
                      '_Far32', '_Fastcall', '_finally', '_Import', '_inline', '_int16', '_int32', '_int64',
                      '_int8', '_leave', '_Optlink', '_Packed', '_Pascal', '_stdcall', '_System', '_try', 'alignas',
                      'alignof', 'and', 'and_eq', 'asm', 'auto', 'bitand', 'bitor', 'bool', 'break', 'case',
                      'catch', 'char', 'char16_t', 'char32_t', 'class', 'compl', 'const', 'const_cast', 'constexpr',
                      'continue', 'decltype', 'default', 'delete', 'do', 'double', 'dynamic_cast', 'else', 'enum',
                      'explicit', 'export', 'extern', 'false', 'final', 'float', 'for', 'friend', 'goto', 'if',
                      'inline', 'int', 'long', 'mutable', 'namespace', 'new', 'noexcept', 'not', 'not_eq', 'nullptr',
                      'operator', 'or', 'or_eq', 'override', 'private', 'protected', 'public', 'register',
                      'reinterpret_cast', 'return', 'short', 'signed', 'sizeof', 'static', 'static_assert',
                      'static_cast', 'struct', 'switch', 'template', 'this', 'thread_local', 'throw', 'true', 'try',
                      'typedef', 'typeid', 'typename', 'union', 'unsigned', 'using', 'virtual', 'void', 'volatile',
                      'wchar_t', 'while', 'xor', 'xor_eq', 'NULL', 'printf', 'STR'})
# holds known non-user-defined functions; immutable set
main_set = frozenset({'main'})
# arguments in main function; immutable set
main_args = frozenset({'argc', 'argv'})

operators3 = {'<<=', '>>='}
operators2 = {
    '->', '++', '--', '**',
    '!~', '<<', '>>', '<=', '>=',
    '==', '!=', '&&', '||', '+=',
    '-=', '*=', '/=', '%=', '&=', '^=', '|='
}
operators1 = {
    '(', ')', '[', ']', '.',
    '+', '&',
    '%', '<', '>', '^', '|',
    '=', ',', '?', ':',
    '{', '}', '!', '~'
}


def to_regex(lst): # 将 lst 中的元素转换为正则表达式
    return r'|'.join([f"({re.escape(el)})" for el in lst])


regex_split_operators = to_regex(operators3) + to_regex(operators2) + to_regex(operators1)

def _removeComments(source) -> []: # 移除源代码每一行的注释 // 和 /* */
    in_block = False # 是否在注释块中
    new_source = []
    # source = source.split('\n')
    for line in source:
        i = 0
        if not in_block: # 如果不在注释块中
            newline = []
        while i < len(line): # 遍历该行
            if line[i:i + 2] == '/*' and not in_block: # 如果遇到 /*
                in_block = True
                i += 1
            elif line[i:i + 2] == '*/' and in_block: # 如果遇到 */
                in_block = False
                i += 1
            elif not in_block and line[i:i + 2] == '//': # 如果遇到 //，就停止存储剩余部分
                break
            elif not in_block:
                newline.append(line[i])
            i += 1
        if newline and not in_block:
            new_source.append("".join(newline))
    return new_source


# input is a list of string lines
# 进行：1. 替换变量名为VAR# 2. 替换函数名为FUN#
def clean_gadget(gadget): 
    # dictionary; map function name to symbol name + number
    fun_symbols = {}
    # dictionary; map variable name to symbol name + number
    var_symbols = {}

    fun_count = 1
    var_count = 1

    # regular expression to find function name candidates
    rx_fun = re.compile(r'\b([_A-Za-z]\w*)\b(?=\s*\()') # 匹配函数名
    # regular expression to find variable name candidates
    # rx_var = re.compile(r'\b([_A-Za-z]\w*)\b(?!\s*\()')
    rx_var = re.compile(r'\b([_A-Za-z]\w*)\b((?!\s*\**\w+))(?!\s*\()') # 匹配变量名，第一个匹配变量名，第二个匹配指针

    # final cleaned gadget output to return to interface
    cleaned_gadget = []

    for line in gadget: # 遍历每一行
        ascii_line = re.sub(r'[^\x00-\x7f]', r'', line) # 将 line 中的非 ASCII 字符替换为空字符串
        hex_line = re.sub(r'0[xX][0-9a-fA-F]+', "HEX", ascii_line) # 将 line 中的十六进制数字替换为 HEX
        user_fun = rx_fun.findall(hex_line) # 在 hex_line 中匹配函数名
        user_var = rx_var.findall(hex_line) # 在 hex_line 中匹配变量名

        
        for fun_name in user_fun:
            # 如果不是 main 函数和关键字
            if len({fun_name}.difference(main_set)) != 0 and len({fun_name}.difference(keywords)) != 0:
                if fun_name not in fun_symbols.keys(): # 如果函数名不在字典中，在字典中就说明之前存过了
                    fun_symbols[fun_name] = 'FUN' + str(fun_count) # 将函数名映射为 FUN#，# 为计数
                    fun_count += 1 # 计数加一
                hex_line = re.sub(r'\b(' + fun_name + r')\b(?=\s*\()', fun_symbols[fun_name], hex_line) # 将函数名替换为 FUN#

        for var_name in user_var: # 遍历变量名
            # next line is the nuanced difference between fun_name and var_name
            if len({var_name[0]}.difference(keywords)) != 0 and len({var_name[0]}.difference(main_args)) != 0:
                # check to see if variable name already in dictionary
                if var_name[0] not in var_symbols.keys(): # var_name 的部分是变量名，如果变量名不在字典中
                    var_symbols[var_name[0]] = 'VAR' + str(var_count) # 将变量名映射为 VAR#，# 为计数
                    var_count += 1 # 计数加一
                # ensure that only variable name gets replaced (no function name with same
                # identifier); uses negative lookforward
                # print(var_name, gadget, user_var)
                hex_line = re.sub(r'\b(' + var_name[0] + r')\b(?:(?=\s*\w+\()|(?!\s*\w+))(?!\s*\()', 
                                  var_symbols[var_name[0]], hex_line) # 将变量名替换为 VAR#

        cleaned_gadget.append(hex_line) # 将处理后的行添加到 cleaned_gadget 中
    # return the list of cleaned lines
    return cleaned_gadget 

def normalize_code(source_code, need_normalize):
    gadget: List[str] = []

    if need_normalize:
        try:
            no_str_lit_line = re.sub(r'["]([^"\\\n]|\\.|\\\n)*["]', '"STR"', source_code) # 将字符串替换为 STR
        except:
            print("Error in string literal replacement")
            print(clean)
        no_char_lit_line = re.sub(r"'.*?'", "", no_str_lit_line) # 将字符替换为""
        source_code = no_char_lit_line
    for line in source_code.splitlines(): # 遍历处理完字符和字符串的代码的每一行
        if line == '':
            continue
        stripped = line.strip() # 去除首尾空格
        gadget.append(stripped) # 将处理完字符和字符串的代码行添加到 gadget 中
    clean = _removeComments(gadget)
    if need_normalize:
        clean = clean_gadget(clean)
    
    dest_code = ""
    for line in clean:
        dest_code += line + '\n'

    return dest_code

def normalize_code_json(json_path, store_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)  # 将 JSON 文件内容解析为 Python 对象

    assert(isinstance(data, dict))

    for CVE_ID, CVE_LIST in data.items():  
        for item in CVE_LIST:
            code_before_change = item["code_before_change"]
            code_after_change = item["code_after_change"]
            code_after_change_normalized = normalize_code(code_after_change, need_normalize=True)
            code_before_change_normalized = normalize_code(code_before_change, need_normalize=True)
            item["code_after_change_normalized"] = code_after_change_normalized
            item["code_before_change_normalized"] = code_before_change_normalized
            
            code_after_change_raw = normalize_code(code_after_change, need_normalize=False)
            code_before_change_raw = normalize_code(code_before_change, need_normalize=False)
            item["code_after_change_raw"] = code_after_change_raw
            item["code_before_change_raw"] = code_before_change_raw
    
    name = json_path.split("/")[-1]
    os.makedirs(store_path, exist_ok=True)
    save_path = store_path / name

    with open(save_path, 'w', encoding='utf-8') as nfile:
        json.dump(data, nfile, indent=2)


def add_id_to_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)  # 将 JSON 文件内容解析为 Python 对象

    assert(isinstance(data, dict))
    id = 0
    for CVE_ID, CVE_LIST in data.items():  
        for item in CVE_LIST:
            item["id"] = id
            id += 1

    name = json_path.split("/")[-1].split(".json")[0]
    os.makedirs(constant.vul_rag_vul_knowledge_with_id_dir, exist_ok=True)
    save_path = constant.vul_rag_vul_knowledge_with_id_dir + "/" + name + "_with_id.json"

    with open(save_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=2)

def main():
    # for file in os.listdir(constant.vul_rag_vul_knowledge_dir):
    #     if file.endswith(".json"):
    #         add_id_to_json(constant.vul_rag_vul_knowledge_dir + "/" + file)

    for file in os.listdir(constant.vul_rag_vul_knowledge_with_id_dir):
        if file.endswith(".json"):
            normalize_code_json(constant.vul_rag_vul_knowledge_with_id_dir + "/" + file, Path(constant.vul_rag_dir) / "normalized")

if __name__ == "__main__":
    main()


