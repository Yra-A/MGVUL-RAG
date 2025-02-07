import re
import codecs
import os
import pandas as pd
from pathlib import Path
import common.constant as constant
from common.tool.progress import print_progress

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


def normalize_code(data_path, store_path):
    files = os.listdir(data_path) # 获取 data_path 下的所有文件
    files_num = len(files) # 文件数量
    count = 0
    if not os.path.exists(store_path): # 如果 store_path 不存在，创建 store_path
        os.mkdir(store_path) # 创建 store_path
    for file in files:
        # 打印进度
        count = count + 1 
        print_progress(count, files_num) 

        path = data_path + '/' + file # 文件路径
        with open(path, "r") as f1: # 打开文件
            code = f1.read() # 读取文件内容
            gadget: List[str] = [] # 用于存储代码行, List[str] 表示列表中的元素是字符串
            # remove all string literals
            no_str_lit_line = re.sub(r'["]([^"\\\n]|\\.|\\\n)*["]', '"STR"', code) # 将字符串替换为 STR
            # remove all character literals
            no_char_lit_line = re.sub(r"'.*?'", "", no_str_lit_line)  # 将字符替换为""
            code = no_char_lit_line 

            for line in code.splitlines(): # 遍历处理完字符和字符串的代码的每一行
                if line == '':
                    continue
                stripped = line.strip() # 去除首尾空格
                # if "\\n\\n" in stripped: print(stripped)
                gadget.append(stripped) # 将处理完字符和字符串的代码行添加到 gadget 中
            clean = _removeComments(gadget) 
            clean = clean_gadget(clean)

            with open(store_path + "/" + file, 'w', encoding='utf-8') as f2: # 将处理后的代码写入到文件
                f2.writelines([line + '\n' for line in clean])


def normalize_code_csv(data_path, store_path):
    data = pd.read_csv(data_path)
    files_num = data.shape[0]
    count = 0

    normalize_code = []
    rc_raw_code = []
    for index, row in data.iterrows():
        count = count + 1
        print("\r", end="")
        print("Process progress: {}%: ".format(count / files_num * 100), end="")
        
        raw_code = row['vulnerable_code']
        code = row['vulnerable_code']
        gadget: List[str] = []
        # remove all string literals
        try:
            no_str_lit_line = re.sub(r'["]([^"\\\n]|\\.|\\\n)*["]', '"STR"', code) # 将字符串替换为 STR
        except:
            print(code)
        # remove all character literals
        no_char_lit_line = re.sub(r"'.*?'", "", no_str_lit_line) # 将字符替换为""
        code = no_char_lit_line

        for line in code.splitlines(): # 遍历处理完字符和字符串的代码的每一行
            if line == '':
                continue
            stripped = line.strip() # 去除首尾空格
            # if "\\n\\n" in stripped: print(stripped) # 如果 stripped 中包含 "\\n\\n"，打印 stripped
            gadget.append(stripped) # 将处理完字符和字符串的代码行添加到 gadget 中
        clean = _removeComments(gadget)
        clean = clean_gadget(clean) # clean 已经做了：1. 移除注释 2. 替换变量名为 VAR# 3. 替换函数名为 FUN# 4. 替换字符串为 STR

        normalize = ""
        for line in clean: 
            normalize = normalize + line + '\n' # 将处理后的代码行拼接成字符串
        normalize_code.append(normalize) # 将处理后的代码添加到 normalize_code 中，一个 normalize 是一段代码

        raw_lines = [] 
        try:
            for line in raw_code.splitlines(): # 遍历原始代码的每一行
                if line == '':
                    continue
                stripped = line.strip() # 去除首尾空格
                raw_lines.append(stripped) # 将处理完字符和字符串的代码行添加到 raw_lines 中
        except:
            print(raw_code)
        rc_raw_lines = _removeComments(raw_lines) # 移除注释
        rc = ""
        for line in rc_raw_lines:
            rc = rc + line + '\n'
        rc_raw_code.append(rc) # 将处理后的代码添加到 rc_raw_code 中

    data["raw_code"] = rc_raw_code # raw code 是去除注释后的代码
    data["normalized_code"] = normalize_code # normalized code 是去除注释后的代码，并且替换变量名为 VAR#，替换函数名为 FUN#，替换字符串为 STR
    data.to_csv(store_path, index=False) # 将处理后的数据保存到 store_path


def main():
    os.makedirs(constant.normalized_dir, exist_ok=True)
    normalize_code_csv(Path(constant.preprocessed_dir) / "bigvul_processed_metadata.csv", Path(constant.normalized_dir) / "bigvul_normalized.csv")

if __name__ == "__main__":
    main()


