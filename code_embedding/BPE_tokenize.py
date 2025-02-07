import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from common.constant import bpe_tokenizer_path, normalized_dir
from pathlib import Path
from common.tool.progress import print_progress

cnt = 0
def read_files_in_batches(file_paths, batch_size=1000):
    batch = []
    for root, dirs, files in os.walk(file_paths):
        for file in files:
            if not file.endswith(".c"):
                continue
            global cnt
            cnt += 1
            with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                for line in f:
                    batch.append(line.strip())
                    if len(batch) >= batch_size:
                        yield batch
                        batch = []
    if batch:
        yield batch

def main():
    codes_dir = Path(normalized_dir) / "raw_code_files"

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

    print("路径: ", str(codes_dir))
    
    length = len(os.listdir(codes_dir))
    print("文件数量: ", length)
    for batch in read_files_in_batches(str(codes_dir), 1000):
        tokenizer.train_from_iterator(batch, trainer, length=length)

    print("训练文件数量: ", cnt)
    print("训练完成")
    tokenizer.save(bpe_tokenizer_path, pretty=True)

if __name__ == "__main__":
    main()