import os
from pathlib import Path

from common import constant

class PathUtil:
    @staticmethod
    def api_keys_data(filename: str, ext: str):
        path = Path(constant.common_dir) / 'api_keys'
        path.mkdir(parents = True, exist_ok = True)
        path = path / f'{filename}.{ext}'
        return str(path)
    
    @staticmethod
    def vul_detection_output(
        filename: str,
        ext: str,
        detection_model_name: str,
    ):
        # 横杠替换为下划线
        filename = filename.replace("-", "_")
        detection_model_name = detection_model_name.replace("-", "_")
        
        path = (
            Path(constant.vul_rag_test_result) /
            f"{detection_model_name}"
        )
        path.mkdir(parents = True, exist_ok = True)
        path = path / f'{filename}.{ext}'
        return str(path)
    
    @staticmethod
    def checkpoint_data(filename: str, ext: str):
        path = Path(constant.vul_rag_test_result) / 'checkpoint'
        path.mkdir(parents = True, exist_ok = True)
        path = path / f'{filename}.{ext}'
        return str(path)
   
    @staticmethod
    def check_file_exists(filename:str):
        path = Path(filename)
        return path.exists()