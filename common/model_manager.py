import common.config as cfg
import openai
import logging
import os

class BaseModel:
    def __init__(self, model_name, base_url, api_key = None):
        self.__base_url = base_url
        self.__model_name = model_name
        self.__api_key = api_key
        self.__client = None
        if api_key:
            try:
                self.__client = openai.OpenAI(api_key = api_key, base_url = base_url)
            except:
                # Lower version of openai package does not support openai.OpenAI
                openai.api_key = api_key
                openai.api_base = base_url
                logging.warning("Outdated openai package. Use the Module-level global client instead.")

    @staticmethod
    def get_messages(user_prompt: str, sys_prompt: str = None) -> list:
        if sys_prompt:
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ]
        else:
            messages = [{"role": "user", "content": user_prompt}]
        return messages

    def get_response_with_messages(self, messages: list, **kwargs) -> str:
        logging.disable(logging.INFO)
        response_content = None
        try:
            if self.__client:
                response = self.__client.chat.completions.create(
                    model = self.__model_name,
                    messages = messages,
                    stream = False,
                    **kwargs
                )
                response_content = response.choices[0].message.content
            else:
                # use the module-level global client
                openai.api_key = self.__api_key
                openai.api_base = self.__base_url
                response = openai.ChatCompletion.create(
                    model = self.__model_name,
                    messages = messages,
                    **kwargs
                )
                response_content = response.choices[0]["message"]["content"]
        except Exception as e:
            logging.error(f"Error while calling {self.__model_name} API: {e}")
        logging.disable(logging.NOTSET)
        return response_content
    
    def get_model_name(self):
        return self.__model_name

    def set_proxy(self, proxy: str = cfg.OPENAI_API_CONNECTION_PROXY):
        if "http_proxy" not in os.environ:
            os.environ["http_proxy"] = proxy
        if "https_proxy" not in os.environ:
            os.environ["https_proxy"] = proxy

    def unset_proxy(self):
        if "http_proxy" in os.environ:
            del os.environ["http_proxy"]
        if "https_proxy" in os.environ:
            del os.environ["https_proxy"]

class DeepSeekModel(BaseModel):
    def __init__(self, model_name):
        super().__init__(
            model_name = model_name,
            base_url = cfg.deepseek_api_base,
            api_key = cfg.deepseek_api_key
        )

class GPTModel(BaseModel):
    def __init__(self, model_name):
        super().__init__(
            model_name = model_name,
            base_url = cfg.openkey_openai_api_base,
            api_key = cfg.openkey_openai_api_key
        )

class QwenModel(BaseModel):
    def __init__(self, model_name):
        super().__init__(
            model_name = model_name,
            base_url = cfg.qwen_api_base,
            api_key = cfg.qwen_api_key
        )

class BailianModel(BaseModel):
    def __init__(self, model_name):
        super().__init__(
            model_name = 'deepseek-r1',
            base_url = cfg.bailian_api_base,
            api_key = cfg.bailian_api_key
        )

class ModelManager:
    __models = {
        "qwen": QwenModel,
        "deepseek": DeepSeekModel,
        "gpt": GPTModel,
        "bailian": BailianModel
    }

    __instances_cache = {}

    @classmethod
    def get_model_instance(cls, model_name: str) -> BaseModel:
        if "qwen" in model_name.lower():
            model_name_kw = "qwen"
        elif "deepseek" in model_name.lower():
            model_name_kw = "deepseek"
        elif "gpt" in model_name.lower():
            model_name_kw = "gpt"
        elif "bailian" in model_name.lower():
            model_name_kw = "bailian"
        
        if model_name_kw not in cls.__models:
            raise ValueError("Unsupported model name")

        if model_name not in cls.__instances_cache: # 如果模型实例不在缓存中
            model_class = cls.__models.get(model_name_kw, None) # 获取模型类，不是实例
            if not model_class: # 如果模型类不存在
                raise ValueError("Unsupported model name") # 抛出异常
            cls.__instances_cache[model_name] = model_class(model_name) # 创建模型实例并缓存
            
        model_instance = cls.__instances_cache[model_name] # 获取模型实例
        return model_instance   # 返回模型实例