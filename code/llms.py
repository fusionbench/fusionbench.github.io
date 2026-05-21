import time
from typing import Dict, List, Optional

from openai import OpenAI

from utils import get_llm_config, get_llm_provider_config, logger


# 从 env.yaml 读取配置
MODEL_PROVIDER_CONFIG = get_llm_provider_config()
LLM_CONFIG = get_llm_config()

class LLMService:
    
    def __init__(self):
        self._clients: Dict[str, OpenAI] = {}
        self._model_to_provider: Dict[str, str] = {}
        self._build_model_to_provider_map()

    def _build_model_to_provider_map(self):
        """构建 {model_id: provider} 的映射表"""
        for provider, config in MODEL_PROVIDER_CONFIG.items():
            for model_id in config["models"]:
                if model_id in self._model_to_provider:
                    raise ValueError(f"模型 ID 冲突: {model_id} 已被注册")
                self._model_to_provider[model_id] = provider
    
    def _get_client(self, provider: str) -> OpenAI:
        if provider not in MODEL_PROVIDER_CONFIG:
            raise ValueError(f"未知的服务商: {provider}")
        if provider not in self._clients:
            config = MODEL_PROVIDER_CONFIG[provider]
            if not config["api_key"]:
                raise ValueError(f"缺少 {provider} 的 API Key")
            self._clients[provider] = OpenAI(
                api_key=config["api_key"],
                base_url=config["base_url"]
            )
        return self._clients[provider]

    def _infer_provider(self, model: str) -> str:
        """根据模型 ID 推断服务商"""
        provider = self._model_to_provider.get(model)
        return provider

    def list_models(self) -> Dict[str, List[str]]:
        """返回各服务商支持的模型列表"""
        return {p: c["models"] for p, c in MODEL_PROVIDER_CONFIG.items()}

    def chat(
        self,
        prompt: str,
        model: Optional[str] = None,
        json_format: bool = False,
        temperature: int = 0.8,
        enable_thinking: bool = False,
        **kwargs
    ) -> str:
        """
        自动根据 model 推导 provider。
        """
        provider = self._infer_provider(model)
        client = self._get_client(provider)

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            **kwargs
        }
        
        if enable_thinking:
            payload["extra_body"] = {"enable_thinking": True}
        
        if json_format:
            payload["response_format"] = {"type": "json_object"}
                
        completion = client.chat.completions.create(**payload)
        content = completion.choices[0].message.content
        
        return content


# 初始化服务实例
llm_service = LLMService()

def get_available_models():
    """获取所有可用的模型"""
    available = llm_service.list_models()
    flat_models = [m for models in available.values() for m in models]
    return flat_models


def get_llm_chat(
    prompt: str,
    model: str,
    json_format: bool = False,
    enable_thinking: bool = False,
    **kwargs
) -> str:
    """
    调用大模型服务并返回结果，添加重试机制。

    Args:
        prompt (str): 用户的输入提示。
        model (Optional[str]): 指定要使用的模型ID。
        json_format (bool): 是否请求 JSON 格式的输出。
        temperature (Optional[float]): 生成温度，如果为 None 则使用配置文件中的值。
        enable_thinking (bool): 是否启用思考模式（如果服务商支持）。
        max_retries (Optional[int]): 最大重试次数，如果为 None 则使用配置文件中的值。
        retry_interval (Optional[float]): 重试间隔（秒），如果为 None 则使用配置文件中的值。
        **kwargs: 传递给 client.chat.completions.create 的额外参数。

    Returns:
        str: 模型的回复内容。
        None: 发生错误
    """
    global llm_service  # 使用全局的 llm_service 实例

    max_retries = LLM_CONFIG.get("max_retries", 3)
    retry_interval = LLM_CONFIG.get("retry_interval", 2.0)
    temperature = LLM_CONFIG.get("temperature", 0.0)

    for attempt in range(1, max_retries + 1):
        try:
            response = llm_service.chat(
                prompt=prompt,
                model=model,
                json_format=json_format,
                temperature=temperature,
                enable_thinking=enable_thinking,
                **kwargs
            )
            return response
        except Exception as e:
            logger.warning(f"调用大模型服务时发生错误 [{attempt}/{max_retries}]: {e}")
            if attempt < max_retries:
                time.sleep(retry_interval)
            else:
                logger.warning("调用大模型服务时发生错误, 已到重试上限, 将返回 None")
                return None


if __name__ == "__main__":
    # 指定模型
    res = get_llm_chat(
        "写一个 Python 冒泡排序", 
        model="gpt-4o-mini",
        # model="deepseek-r1",
        # model="gpt-4o",
        json_format=False,
        enable_thinking=False
    )
    print(res)
