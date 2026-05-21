import json
import logging
import os
import sys

import colorlog
import yaml


# ==================== 配置读取函数 ====================

def load_config(config_path: str = "env.yaml") -> dict:
    """
    读取 YAML 配置文件
    
    :param config_path: 配置文件路径，默认为 env.yaml
    :return: 配置字典
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件未找到: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    return config


def get_llm_provider_config(config_path: str = "env.yaml") -> dict:
    """
    从配置文件中获取 LLM 提供商配置
    
    :param config_path: 配置文件路径，默认为 env.yaml
    :return: LLM 提供商配置字典，格式与 MODEL_PROVIDER_CONFIG 相同
    """
    config = load_config(config_path)
    llm_providers = config.get("llm_providers", {})
    
    # 转换为 MODEL_PROVIDER_CONFIG 格式
    provider_config = {}
    for provider, provider_info in llm_providers.items():
        provider_config[provider] = {
            "api_key": provider_info.get("api_key", ""),
            "base_url": provider_info.get("base_url", ""),
            "models": provider_info.get("models", [])
        }
    
    return provider_config


def get_llm_config(config_path: str = "env.yaml") -> dict:
    """
    从配置文件中获取 LLM 通用配置
    
    :param config_path: 配置文件路径，默认为 env.yaml
    :return: LLM 配置字典（temperature, max_retries, retry_interval, timeout）
    """
    config = load_config(config_path)
    return config.get("llm_config", {})


def get_logging_config(config_path: str = "env.yaml") -> dict:
    """
    从配置文件中获取日志配置
    
    :param config_path: 配置文件路径，默认为 env.yaml
    :return: 日志配置字典
    """
    config = load_config(config_path)
    return config.get("logging", {})


# ==================== 日志系统 ====================

def get_logger(level=None):
    """
    配置带颜色的日志系统
    
    :param level: 日志级别，如果为 None 则从配置文件读取
    """
    # 如果未指定级别，尝试从配置文件读取
    if level is None:
        try:
            logging_config = get_logging_config()
            level_str = logging_config.get("level", "INFO").upper()
            level = getattr(logging, level_str, logging.INFO)
        except Exception:
            # 如果读取配置失败，使用默认值
            level = logging.INFO
    
    # 1. 定义颜色格式
    log_format = "%(log_color)s%(asctime)s - %(levelname)s - %(message)s"
    
    colors = {
        'DEBUG':    'cyan',
        'INFO':     'green',
        'WARNING':  'yellow',
        'ERROR':    'red',
        'CRITICAL': 'red,bg_white',
    }

    formatter = colorlog.ColoredFormatter(
        log_format,
        datefmt='%Y-%m-%d %H:%M:%S',
        reset=True,
        log_colors=colors
    )

    # 2. 设置标准输出 Handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    # 3. 配置根日志记录器
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # 防止重复添加 Handler
    if not logger.handlers:
        logger.addHandler(handler)

    # 4. 屏蔽第三方库的冗余日志
    logging.getLogger("httpx").setLevel(logging.ERROR)
    
    return logger

# 创建单例
logger = get_logger()

def load_json(filepath):
    """
    读取 .json 或 .jsonl 文件
    :param filepath: 文件路径
    :return: Python 对象(list 或 dict)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件未找到: {filepath}")

    _, ext = os.path.splitext(filepath)
    ext = ext.lower()

    with open(filepath, "r", encoding="utf-8") as f:
        if ext == ".json":
            return json.load(f)
        elif ext == ".jsonl":
            return [json.loads(line) for line in f.readlines() if line.strip()]
        else:
            raise ValueError(f"不支持的文件类型: {ext}")


def save_json(data, filepath):
    """
    保存数据到 .json 文件
    :param data: Python 对象(list 或 dict)
    :param filepath: 文件路径
    """
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return filepath


def convert_json_to_jsonl(input_dir, output_dir='temp'):
    """将 JSON 文件转换为 JSONL 文件"""

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):

        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename.replace(".json", ".jsonl"))

        data = load_json(input_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"转换完成: {input_path} -> {output_path}")


def restore_progress(
    last_output_path: str, 
    output_path: str
) -> int:
    """加载上次保存的输出, 返回索引"""
    
    if not os.path.exists(last_output_path):
        logger.info(f"无法从 {last_output_path} 恢复进度")
        return -1

    last_output = load_json(last_output_path)
    with open(output_path, "a", encoding="utf-8") as f:
        for idx, item in enumerate(last_output):
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return len(last_output) - 1


def sanitize_filename(name: str) -> str:
    # 替换 Windows 禁用字符 + 其他高风险字符
    unsafe_chars = r'<>:"/\\|?*'
    for ch in unsafe_chars:
        name = name.replace(ch, "-")
    
    # 可选：进一步只保留字母、数字、点、连字符、下划线
    # name = re.sub(r"[^a-zA-Z0-9._\-]", "_", name)
    
    # 去除首尾空格和点（避免隐藏文件或无效名）
    name = name.strip().strip(".")
    
    # 避免保留设备名（Windows）
    if name.upper() in {"CON", "PRN", "AUX", "NUL"} or \
       (name.upper().startswith(("COM", "LPT")) and len(name) <= 4 and name[3:].isdigit()):
        name = "_" + name
    
    # 限制长度（可选，Windows 路径总长 ≤ 260）
    return name[:255]  # 大多数文件系统支持的最大文件名长度