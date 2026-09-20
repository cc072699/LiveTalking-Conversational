import time
import os
import yaml
import httpx
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from avatars.base_avatar import BaseAvatar
from utils.logger import logger

# ── LLM Client 单例缓存（避免每次请求重新建立 HTTP 连接）────────────────────
_llm_clients: dict = {}

def _get_llm_client(provider: str):
    """按 provider 返回复用的 OpenAI client 单例，避免每次请求重建连接"""
    global _llm_clients
    if provider in _llm_clients:
        return _llm_clients[provider]
    from openai import OpenAI
    if provider == "sensenova":
        client = OpenAI(
            api_key=os.getenv("SENSNOVA_API_KEY"),
            base_url=os.getenv("SENSNOVA_BASE_URL", "https://token.sensenova.cn/v1"),
            timeout=httpx.Timeout(60.0, connect=10.0),
        )
    else:
        client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            timeout=httpx.Timeout(60.0, connect=10.0),
        )
    _llm_clients[provider] = client
    logger.info(f"LLM client created for provider={provider}")
    return client


def _load_prompts():
    """从配置文件加载所有 prompt 预设，支持热更新"""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prompt_config.yaml')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return {
            'prompts': config.get('prompts', {}),
            'default': config.get('default', ''),
        }
    except Exception as e:
        logger.warning(f"Failed to load prompt config: {e}")
        return {'prompts': {}, 'default': ''}


def _get_system_prompt(prompt_key: str = ''):
    """根据 key 获取 system prompt；空/未找到时取默认"""
    cfg = _load_prompts()
    prompts = cfg.get('prompts', {})
    if prompt_key and prompt_key in prompts:
        return prompts[prompt_key].get('content', '你是一位专业的AI助手。')
    default_key = cfg.get('default', '')
    if default_key and default_key in prompts:
        return prompts[default_key].get('content', '你是一位专业的AI助手。')
    # 兜底：取第一个
    if prompts:
        first = next(iter(prompts.values()))
        return first.get('content', '你是一位专业的AI助手。')
    return '你是一位专业的AI助手。'


def _load_system_prompt():
    """向后兼容：返回默认 prompt"""
    return _get_system_prompt()


def llm_response(message, avatar_session: 'BaseAvatar', datainfo: dict = {}):
    try:
        opt = avatar_session.opt
        start = time.perf_counter()

        provider = os.getenv("LLM_PROVIDER", "dashscope")
        client = _get_llm_client(provider)

        if provider == "sensenova":
            model = os.getenv("SENSNOVA_MODEL", "sensenova-6.7-flash-lite")
        else:
            model = os.getenv("DASHSCOPE_MODEL", "qwen-plus")

        end = time.perf_counter()
        logger.info(f"llm provider={provider}, model={model}, init={end-start:.3f}s")

        prompt_key = getattr(opt, 'PROMPT_KEY', '') or ''
        system_prompt = _get_system_prompt(prompt_key)
        completion = client.chat.completions.create(
            model=model,
            messages=[{'role': 'system', 'content': system_prompt},
                      {'role': 'user', 'content': message}],
            stream=True,
            max_tokens=1024,
        )
        result = ""
        first = True
        for chunk in completion:
            if len(chunk.choices) > 0:
                if first:
                    end = time.perf_counter()
                    logger.info(f"llm Time to first chunk: {end-start:.3f}s")
                    first = False
                msg = chunk.choices[0].delta.content
                if msg is None:
                    continue
                lastpos = 0
                for i, char in enumerate(msg):
                    if char in ",.!;:，。！？：；":
                        result = result + msg[lastpos:i+1]
                        lastpos = i + 1
                        if len(result) > 10:
                            avatar_session.put_msg_txt(result, datainfo)
                            result = ""
                result = result + msg[lastpos:]
        end = time.perf_counter()
        logger.info(f"llm Time to last chunk: {end-start:.3f}s")
        if result:
            avatar_session.put_msg_txt(result, datainfo)

    except Exception as e:
        logger.exception('llm exception:')
        return
