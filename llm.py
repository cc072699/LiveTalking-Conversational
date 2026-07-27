import time
import os
import yaml
import httpx
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from avatars.base_avatar import BaseAvatar
from utils.logger import logger


def _load_system_prompt():
    """从配置文件加载 System Prompt，支持热更新（每次请求重新读取）"""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prompt_config.yaml')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config.get('system_prompt', '你是一位专业的AI助手。')
    except Exception as e:
        logger.warning(f"Failed to load prompt config: {e}")
        return '你是一位专业的AI助手。'


def llm_response(message, avatar_session: 'BaseAvatar', datainfo: dict = {}):
    try:
        opt = avatar_session.opt
        start = time.perf_counter()

        from openai import OpenAI

        provider = os.getenv("LLM_PROVIDER", "dashscope")

        if provider == "sensenova":
            client = OpenAI(
                api_key=os.getenv("SENSNOVA_API_KEY"),
                base_url=os.getenv("SENSNOVA_BASE_URL", "https://token.sensenova.cn/v1"),
                timeout=httpx.Timeout(60.0, connect=10.0),
            )
            model = os.getenv("SENSNOVA_MODEL", "sensenova-6.7-flash-lite")
        else:
            client = OpenAI(
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                timeout=httpx.Timeout(60.0, connect=10.0),
            )
            model = os.getenv("DASHSCOPE_MODEL", "qwen-plus")

        end = time.perf_counter()
        logger.info(f"llm Time init: {end-start}s, provider={provider}, model={model}")

        completion = client.chat.completions.create(
            model=model,
            messages=[{'role': 'system', 'content': _load_system_prompt()},
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
                    logger.info(f"llm Time to first chunk: {end-start}s")
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
        logger.info(f"llm Time to last chunk: {end-start}s")
        if result:
            avatar_session.put_msg_txt(result, datainfo)

    except Exception as e:
        logger.exception('llm exception:')
        return
