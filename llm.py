"""
NovelWeaver API - LLM 调用层
封装 OpenAI 兼容接口的同步/异步调用逻辑
"""

import asyncio
import logging
import time
from typing import Optional, AsyncGenerator, List, Dict
from functools import partial

import httpx
from openai import OpenAI

logger = logging.getLogger(__name__)


def create_client(model_config: dict, httpx_limits: dict) -> OpenAI:
    """
    根据模型配置创建 OpenAI 客户端。
    支持通过环境变量覆盖 api_key: 若 config 中 api_key 以 'env:' 开头，
    则从环境变量读取（如 "env:DEEPSEEK_API_KEY" → os.environ["DEEPSEEK_API_KEY"]）。
    """
    import os

    api_key = model_config["api_key"]
    if api_key.startswith("env:"):
        env_var = api_key[4:]
        api_key = os.environ.get(env_var)
        if not api_key:
            raise ValueError(f"Environment variable '{env_var}' not set (required by model config)")

    http_client = httpx.Client(
        limits=httpx.Limits(
            max_keepalive_connections=httpx_limits.get("max_keepalive_connections", 200),
            max_connections=httpx_limits.get("max_connections", 1000),
        ),
        timeout=httpx.Timeout(
            httpx_limits.get("timeout", 300.0),
            connect=httpx_limits.get("connect_timeout", 15.0),
        ),
    )
    return OpenAI(
        api_key=api_key,
        base_url=model_config["base_url"],
        http_client=http_client,
    )


def _build_messages(prompt: str, system_prompt: Optional[str] = None) -> List[Dict]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def call_llm_sync(
    client: OpenAI,
    model_name: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0,
    max_retries: int = 3,
    retry_delay: float = 5.0,
    max_tokens: Optional[int] = None,
    reasoning_effort: Optional[str] = None,
) -> str:
    """同步调用 LLM（阻塞式，仅在线程池中使用）"""
    messages = _build_messages(prompt, system_prompt)
    kwargs = {"model": model_name, "messages": messages, "temperature": temperature}
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    if reasoning_effort:
        kwargs["extra_body"] = {"thinking": {"type": "enabled", "reasoning_effort": reasoning_effort}}

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(**kwargs)
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            logger.warning(f"LLM call attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise


async def call_llm_async(
    client: OpenAI,
    model_name: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0,
    max_retries: int = 3,
    retry_delay: float = 5.0,
    max_tokens: Optional[int] = None,
    reasoning_effort: Optional[str] = None,
) -> str:
    """
    异步调用 LLM。
    将阻塞的同步调用放入线程池，避免阻塞 asyncio 事件循环。
    """
    loop = asyncio.get_running_loop()
    func = partial(
        call_llm_sync,
        client=client,
        model_name=model_name,
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        max_retries=max_retries,
        retry_delay=retry_delay,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
    )
    return await loop.run_in_executor(None, func)


def call_llm_stream_sync(
    client: OpenAI,
    model_name: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0,
    max_tokens: Optional[int] = None,
    reasoning_effort: Optional[str] = None,
):
    """
    同步流式调用 LLM，返回 (chunk_text) 迭代器。
    仅在线程池中使用。
    """
    messages = _build_messages(prompt, system_prompt)
    kwargs = {"model": model_name, "messages": messages, "temperature": temperature, "stream": True}
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    if reasoning_effort:
        kwargs["extra_body"] = {"thinking": {"type": "enabled", "reasoning_effort": reasoning_effort}}

    response = client.chat.completions.create(**kwargs)
    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


async def call_llm_stream_async(
    client: OpenAI,
    model_name: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0,
    max_tokens: Optional[int] = None,
    reasoning_effort: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """
    异步流式调用 LLM，返回 async generator。
    内部使用线程池 + queue 桥接同步流到异步。
    """
    import queue
    import threading

    _SENTINEL = object()
    q: queue.Queue = queue.Queue(maxsize=64)

    def _producer():
        try:
            for chunk in call_llm_stream_sync(
                client=client,
                model_name=model_name,
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
            ):
                q.put(chunk)
        except Exception as e:
            q.put(e)
        finally:
            q.put(_SENTINEL)

    thread = threading.Thread(target=_producer, daemon=True)
    thread.start()

    loop = asyncio.get_running_loop()
    while True:
        item = await loop.run_in_executor(None, q.get)
        if item is _SENTINEL:
            break
        if isinstance(item, Exception):
            raise item
        yield item
