"""
NovelWeaver API Server
提供两个核心接口：固定节点拆分、AI节点扩展
"""

import json
import os
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import AliasChoices, BaseModel, ConfigDict, Field
import re
import glob as glob_module

import prompt_builder
import llm
import parsers

# ---------------------------------------------------------------------------
# 配置加载
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("novelweaver-api")


# ---------------------------------------------------------------------------
# 客户端缓存（asyncio.Lock 保护，防止并发重复创建）
# ---------------------------------------------------------------------------

_clients = {}
_clients_lock = asyncio.Lock()


async def get_client(model_name: str):
    """获取或创建 OpenAI 客户端（带缓存 + 异步锁）"""
    if model_name in _clients:
        return _clients[model_name]
    async with _clients_lock:
        # double-check：拿锁后再检查一次
        if model_name not in _clients:
            model_config = CONFIG["models"].get(model_name)
            if not model_config:
                raise ValueError(f"Model '{model_name}' not found in config")
            _clients[model_name] = llm.create_client(model_config, CONFIG.get("httpx_limits", {}))
        return _clients[model_name]


# ---------------------------------------------------------------------------
# Pydantic 请求/响应模型
# ---------------------------------------------------------------------------

# ===== 共用模型 =====

class PrologueItem(BaseModel):
    """开场白条目"""
    role: str = Field(..., description="角色名，如'旁白'、'林亦扬'")
    content: str = Field(..., description="台词/叙述内容")


class PlotEnding(BaseModel):
    """剧情结尾"""
    id: int = Field(..., description="结尾编号")
    description: str = Field(..., description="结尾描述")


# ===== 固定节点 =====

class FixedNodeRequest(BaseModel):
    """固定节点拆分请求"""
    plot_text: List[PrologueItem] = Field(..., description="固定剧情对话列表，每项含 role 和 content")
    available_music_types: List[str] = Field(
        default_factory=list,
        description="已录入的音乐类型列表，如 ['快乐', '轻松', '悲伤', '悬疑', '紧张']"
    )
    model: Optional[str] = Field(None, description="覆盖默认模型名称（测试用）")


class FixedNodeItem(BaseModel):
    """单个固定节点"""
    order: int = Field(..., description="节点顺序（从1开始）")
    plot_text: List[PrologueItem] = Field(default_factory=list, description="本节点的对话内容")
    image_prompt: str = Field(..., description="背景图提示词")
    music_type: str = Field(..., description="音乐类型")
    is_new_music_type: bool = Field(False, description="是否为建议新增的音乐类型")


class FixedNodeResponse(BaseModel):
    """固定节点拆分响应"""
    nodes: List[FixedNodeItem]
    raw_output: Optional[str] = Field(None, description="LLM原始输出（调试用）")


# ===== AI节点 =====


class AINodeRequest(BaseModel):
    """AI节点扩展请求"""
    model_config = ConfigDict(populate_by_name=True)

    # --- 核心剧本信息 ---
    previous_plot: str = Field("", description="之前节点的全部剧情（背景摘要）。若为空且 previous_nodes 有值，将自动调用 LLM 生成。")
    previous_nodes: Optional[List[Dict]] = Field(None, description="前序节点数组（novelweaver-api 输出格式）。当 previous_plot 为空时，服务端会自动调用 LLM 生成背景摘要。")
    prologue: List[PrologueItem] = Field(..., description="开场白角色及内容")
    task: str = Field("", validation_alias=AliasChoices("task", "strategy"), description="任务")
    plot_start: str = Field(..., description="剧情开端")
    plot_process: str = Field(..., description="剧情过程")
    plot_endings: List[PlotEnding] = Field(..., description="剧情结尾（可多个）")
    ai_reply_style: Optional[str] = Field(None, description="AI回复要求（语气风格、互动规则等）")
    image_requirements: Optional[str] = Field(None, description="背景图要求（作者提出）")

    # --- 角色信息 ---
    user_role: str = Field("用户", description="用户扮演的角色名")
    roles: List[Dict[str, str]] = Field(
        default_factory=list,
        description="小说完整角色表，每个元素 {role_name, description}。API 通过字符串匹配自动提取参与本场景的角色。"
    )

    # --- 小说全局元数据 ---
    attributes: List[str] = Field(
        default_factory=list,
        description="本小说的全部属性值名称，如 ['灵敏值', '智慧值', '精神值']"
    )
    favorabilities: List[str] = Field(
        default_factory=list,
        description="本小说的全部好感度名称，如 ['林亦扬好感度', '吴魏好感度']"
    )
    existing_events: List[str] = Field(
        default_factory=list,
        description="本小说已有的全部事件名称（用于排重）"
    )
    existing_easter_eggs: List[str] = Field(
        default_factory=list,
        description="本小说已有的全部彩蛋名称（用于排重）"
    )
    available_music_types: List[str] = Field(
        default_factory=list,
        description="本小说已录入的音乐类型列表"
    )
    model: Optional[str] = Field(None, description="覆盖默认模型名称（测试用）")


class UserMessageBranch(BaseModel):
    """用户消息分支"""
    trigger: str = Field(..., description="触发条件")
    ai_reply: str = Field(..., description="AI回复策略描述")
    progress_name: str = Field(..., description="进度指标名称")
    progress_gain: int = Field(..., description="进度增量 (0/40/50/100)")
    examples: List[str] = Field(default_factory=list, description="回复示例")


class EasterEgg(BaseModel):
    """彩蛋"""
    name: str = Field(..., description="彩蛋名称（≤8字）")
    trigger: str = Field(..., description="触发条件（用户的具体输入内容）")


class AttributeReward(BaseModel):
    """属性奖励条目"""
    name: str = Field(..., description="属性名称，如 '智慧值'")
    value: int = Field(..., description="变化量，如 10 或 -5")


class FavorabilityReward(BaseModel):
    """好感度奖励条目"""
    name: str = Field(..., description="好感度名称，如 '林亦扬好感度'")
    value: int = Field(..., description="变化量，如 10 或 -5")


class RewardInfo(BaseModel):
    """奖励信息"""
    event: str = Field(..., description="事件名称（≤8字）")
    attributes: List[AttributeReward] = Field(
        default_factory=list,
        description="属性奖励列表"
    )
    favorabilities: List[FavorabilityReward] = Field(
        default_factory=list,
        description="好感度变化列表"
    )


class ProgressIndicator(BaseModel):
    """进度指标"""
    name: str = Field(..., description="进度指标名称")
    min_value: int = Field(0, description="最小值")
    max_value: int = Field(100, description="最大值")
    linked_ending: str = Field(..., description="关联的结尾描述")
    ending_example: Optional[str] = Field(None, description="结尾输出示例")
    rewards: Optional[RewardInfo] = Field(None, description="通关奖励（无关值为null）")


class CharacterStatus(BaseModel):
    """角色状态"""
    role: str = Field(..., description="角色名")
    description: str = Field(..., description="状态描述")


class CurrentPlot(BaseModel):
    """当前剧情"""
    start: str = Field(..., description="剧情开始")
    process: str = Field(..., description="剧情过程")
    plot_endings: List[PlotEnding] = Field(default_factory=list, description="剧情结尾列表")


class AINodeResponse(BaseModel):
    """AI节点扩展响应"""
    node_title: str = Field(..., description="节点标题（≤8字）")
    image_prompt: str = Field(..., description="背景图提示词")
    music_type: str = Field(..., description="音乐类型")
    is_new_music_type: bool = Field(False, description="是否为建议新增的音乐类型")
    prologue: List[PrologueItem] = Field(..., description="开场白角色及内容")
    background: str = Field(..., description="故事背景")
    character_status: List[CharacterStatus] = Field(..., description="人物空间关系与状态")
    current_plot: CurrentPlot = Field(..., description="当前剧情")
    task: str = Field(..., description="任务描述")
    user_messages: List[UserMessageBranch] = Field(..., description="用户消息分支")
    easter_eggs: List[EasterEgg] = Field(default_factory=list, description="彩蛋列表")
    progress_indicators: List[ProgressIndicator] = Field(..., description="进度指标")
    needs_regeneration: bool = Field(False, description="是否需要重新生成（进度格式不合规时为True）")
    raw_output: Optional[str] = Field(None, description="LLM原始输出（调试用）")


# ---------------------------------------------------------------------------
# FastAPI 应用
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 线程池配置（支撑高并发 LLM 调用）
# ---------------------------------------------------------------------------
# 每个非流式 LLM 调用占 1 个线程，流式调用占 2 个线程。
# 默认线程池只有 ~8-12 个线程，无法支撑 50 并发。
# 此处显式设置为 128 个线程，足够支撑 50+ 并发请求。
LLM_THREAD_POOL = ThreadPoolExecutor(
    max_workers=CONFIG.get("max_llm_threads", 128),
    thread_name_prefix="llm-worker",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 将默认线程池替换为大容量池，所有 run_in_executor(None, ...) 都会用这个池
    loop = asyncio.get_running_loop()
    loop.set_default_executor(LLM_THREAD_POOL)
    logger.info(f"NovelWeaver API starting... (thread pool: {LLM_THREAD_POOL._max_workers} workers)")
    yield
    LLM_THREAD_POOL.shutdown(wait=False)
    logger.info("NovelWeaver API shutting down...")


app = FastAPI(
    title="NovelWeaver API",
    description="互动叙事节点生成 API：固定节点拆分 & AI节点扩展",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS（允许前端跨域访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=CONFIG.get("cors_origins", ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# 公共辅助：提示词组装 + 客户端获取
# ---------------------------------------------------------------------------

async def _prepare_fixed_node(request: FixedNodeRequest):
    """组装固定节点提示词并获取 LLM 客户端，返回 (prompt, client, model_name, temperature, reasoning_effort)"""
    # 将 PrologueItem 列表转为 "角色：内容" 的多行字符串
    plot_text_str = "\n".join(f"{item.role}：{item.content}" for item in request.plot_text)
    prompt = prompt_builder.generate_fixed_node_split_prompt(
        plot_text=plot_text_str,
        available_music_types=request.available_music_types,
    )
    agent_config = CONFIG["agents"]["fixed_node_split"]
    model_name = request.model or agent_config["model"]
    client = await get_client(model_name)
    temperature = agent_config.get("temperature", 0)
    # 从模型配置读取 reasoning_effort 和 actual_model（如有）
    model_config = CONFIG["models"].get(model_name, {})
    reasoning_effort = model_config.get("reasoning_effort")
    api_model_name = model_config.get("actual_model", model_name)
    return prompt, client, api_model_name, temperature, reasoning_effort


async def _generate_background_summary(previous_nodes_text: str, model_override: Optional[str] = None) -> str:
    """调用 LLM 将前序节点结构化文本生成背景摘要"""
    prompt = prompt_builder.generate_background_prompt(previous_nodes_text)
    agent_config = CONFIG["agents"].get("background_summary", CONFIG["agents"]["ai_node_expansion"])
    model_name = model_override or agent_config["model"]
    client = await get_client(model_name)
    temperature = agent_config.get("temperature", 0)
    model_config = CONFIG["models"].get(model_name, {})
    reasoning_effort = model_config.get("reasoning_effort")
    api_model_name = model_config.get("actual_model", model_name)

    logger.info(f"Auto-generating background summary with model={api_model_name}")
    summary = await llm.call_llm_async(
        client=client,
        model_name=api_model_name,
        prompt=prompt,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
    )
    logger.info(f"Background summary generated: {len(summary)} chars")
    return summary


async def _prepare_ai_node(request: AINodeRequest):
    """组装 AI 节点提示词并获取 LLM 客户端，返回 (prompt, client, model_name, temperature, reasoning_effort)"""
    # 如果 previous_plot 为空但 previous_nodes 有值，先自动生成背景摘要
    background = request.previous_plot
    if not background.strip() and request.previous_nodes:
        prev_node_text = prompt_builder.serialize_previous_nodes(request.previous_nodes)
        background = await _generate_background_summary(prev_node_text, request.model)

    # 从完整角色表中自动提取参与本场景的角色
    plot_text = "\n".join([
        request.plot_start,
        request.plot_process,
        request.task,
        "\n".join(p.content for p in request.prologue),
        "\n".join(e.description for e in request.plot_endings),
    ])
    involved_roles = []
    for role in request.roles:
        role_name = role.get("role_name", "")
        if not role_name:
            continue
        if role_name in plot_text or role_name == request.user_role:
            involved_roles.append({
                "name": role_name,
                "description": role.get("description", ""),
            })

    prompt = prompt_builder.generate_ai_node_expansion_prompt(
        background=background,
        prologue=[p.model_dump() for p in request.prologue],
        task=request.task,
        plot_start=request.plot_start,
        plot_process=request.plot_process,
        plot_endings=[e.model_dump() for e in request.plot_endings],
        ai_reply_style=request.ai_reply_style,
        image_requirements=request.image_requirements,
        user_role=request.user_role,
        involved_roles=involved_roles,
        attributes=request.attributes,
        favorabilities=request.favorabilities,
        existing_events=request.existing_events,
        existing_easter_eggs=request.existing_easter_eggs,
        available_music_types=request.available_music_types,
    )
    agent_config = CONFIG["agents"]["ai_node_expansion"]
    model_name = request.model or agent_config["model"]
    client = await get_client(model_name)
    temperature = agent_config.get("temperature", 0)
    # 从模型配置读取 reasoning_effort 和 actual_model（如有）
    model_config = CONFIG["models"].get(model_name, {})
    reasoning_effort = model_config.get("reasoning_effort")
    api_model_name = model_config.get("actual_model", model_name)
    return prompt, client, api_model_name, temperature, reasoning_effort


# ---------------------------------------------------------------------------
# 端点: 固定节点拆分
# ---------------------------------------------------------------------------

@app.post("/api/fixed_node", response_model=FixedNodeResponse, summary="固定节点拆分")
async def split_fixed_node(request: FixedNodeRequest):
    """
    将一整段固定剧情拆分为多个固定节点。
    每个节点包含：剧情文本、背景图提示词、音乐类型。
    """
    try:
        prompt, client, model_name, temperature, reasoning_effort = await _prepare_fixed_node(request)

        raw_output = await llm.call_llm_async(
            client=client,
            model_name=model_name,
            prompt=prompt,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )

        nodes = parsers.parse_fixed_node_output(raw_output)

        return FixedNodeResponse(
            nodes=[FixedNodeItem(**n) for n in nodes],
            raw_output=raw_output,
        )

    except Exception as e:
        logger.error(f"Fixed node split failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/fixed_node/stream", summary="固定节点拆分（流式）")
async def split_fixed_node_stream(request: FixedNodeRequest):
    """
    固定节点拆分的流式版本，实时返回 LLM 输出。
    最后一条消息包含解析后的结构化 JSON。
    """
    prompt, client, model_name, temperature, reasoning_effort = await _prepare_fixed_node(request)

    async def generate():
        full_text = []
        try:
            async for chunk in llm.call_llm_stream_async(
                client=client,
                model_name=model_name,
                prompt=prompt,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
            ):
                full_text.append(chunk)
                yield json.dumps({"type": "chunk", "content": chunk}, ensure_ascii=False) + "\n"

            raw_output = "".join(full_text)
            nodes = parsers.parse_fixed_node_output(raw_output)
            yield json.dumps({"type": "result", "data": {"nodes": nodes}}, ensure_ascii=False) + "\n"
        except Exception as e:
            logger.error(f"Fixed node stream error: {e}", exc_info=True)
            yield json.dumps({"type": "error", "message": str(e)}, ensure_ascii=False) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


# ---------------------------------------------------------------------------
# 端点: AI 节点扩展
# ---------------------------------------------------------------------------

MAX_REGENERATION_RETRIES = 2  # 进度格式不合规时最多重试次数


@app.post("/api/ai_node", response_model=AINodeResponse, summary="AI节点扩展")
async def expand_ai_node(request: AINodeRequest):
    """
    将简略的AI节点剧本大纲扩展为完整的互动节点配置。
    输出包含：背景、分支逻辑、进度指标、奖励、彩蛋、图片提示词、音乐类型等。
    当进度格式不合规时，自动重试最多 MAX_REGENERATION_RETRIES 次。
    """
    try:
        prompt, client, model_name, temperature, reasoning_effort = await _prepare_ai_node(request)

        parsed = None
        for attempt in range(1 + MAX_REGENERATION_RETRIES):
            raw_output = await llm.call_llm_async(
                client=client,
                model_name=model_name,
                prompt=prompt,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
            )
            parsed = parsers.parse_ai_node_output(raw_output)
            parsed["raw_output"] = raw_output

            if not parsed.get("needs_regeneration"):
                break
            logger.warning(
                f"Progress format invalid, regenerating (attempt {attempt + 1}/{1 + MAX_REGENERATION_RETRIES})"
            )

        return AINodeResponse(**parsed)

    except Exception as e:
        logger.error(f"AI node expansion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ai_node/stream", summary="AI节点扩展（流式）")
async def expand_ai_node_stream(request: AINodeRequest):
    """
    AI节点扩展的流式版本，实时返回 LLM 输出。
    最后一条消息包含解析后的结构化 JSON。
    当进度格式不合规时，自动重试最多 MAX_REGENERATION_RETRIES 次。
    """
    prompt, client, model_name, temperature, reasoning_effort = await _prepare_ai_node(request)

    async def generate():
        for attempt in range(1 + MAX_REGENERATION_RETRIES):
            full_text = []
            try:
                if attempt > 0:
                    yield json.dumps({
                        "type": "retry",
                        "message": f"进度格式不合规，正在重新生成（第 {attempt + 1} 次）…"
                    }, ensure_ascii=False) + "\n"

                async for chunk in llm.call_llm_stream_async(
                    client=client,
                    model_name=model_name,
                    prompt=prompt,
                    temperature=temperature,
                    reasoning_effort=reasoning_effort,
                ):
                    full_text.append(chunk)
                    yield json.dumps({"type": "chunk", "content": chunk}, ensure_ascii=False) + "\n"

                raw_output = "".join(full_text)
                parsed = parsers.parse_ai_node_output(raw_output)

                if parsed.get("needs_regeneration") and attempt < MAX_REGENERATION_RETRIES:
                    logger.warning(
                        f"Progress format invalid in stream, regenerating "
                        f"(attempt {attempt + 1}/{1 + MAX_REGENERATION_RETRIES})"
                    )
                    continue

                yield json.dumps({"type": "result", "data": parsed}, ensure_ascii=False) + "\n"
                return
            except Exception as e:
                logger.error(f"AI node stream error: {e}", exc_info=True)
                yield json.dumps({"type": "error", "message": str(e)}, ensure_ascii=False) + "\n"
                return

    return StreamingResponse(generate(), media_type="application/x-ndjson")


# ---------------------------------------------------------------------------
# 健康检查
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# 测试输出文件发现
# ---------------------------------------------------------------------------

@app.get("/api/test_manifest", summary="扫描 tests/outputs 目录，返回可用的输入和模型列表")
async def test_manifest():
    outputs_dir = os.path.join(BASE_DIR, "tests", "outputs")
    if not os.path.isdir(outputs_dir):
        return {"ai_inputs": [], "fixed_inputs": [], "models": []}

    ai_inputs = set()
    fixed_inputs = set()
    models = set()

    pattern_ai = re.compile(r"^test_ai_node_input_(\d+)_output_(.+)\.json$")
    pattern_fixed = re.compile(r"^test_fixed_node_input_(\d+)_output_(.+)\.json$")

    for fname in os.listdir(outputs_dir):
        m = pattern_ai.match(fname)
        if m:
            ai_inputs.add(int(m.group(1)))
            models.add(m.group(2))
            continue
        m = pattern_fixed.match(fname)
        if m:
            fixed_inputs.add(int(m.group(1)))
            models.add(m.group(2))

    return {
        "ai_inputs": sorted(ai_inputs),
        "fixed_inputs": sorted(fixed_inputs),
        "models": sorted(models),
    }


# ---------------------------------------------------------------------------
# 静态文件 & 查看器
# ---------------------------------------------------------------------------

# 挂载 tests/outputs 目录，使 viewer 可以通过 /test_outputs/ 访问 JSON 文件
_test_outputs_dir = os.path.join(BASE_DIR, "tests", "outputs")
if os.path.isdir(_test_outputs_dir):
    app.mount("/test_outputs", StaticFiles(directory=_test_outputs_dir), name="test_outputs")

# 提供 viewer.html
@app.get("/viewer", include_in_schema=False)
async def viewer():
    viewer_path = os.path.join(BASE_DIR, "viewer.html")
    if os.path.isfile(viewer_path):
        return FileResponse(viewer_path, media_type="text/html")
    raise HTTPException(status_code=404, detail="viewer.html not found")


# ---------------------------------------------------------------------------
# 启动入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    server_config = CONFIG.get("server", {})
    is_dev = server_config.get("dev_mode", True)

    uvicorn.run(
        "server:app",
        host=server_config.get("host", "0.0.0.0"),
        port=server_config.get("port", 8100),
        reload=is_dev,
        workers=1 if is_dev else server_config.get("workers", 4),
    )
