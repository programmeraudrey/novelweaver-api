"""
NovelWeaver API - 输出解析器
将 LLM 生成的结构化文本转换为 JSON 字典
"""

import re
import logging
import textwrap
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Section 标记常量 (用于 _extract_section)
# ---------------------------------------------------------------------------

_SECTION_MARKERS = [
    r"【节点】", r"\[图片提示词\]", r"\[音乐类型\]", r"\[开场白\]",
    r"\[背景\]", r"\[人物空间关系与状态\]",
    r"\[当前剧情\]", r"\[任务\]", r"\[用户消息\]",
    r"\[进度指标\]",
]


# ---------------------------------------------------------------------------
# 公共工具
# ---------------------------------------------------------------------------

def _strip_line_indents(text: str) -> str:
    """去除每行的前导缩进空格（LLM 输出格式的缩进残留）"""
    return "\n".join(line.lstrip() for line in text.split("\n"))


def _parse_prologue_text(text: str) -> List[Dict[str, str]]:
    """将开场白纯文本解析为 [{role, content}] 格式"""
    items = []
    for line in _strip_line_indents(text).split("\n"):
        line = line.strip()
        if not line:
            continue
        m = re.match(r'^(.+?)[：:]\s*(.+)$', line)
        if m:
            items.append({"role": m.group(1).strip(), "content": m.group(2).strip()})
        else:
            items.append({"role": "旁白", "content": line})
    return items


def _strip_think_tags(text: str) -> str:
    """移除 <think>...</think> 推理标签"""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)


def _extract_section(text: str, header_pattern: str) -> str:
    """
    提取以 header_pattern 开头的节内容，直到下一个同级 section 标记或文末。
    不依赖 section 出现顺序——使用前瞻断言匹配任意下一个 section。
    """
    other_markers = [m for m in _SECTION_MARKERS if m != header_pattern]
    end_pattern = "|".join(other_markers)
    pattern = rf"{header_pattern}[:：]?\s*\n?(.*?)(?={end_pattern}|\Z)"
    m = re.search(pattern, text, re.DOTALL)
    if m:
        return m.group(1).strip()
    logger.debug(f"Section '{header_pattern}' not found in output")
    return ""


def _safe_int(value: Optional[str], default: int = 0) -> int:
    """安全转换整数"""
    if value is None:
        return default
    try:
        return int(value.strip())
    except (ValueError, AttributeError):
        return default


# ---------------------------------------------------------------------------
# AI 节点解析
# ---------------------------------------------------------------------------

def parse_ai_node_output(raw_text: str) -> Dict[str, Any]:
    """
    将 LLM 输出的 AI 节点文本解析为结构化字典。
    
    返回字段: node_title, image_prompt, music_type, is_new_music_type,
              prologue, background, character_status,
              current_plot, task, user_messages, progress_indicators
    """
    raw_text = _strip_think_tags(raw_text)
    result: Dict[str, Any] = {}
    parse_warnings: List[str] = []

    # --- 节点标题 ---
    m = re.search(r"【节点】[:：]\s*(.+)", raw_text)
    result["node_title"] = m.group(1).strip() if m else ""
    if not result["node_title"]:
        parse_warnings.append("missing: node_title")

    # --- 图片提示词 ---
    result["image_prompt"] = _extract_section(raw_text, r"\[图片提示词\]")

    # --- 音乐类型 ---
    music_raw = _extract_section(raw_text, r"\[音乐类型\]")
    if "建议新增" in music_raw:
        result["music_type"] = re.sub(r"建议新增[:：]?\s*", "", music_raw).strip()
        result["is_new_music_type"] = True
    else:
        result["music_type"] = music_raw.strip()
        result["is_new_music_type"] = False

    # --- 开场白 ---
    result["prologue"] = _parse_prologue_text(
        _extract_section(raw_text, r"\[开场白\]")
    )

    # --- 背景 ---
    result["background"] = _extract_section(raw_text, r"\[背景\]")

    # --- 人物空间关系与状态 ---
    result["character_status"] = _parse_character_status(
        _extract_section(raw_text, r"\[人物空间关系与状态\]")
    )

    # --- 当前剧情 ---
    result["current_plot"] = _parse_current_plot(
        _extract_section(raw_text, r"\[当前剧情\]")
    )

    # --- 任务 ---
    result["task"] = _extract_section(raw_text, r"\[任务\]")

    # --- 用户消息 ---
    result["user_messages"], has_invalid_progress = _parse_user_messages(
        _extract_section(raw_text, r"\[用户消息\]")
    )
    if not result["user_messages"]:
        parse_warnings.append("missing: user_messages (no branches parsed)")
    if has_invalid_progress:
        result["needs_regeneration"] = True
        parse_warnings.append("invalid_progress_format: 进度格式必须为 '进度值名+数字'")

    # --- 进度指标 ---
    result["progress_indicators"] = _parse_progress_indicators(
        _extract_section(raw_text, r"\[进度指标\]")
    )
    if not result["progress_indicators"]:
        parse_warnings.append("missing: progress_indicators")

    # --- 记录解析警告 ---
    if parse_warnings:
        logger.warning(f"AI node parse warnings: {parse_warnings}")

    return result


# ---------------------------------------------------------------------------
# 子解析器
# ---------------------------------------------------------------------------

def _parse_character_status(raw: str) -> List[Dict[str, str]]:
    """解析 [人物空间关系与状态]"""
    items = []
    # 支持多行描述：以 "- 角色名：" 开头，直到下一个 "- " 或结束
    entries = re.split(r"\n\s*-\s+", raw)
    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue
        # 去掉开头残留的 "- " 前缀（首个条目可能未被 split 切掉）
        entry = re.sub(r"^-\s+", "", entry)
        # 用第一个中/英文冒号分割角色名和描述
        for sep in ("：", ":"):
            if sep in entry:
                role, desc = entry.split(sep, 1)
                items.append({"role": role.strip(), "description": desc.strip()})
                break
    return items


def _parse_current_plot(raw: str) -> Dict[str, Any]:
    """解析 [当前剧情]（开始/过程/结尾N）"""
    plot: Dict[str, Any] = {"start": "", "process": "", "plot_endings": []}
    _ending_counter = 0

    current_key: Optional[str] = None
    current_lines: List[str] = []

    def _flush():
        nonlocal current_key, current_lines, _ending_counter
        if current_key is None:
            return
        text = "\n".join(line for line in current_lines if line)
        if current_key == "start":
            plot["start"] = text
        elif current_key == "process":
            plot["process"] = text
        elif current_key == "ending":
            _ending_counter += 1
            plot["plot_endings"].append({"id": _ending_counter, "description": text})
        current_key = None
        current_lines = []

    for line in raw.split("\n"):
        stripped = line.strip()
        if re.match(r"开始[:：]", stripped):
            _flush()
            current_key = "start"
            current_lines = [re.sub(r"^开始[:：]\s*", "", stripped)]
        elif re.match(r"过程[:：]", stripped):
            _flush()
            current_key = "process"
            current_lines = [re.sub(r"^过程[:：]\s*", "", stripped)]
        elif re.match(r"结尾\d*[:：]", stripped):
            _flush()
            current_key = "ending"
            current_lines = [re.sub(r"^结尾\d*[:：]\s*", "", stripped)]
        elif stripped:
            current_lines.append(stripped)

    _flush()
    return plot


def _parse_user_messages(raw: str) -> tuple:
    """
    解析 [用户消息] 的分支列表。
    返回 (branches, has_invalid_progress)
    """
    branches: List[Dict[str, Any]] = []
    has_invalid_progress = False
    parts = re.split(r"分支\d+[：:]", raw)

    for part in parts[1:]:  # 跳过首段空内容
        branch: Dict[str, Any] = {}

        # 触发条件
        m = re.search(r"触发条件[:：]\s*(.+?)(?=\n\s*AI回复[:：]|\Z)", part, re.DOTALL)
        branch["trigger"] = m.group(1).strip() if m else ""

        # AI回复
        m = re.search(r"AI回复[:：]\s*(.+?)(?=\n\s*进度[:：]|\Z)", part, re.DOTALL)
        branch["ai_reply"] = m.group(1).strip() if m else ""

        # 进度 —— 先提取原始进度文本进行格式校验
        progress_line_match = re.search(r"进度[:：]\s*(.+?)(?:\n|$)", part)
        progress_line = progress_line_match.group(1).strip() if progress_line_match else ""
        # 校验：必须是单一的 "进度值名+数字" 格式
        if progress_line and not re.fullmatch(r"[^\s+，,]+\s*\+\s*\d+", progress_line):
            has_invalid_progress = True
            logger.warning(f"Invalid progress format: '{progress_line}'")

        m = re.search(r"进度[:：]\s*(\S+)\s*\+(\d+)", part)
        branch["progress_name"] = m.group(1) if m else ""
        branch["progress_gain"] = int(m.group(2)) if m else 0

        # 彩蛋（可选）
        m = re.search(r"彩蛋[:：]\s*(.+?)(?=\n|$)", part)
        branch["egg_name"] = m.group(1).strip() if m else None

        # 示例（可选，可能多个）
        examples: List[str] = []
        example_parts = re.split(r"示例\d+[:：]", part)
        for ep in example_parts[1:]:
            # 截断到下一个分支标记前
            ep_clean = re.split(r"\n\s*分支\d+[：:]", ep)[0].strip()
            if ep_clean:
                # 去除每行的前导缩进空格
                ep_clean = _strip_line_indents(ep_clean)
                examples.append(ep_clean)
        branch["examples"] = examples

        branches.append(branch)

    # --- 彩蛋后处理：拆分带括号描述的彩蛋 ---
    branches = _postprocess_egg_branches(branches)

    return branches, has_invalid_progress


def _postprocess_egg_branches(branches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    彩蛋后处理：如果彩蛋名后包含括号描述（如 "彩蛋名 (触发条件)"），
    则从原分支移除彩蛋，并将括号内容拆分为一个独立的新分支。
    """
    result = []
    for branch in branches:
        egg_name = branch.get("egg_name")
        if egg_name:
            # 匹配彩蛋名后的中文或英文括号内容
            m = re.match(r"^(.+?)\s*[（(](.+)[）)]\s*$", egg_name)
            if m:
                clean_egg_name = m.group(1).strip()
                egg_trigger = m.group(2).strip()

                # 从原分支移除彩蛋
                branch["egg_name"] = None
                result.append(branch)

                # 创建新分支：触发条件=括号内容，进度与原分支相同
                new_branch = {
                    "trigger": egg_trigger,
                    "ai_reply": "",
                    "progress_name": branch["progress_name"],
                    "progress_gain": branch["progress_gain"],
                    "egg_name": clean_egg_name,
                    "examples": [],
                }
                result.append(new_branch)
                continue

        result.append(branch)
    return result


def _parse_progress_indicators(raw: str) -> List[Dict[str, Any]]:
    """解析 [进度指标]"""
    indicators: List[Dict[str, Any]] = []

    # 先找到所有标签位置
    labels = list(re.finditer(r"(进度指标\d+|无关值)[:：]", raw))
    if not labels:
        return indicators

    for idx, label_match in enumerate(labels):
        # 提取当前标签到下一个标签之间的内容
        start = label_match.end()
        end = labels[idx + 1].start() if idx + 1 < len(labels) else len(raw)
        part = raw[start:end]

        is_irrelevant = "无关值" in label_match.group(1)
        indicator: Dict[str, Any] = {}

        # 名称
        m = re.search(r"名称[:：]\s*(.+?)(?=\n|$)", part)
        indicator["name"] = m.group(1).strip() if m else ("无关值" if is_irrelevant else "")

        # 最小值 & 最大值
        m = re.search(r"最小值[:：]\s*(\d+)", part)
        indicator["min_value"] = _safe_int(m.group(1) if m else None, 0)

        m = re.search(r"最大值[:：]\s*(\d+)", part)
        indicator["max_value"] = _safe_int(m.group(1) if m else None, 100)

        # 关联结尾
        m = re.search(r"关联结尾[:：]\s*(.+?)(?=\n|$)", part)
        indicator["linked_ending"] = m.group(1).strip() if m else ""

        # 结尾示例
        m = re.search(r"结尾示例[:：]\s*\n(.*?)(?=奖励[:：]|\Z)", part, re.DOTALL)
        indicator["ending_example"] = _strip_line_indents(m.group(1).strip()) if m else None

        # 奖励
        if is_irrelevant or re.search(r"奖励[:：]\s*无\s*$", part, re.MULTILINE):
            indicator["rewards"] = None
        else:
            indicator["rewards"] = _parse_rewards(part)

        indicators.append(indicator)

    return indicators


def _parse_rewards(part: str) -> Dict[str, Any]:
    """解析进度指标下的奖励部分"""
    rewards: Dict[str, Any] = {"event": "", "attributes": [], "favorabilities": []}

    # 事件
    m = re.search(r"事件[:：]\s*(.+?)(?=\n|$)", part)
    rewards["event"] = m.group(1).strip() if m else ""

    # 属性 (格式: "属性名 +10, 属性名 +10")
    m = re.search(r"属性[:：]\s*(.+?)(?=\n|$)", part)
    if m:
        for segment in m.group(1).split(","):
            segment = segment.strip()
            m2 = re.match(r"(.+?)\s*([+-]\d+)", segment)
            if m2:
                rewards["attributes"].append({
                    "name": m2.group(1).strip(),
                    "value": int(m2.group(2)),
                })

    # 好感度 (格式: "角色好感度 +10, 角色好感度 -5")
    m = re.search(r"好感度[:：]\s*(.+?)(?=\n|$)", part)
    if m:
        for segment in m.group(1).split(","):
            segment = segment.strip()
            m2 = re.match(r"(.+?)\s*([+-]\d+)", segment)
            if m2:
                rewards["favorabilities"].append({
                    "name": m2.group(1).strip(),
                    "value": int(m2.group(2)),
                })

    return rewards


def _parse_easter_eggs(raw: str) -> List[Dict[str, str]]:
    """解析 [彩蛋]"""
    if not raw or raw.strip() == "无":
        return []

    eggs: List[Dict[str, str]] = []
    parts = re.split(r"彩蛋\d+[:：]", raw)

    for part in parts[1:]:
        m_name = re.search(r"彩蛋名[:：]\s*(.+?)(?=\n|$)", part)
        m_cond = re.search(r"触发条件[:：]\s*(.+?)(?=\n|$)", part)

        name = m_name.group(1).strip() if m_name else ""
        if name:
            eggs.append({
                "name": name,
                "trigger_condition": m_cond.group(1).strip() if m_cond else "",
            })

    return eggs


# ---------------------------------------------------------------------------
# 固定节点解析
# ---------------------------------------------------------------------------

def parse_fixed_node_output(raw_text: str) -> List[Dict[str, Any]]:
    """将 LLM 输出的固定节点文本解析为结构化列表"""
    raw_text = _strip_think_tags(raw_text)
    nodes: List[Dict[str, Any]] = []

    parts = re.split(r"---固定节点\s*\d+---", raw_text)

    for i, part in enumerate(parts[1:], 1):
        node: Dict[str, Any] = {"order": i}

        m = re.search(r"\[剧情文本\][:：]\s*\n(.*?)(?=\[图片提示词\]|\Z)", part, re.DOTALL)
        node["plot_text"] = _parse_prologue_text(m.group(1).strip()) if m else []

        m = re.search(r"\[图片提示词\][:：]\s*\n(.*?)(?=\[音乐类型\]|\Z)", part, re.DOTALL)
        node["image_prompt"] = m.group(1).strip() if m else ""

        m = re.search(r"\[音乐类型\][:：]\s*\n?(.*?)(?=---固定节点|\Z)", part, re.DOTALL)
        music_raw = m.group(1).strip() if m else ""
        if "建议新增" in music_raw:
            node["music_type"] = re.sub(r"建议新增[:：]?\s*", "", music_raw).strip()
            node["is_new_music_type"] = True
        else:
            node["music_type"] = music_raw
            node["is_new_music_type"] = False

        nodes.append(node)

    if not nodes:
        logger.warning("Fixed node parser produced 0 nodes from LLM output")

    return nodes
