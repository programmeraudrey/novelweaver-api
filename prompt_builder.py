"""
NovelWeaver API - 提示词组装
仅负责将输入数据 + 模板文件组装为完整提示词。
LLM 调用 → llm.py, 输出解析 → parsers.py
"""

import os
import logging
from functools import lru_cache
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPTS_DIR = os.path.join(BASE_DIR, "prompts")


# ---------------------------------------------------------------------------
# 模板加载（带缓存）
# ---------------------------------------------------------------------------

@lru_cache(maxsize=32)
def load_template(file_path: str) -> str:
    """加载提示词模板文件（缓存，避免重复磁盘IO）"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def reload_templates():
    """清除模板缓存（开发/热更新时调用）"""
    load_template.cache_clear()


# ---------------------------------------------------------------------------
# 提示词组装 - AI 节点
# ---------------------------------------------------------------------------

def generate_ai_node_expansion_prompt(
    background: str,
    prologue: List[Dict[str, str]],
    task: str,
    plot_start: str,
    plot_process: str,
    plot_endings: List[Dict[str, str]],
    ai_reply_style: Optional[str],
    image_requirements: Optional[str],
    user_role: str,
    involved_roles: List[Dict[str, str]],
    attributes: List[str],
    favorabilities: List[str],
    existing_events: List[str],
    existing_easter_eggs: List[str],
    available_music_types: List[str],
) -> str:
    """
    组装 AI 节点扩展提示词
    """
    template_file = os.path.join(PROMPTS_DIR, "expand_ai_node_prompt.txt")
    standard_file = os.path.join(PROMPTS_DIR, "standard_weaver.txt")

    template = load_template(template_file)
    standard = load_template(standard_file)

    H = "  "    # section header indent (2 spaces)
    C = "    "  # content indent (4 spaces)

    # --- 组装各输入 section（与输出格式对齐：2空格header + 4空格content） ---
    sections = []

    # 背景摘要
    sections.append(f"{H}[背景摘要]:\n{C}{background or '无背景信息'}")

    # 开场白
    if prologue:
        lines = [f"{H}[开场白]:"]
        for p in prologue:
            role = p.get("role", "旁白")
            content = p.get("content", "")
            lines.append(f"{C}- {role}：{content}")
        sections.append("\n".join(lines))

    # 任务
    if task:
        sections.append(f"{H}[任务]:\n{C}{task}")

    # 剧情
    if plot_start:
        sections.append(f"{H}[剧情开端]:\n{C}{plot_start}")
    if plot_process:
        # 剧情过程可能多行，每行都缩进
        process_lines = plot_process.splitlines()
        formatted = "\n".join(C + line if line.strip() else "" for line in process_lines)
        sections.append(f"{H}[剧情过程]:\n{formatted}")
    if plot_endings:
        ending_lines = [f"{H}[剧情结尾]:"]
        for i, ending in enumerate(plot_endings, 1):
            desc = ending.get("description", "")
            ending_lines.append(f"{C}结尾{i}: {desc}")
        sections.append("\n".join(ending_lines))

    # AI回复要求
    if ai_reply_style:
        sections.append(f"{H}[AI回复要求]:\n{C}{ai_reply_style}")

    # 背景图要求
    if image_requirements:
        sections.append(f"{H}[背景图要求]:\n{C}{image_requirements}")

    # 参与角色（含描述，标注用户角色）
    role_names = []
    if involved_roles:
        role_lines = [f"{H}[参与角色]:"]
        for role in involved_roles:
            name = role.get("name", "")
            desc = role.get("description", "")
            role_names.append(name)
            tag = "（用户扮演）" if user_role == name else ""
            if desc:
                role_lines.append(f"{C}- {name}{tag}：{desc}")
            else:
                role_lines.append(f"{C}- {name}{tag}")
        sections.append("\n".join(role_lines))

    # 元数据
    meta_lines = [f"{H}[元数据]:"]
    meta_lines.append(f"{C}属性列表: [{', '.join(attributes)}]" if attributes else f"{C}属性列表: []")
    # 好感度仅保留参与当前节点的角色
    relevant_favs = [f for f in favorabilities if any(r in f for r in role_names)] if role_names else favorabilities
    meta_lines.append(f"{C}好感度列表: [{', '.join(relevant_favs)}]" if relevant_favs else f"{C}好感度列表: []")
    meta_lines.append(f"{C}已有事件列表: [{', '.join(existing_events)}]" if existing_events else f"{C}已有事件列表: []")
    meta_lines.append(f"{C}已有彩蛋列表: [{', '.join(existing_easter_eggs)}]" if existing_easter_eggs else f"{C}已有彩蛋列表: []")
    meta_lines.append(f"{C}已录入音乐类型: [{', '.join(available_music_types)}]" if available_music_types else f"{C}已录入音乐类型: []")
    sections.append("\n".join(meta_lines))

    input_data = "\n\n".join(sections)

    # --- 替换占位符 ---
    prompt = template.replace("{standard}", standard)
    prompt = prompt.replace("{user_role}", user_role)
    prompt = prompt.replace("{input_data}", input_data)

    return prompt


# ---------------------------------------------------------------------------
# 提示词组装 - 背景摘要生成
# ---------------------------------------------------------------------------

def serialize_previous_nodes(nodes: List[Dict]) -> str:
    """
    将 novelweaver-api 格式的前序节点数组序列化为结构化文本。
    最后 3 个节点标记为【最近剧情(重点)】，其余为【历史剧情】。
    支持两种节点类型：AI 节点（含 background/current_plot/user_messages 等）和固定节点（含 plot_text）。
    """
    if not nodes:
        return "无前序节点信息（这是第一个节点）"

    total = len(nodes)
    all_text = []

    for i, node in enumerate(nodes):
        is_recent = (i >= total - 3)
        prefix = "【最近剧情(重点) - 节点】" if is_recent else "【历史剧情 - 节点】"
        title = node.get("node_title", node.get("title", "未命名"))
        lines = [f"{prefix}: {title}"]

        # 固定节点（无 current_plot 且无 background）
        is_fixed = "current_plot" not in node and "background" not in node
        if is_fixed and "plot_text" in node:
            lines.append(f"  [类型]: 固定")
            lines.append(f"\n  [剧情文本]:")
            plot_text = node["plot_text"]
            if isinstance(plot_text, list):
                # 新格式 [{role, content}] 数组
                for p in plot_text:
                    role = p.get("role", "旁白") if isinstance(p, dict) else "旁白"
                    content = p.get("content", "") if isinstance(p, dict) else str(p)
                    lines.append(f"    {role}：{content}")
            else:
                # 兼容旧格式字符串
                for line in str(plot_text).split("\n"):
                    if line.strip():
                        lines.append(f"    {line.strip()}")
        elif "nodes" in node:
            # fixed_node 拆分结果（含多个子节点）
            lines.append(f"  [类型]: 固定")
            for sub in node["nodes"]:
                lines.append(f"\n  [子节点 {sub.get('order', '?')}]:")
                sub_plot_text = sub.get("plot_text", [])
                if isinstance(sub_plot_text, list) and sub_plot_text:
                    for p in sub_plot_text:
                        role = p.get("role", "旁白") if isinstance(p, dict) else "旁白"
                        content = p.get("content", "") if isinstance(p, dict) else str(p)
                        lines.append(f"    {role}：{content}")
                elif isinstance(sub_plot_text, str) and sub_plot_text:
                    # 兼容旧格式字符串
                    for line in sub_plot_text.split("\n"):
                        if line.strip():
                            lines.append(f"    {line.strip()}")
        else:
            # AI 节点
            lines.append(f"  [类型]: AI")

            # 开场白
            prologue = node.get("prologue", [])
            if isinstance(prologue, list) and prologue:
                lines.append(f"\n  [开场白]:")
                for p in prologue:
                    role = p.get("role", "旁白") if isinstance(p, dict) else "旁白"
                    content = p.get("content", "") if isinstance(p, dict) else str(p)
                    lines.append(f"    {role}：{content}")
            elif isinstance(prologue, str) and prologue:
                # 兼容旧格式 prologue_text 字符串
                lines.append(f"\n  [开场白]:")
                for line in prologue.split("\n"):
                    if line.strip():
                        lines.append(f"    {line.strip()}")

            # 背景
            bg = node.get("background", "")
            if bg:
                lines.append(f"\n  [背景]:")
                lines.append(f"    {bg}")

            # 人物状态
            char_status = node.get("character_status", [])
            if char_status:
                lines.append(f"\n  [人物空间关系与状态]:")
                for cs in char_status:
                    role = cs.get("role", "未知")
                    desc = cs.get("description", "")
                    lines.append(f"    - {role}: {desc}")

            # 当前剧情
            plot = node.get("current_plot", {})
            if plot:
                lines.append(f"\n  [当前剧情]:")
                if plot.get("start"):
                    lines.append(f"    开始: {plot['start']}")
                if plot.get("process"):
                    lines.append(f"    过程: {plot['process']}")
                endings = plot.get("plot_endings", plot.get("endings", []))
                for j, ending in enumerate(endings):
                    if isinstance(ending, dict):
                        lines.append(f"    结尾{ending.get('id', j+1)}: {ending.get('description', '')}")
                    else:
                        lines.append(f"    结尾{j+1}: {ending}")

        all_text.append("\n".join(lines))

    return "\n\n".join(all_text)


def generate_background_prompt(prev_node_text: str) -> str:
    """
    组装前情提要生成提示词。
    将前序节点的结构化文本传入模板，让 LLM 梳理出连贯的剧情摘要。
    """
    template_file = os.path.join(PROMPTS_DIR, "generate_background_prompt.txt")
    template = load_template(template_file)
    return template.replace("{prev_node_text}", prev_node_text)


# ---------------------------------------------------------------------------
# 提示词组装 - 固定节点
# ---------------------------------------------------------------------------

def generate_fixed_node_split_prompt(
    plot_text: str,
    available_music_types: List[str],
) -> str:
    """
    组装固定节点拆分提示词
    """
    template_file = os.path.join(PROMPTS_DIR, "split_fixed_node_prompt.txt")
    template = load_template(template_file)

    music_types_str = ", ".join(available_music_types) if available_music_types else "无"

    prompt = template.replace("{plot_text}", plot_text)
    prompt = prompt.replace("{available_music_types}", music_types_str)

    return prompt
