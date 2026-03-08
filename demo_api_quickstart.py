"""
NovelWeaver API 快速演示脚本

用法：
1) 在另一个终端启动服务器：
   python server.py
2) 运行本演示：
   python demo_api_quickstart.py

可选参数：
   --no-stream      跳过流式接口演示
   --base-url URL   指定 API 地址（默认 http://127.0.0.1:8100）

示例：
   python demo_api_quickstart.py
   python demo_api_quickstart.py --no-stream
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


# ---------------------------------------------------------------------------
# 通用工具函数
# ---------------------------------------------------------------------------

def get_json(url: str, timeout: int = 15):
    """发送 GET 请求并返回 JSON 响应"""
    req = urllib.request.Request(url=url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def post_json(url: str, payload: dict, timeout: int = 120):
    """发送 POST 请求（JSON 格式）并返回 JSON 响应"""
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def post_ndjson_stream(url: str, payload: dict, timeout: int = 180):
    """发送 POST 请求并逐行读取 NDJSON 流式响应"""
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for line in resp:
            line = line.decode("utf-8").strip()
            if line:
                yield json.loads(line)


def load_json(path: str):
    """从文件加载 JSON 数据"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output-example")


def save_result(filename: str, data):
    """将结果保存到 output-example/ 目录"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"结果已保存: {path}")


# ---------------------------------------------------------------------------
# 演示：健康检查
# ---------------------------------------------------------------------------

def demo_health(base_url: str):
    print("\n=== 1) GET /health — 健康检查 ===")
    result = get_json(f"{base_url}/health")
    print(json.dumps(result, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# 演示：固定节点拆分
# ---------------------------------------------------------------------------

def demo_fixed_node(base_url: str):
    print("\n=== 2) POST /api/fixed_node — 固定节点拆分 ===")
    payload_path = os.path.join("input-example", "test_fixed_node_input.json")
    payload = load_json(payload_path)
    print(f"已加载 payload: {payload_path}")

    result = post_json(f"{base_url}/api/fixed_node", payload, timeout=300)
    print("固定节点结果:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    save_result("fixed_node_output.json", result)


def demo_fixed_node_stream(base_url: str):
    """固定节点拆分 — 流式版本"""
    print("\n=== 2s) POST /api/fixed_node/stream — 固定节点拆分（流式） ===")
    payload_path = os.path.join("input-example", "test_fixed_node_input.json")
    payload = load_json(payload_path)
    print(f"已加载 payload: {payload_path}")

    print("流式输出:")
    stream_result = None
    for msg in post_ndjson_stream(f"{base_url}/api/fixed_node/stream", payload):
        if msg["type"] == "chunk":
            # 实时打印 LLM 的文本片段
            print(msg["content"], end="", flush=True)
        elif msg["type"] == "result":
            stream_result = msg["data"]
            print("\n\n解析后的结构化结果:")
            print(json.dumps(stream_result, ensure_ascii=False, indent=2))
        elif msg["type"] == "error":
            print(f"\n错误: {msg['message']}")
    if stream_result:
        save_result("fixed_node_stream_output.json", stream_result)


# ---------------------------------------------------------------------------
# 演示：AI 节点扩展（已有前情提要）
# ---------------------------------------------------------------------------

def demo_ai_node_with_bg(base_url: str):
    """AI 节点扩展 — 已有前情提要（直接使用 previous_plot）"""
    print("\n=== 3) POST /api/ai_node — AI节点扩展（已有前情提要） ===")
    payload_path = os.path.join("input-example", "test_ai_node_input_with_bg.json")
    payload = load_json(payload_path)
    print(f"已加载 payload: {payload_path}")
    print(f"previous_plot 长度: {len(payload.get('previous_plot', ''))} 字符（已提供）")

    result = post_json(f"{base_url}/api/ai_node", payload, timeout=300)
    print("AI节点结果:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    save_result("ai_node_with_bg_output.json", result)


def demo_ai_node_auto_bg(base_url: str):
    """AI 节点扩展 — 自动生成前情提要（previous_plot 为空，使用 previous_nodes）"""
    print("\n=== 4) POST /api/ai_node — AI节点扩展（自动生成前情提要） ===")
    payload_path = os.path.join("input-example", "test_ai_node_input_auto_bg.json")
    payload = load_json(payload_path)
    print(f"已加载 payload: {payload_path}")
    print(f"previous_plot 为空，previous_nodes 包含 {len(payload.get('previous_nodes', []))} 个节点")
    print("服务端将自动调用 LLM 生成前情提要...")

    result = post_json(f"{base_url}/api/ai_node", payload, timeout=300)
    print("AI节点结果:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    save_result("ai_node_auto_bg_output.json", result)


# ---------------------------------------------------------------------------
# 演示：AI 节点扩展（流式）
# ---------------------------------------------------------------------------

def demo_ai_node_stream(base_url: str):
    """AI 节点扩展 — 流式版本（使用已有前情提要）"""
    print("\n=== 5) POST /api/ai_node/stream — AI节点扩展（流式） ===")
    payload_path = os.path.join("input-example", "test_ai_node_input_with_bg.json")
    payload = load_json(payload_path)
    print(f"已加载 payload: {payload_path}")

    print("流式输出:")
    stream_result = None
    for msg in post_ndjson_stream(f"{base_url}/api/ai_node/stream", payload, timeout=300):
        if msg["type"] == "chunk":
            print(msg["content"], end="", flush=True)
        elif msg["type"] == "retry":
            print(f"\n{msg['message']}")
        elif msg["type"] == "result":
            stream_result = msg["data"]
            print("\n\n解析后的结构化结果:")
            print(json.dumps(stream_result, ensure_ascii=False, indent=2))
        elif msg["type"] == "error":
            print(f"\n错误: {msg['message']}")
    if stream_result:
        save_result("ai_node_stream_output.json", stream_result)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NovelWeaver API 快速演示脚本")
    parser.add_argument("--base-url", default="http://127.0.0.1:8100", help="API 基础地址")
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="跳过流式接口演示",
    )
    args = parser.parse_args()

    try:
        # 1) 健康检查（始终运行）
        demo_health(args.base_url)

        # 2) 固定节点拆分
        demo_fixed_node(args.base_url)

        # 2s) 固定节点拆分 — 流式
        if not args.no_stream:
            demo_fixed_node_stream(args.base_url)

        # 3-4) AI 节点扩展
        demo_ai_node_with_bg(args.base_url)
        demo_ai_node_auto_bg(args.base_url)

        # 5) AI 节点扩展 — 流式
        if not args.no_stream:
            demo_ai_node_stream(args.base_url)

    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        print(f"HTTP 错误: {e.code} {e.reason}")
        if body:
            print("响应内容:")
            print(body)
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"无法连接到 API: {e}")
        print("请确认服务器已启动: python server.py")
        sys.exit(1)


if __name__ == "__main__":
    main()
