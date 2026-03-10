# NovelWeaver API

互动叙事节点生成 API，提供两个核心能力：**固定节点拆分**与**AI节点扩展**。

## 快速开始

```bash
# 启动服务器（默认端口 8100）
python server.py

# 运行演示脚本
python demo_api_quickstart.py
python demo_api_quickstart.py --no-stream   # 跳过流式演示
```

## API 接口一览

| 接口 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/api/fixed_node` | POST | 固定节点拆分 |
| `/api/fixed_node/stream` | POST | 固定节点拆分（流式） |
| `/api/ai_node` | POST | AI节点扩展 |
| `/api/ai_node/stream` | POST | AI节点扩展（流式） |

---

## 输入输出格式与字段定义
输入输出格式请见`input-example/`和`output-example/`，字段定义请见 server.py 63-223行。

---

## 示例文件

### 输入示例（`input-example/`）

| 文件 | 说明 |
|------|------|
| `test_fixed_node_input.json` | 固定节点拆分输入 |
| `test_ai_node_input.json` | AI节点扩展输入（原始示例） |
| `test_ai_node_input_with_bg.json` | AI节点扩展输入——已有前情提要（直接提供 `previous_plot`） |
| `test_ai_node_input_auto_bg.json` | AI节点扩展输入——需自动生成前情提要（`previous_plot` 为空，提供 `previous_nodes`） |

### 输出示例（`output-example/`）

运行 `demo_api_quickstart.py` 后自动生成。
