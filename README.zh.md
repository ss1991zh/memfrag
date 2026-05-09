# MemFrag — Claude 的碎片化记忆系统
### 以 MCP Server 形式运行，直接在 Claude Code 和 Claude Desktop 中提供持久化记忆

[English README](README.md)

---

## 一、产品定位

**MemFrag** 让 Claude 拥有跨会话的长期记忆，工作方式类似人脑——把信息存成可重组的碎片，而不是把完整对话历史塞进上下文。

它以 **MCP（Model Context Protocol）Server** 的形式运行，Claude Code 和 Claude Desktop 可以把它当作原生工具直接调用，无需任何第三方平台。

> 一句话：**Claude 跨会话记住你，而不是每次对话都从零开始。**

---

## 二、解决什么问题

| 现有方案的痛点 | MemFrag 的解法 |
|---|---|
| 对话历史越来越长，token 成本线性增长 | 只召回相关碎片，上下文始终精简 |
| RAG 切片粒度粗（几百字），语义噪声大 | 碎片粒度到关键词/短句级，精准匹配 |
| 记忆孤岛：各条记忆互相不知道对方存在 | 关系图层显式建立碎片间的语义连接 |
| 模型"脑补"无法溯源，出处不明 | 每条碎片绑定来源 ID，回答可追溯 |
| 记忆库无限膨胀，检索越来越慢 | 遗忘曲线机制，低频碎片自动冷存储 |
| 需要额外平台才能运行 | 原生 MCP，Claude Code/Desktop 直接用 |

---

## 三、整体架构

```
┌──────────────────────────────────────┐
│     Claude Code / Claude Desktop     │
│                                      │
│  "记住我用 Python 做后端"             │
│  "我的项目截止日是什么时候？"          │
└────────────┬─────────────────────────┘
             │ MCP 协议（stdio）
             ▼
┌──────────────────────────────────────┐
│        MemFrag MCP Server            │
│                                      │
│  暴露给 Claude 的工具：               │
│  ├── memfrag_ingest(turns)           │
│  ├── memfrag_recall(query)           │
│  ├── memfrag_list_fragments()        │
│  ├── memfrag_delete_fragment(id)     │
│  ├── memfrag_run_decay()             │
│  └── memfrag_stats()                 │
│                                      │
│  内部组件：                           │
│  ├── 碎片提取器   (Claude LLM)        │
│  ├── 语义指纹引擎 (Embedding)         │
│  ├── 关系图层     (NetworkX)          │
│  ├── 存储层       (SQLite)            │
│  ├── 召回引擎     (向量+图谱)          │
│  └── 遗忘曲线调度器                   │
└──────────────────────────────────────┘
```

### 写入路径（每次对话后）
```
Claude 调用 memfrag_ingest(turns)
  → LLM 从对话中提取关键碎片（实体、偏好、约束…）
  → 每个碎片生成语义指纹（embedding）
  → 重复碎片合并；更新内容建立 override 边
  → 碎片存入 SQLite；原文归档并建立反向链接
  → 基于相似度自动建立 co-topic 关系
```

### 召回路径（生成回答前）
```
Claude 调用 memfrag_recall(query)
  → 查询向量化 → Top-K 向量搜索
  → 图谱扩展：沿 co-topic / causal / override 边遍历（1-2跳）
  → 按强度 × 相似度排序，截断至 token 预算
  → 重组为自然语言上下文块
  → Claude 基于该上下文块生成有据可查的回答
```

---

## 四、三层记忆模型

| 层级 | 存储内容 | 生命周期 |
|---|---|---|
| **碎片层** | 关键词、短句、实体、偏好 | 长期（几乎不删） |
| **关系层** | 碎片间连接（同主题 / 时间序 / 因果 / 覆盖） | 中期（强度动态变化） |
| **子记忆归档** | 完整原始对话文本 | 按需回滚 |

### 遗忘曲线参数
```
初始强度          = 1.0
每次被召回        → 强度 × 1.2（上限 10）
每 7 天未被调用   → 强度 × 0.85
强度 < 0.3        → 冷存储（不参与主动召回）
强度 < 0.1        → 自动删除
```

---

## 五、快速开始

### 1. 安装

```bash
git clone https://github.com/ss1991zh/memfrag.git
cd memfrag
pip install -e .
```

### 2. 配置 Claude Code

在 `~/.claude/settings.json` 中添加：

```json
{
  "mcpServers": {
    "memfrag": {
      "command": "python",
      "args": ["-m", "memfrag.mcp_server"],
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-...",
        "MEMFRAG_DB": "/path/to/memfrag.db"
      }
    }
  }
}
```

### 3. 配置 Claude Desktop

在 `~/Library/Application Support/Claude/claude_desktop_config.json` 中添加：

```json
{
  "mcpServers": {
    "memfrag": {
      "command": "python",
      "args": ["-m", "memfrag.mcp_server"],
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-...",
        "MEMFRAG_DB": "/Users/你的用户名/memfrag.db"
      }
    }
  }
}
```

### 4. 在 Claude 中使用

配置好后，Claude 可以自然地使用记忆：

```
你：    记住我正在用 Python 开发一个碎片化记忆系统，
        通过 MCP 集成到 Claude 中。

Claude：[调用 memfrag_ingest] ✓ 已存储 3 条碎片。

你：    我在做什么项目？

Claude：[调用 memfrag_recall] 根据记忆：
        你正在开发 MemFrag —— 一个 Python 实现的碎片化
        记忆层，通过 MCP 协议集成到 Claude。
```

---

## 六、MCP 工具说明

| 工具名 | 参数 | 功能 |
|---|---|---|
| `memfrag_ingest` | `turns: [{role, content}]` | 从对话中提取并存储碎片 |
| `memfrag_recall` | `query: str` | 召回相关碎片，返回上下文块 |
| `memfrag_list_fragments` | `include_cold?: bool` | 列出所有活跃碎片 |
| `memfrag_delete_fragment` | `fragment_id: str` | 删除指定碎片 |
| `memfrag_run_decay` | — | 手动触发遗忘曲线 |
| `memfrag_stats` | — | 返回存储统计信息 |

---

## 七、技术栈

| 模块 | 技术选型 |
|---|---|
| MCP Server | `mcp` Python SDK（stdio 传输） |
| LLM（提取 + 重组） | Claude Haiku（Anthropic SDK） |
| Embedding | `sentence-transformers`（本地，无需额外 API） |
| 向量搜索 | NumPy 余弦相似度 |
| 关系图 | NetworkX |
| 存储 | SQLite（内置，零配置） |

---

## 八、与现有方案对比

| | MemFrag | Mem0 | Letta | GraphRAG |
|---|---|---|---|---|
| Claude 原生（MCP） | ✅ | ❌ | ❌ | ❌ |
| 碎片粒度 | 词/短句级 | 短句级 | 段落级 | 段落级 |
| 关系图层 | ✅ | ❌ | ❌ | ✅ |
| 遗忘曲线 | ✅ | ❌ | ❌ | ❌ |
| 零配置存储 | ✅ SQLite | ❌ | ❌ | ❌ |

---

## 九、REST API（可选）

MemFrag 同时提供 FastAPI 服务，供非 MCP 场景使用：

```bash
ANTHROPIC_API_KEY=sk-ant-... uvicorn memfrag.api:app --port 8765
```

接口：`POST /ingest`、`POST /recall`、`POST /decay`、`GET /stats`、`GET /fragments`、`DELETE /fragments/{id}`

---

## 十、运行测试

```bash
pip install -e ".[dev]"
pytest tests/     # 61+ 个测试，无需 API Key
```

---

## 十一、路线图

- [x] 碎片提取、关系图、存储、召回
- [x] 遗忘曲线
- [x] REST API
- [x] MCP Server
- [ ] 真实 API Key 端到端测试
- [ ] Neo4j 图谱后端（生产规模）
- [ ] 多用户支持

---

*版本 0.2.0 · 2026-05-09*
