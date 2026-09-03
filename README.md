<div align="center">

# StarOracle_Agent 星座运势助手

**基于 LangChain / LangGraph / Streamlit 构建的星座运势问答助手，支持 ReAct Agent 推理、多工具调用、RAG 知识库检索、长期记忆、每日运势占卜，以及 URL / PDF / 文本知识入库。**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/) &nbsp;![FastAPI](https://img.shields.io/badge/FastAPI-0.141.1-009688?style=flat-square&logo=fastapi&logoColor=white) ![Uvicorn](https://img.shields.io/badge/Uvicorn-0.52.4-29BEB0?style=flat-square&logo=uvicorn&logoColor=white) ![Streamlit](https://img.shields.io/badge/Streamlit-1.62.0-FF4B4B?style=flat-square&logo=streamlit&logoColor=white) ![LangChain](https://img.shields.io/badge/LangChain-1.3.17-1C3C84?style=flat-square&logo=langchain&logoColor=white) ![LangChain-Chroma](https://img.shields.io/badge/LangChain--Chroma-1.1.0-9333EA?style=flat-square)[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](./LICENSE)

</div>

---

## 📖 项目简介

本项目以“星座运势问答”为主场景，后端使用 FastAPI 提供接口服务，Agent 侧通过 LangChain ReAct 模式完成推理与工具选择，前端使用 Streamlit 提供聊天式交互界面。

系统包含以下能力：

- 结合用户输入进行情绪识别与角色风格切换
- 通过 Chroma 向量库实现知识库检索与长期记忆
- 支持 URL、PDF、文本三种方式写入知识库
- 支持每日运势占卜工具
- 支持滚动日志与统一 YAML 配置管理
- 支持 Streamlit 历史对话、当前聊天与多轮问答

## 🧱 项目结构

```text
StarOracle_Agent/
├── evaluation/               # 离线评测集与基线评测脚本
├── server.py                 # FastAPI 入口
├── app.py                    # Streamlit 前端
├── config/                   # YAML 配置
├── prompts/                  # 提示词模板
├── services/                 # 聊天、记忆、知识库、情绪等服务
├── tools/                    # 工具函数
├── utils/                    # 配置、日志、路径、Prompt 加载
├── README.assets/            # README 图片资源
└── README.md
```

## ✨ 主要特性

| 特性 | 说明 |
| --- | --- |
| ReAct Agent | 基于 Thought / Action / Observation 的推理链路 |
| RAG 检索增强 | 使用 Chroma + DashScope Embedding 对知识库内容做向量检索 |
| 长期记忆 | 通过独立记忆库保存用户历史偏好和上下文 |
| 多工具调用 | 集成搜索、当前时间、每日运势占卜等工具 |
| 知识入库 | 支持 URL / PDF / 文本入库，并支持去重 |
| 流式前端 | Streamlit 聊天界面支持历史消息留存与交互式问答 |
| 统一配置 | 模型、日志、提示词路径均由 YAML 管理 |

## 🏗️ 技术架构

```mermaid
flowchart TB
    U[Streamlit 前端] --> API[FastAPI 接口]
    API --> CHAT[Chat Service]
    CHAT --> EMO[情绪识别]
    CHAT --> MEM[长期记忆检索]
    CHAT --> RAG[知识库检索]
    CHAT --> AGENT[Agent Factory]
    AGENT --> LLM[ChatTongyi / DashScope]
    AGENT --> TOOLS[搜索 / 当前时间 / 每日运势]
    RAG --> CHROMA[Chroma 向量库]
    MEM --> CHROMA
    API --> INGEST[知识入库接口]
    INGEST --> CHROMA
```

## 🧰 技术栈

- Python
- FastAPI
- Streamlit
- LangChain
- LangGraph
- Chroma
- DashScope / 通义千问
- YAML 配置
- Requests

## 🧭 功能说明

### 💬 聊天能力

- 支持普通问答
- 支持快捷问题入口
- 支持历史对话保存和切换
- 支持“新对话”和“清空当前聊天”

### 📚 知识库能力

- `add_urls`：从 URL 页面抽取文本并入库
- `add_pdfs`：从 PDF 文档入库
- `add_texts`：从纯文本入库
- 聊天时自动进行知识库检索
- 支持去重，重复内容不会反复写入

### 🧠 记忆能力

- 根据用户输入和回答，提取长期记忆
- 将用户偏好、身份信息和历史上下文保存到 Chroma
- 下次聊天时自动召回

### 🛠️ 工具能力

- 搜索工具
- 当前时间工具
- 每日运势占卜工具

### 📏 评测能力

- 提供离线评测集，覆盖知识检索、长期记忆召回和工具调用
- 支持一键生成基线指标，包括命中率、工具成功率和平均延迟
- 支持导出 Markdown / JSON 报告，便于迭代前后对比

## 🚀 快速开始

### 1️⃣ 配置环境

```bash
pip install -r requirements.txt
```

### 2️⃣ 申请 API Key

```bash
DashScope_API_KEY=your-dashscope-key
YUANFENJU_API_KEY=your-yuanfenju-key
TAVILY_API_KEY=your-tavily-key
```

### 3️⃣ 启动后端服务

```bash
python server.py
```

### 4️⃣ 启动前端交互

```bash
streamlit run app.py
```

前端默认连接：`http://127.0.0.1:8000`

### 5️⃣ 前后端部署

- 后端部署：Render
    - 地址：<https://staroracle-agent.onrender.com>
- 前端部署：Streamlit Community Cloud
    - 地址：<https://staroracleagent-be5z9zd7r7z5anbcf5idjg.streamlit.app/>

### 6️⃣ 运行离线评测

三个核心能力：

1. 知识检索是否能命中正确内容
2. 长期记忆是否能召回用户偏好和画像
3. 工具调用是否能按预期工作，比如当前时间工具

基线指标：

- retrieval_hit_rate：知识检索和记忆检索的命中率。
- tool_pass_rate：工具输出是否符合预期格式。
- retrieval_avg_latency_ms：检索平均耗时。
- tool_avg_latency_ms：工具平均耗时。
- live_keyword_coverage：在线 Agent 输出里预期关键词的覆盖率。
- live_avg_latency_ms：在线 Agent 平均耗时。


评测会自动构建本地知识库和记忆库样本，输出知识检索、记忆召回和工具调用的基线指标。
```bash
python evaluation/run_benchmark.py --markdown-output evaluation/benchmark_report.md --output evaluation/benchmark_report.json
```


## 🖥️ 使用方式

![界面示意图](README.assets/image-20260902161329699.png)

1. 在左上角输入个人 API Key。
2. 选择或设置用户 ID。
3. 在输入框中直接输入问题开始聊天。
4. 点击快捷问答按钮，可快速发起星座相关提问。
5. 在左侧控制台进行知识入库，支持 URL、PDF 和 TXT。
6. 在历史对话中切换、继续或删除已有会话。