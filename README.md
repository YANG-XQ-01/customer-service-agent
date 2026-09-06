# 电商智能客服 Agent（求职实战项目）

一个用 **FastAPI + LangChain/LangGraph + RAG** 搭建的电商客服机器人：
用户在网页上提问，系统判断意图后查订单、查售后、检索商品/政策知识，最后组织成人话回答；
处理不了会带着完整上下文转人工。

本项目是边学边做的教学项目，路线是：先写一个会调工具的单 Agent 建立直觉，
再演进成 LangGraph 多智能体，最后做评测、优化与部署。

## 技术栈

| 层 | 选择 | 说明 |
|---|---|---|
| 大模型 | 通义千问 qwen-plus | 走阿里云百炼的 OpenAI 兼容接口，配置可换模型 |
| 编排 | LangChain 高层 Agent → LangGraph | 阶段 2 先单 Agent，阶段 3 起多智能体 |
| 后端 | FastAPI + uvicorn | REST API + 静态网页 |
| 知识库 | Milvus | 商品说明/售后政策等非结构化文档的向量检索 |
| 业务数据 | MySQL 8 | 订单、售后单等结构化数据 |
| 前端 | 原生 HTML/CSS/JS | 简单聊天网页，无框架 |
| 部署 | Docker Compose | 一键编排 MySQL + Milvus + 应用 |

## 阶段路线（共 8 个阶段）

| 阶段 | 目标 | 状态 |
|---|---|---|
| 0 | 设计方案定稿 + 初始化项目 | 进行中（待确认进入阶段 1） |
| 1 | FastAPI 骨架 + 千问直连聊天（带会话记忆） | 未开始 |
| 2 | 工具调用 + Milvus/MySQL 接入（单 Agent） | 未开始 |
| 3 | LangGraph 多智能体：路由 + 专家 Agent + 状态 | 未开始 |
| 4 | 转人工（三种触发场景 + 上下文打包） | 未开始 |
| 5 | 评测体系（完成率 / 轨迹正确率 / 耗时 / 成本） | 未开始 |
| 6 | 优化迭代（一次只改一个点，前后对比） | 未开始 |
| 7 | 部署 + 验证清单 + 演示脚本 + 简历描述 | 未开始 |

## 目录结构

```text
customer-service-agent/
├─ docs/      # 设计方案、阶段记录、踩坑笔记
├─ app/       # FastAPI 入口、Agent、工具、RAG（随阶段填充）
├─ data/      # 知识文档、演示数据脚本
├─ eval/      # 评测集与评测脚本（阶段 5）
├─ static/    # 聊天网页
└─ tests/     # 自动化测试
```

## 文档

- [总体设计方案](docs/design.md)：阶段 0 交付物，记录已确认的选型、架构与路线。

## 运行方式

### 1. 准备环境（首次）

```powershell
# 激活已实测的 conda 环境（Python 3.13 + langchain 1.2）
conda activate langchain1.2

# 复制 .env.example 为 .env，填入你的 DASHSCOPE_API_KEY
# （密钥只放 .env，已被 git 忽略，不会提交）
```

如果在新机器上从零安装依赖：

```powershell
pip install -r requirements.txt
```

### 2. 启动服务

```powershell
python -m uvicorn app.main:app --reload
```

浏览器打开 <http://127.0.0.1:8000> 即可聊天。
若端口 8000 被占用（Windows 报错 10013 或 address already in use），换一个端口：

```powershell
python -m uvicorn app.main:app --port 8001
```

### 3. 验证会话记忆（阶段 1 通过标准）

同一会话里连发三句：

1. `你好`
2. `我叫小明，喜欢蓝色`
3. `我叫什么名字？`

第三句能答出“小明”即记忆生效；服务端日志会打印每次请求的
用户消息、耗时、token 用量和助手回复。

### 4. 初始化业务数据与知识库（阶段 2 起需要）

```powershell
# 建库建表 + 灌入演示订单/售后数据（可重复执行，会重建本项目 4 张表）
python -m app.seed_data

# 把 data/knowledge 下的知识文档向量化存入 Milvus（可重复执行，会重建集合）
python -m app.rag.indexer
```

两条命令都设计成幂等：重复执行不会产生重复数据。
运行前确认 `.env` 里 MYSQL_*（本地开发用 root）和 MILVUS_* 已填好。
