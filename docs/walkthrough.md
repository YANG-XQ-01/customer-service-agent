# 电商智能客服 Agent：从零到一完整教程

> 这份文档按真实建设顺序复盘整个项目：每一步「为什么做 → 关键文件 → 核心机制 → 怎么验证 → 踩过的坑」。
> 建议边看边打开对应代码文件，配合仓库的 README 与 `eval/` 使用。

## 0. 项目全景

**目标**：做一个能上线的电商客服 Agent —— 识别用户意图，分派给订单/售后/知识专家节点，
结构化事实走 MySQL 工具、开放知识走 Milvus 检索，高风险场景转人工并生成工单，
自带回归评测与 Docker 一键部署。

**技术栈**：FastAPI · LangGraph · LangChain · 通义千问（OpenAI 兼容接口）·
text-embedding-v3（原生 HTTP）· Milvus · MySQL（SQLAlchemy）· Redis · Docker Compose。

**架构**：

```text
浏览器 → FastAPI /api/chat
            │  （登录用户：Redis 会话；匿名：进程内存）
            ▼
        LangGraph 图
            │
         路由节点（结构化输出分类）
            ├─ order        → 订单专家（查订单/查物流工具）
            ├─ after_sale   → 售后专家（查订单/售后单/知识库/转人工工具）
            ├─ knowledge    → 知识专家（Milvus 检索）
            ├─ chat         → 闲聊节点
            ├─ clarify      → 澄清节点（连续失败计数 ≥2 时改道转人工）
            └─ handoff      → 转人工节点（生成工单 → Redis）
```

## 1. 环境与配置

需要准备：Python 3.13、MySQL 8、Milvus 3.0、Redis、Docker Desktop、阿里云百炼 API Key。

```powershell
# .env（不要提交到 git）
DASHSCOPE_API_KEY=sk-xxx
QWEN_CHAT_MODEL=qwen-plus
QWEN_EMBEDDING_MODEL=text-embedding-v3
MYSQL_HOST=127.0.0.1 / PORT=3306 / USER=root / PASSWORD=xxx / DB=customer_service
MILVUS_HOST=127.0.0.1 / PORT=19530 / COLLECTION=customer_service_kb
REDIS_HOST=127.0.0.1 / PORT=6380 / DB=0
REFUND_THRESHOLD=500
RATE_LIMIT_CHAT_PER_MIN=20 / RATE_LIMIT_AUTH_PER_MIN=10
LOGIN_MAX_FAILURES=5 / LOGIN_LOCK_SECONDS=300
```

**为什么**：配置集中在 `app/config.py` 读取，代码里不出现密钥与魔法数字；换模型/换库只改 `.env`。

## 2. 第一步：让机器人能聊天（阶段 1）

**关键文件**：[app/main.py](../app/main.py)、[app/llm.py](../app/llm.py)、[app/memory.py](../app/memory.py)

**核心机制**：
- `ChatOpenAI(base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")` 用 OpenAI 协议调千问，模型可插拔；
- 会话记忆 = 外部存储 + 每次把历史一起发给模型（模型本身没有记忆）；
- 历史按 `role` 分开传（user/assistant），不拼成大字符串；
- 历史必须截断（`MEMORY_MAX_TURNS`），否则 token 成本线性上涨。

**验证**：连发「你好 → 我叫小明 → 我叫什么名字」，第三句答出“小明”。

**踩坑**：Windows 控制台 GBK 导致日志乱码（启动时把 stdout/stderr 切 UTF-8）；
token 用量返回是 dict 不是对象（两种格式都兼容）。

## 3. 第二步：接数据（阶段 2）

**关键文件**：`app/models.py`、`app/db.py`、`app/seed_data.py`、`app/rag/*`、`data/knowledge/*.md`

**核心机制**：
- MySQL 存结构化事实：`users / orders / order_items / logistics_events / after_sales`，
  `seed_data.py` 幂等重建演示数据（3 用户 / 4 订单 / 物流轨迹 / 售后单）；
- Milvus 存开放知识：13 篇商品与政策文档 → 切分 → 向量化 → 入库（`drop_old=True` 幂等重建）；
- **数据边界**：精确事实走 SQL 工具，开放知识走向量检索——这是防幻觉第一道防线；
- 向量接口两个坑：OpenAI 兼容通道参数不兼容（改用百炼原生 HTTP）、单次批量上限 10 条（自动分批）。

**验证**：`python -m app.seed_data`；`python -m app.rag.indexer`；检索“耳机能游泳吗”命中 `商品-星音蓝牙耳机X9.md`。

## 4. 第三步：工具层（阶段 2 后半）

**关键文件**：[app/tools.py](../app/tools.py)

**核心机制**：
- 每个工具 = Python 函数 + `@tool` 装饰器；**函数名/描述/参数说明就是给模型看的说明书**，描述质量决定调用正确率；
- 4 个业务工具 + 1 个转人工工具：`query_order_by_no`、`query_logistics_by_no`、
  `query_after_sale_by_no`、`search_service_knowledge`、`request_human_handoff`；
- 工具内统一打印调用日志（名称/参数/返回/耗时）——嵌套 Agent 场景比回调更稳；
- 查无数据必须返回结构化“未找到”，不能抛异常也不能编造。

**踩坑**：物流工具曾对“订单不存在”和“未发货”返回同一句话，导致模型说不清 → 先校验订单存在性。

## 5. 第四步：LangGraph 多智能体（阶段 3）

**关键文件**：`app/state.py`、`app/graph.py`、`app/agents/{router,experts,prompts,handoff}.py`

**核心机制**：
- **状态**（`AgentState`）：`messages / intent / extracted / session_id / attempts /
  handoff_* / answer / trace`；`messages` 用 `Annotated[list, operator.add]` 追加而非覆盖；
- **路由节点**：`with_structured_output` + 枚举约束输出意图，并把最近 8 条对话一起喂给路由
  （解决“先说订单号，再问能退吗”的上下文依赖）；解析失败降级为 `clarify`；
- **专家节点**：每个只持有本职工具子集（订单 2 个 / 售后 4 个 / 知识 1 个），
  降低工具误选；业务规则挂在对应分支，而不是全局提示词；
- **条件边**：route → 各专家；`clarify` 且 `attempts ≥ 2` 改道 `handoff`；
  售后节点若检测到超阈值也改道 `handoff`；
- **日志**：节点进出 + 工具调用 + 图轨迹，一次请求可完整复盘。

**踩坑**：路由正则用 `\b` 在中文场景失效（Python 的 `\w` 含汉字）→ 改用环视 `(?<!\d)\d{10,12}(?!\d)`；
`state["extracted"]` 首轮不存在 → 用 `.get()` 兜底。

## 6. 第五步：转人工（阶段 4）

**关键文件**：`app/agents/handoff.py`、`static/human.html`、`main.py` 的 `/api/handoffs`

**三种触发**：
1. 用户明确要求人工 → 路由枚举 `handoff` → 直达转人工节点；
2. 退款金额超阈值 → 售后专家调 `request_human_handoff`，或由**代码层校验**强制改道；
3. 连续两次无法理解 → `attempts ≥ 2` 条件边改道。

**工单内容**：`ticket_id / created_at / session_id / trigger / reason / intent / extracted / 最近 10 轮对话`。

**验证**：三种场景各跑一遍，人工工作台 `http://127.0.0.1:8010/human` 查看工单。

## 7. 第六步：评测体系（阶段 5）

**关键文件**：`eval/cases.json`（17 用例）、`eval/run_eval.py`、`eval/report.json`

**指标定义**：
- **任务完成率**（`passed`）= 意图 + 必需工具及参数 + 答案关键词 + 转人工标志与触发场景，全部满足；
- **轨迹正确率**（`trajectory_ok`）= 只审工具调用（该调的工具调了、参数对）；
- 平均耗时、估算成本（按 token × 单价）。

**用例来源**：真实场景样本 + **已修复 bug 的回归用例**（查无订单不编造、超阈值必须转人工、订单号格式）。

**多轮稳定性**：`python -m eval.run_eval --rounds 3` —— 单轮是抽样，
多轮才能暴露 flaky 用例（曾抓到 1/48 的波动）。

## 8. 第七步：优化迭代（阶段 6）

**方法**：一次只改一个变量 → 跑全量评测 → 前后对比四指标 → 没变好就回滚。

| 改动 | 结果 |
|---|---|
| 路由提示词区分“纯政策咨询” | 完成率 81.2% → 87.5%（修掉意图误判） |
| 售后金额代码级强制转人工 | 完成率 → 100%，轨迹 100%（不再依赖模型自觉） |

**结论**：硬规则（金额阈值、权限）必须由代码兜底；提示词只负责“识别时机”。

## 9. 第八步：用户体系与 Redis 会话（本次新增）

**关键文件**：`app/store.py`、`static/index.html`、`static/chat.js`

**为什么**：进程内存的会话在**重启**和**多 worker** 下会丢/分裂 —— 用户觉得“刚说过就忘”、
连续失败计数失效、工单看不到。

**实现**：
- 注册/登录：密码加盐 SHA256 存 Redis Hash；登录签发令牌（TTL 7 天，可吊销）；
- 聊天记录：Redis List `cs:chat:{user_id}`，保留最近 200 条，TTL 7 天；
- 失败计数：`cs:attempts:{user_id}`，跨进程一致；
- 前端：登录/注册面板、顶部用户信息与退出、打开页面自动拉 `/api/history` 回显历史。

**验证**：注册 → 聊天 → 刷新页面历史仍在 → 重启 app 容器历史仍在。

## 10. 第九步：工单持久化 + 限流与登录安全（本次新增）

**工单持久化**：`cs:ticket:{id}`（TTL 30 天）+ `cs:tickets:index` 有序集合（按时间倒序）；
人工工作台从 Redis 读取，重启不丢；索引只保留最近 500 张。

**限流**：
- 聊天：登录用户按 user_id、匿名按 IP，默认 20 次/分钟；
- 注册/登录：按 IP，默认 10 次/分钟；
- 超限统一返回 429，前端展示“请求过于频繁”。

**登录安全**：连续失败 5 次锁定 5 分钟（Redis 计数），成功登录清零。

**验证结果**：工单跨容器重启仍存在；连续 5 次错误登录后第 6 次返回 429。

## 11. 部署（阶段 7）

**关键文件**：`Dockerfile`、`docker-compose.yml`、`.dockerignore`

```powershell
docker compose build app
docker compose up -d mysql milvus redis          # 等 healthy
docker compose run --rm app python -m app.seed_data
docker compose run --rm app python -m app.rag.indexer
docker compose up -d app
```

- 四个容器：mysql / milvus / redis / app；只有 app 对外映射 8010（Redis 额外映射本机 6381 供调试）；
- 数据落命名卷，`docker compose down` 不丢数据；
- 排错：`docker compose ps`、`docker compose logs app --tail 50`；
- 踩坑：容器缺 `langchain-community` → 改用轻量 HTTP embedding 客户端；
  Dockerfile 使用 pip cache mount，后续重建从 30 分钟降到 1 分钟内。

## 12. 上线后的监控与稳定性

**四类指标**：
| 类别 | 指标 |
|---|---|
| 基础设施 | 容器 healthy/重启次数、MySQL/Milvus/Redis 连接与健康、队列积压 |
| 模型与成本 | 每轮调用次数、token 用量、成本、模型 p95 延迟、429/5xx |
| 业务质量 | 路由分布、工具失败分类、检索无结果率、转人工率与触发分布、承诺类话术次数 |
| 安全合规 | 密钥/PII 泄漏扫描、越权查询、注入尝试、登录失败与锁定次数 |

**处置原则**：依赖失败率升高 → 熔断降级转人工；429 增多 → 限流 + 降级小模型；
转人工率突增 → 先查路由与规则变更；幻觉 → 关话术开关 + 回滚提示词/知识版本 + 补回归用例。

## 13. 面试高频追问速查

| 问题 | 一句话答案 |
|---|---|
| 怎么防死循环 | 图层面 clarify 计数改道转人工；节点内部工具循环上限；会话计数按 session 隔离、成功清零 |
| 工具失败怎么办 | 分三类：查无 → 如实告知；参数错 → 引导补全；依赖故障 → 降级重试，必要时转人工 |
| 怎么评估 Agent | 17 用例回归，任务完成率 + 轨迹正确率 + 耗时 + 成本，多轮稳定性模式 |
| 状态怎么管理 | 单次请求状态在 AgentState；跨请求在 Redis（会话/计数/工单）；业务事实在 MySQL/Milvus |
| 转人工策略 | 三种触发 + 工单上下文（trigger/reason/槽位/最近对话）+ 限流与锁定防滥用 |
| 成本与延迟 | 路由 + 专家（含工具循环）+ embedding；优化：小模型路由、语义缓存、上下文摘要 |
| 和 RAG 项目怎么配合 | 检索能力抽公共底座（多 collection 隔离、embedding 版本校验、缓存命名空间），业务层各自保留 |
| 幻觉怎么防 | 事实走工具、知识走检索引用、行为代码化、参数显式规则；再配回归用例锁住 |
| 并发怎么处理 | timeout/重试/熔断 → 限流 → 同步 DB 放线程池 → 状态外置 → 队列削峰 |
| 上线怎么监控 | 四类指标 + 阈值动作 + 幻觉发现/止损/回归闭环 |

