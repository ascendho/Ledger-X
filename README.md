# Ledger-X

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Tests](https://img.shields.io/badge/tests-61%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)
![Data](https://img.shields.io/badge/data-synthetic%20only-lightgrey)

Ledger-X 是一个面向商户结算对账的智能分析系统。项目使用合成账本模拟支付、退款、结算批次、手续费规则和到账凭证；模型负责理解问题、选择只读工具和组织解释；工具层负责确定性账务计算；运行时在结构化工具调用阶段提供可验证的推测解码加速。

代码中使用 `ledger_x` 作为 Python namespace、`ledger-x` 作为安装包名、`LEDGERX_*` 作为环境变量前缀。

项目关注三个目标：

- 对账结果可验证：金额、差额、异常类型都来自只读工具和独立 oracle。
- Agent 行为可控：上下文、记忆、工具调用和安全策略都写入 trace。
- 工具调用更快：针对重复的结构化工具调用输出，在目标模型校验下减少响应延迟。

## 核心能力

- 合成账本生成：生成商户、手续费规则、支付、退款、结算批次、结算明细和到账凭证。
- 只读工具接口：模型只能调用受控 typed tools，不能直接执行 SQL、写库或发起资金操作。
- 对账工具集：支持结算汇总、异常批次列表、批次复算、交易追踪、规则查询和商户解析。
- 策略护栏：限制授权商户 scope，拦截写操作意图、越权商户和异常工具调用。
- 上下文管理：按预算注入 system policy、working memory、case memory 和 retrieved context。
- 证据记忆：只把工具返回的结构化证据写入记忆，不写入模型自由文本。
- 审计轨迹：每次运行记录模型消息、工具调用、工具结果、策略判定、记忆读写和耗时。
- 结构化调用加速：在工具调用生成阶段提出 draft tokens，并由同一个目标模型验证后接受。

## 项目结构

```text
.
├── README.md
├── pyproject.toml
├── requirements.txt
└── src/ledger_x/
    ├── app/        # Agent、工具、策略、合成账本和 Web demo
    ├── specdec/    # 结构化工具调用推测解码运行时
    ├── __main__.py
    └── paths.py
```

`benchmarks/` 中包含可提交的端到端评测脚本；`experiments/`、`artifacts/`、`runtime/` 是本地开发和实验输出目录，默认不提交。

## 快速开始

安装依赖：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

生成一个本地合成账本：

```bash
.venv/bin/python -m ledger_x seed --db runtime/ledger.duckdb
```

直接调用只读工具：

```bash
.venv/bin/python -m ledger_x tool query_settlement_summary \
  '{"merchant_id":"M001","start_date":"2026-03-01","end_date":"2026-09-01","group_by":"channel"}' \
  --db runtime/ledger.duckdb
```

连接本地或内网 OpenAI-compatible 模型服务后运行 Agent：

```bash
LEDGERX_BASE_URL=http://127.0.0.1:2011/v1 \
LEDGERX_MODEL=qwen3-14b-fp8-ledger-x \
.venv/bin/python -m ledger_x ask \
  '查询 M001 从2026年3月1日到9月1日的结算异常，选出差额最大的批次并解释原因。'
```

启动本地 Web 工作台：

```bash
LEDGERX_BASE_URL=http://127.0.0.1:2011/v1 \
LEDGERX_MODEL=qwen3-14b-fp8-ledger-x \
.venv/bin/python -m ledger_x serve --port 8090
```

## 数据与数据库

Ledger-X 使用 DuckDB 作为本地合成账本数据库。默认数据库文件是 `runtime/ledger.duckdb`，由命令行生成，不连接真实银行、支付平台或商户账本。

数据库里保存了完整对账链路需要的只读事实：

- `merchants`：合成商户信息。
- `fee_rules`：不同渠道在不同日期生效的手续费规则。
- `payments`：支付流水，金额单位为分。
- `refunds`：退款流水。
- `settlement_batches`：商户、渠道、月份维度的结算批次。
- `settlement_items`：批次内的支付/退款明细和渠道侧记录的手续费。
- `bank_receipts`：到账凭证，用来判断缺失到账、延迟到账和短款。

数据库还定义了 `expected_items` 和 `expected_batches` 两个 view，用于按规则确定性复算“应结算金额”。生成账本时会同时写出一个独立的 `ledger.oracle.json`，它不是由工具 SQL 计算出来的，而是由数据生成器直接维护的校验答案。benchmark 用这个 oracle 判断工具调用、证据和最终回答是否正确。

模型不会直接写 SQL。Agent 只把受控工具 schema 交给模型，模型选择工具和参数；工具层再用参数化只读 SQL 查询 DuckDB。这样数据库承担四个作用：

- 提供可复现的合成业务事实。
- 执行确定性的金额计算和异常识别。
- 作为 Agent 最终回答的证据来源。
- 为 benchmark 提供可自动评分的 oracle。

## Agent 上下文与记忆

Ledger-X 不把完整历史消息直接塞回模型，而是把上下文分成四类：

1. `system policy`：合成数据边界、只读工具约束、授权商户、金额单位和日期口径。
2. `working memory`：当前任务内从工具结果提取的证据，例如批次号、差额、异常类型、规则号。
3. `case memory`：同一 tenant 和 merchant scope 下的长期已验证事实。
4. `retrieved context`：根据批次号、交易号、规则号、异常类型或商户 scope 检索出来的少量历史事实。

记忆写入遵循证据优先策略：

- 允许写入：工具返回的 `batch_id`、`merchant_id`、`transaction_id`、`rule_id`、`difference_fen`、异常类型等结构化证据。
- 禁止写入：模型最终回答、用户原文中的未验证金额、工具错误文本。
- 作用域隔离：M001 的证据不会注入到 M002 的请求。

每次运行的 trace 会记录：

- `context`：本轮注入模型的上下文块。
- `context_budget`：压缩前后字符数、working facts 数和 retrieved facts 数。
- `memory_reads`：本轮检索到的历史证据。
- `memory_writes`：本轮从工具结果写入的证据。
- `filled_from_context`：从授权 scope 或历史上下文安全补全的字段。

如果问题缺少必要上下文，例如没有商户或没有日期范围，Agent 会返回 `needs_clarification`，不会猜测参数后直接调用工具。

## 结构化工具调用加速

Ledger-X 的加速发生在模型生成工具调用的阶段。流程如下：

1. Agent 将问题、商户 scope、工具 schema、工作记忆和检索上下文组成 prompt。
2. 目标模型正常生成工具调用。
3. 运行时根据 schema、当前 token 后缀和历史已验证工具调用提出一小段 draft tokens。
4. 同一个目标模型验证 draft tokens。
5. 只有验证通过的 tokens 才会进入最终输出。

这种方式不缓存金融结果，不绕过目标模型，也不会把历史参数强行写入当前请求。它只加速稳定的结构化 token，例如工具名、JSON key、枚举值、括号和固定 wrapper。

## 加速效果

### 1. 工具调用生成阶段

在同一个 `Qwen/Qwen3-14B-FP8` 目标模型上，对 400 个测试问题重复 3 轮，共比较每种模式 1,200 次结构化工具调用生成。

最核心的数据是：

| 模式 | p50 响应时间 | p95 响应时间 | 工具调用正确率 |
| --- | ---: | ---: | ---: |
| 普通生成（AR） | 2143.3 ms | 2592.2 ms | 1200 / 1200 |
| Ledger-X 加速 | 371.1 ms | 488.6 ms | 1200 / 1200 |

也就是说，模型生成一次工具调用的中位延迟降低 **82.68%**，约 **5.8×** 更快。

正确性保持不变：

- 数值校验：609 / 609 次 oracle 校验一致。
- 结构化参数归一化后一致：1,200 / 1,200。
- speculative draft token 接受率：83.15%。

测试使用的合成账本包含 100,000 笔支付、7,056 笔退款和 90 个结算批次。这里的加速只针对“生成工具调用”的响应时间。

### 2. 完整 Agent 端到端耗时

完整 Agent 任务还包括上下文整理、策略检查、工具执行、证据记忆和最终回答生成，所以端到端提升通常会小于单次工具调用生成提升。

当前端到端评测入口会记录：

- `agent_p50_ms / agent_p95_ms`：完整任务耗时。
- `llm_p50_ms / llm_p95_ms`：所有模型轮次耗时。
- `tool_p50_ms`：工具执行耗时。
- `mean_llm_rounds / mean_tool_calls`：平均模型轮次和工具调用数。
- `task_success_rate`：工具链、证据和最终答案是否全部匹配 oracle。

在 80 条完整 Agent test case 上，分别运行普通生成和 Ledger-X full 加速模式，结果如下：

| 模式 | 任务成功率 | 端到端 p50 | 端到端 p95 | LLM p50 | 平均 LLM 轮数 | 平均工具调用 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 普通生成（AR） | 64 / 80 | 9298.9 ms | 16923.5 ms | 9240.5 ms | 2.13 | 1.13 |
| Ledger-X full | 66 / 80 | 5810.3 ms | 13386.6 ms | 5750.6 ms | 2.16 | 1.16 |

这里的 AR 是 autoregressive baseline，即不开启推测解码加速的同模型普通生成模式。按两种模式都成功的 64 个任务做配对比较，完整 Agent 端到端中位延迟下降 **19.95%**，95% bootstrap CI 为 **10.73%–40.48%**。full 模式在这轮评测中提出 14,082 个 draft tokens，目标模型接受 6,841 个，接受率约 **48.58%**。

`benchmarks/` 中包含配对评测脚本，可按下面流程生成端到端对比报告。

```bash
.venv/bin/python -m benchmarks.agent_workload \
  --db runtime/ledger.duckdb \
  --out runtime/agent-workload.jsonl

LEDGERX_BASE_URL=http://127.0.0.1:2011/v1 \
LEDGERX_MODEL=qwen3-14b-fp8-ledger-x \
.venv/bin/python -m benchmarks.agent_benchmark \
  --workload runtime/agent-workload.jsonl \
  --db runtime/ledger.duckdb \
  --manifest runtime/service-manifest.json \
  --out artifacts/evidence/agent-e2e-ar \
  --split test --limit 80 --repeats 1

LEDGERX_BASE_URL=http://127.0.0.1:2011/v1 \
LEDGERX_MODEL=qwen3-14b-fp8-ledger-x \
.venv/bin/python -m benchmarks.agent_benchmark \
  --workload runtime/agent-workload.jsonl \
  --db runtime/ledger.duckdb \
  --manifest runtime/service-manifest.json \
  --out artifacts/evidence/agent-e2e-full \
  --split test --limit 80 --repeats 1

.venv/bin/python -m benchmarks.agent_report \
  --ar artifacts/evidence/agent-e2e-ar \
  --candidate artifacts/evidence/agent-e2e-full \
  --candidate-name full \
  --out artifacts/evidence/agent-e2e-comparison.json
```

## 测试

```bash
.venv/bin/python -m pytest tests -q
```

当前本地测试覆盖：

- 只读工具和合成账本生成
- 工具参数校验和商户 scope
- Agent 策略护栏
- 上下文压缩和证据记忆
- 结构化调用 draft engine
- benchmark 结果汇总逻辑

## 项目边界

- 只使用合成 CNY 数据，不连接真实银行、支付平台或商户账本。
- 工具只读，不执行转账、退款、写库或外部副作用。
- 模型输出不能作为账务事实来源，最终回答必须基于工具证据。
- 延迟实验只针对固定模型、固定工具集、batch size 1 和 frozen workload。

## License

MIT
