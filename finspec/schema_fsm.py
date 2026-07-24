"""从工具文档提取 Schema，并为确定性结构生成 token 草稿。

建议阅读顺序
============

请按照数据实际流动的方向阅读，而不是按照函数长短阅读：

``__init__``
    → ``extract_schema_from_system_prompt``
    → ``_extract_param_names``
    → ``convert_schema_values_to_token_ids``
    → ``find_candidate_pred_tokens``
    → ``state_transition``

``__init__`` 是总入口；它先把 system prompt 解析成字符串 Schema，再把字符串编码成
token Schema，最后把状态设为 ``init``。``find_candidate_pred_tokens`` 消费这些
中间结果，为主解码器提供候选草稿。

贯穿示例：API-Bank 第一条真实数据
=================================

示例来源：``data/level-1-api_processed.json`` 的第 0 条记录。

用户希望把预约改到 3 月 26 日并更换医生。该记录的标准答案是：

.. code-block:: text

    ModifyRegistration(
        appointment_id="34567890",
        new_appointment_date="2023-03-26",
        new_appointment_doctor="Dr. Lee",
    )

system prompt 中提供了三个工具。第一次数据变换发生在
``extract_schema_from_system_prompt``：大段自然语言工具文档被压缩为只包含
“工具名 → 参数名”的 Python 字典：

.. code-block:: python

    {
        "QueryHealthData": ["user_id", "start_time", "end_time"],
        "CancelRegistration": ["appointment_id"],
        "ModifyRegistration": [
            "appointment_id",
            "new_appointment_date",
            "new_appointment_doctor",
        ],
    }

参数的类型、描述和取值不会进入该字典。SchemaFSM 只利用确定性较高的调用结构，
不会从 Schema 中得到 ``34567890``、``2023-03-26`` 或 ``Dr. Lee``。

第二次数据变换发生在 ``convert_schema_values_to_token_ids``。字符串会通过当前目标
模型的 tokenizer 变成 token ids：

.. code-block:: text

    "ModifyRegistration"
        → tokenizer.encode(...)
        → (tool_token_1, tool_token_2, ...)

    "appointment_id"
        → tokenizer.encode(...)
        → [param_token_1, param_token_2, ...]

这一步不是为了改变 Schema 的含义，而是把它转换为模型真正使用的“语言”。LLM 的
输入、输出以及 Tree Attention 验证树中保存的都是 token id，而不是 Python 字符串。
如果不提前转换，解码循环每一轮都要重复调用 tokenizer，而且无法直接把工具名与固定
JSON 片段拼成 ``input_ids``，也无法与目标模型的预测 token 做逐项相等比较。

转换后的数据会沿着下面的路径继续流动：

.. code-block:: text

    字符串工具名/参数名
        → tokenized_schema
        → find_candidate_pred_tokens 拼接固定 JSON token
        → candidate token id lists
        → GPU Tensor + 候选树
        → 目标模型并行验证每个候选 token

具体数字取决于 Qwen/Llama tokenizer，不能写死。工具 token 使用 tuple，是因为它
需要成为 ``tokenized_schema`` 的字典键；参数 token 保持 list，便于与固定 JSON
前后缀拼接。这里的 tuple/list 区别只是 Python 数据结构需求，不代表两种 token。

第三次数据变换发生在 ``find_candidate_pred_tokens``，分为三个阶段：

1. ``init``：4 种 ``init_prefix`` 分别与 3 个工具名组合，共产生 12 条候选。可读形式
   类似 ``{"name": "QueryHealthData``、``{"name": "CancelRegistration`` 和
   ``{"name": "ModifyRegistration``。目标模型验证后，示例应选择最后一条工具分支。
2. ``first_param``：主解码器识别出 ``ModifyRegistration`` 后，传回其工具 token
   tuple。FSM 概念上产生以下三条候选：

   .. code-block:: text

       ", "parameters": {"appointment_id": "
       ", "parameters": {"new_appointment_date": "
       ", "parameters": {"new_appointment_doctor": "

   实际返回值从 ``tokenized_param_prefix[1:]`` 开始，因为第一个 token 已作为当前
   验证树的根 token 存在。FSM 不理解业务语义，因此三个参数都可能成为“首参数”；
   目标模型结合用户对话选择 ``appointment_id``。
3. ``remaining_param``：当模型生成完 ``"34567890",`` 后，主解码器只把最近 4 个
   token 作为 ``param_span`` 传入。FSM 找到字符串结束分隔符，再根据分隔符之后已经
   生成的 token 前缀补全下一个参数名，例如 ``new_appointment_date``。

参数值为什么不能由 SchemaFSM 生成
=================================

Schema 只声明 ``new_appointment_date`` 是字符串以及它的格式，并没有当前用户要求的
具体日期；``new_appointment_doctor`` 的定义也不包含 ``Dr. Lee``。这些值只能从当前
对话上下文中推断。因此更准确的职责划分不是“三种机制各自决定参数值”，而是：

* **目标模型始终负责最终决定。** 它读取完整 system/user 上下文，并验证每个草稿
  token。草稿错误时，第一个不匹配位置直接采用目标模型当前的预测。
* **历史检索负责提出长草稿。** 如果相似历史调用具有相同结构或部分相同值，可以
  一次提出多个 continuation token。例如历史航班调用和当前请求都包含
  ``New York → Los Angeles``，这部分值可以复用；历史日期是 2024、当前日期是 2022
  时，目标模型会在年份首次不一致处停止接受，再改为 2022。历史值因此有加速作用，
  但不会被盲目复制。
* **Token Recycling 负责提出通用短草稿。** 它复用目标模型在此前前向计算中已经
  得到的 top-k 后继 token，构造多分支候选树；它不理解日期或医生语义，也不是最终
  生成者。

所以本例中的 ``34567890``、``2023-03-26`` 和 ``Dr. Lee`` 应由目标模型结合用户
对话确定；历史检索或 Token Recycling 只可能提前猜中其中一部分，猜中的部分经验证
后批量接受，猜错的部分由目标模型纠正。

Token Recycling 是什么
======================

Token Recycling 不是另一个小模型，也不是向量数据库。目标模型每次前向本来就会
为各位置计算 logits；当前解码器取其中 top-k 后继 token，保存成
``adj_matrix[token_id, rank]``。当 Schema 与历史检索都无法提供草稿时，解码器从
当前 token 出发，沿这个邻接矩阵展开一棵“过去认为可能性较高”的候选树，再让同一个
目标模型一次并行验证整棵树。

因此它比普通逐 token 生成多了一步“回收以前已经算过的 top-k 结果并提前展开多条
路径”，但裁判仍是目标模型。完整实现位于 ``decoding.py``：
``prepare_data_input_ids`` 负责用邻接矩阵填树，``finspec_forward`` 负责验证树并
更新邻接矩阵。

它是当前解码器原创的吗
======================

不是。Token Recycling 来自 Luo 等人的论文：
``Turning Trash into Treasure: Accelerating Inference of Large Language Models with
Token Recycling``（arXiv:2408.08696，后发表于 ACL 2025）。当前代码把这个已有的
通用推测解码方法作为兜底：Schema 草稿与历史检索草稿均不可用时，才执行这里的
Token Recycling。

还要区分“模型”和“推理引擎”：Token Recycling 不会写进模型参数，也不改变模型
架构；它是在模型外部管理旧 top-k、候选树、Tree Attention 和 KV Cache 的**运行时
解码策略**。所以“为什么现在的大模型不都用”更准确的问法是：“为什么推理框架不把
它默认用于所有请求？”

它与目标模型普通 decode 的区别
==============================

两种方式使用的是**同一个目标模型**，差别不在“谁来生成”，而在“一次模型前向希望
推进多少个 token”以及“前向之前有没有先准备草稿”。

普通自回归 decode 的流程是：

.. code-block:: text

    当前完整上下文
        → 目标模型做一次当前前向
        → 读取 next-token 分布
        → 选择当前 top-1
        → 只追加 1 个 token
        → 带着更长的上下文再次前向

即使使用 KV Cache，它通常仍需要“一次串行前向推进一个 token”。

Token Recycling 的流程是：

.. code-block:: text

    目标模型过去某轮前向产生的 top-k logits
        → 保存到 adj_matrix，成为旧预测

    当前完整上下文
        → 查询 adj_matrix（不需要额外模型前向）
        → 用旧预测提前展开多条未来 token 路径
        → 目标模型对整棵候选树做一次当前前向
        → 只接受与当前模型预测连续一致的最长路径
        → 最理想时一次追加多个 token

因此可以把两者理解为：

* 普通 decode：**当前模型算一次，只走当前最可能的一步**；
* Token Recycling：**先用模型过去算过的结果猜多步，再让当前模型一次批改整棵树**。

“旧预测”和“当前验证”必须分开理解。``adj_matrix`` 只按 token id 保存最近一次观察到
的 top-k 后继，会丢失当时的完整上下文，所以它可能过时或不适合当前请求。当前目标
模型仍会读取完整 prompt、对话历史和已接受前缀；只有旧草稿与它在当前上下文下的
预测相同，才会被接受。

以预约示例说明（为了可读性按文本片段展示，真实 token 边界取决于 tokenizer）：

.. code-block:: text

    当前已生成：
    ..."new_appointment_doctor": "Dr.

    普通 decode：
        forward 1 → " Lee"
        forward 2 → '"'
        forward 3 → "}"
        三次串行前向分别推进一个 token/片段

    Token Recycling：
        旧邻接表中可能存在 Dr. → Lee → " → }
        → 同时展开这条路径和其他 top-k 分支
        → 当前目标模型一次验证候选树

如果当前用户确实要求 ``Dr. Lee``，模型可能连续接受整条路径，从而减少串行前向次数。
如果当前用户要求 ``Dr. Kim``，旧草稿会在 ``Lee`` 首次不匹配处停止；解码器采用当前
目标模型预测的 ``Kim`` 作为下一步，不会为了复用缓存而输出错误医生。

它的收益也不是免费的：一次树验证需要并行计算更多候选节点并占用额外显存。只有
“减少串行前向次数”带来的收益大于“验证额外分支”的成本时才会加速。若旧草稿全部
不匹配，本轮可能只接受根 token，效果接近普通 decode，却仍支付了候选树计算开销。

为什么它没有成为所有场景的默认方案
==================================

原始 Token Recycling 实验展示的是一组明确条件下的收益，而不是“对所有服务负载都
更快”的保证：论文重点评估贪心解码、batch size 1，并使用单张 A100-80GB。真实在线
系统还要考虑连续批处理、Paged KV Cache、CUDA Graph、不同请求的可变接受长度和
采样策略。把树状候选正确接入这些组件，需要额外的调度、mask 和 KV 搬运逻辑。

它的主要限制可以按数据流理解：

1. **上下文被压缩得过于激进。** ``adj_matrix`` 的键只有当前 token id。同一个
   ``"Dr"`` 在“医生姓名”和普通文本里会共用一行，完整 prompt、工具名和字段名都
   没有进入键。因此旧 top-k 只是廉价候选，不能看作更“底层”或更准确的语义表示。
2. **存在冷启动。** 新建邻接矩阵时各行都是 0；只有目标模型实际处理过某些 token
   后，对应行才有有意义的 top-k。请求少、输出模式变化快或模型刚启动时，命中率会
   较低。可以持久化或用代表性流量预热，但也要防止过期模式污染新请求。
3. **树的大小存在两难。** 更宽、更深的树覆盖更多未来路径，却也让一次验证计算更多
   无效节点。树太小可能猜不中，树太大则额外计算超过省下的串行前向，反而变慢。
4. **固定树不能适应请求。** 当前 ``tree_template.py`` 对所有位置使用同一棵静态树。
   JSON 标点和常见字段可能适合深挖 top-1 路径；姓名、日期等高不确定值通常需要更
   谨慎的节点预算。固定模板无法针对这种差异动态调整。
5. **采样解码更复杂。** 当前实现用 ``argmax`` 和 token 相等做验证，适合贪心解码。
   temperature/top-p 下，目标 token 是抽样结果，不能简单要求等于旧 top-k 的第一
   项；必须实现推测采样的接受概率与拒绝后重采样，否则可能改变原分布。
6. **高并发时收益可能缩小。** batch 很小时，GPU 往往还有并行余量，拿来验证一棵树
   可能很划算；连续批处理已经让 GPU 饱和时，额外树节点会和其他请求争夺算力。此时
   普通批量 decode 可能吞吐更高，即使单请求需要更多串行轮次。
7. **收益依赖重复性。** 结构化 JSON、SQL 关键字和固定协议有大量重复局部转移，容易
   命中；开放式文本、唯一账号、金额、日期和姓名更依赖完整上下文，旧邻接行未必有
   帮助。最终是否值得启用，应以实际 ``accept_len``、端到端延迟和吞吐衡量。

FinSpec 中应如何定位
===================

Token Recycling 适合继续作为最后一级通用兜底，优先加速 JSON 标点、字段名、SQL
关键字和反复出现的调用骨架；它不应被当作账户号、金额、日期、姓名等业务事实的
来源。合理的优先级仍是：

``Schema 确定性结构 → 相似历史长草稿 → Token Recycling 短草稿 → 普通 decode``。

目标模型会验证所有来源，因此历史或 Token Recycling 猜错不会直接改变贪心解码
结果；但错误草稿仍消耗计算。FinSpec 后续可以按工具/字段记录平均 ``accept_len``，
接受率或延迟收益低时自动缩小树或暂时关闭 Token Recycling。

论文概念上把输出划分为工具名、参数名、参数值和其他文本四类状态；当前 API-Bank
实现进一步简化为 ``init``、``first_param``、``remaining_param`` 三个单向状态。
开放式参数值没有独立状态，而是交给历史检索或 Token Recycling 提供草稿。

需要注意：当前 FSM 不记录已经使用过的参数，也不了解 required/optional、参数顺序
或字段依赖，所以在后续阶段仍可能再次提出 ``appointment_id``。Schema 候选始终只是
草稿，最终是否接受仍由目标模型逐 token 验证。
"""

import re
import ast
from typing import Dict, List


class SchemaFSM:
    """解析 system prompt 中的工具文档，并生成结构化候选 token。

    ``schema`` 保存便于阅读的工具名/参数名字符串；``tokenized_schema`` 在请求开始时
    一次性编码，避免解码热路径反复调用 tokenizer。这里产生的内容仍只是草稿，每个
    token 最终都必须通过目标模型验证。
    """

    def __init__(self, tools_metadata=None, tokenizer=None):
        """初始化当前请求独享的 Schema 状态机。

        参数:
            tools_metadata:
                - ``dict``：已经解析好的 ``{"工具名": ["参数名", ...]}``；
                - ``str``：包含 API-Bank 工具文档的 system prompt；
                - ``None``：创建空 Schema。
            tokenizer: 必须与目标模型一致，否则候选 token 无法与目标输出正确比较。

        TODO(优化): 参数签名目前允许 ``tokenizer=None``，但初始化固定前缀时仍会调用
        tokenizer。应拆分“纯 Schema 解析器”和“token 草稿编译器”，或在缺少 tokenizer
        时延迟编码，修复 ``extract_schema_from_query`` 等纯解析用法。
        """
        self.tokenizer = tokenizer
        # 真实示例进入这里时，tools_metadata 是第一条记录的完整 system prompt 字符串。
        # 经过下面的 str 分支后：
        #
        #   自然语言工具文档
        #       ↓ extract_schema_from_system_prompt
        #   {
        #       "QueryHealthData": [...],
        #       "CancelRegistration": ["appointment_id"],
        #       "ModifyRegistration": [
        #           "appointment_id",
        #           "new_appointment_date",
        #           "new_appointment_doctor",
        #       ],
        #   }
        #
        # 注意：用户请求和 answer 不会传进 SchemaFSM。选择 ModifyRegistration 的
        # 语义判断由目标模型完成，FSM 只知道 system prompt 中“有哪些合法结构”。
        if tools_metadata is None:
            self.schema = {}
        elif isinstance(tools_metadata, dict):
            self.schema = tools_metadata
        elif isinstance(tools_metadata, str):
            self.schema = self.extract_schema_from_system_prompt(tools_metadata)
        else:
            raise ValueError("tools_metadata must be dict, str, or None")
        # 这些前缀来自 API-Bank 上观察到的 Qwen/Llama 输出风格，包括不同空格、
        # 代码块和 <tool_call> 包装。一个工具会与每种 init_prefix 组合成候选。
        # TODO(优化): FinSpec 应直接消费标准 JSON Schema/Pydantic 工具定义，并通过
        # grammar/constrained decoding 生成结构，避免硬编码空白、引号和字段顺序。
        # 不同模型的历史前缀参考：
        # Qwen2.5, llama3.2-3B: self.init_prefix = ['{\"name\": \"', '<tool_call>\n{\"name\": \"', '<tool_call>\n  {\"name\": \"', '```<tool_call>\n{\"name\": \"']
        # llama3.1-8B ['<tool_call>\n{\"name\": \"', '</tool_call>\n{\"name\": \"']
        self.init_prefix =   ['{\"name\": \"', '<tool_call>\n{\"name\": \"', '<tool_call>\n  {\"name\": \"', '```<tool_call>\n{\"name\": \"']
        self.param_prefix = '\", \"parameters\": {\"'
        self.param_suffix = '\": \"'
        self.remaining_param_prefix = ' \"'

        self.tokenized_init_prefix = [self._encode(prefix) for prefix in self.init_prefix]
        self.tokenized_param_prefix = self._encode(self.param_prefix)
        self.tokenized_param_suffix = self._encode(self.param_suffix)
        self.tokenized_remaining_param_prefix = self._encode(self.remaining_param_prefix)
        # 当前假设 `",` 的第一个 token 足以标识字符串参数值结束。
        # TODO(优化): 应匹配完整分隔符 token 序列；不同 tokenizer 可能拆分方式不同，
        # 仅检查第一个 token 会提前触发或漏掉字段切换。
        self.seperator_id = self._encode("\",")[0]
        # 此时发生第二次数据变换。以 ModifyRegistration 为例，结构从：
        #
        #   "ModifyRegistration": ["appointment_id", ...]
        #
        # 变为：
        #
        #   (ModifyRegistration 的 token ids):
        #       [
        #           [appointment_id 的 token ids],
        #           [new_appointment_date 的 token ids],
        #           [new_appointment_doctor 的 token ids],
        #       ]
        #
        # token id 数字与目标模型 tokenizer 绑定，所以源码只保存运行时结果。
        self.tokenized_schema = self.convert_schema_values_to_token_ids()

        # 新请求总是从预测“调用前缀 + 工具名”开始。
        self.state = "init"  # init, first_param, remaining_param
    
    def state_transition(self):
        """将状态按 ``init → first_param → remaining_param`` 单向推进。

        主解码器在构造一次 Schema 草稿树后调用本方法。``remaining_param`` 之后不再
        自动推进，因为后续可能反复出现多个参数名/参数值组合。
        """
        if self.state == "init":
            # 示例：目标模型刚刚验证了 `{"name": "ModifyRegistration` 草稿，
            # 下一轮应该构造 `"parameters"` 与首参数名候选。
            self.state = "first_param"
        elif self.state == "first_param":
            # 示例：目标模型刚刚验证了 `"appointment_id": "`，后续进入参数值和
            # 剩余参数交替生成阶段；该状态此后不会自动离开。
            self.state = "remaining_param"
        else:
            raise ValueError("Invalid state")

    def find_candidate_pred_tokens(self, tool_name=None, param_span=None):
        """根据当前状态返回一组结构化候选 token 序列。

        - ``init``：枚举“允许的调用前缀 + 工具名”；
        - ``first_param``：工具名已确定时枚举该工具的首个参数名，否则只补固定结构；
        - ``remaining_param``：根据最近 token 已生成的前缀，补全可能的后续参数名。

        ``tool_name`` 使用 token tuple，而非字符串，因为主循环直接通过输出 token
        判断已经选择了哪个工具。``param_span`` 是当前输出末尾的短 token 窗口。

        TODO(优化): 当前没有记录已经使用过的参数，也不了解 required/optional、
        嵌套对象、数组、数字/布尔值或字段依赖。应使用真正的 JSON grammar FSM，
        并让状态随每个已接受 token 更新。
        """
        if self.state == "init":
            candidates = []
            # 枚举所有协议前缀与工具名的笛卡尔积，让目标模型选择正确格式和工具。
            # 示例中 4 个 init_prefix × 3 个工具 = 12 条候选。以下只是其中三条
            # 可读形式，真实返回值全部是 token id list：
            #
            #   {"name": "QueryHealthData
            #   {"name": "CancelRegistration
            #   {"name": "ModifyRegistration
            for init_prefix in self.tokenized_init_prefix:
                for tool_name_token_ids_tuple in self.tokenized_schema.keys():
                    # dict key 使用 tuple 以便哈希，这里转回 list 进行 token 拼接。
                    tool_name_token_ids = list(tool_name_token_ids_tuple)
                    candidate_token_ids = init_prefix + tool_name_token_ids
                    candidates.append(candidate_token_ids)
            return candidates
        elif self.state == "first_param":
            candidates = []
            if tool_name == None:
                # 尚未识别出工具名时无法限制参数，只补齐固定 parameters 结构。
                candidate_token_ids = self.tokenized_param_prefix[1:]
                candidates.append(candidate_token_ids)
            else:
                if tool_name not in self.tokenized_schema:
                    raise ValueError(f"Tool name {tool_name} not found in schema")
                param_list = self.tokenized_schema[tool_name]
                if not param_list:
                    # 无参数工具仍需生成固定 JSON 结构，后续由模型负责闭合对象。
                    candidate_token_ids = self.tokenized_param_prefix[1:]
                    candidates.append(candidate_token_ids)
                else:
                    # 示例中 tool_name 是 ModifyRegistration 的 token tuple，
                    # param_list 含 3 个参数，因此会产生 3 条参数名候选：
                    #
                    #   `", "parameters": {"appointment_id": "`
                    #   `", "parameters": {"new_appointment_date": "`
                    #   `", "parameters": {"new_appointment_doctor": "`
                    #
                    # [1:] 表示候选的首 token 已由主解码循环作为树根持有，不应重复。
                    for param_token_ids in param_list:
                        candidate_token_ids = self.tokenized_param_prefix[1:] + param_token_ids + self.tokenized_param_suffix
                        candidates.append(candidate_token_ids)
            return candidates
        elif self.state == "remaining_param":
            candidates = []
            if tool_name == None:
                return candidates
            if tool_name not in self.tokenized_schema:
                raise ValueError(f"Remaining param: tool name {tool_name} not found in schema")
            # 定位上一个字符串值结束标记，并取得它之后已经生成的参数名前缀。
            # 示例：目标模型已经生成 `"appointment_id": "34567890",`。主循环只传入
            # 最近 4 个 token；若其中出现 separator_id，就从该位置之后检查参数名前缀。
            separator_idx = param_span.index(self.seperator_id)
            tokens_after_separator = param_span[separator_idx + 1:]
            
            param_list = self.tokenized_schema[tool_name]
            for param_token_ids in param_list:
                candidate_token_ids = self.tokenized_remaining_param_prefix + param_token_ids + self.tokenized_param_suffix
                # 只保留与已确认 token 前缀一致的参数名，并仅返回尚未生成的剩余部分。
                # 假设模型已经生成了新字段开头和 `new_` 前缀，那么 appointment_id
                # 会因前缀不符被过滤，日期/医生字段仍可能保留。若分隔符之后还没有
                # 可区分前缀，三个参数（包括已经用过的 appointment_id）都可能出现；
                # 这是当前 FSM 不记录已用字段的直接结果。
                if len(candidate_token_ids) >= len(tokens_after_separator):
                    candidate_prefix = candidate_token_ids[:len(tokens_after_separator)]
                    if candidate_prefix == tokens_after_separator:
                        candidates.append(candidate_token_ids[len(tokens_after_separator):])
            return candidates
        else:
            raise ValueError("Invalid state")

    def extract_schema_from_system_prompt(self, system_prompt: str) -> Dict[str, List[str]]:
        """从 API-Bank system prompt 中提取工具名与参数名。

        两种模板表达的是同一种工具 Schema，仅仅是数据集的文本排版不同。

        模板 1（level-1/2）显式使用 ``Name:`` 和独立 ``Description:`` 行：

        .. code-block:: text

            1. Name: ModifyRegistration
            Description: This API modifies the registration ...
            Parameters: {'appointment_id': {...},
                         'new_appointment_date': {...},
                         'new_appointment_doctor': {...}}

        正则直接读取 ``Name:`` 后面的 ``ModifyRegistration``。

        模板 2（level-3）没有 ``Name:`` 标签，工具名和描述写在同一行：

        .. code-block:: text

            1. QueryMeeting: The API for retrieving the meeting details ...
            Parameters: {'user_name': {...}}

        正则把编号 ``1.`` 之后、冒号之前的 ``QueryMeeting`` 当作工具名。两种输入
        最终分别得到：

        .. code-block:: python

            {"ModifyRegistration": [
                "appointment_id",
                "new_appointment_date",
                "new_appointment_doctor",
            ]}

            {"QueryMeeting": ["user_name"]}

        返回 ``{"工具名": ["参数名", ...]}``。此函数只关心候选结构，不保留参数
        类型、描述和约束。支持两种模板只是为了兼容 API-Bank 的历史数据格式，不是
        两套解析目标，也不会改变后续 FSM 算法。

        TODO(优化): 正则解析 prompt 对格式变化非常敏感。FinSpec 应把标准化工具
        Schema 作为显式输入，并在进入模型前完成校验，而不是从展示文本反向恢复。
        """
        schema = {}
        
        # 模板 1：
        # - "1. Name: ToolName\nDescription: ...\nParameters: {...}"
        # - "1. Name: ToolName\nDescription: Parameters: {...}" (inline Parameters)
        tool_entry_pattern1 = r'\d+\.\s*Name:\s*([^\n]+)\nDescription:[^\n]*?(?:\nParameters:\s*|Parameters:\s*)'
        matches1 = list(re.finditer(tool_entry_pattern1, system_prompt, re.MULTILINE))
        
        # 模板 2：工具名位于冒号前，Parameters 位于下一行。
        tool_entry_pattern2 = r'\d+\.\s*([^:]+):\s*[^\n]*\nParameters:\s*'
        matches2 = list(re.finditer(tool_entry_pattern2, system_prompt, re.MULTILINE))
        
        # 两种正则都命中时优先模板 1，保持与较明确的 Name 字段一致。
        if matches1:
            # 第一条真实记录命中模板 1 三次，对应 QueryHealthData、
            # CancelRegistration、ModifyRegistration。
            matches = matches1
        elif matches2:
            # level-3 第 0 条真实记录命中模板 2，冒号前的 QueryMeeting 是工具名，
            # 最终得到 {"QueryMeeting": ["user_name"]}。
            matches = matches2
        else:
            return schema
        
        for i, match in enumerate(matches):
            # 第 i 次循环只处理一个工具。以 i=2 为例：
            #   match.group(1) == "ModifyRegistration"
            tool_name = match.group(1).strip()
            start_pos = match.end()
            
            # 先用下一工具条目的起点切出当前工具的参数区域。
            if i < len(matches) - 1:
                end_pos = matches[i + 1].start()
                params_str = system_prompt[start_pos:end_pos]
            else:
                params_str = system_prompt[start_pos:]
            
            # 从第一个左花括号开始计数，找到与之配对的右花括号。相比简单的非贪婪
            # 正则，这种方式可以容忍参数描述中的嵌套字典。
            brace_start = params_str.find('{')
            if brace_start != -1:
                brace_count = 0
                brace_end = brace_start
                for j, char in enumerate(params_str[brace_start:], start=brace_start):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            brace_end = j + 1
                            break
                
                params_dict_str = params_str[brace_start:brace_end]
                
                # 只提取顶层参数名；当前草稿器并不消费类型、默认值和描述。
                # ModifyRegistration 在这里得到：
                # [
                #     "appointment_id",
                #     "new_appointment_date",
                #     "new_appointment_doctor",
                # ]
                param_names = self._extract_param_names(params_dict_str)
                # 即使参数列表为空也保留工具，否则无参数工具无法成为候选。
                schema[tool_name] = param_names
        
        return schema

    def _extract_param_names(self, params_str: str) -> List[str]:
        """从 Parameters 字典文本中提取顶层键。

        优先使用安全的 ``ast.literal_eval`` 解析 Python 风格的单引号字典；解析失败
        时才退回正则。回退正则可能把嵌套字典的键也当作参数名，因此仅作为对脏数据
        的容错方案。

        TODO(优化): 为解析失败、重复参数和异常嵌套增加显式告警与测试，避免静默
        生成错误 Schema。
        """
        param_names = []
        
        try:
            # literal_eval 不执行任意代码，并原生支持 API-Bank 的单引号字典。
            # 示例输入仍包含 type/description 等嵌套信息；解析成 dict 后这里只读取
            # 最外层 keys，因此这些描述自然被丢弃。
            params_dict = ast.literal_eval(params_str)
            
            if isinstance(params_dict, dict):
                param_names = list(params_dict.keys())
        except (ValueError, SyntaxError):
            # 容错匹配单引号或双引号包围的 ``key:``。
            key_pattern = r"['\"]([^'\"]+)['\"]\s*:"
            param_names = re.findall(key_pattern, params_str)
        
        return param_names

    def get_tool_names(self):
        """返回当前 Schema 的全部工具名。"""
        return list(self.schema.keys())

    def get_all_params(self, tool_name: str) -> List[str]:
        """返回指定工具的参数名；工具不存在时返回空列表。"""
        return self.schema.get(tool_name, [])

    def convert_schema_values_to_token_ids(self) -> Dict[tuple, List[List[int]]]:
        """一次性把工具名和参数名编码为目标模型 token ids。

        转换前的字符串适合人阅读，但不能直接放进模型输入或与模型输出比较。转换后，
        ``find_candidate_pred_tokens`` 可以把固定 JSON token、工具名 token 和参数名
        token 直接拼接；主解码器再把这些 list 转为 GPU Tensor、装入候选树，并与
        目标模型预测的 token ids 逐项验证。提前编码也避免了解码热路径重复 tokenizer。

        返回映射的 key 是工具名 token tuple，value 是多个参数名 token list。工具名
        使用 tuple 是为了可作为字典键；参数名保留 list 便于与固定 JSON token 拼接。
        tuple 与 list 中保存的都是普通 token id，区别仅来自 Python 的可哈希/可拼接
        需求。tokenizer 为空时返回空字典。

        TODO(优化): 相同工具集合会被每个问题重复编码，可按“tokenizer 标识 +
        Schema hash”缓存结果；同时验证不同字符串是否意外映射为相同 token tuple。
        """
        if self.tokenizer is None:
            return {}
        
        tokenized_schema = {}
        for tool_name, param_names in self.schema.items():
            # 禁止自动添加 BOS/EOS，否则候选无法嵌入正在生成的 JSON 中间。
            tool_token_ids = self._encode(tool_name)
            if not tool_token_ids:
                continue
            # list 不可哈希，转换为 tuple 后才能作为工具查找键。
            tool_token_id_tuple = tuple(tool_token_ids)
            
            param_token_ids = []
            for param_name in param_names:
                token_ids = self._encode(param_name)
                if token_ids:
                    param_token_ids.append(token_ids)
            tokenized_schema[tool_token_id_tuple] = param_token_ids
            # 示例循环处理 ModifyRegistration 后，新增的条目概念上是：
            #
            #   key   = tuple(tokenizer.encode("ModifyRegistration"))
            #   value = [
            #       tokenizer.encode("appointment_id"),
            #       tokenizer.encode("new_appointment_date"),
            #       tokenizer.encode("new_appointment_doctor"),
            #   ]
            #
            # 主解码器稍后通过输出末尾 token 是否等于 key，确认模型选择了哪个工具。
        
        return tokenized_schema

    def _encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    @staticmethod
    def extract_schema_from_query(query_data: Dict) -> Dict[str, List[str]]:
        """便捷方法：从包含 ``system`` 字段的单条数据中提取 Schema。"""
        if 'system' not in query_data:
            return {}
        
        schema_fsm = SchemaFSM(query_data['system'])
        return schema_fsm.schema

    def __repr__(self):
        return f"SchemaFSM(schema={self.schema})"
