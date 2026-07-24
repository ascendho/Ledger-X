"""FinSpec 的推测式工具调用解码核心。

先从哪里看
==========

不要从第一个辅助函数开始逐行向下读。推荐把 ``finspec_forward`` 当作主线，按下面
顺序阅读：

1. ``finspec_forward``：先看清一次请求从 prompt 到完整输出的生命周期；
2. ``get_topk_similar_outputs``：理解如何用当前请求表示筛选历史调用；
3. ``find_candidate_pred_tokens``：理解如何对筛出的历史做精确 token 后缀匹配；
4. ``create_template_from_candidates`` 和 ``build_retrieval_tree_from_candidates``：
   理解若干线性草稿怎样变成一棵候选树；
5. ``prepare_data``：理解 Tree Attention mask、position ids 与 ``search_path``；
6. 回到 ``finspec_forward`` 中间的 ``model(...)``、``reward``、``accept_len`` 和
   KV Cache 压缩，这是推测解码真正决定“接受哪些 token”的部分；
7. 最后看 ``prepare_data_input_ids`` 和 ``adj_matrix`` 更新，理解没有 Schema/历史
   草稿时的 Token Recycling；
8. ``_truncate_generated_suffix`` 等停止辅助函数最后再看。

未来 Agent runtime 的调用链应是：

.. code-block:: text

    finance agent runtime
        → tokenizer.apply_chat_template(system + user)
        → SchemaFSM(system prompt)
        → finspec_forward(inputs, output_memory, schema_fsm, ...)
        → 完成输出后，把 hidden state 与 output token 写回 output_memory
        → 后续请求才可能检索到这条历史

三种草稿来源
============

代码并不是在所有位置机械执行同一个 ``Schema → 历史 → TR`` 优先队列，而是受
SchemaFSM 状态控制：

* ``init`` / ``first_param``：直接使用 Schema 枚举工具名、固定 JSON 和首参数名；
* ``remaining_param``：先尝试历史 continuation；若历史没有命中且刚到字段边界，
  再使用 Schema 补参数名；两者都不可用时才使用 Token Recycling。

三种来源最终都会变成同一种“候选 token 树”，目标模型的验证逻辑并不知道草稿来自
哪里。草稿只负责猜，不会绕过目标模型；最终只接受从根开始、与目标模型当前贪心预测
连续一致的最长前缀。

贯穿示例：API-Bank 第一条真实数据
=================================

示例来自 ``data/level-1-api_processed.json`` 第 0 条。用户要求把预约修改为：

.. code-block:: python

    {
        "name": "ModifyRegistration",
        "parameters": {
            "appointment_id": "34567890",
            "new_appointment_date": "2023-03-26",
            "new_appointment_doctor": "Dr. Lee",
        },
    }

这一条数据在本文件中的概念流动如下：

.. code-block:: text

    system + user
        → chat template
        → inputs.input_ids，形状 [1, P]
        → 前 P-1 个 token 做 prefill，最后 1 个 token 留作首轮树根
        → Schema 草稿枚举 QueryHealthData / CancelRegistration /
          ModifyRegistration 等工具分支
        → 目标模型验证并选择与 ModifyRegistration 连续一致的分支
        → Schema 草稿枚举 appointment_id 等首字段
        → 参数值阶段尝试历史后缀；无候选时使用 Token Recycling
        → 每轮只接受目标模型重新验证后连续匹配的最长路径
        → 最终得到完整、仍由目标模型决定的工具调用

如果调用方第一次传入 ``output_memory=[]``，当前请求无法检索历史；它完成后，未来
Agent runtime 可以把返回的 hidden state 与已接受输出保存成一条记忆，供后续相似
请求检索。核心解码器只消费这份记忆，不规定其存储、持久化或租户隔离方式。

注释中的 ``ROOT``、``t1`` 等是为了展示张量变化而使用的符号 token。真实 token id、
一个英文单词会被拆成几个 token，以及实际接受长度，都取决于所用 Qwen/Llama
tokenizer 和目标模型，不能在源码里写死。
"""

from transformers import StoppingCriteriaList, MaxLengthCriteria

from finspec.kv_cache import initialize_past_key_values
from finspec.tree_template import choose_tree_template

import torch
import torch.nn.functional as F


def _normalize_token_ids(token_ids):
    """把 tokenizer/model 中不同形态的停止 token 配置统一成 ``list[int]``。"""
    if token_ids is None:
        return []
    if isinstance(token_ids, (list, tuple)):
        return [int(x) for x in token_ids]
    return [int(token_ids)]


def _first_token_index(tokens_1d: torch.Tensor, token_ids_1d: torch.Tensor):
    """返回任意目标 token 在一维序列中第一次出现的位置；没有匹配时返回 ``None``。"""
    if tokens_1d.numel() == 0 or token_ids_1d.numel() == 0:
        return None
    pos = torch.nonzero(torch.isin(tokens_1d, token_ids_1d), as_tuple=False)
    if pos.numel() == 0:
        return None
    return int(pos[0].item())


def _truncate_generated_suffix(
    verify_input_ids: torch.Tensor,
    start_len: int,
    eos_token_ids: torch.Tensor,
    header_token_ids: torch.Tensor,
    accept_length_list,
):
    """裁掉 EOS 或下一轮 assistant header 之后被一并接受的多余 token。

    推测解码一次可能接受多个 token，因此胜出分支可能在同一轮跨过停止标记。
    EOS 自身会保留，assistant header 及其后内容会删除；同时修正最后一轮的接受长度，
    避免性能统计把停止标记之后的 token 也算成有效接受。

    例如某次树验证一口气接受了：

    ``... "Dr. Lee"}} → EOS → 无效尾巴``

    那么返回值保留 EOS、删除“无效尾巴”。如果先遇到的是下一轮 assistant header，
    则 header 本身也会删除，因为它不属于当前工具调用答案。普通逐 token decode
    通常在生成停止 token 后立刻退出；树解码需要这个额外裁剪，是因为一次可能接受
    跨过停止边界的多个 token。

    参数:
        verify_input_ids: 已验证的完整序列，形状 ``[1, total_length]``。
        start_len: 原始 prompt 长度，用于定位生成区间。
        eos_token_ids/header_token_ids: 允许有多个候选 id 的停止标记。
        accept_length_list: 每轮接受长度，函数会原地修正最后一个元素。

    TODO(优化): 将停止条件抽象成可组合的 token 序列匹配器。目前 header 只按单个
    token 检测，不能覆盖被 tokenizer 拆成多个 token 的自定义协议标记。
    """
    # Keep EOS if it appears first; drop assistant header token and anything after it.
    gen_tokens = verify_input_ids[:, start_len:]
    first_eos_idx = _first_token_index(gen_tokens[0], eos_token_ids)
    first_header_idx = _first_token_index(gen_tokens[0], header_token_ids)

    cut_generated_len = None
    if first_eos_idx is not None and (first_header_idx is None or first_eos_idx <= first_header_idx):
        cut_generated_len = first_eos_idx + 1
    elif first_header_idx is not None:
        cut_generated_len = first_header_idx

    if cut_generated_len is None:
        return verify_input_ids

    original_len = verify_input_ids.size(1)
    verify_input_ids = verify_input_ids[:, : start_len + cut_generated_len]
    truncated = original_len - verify_input_ids.size(1)
    if truncated > 0 and len(accept_length_list) > 0:
        accept_length_list[-1] = max(1, accept_length_list[-1] - truncated)
    return verify_input_ids


@torch.no_grad()
def prepare_data(template, input_ids):
    """把逻辑候选路径编译为 Tree Attention 所需的结构张量。

    ``template`` 中的每个元素是一条“top-k 排名路径”，例如 ``[0, 1]`` 表示先选择
    根节点排名第 0 的 token，再选择该节点下排名第 1 的 token。这里额外加入编号为
    0 的虚拟根节点，因此返回张量的节点数均为 ``len(template) + 1``。

    先用一个不依赖 tokenizer 的小例子理解。假设有两条草稿：

    .. code-block:: text

        候选 A = [a1, a2]
        候选 B = [b1]

        逻辑 template = [[0], [0, 0], [1]]

        扁平节点：
          node 0 = ROOT（当前已经确认的根 token）
          node 1 = a1
          node 2 = a2
          node 3 = b1

    ``prepare_data`` 不知道 a1/a2/b1 的真实 token id，只编译拓扑。对应的关键张量
    可读形式为：

    .. code-block:: text

        position_ids = [0, 1, 2, 1]

        attention_mask =
          ROOT [1, 0, 0, 0]
          a1   [1, 1, 0, 0]   # 看 ROOT 和自己
          a2   [1, 1, 1, 0]   # 看 ROOT、a1 和自己
          b1   [1, 0, 0, 1]   # 不能偷看候选 A

        search_path =
          [0, 1, 2]           # ROOT → a1 → a2
          [0, 3, -1]          # ROOT → b1；-1 是补齐占位

    Tree Attention 的核心就在这里：GPU 可以一次计算全部节点，但每个节点只能看到
    自己分支上的祖先，因此 a2 的预测等价于“在 ROOT→a1 上继续”，不会受到 b1 污染。

    返回值:
        attention_mask: ``[batch, node_count, node_count]``。每个节点只能看到虚拟根、
            自己及其祖先，不能看到兄弟分支。
        position_ids: ``[batch, node_count]``，同一深度的树节点共享位置编号。
        search_path: 补齐后的根到叶路径，``-1`` 表示 padding；后续用它并行比较
            各候选路径与目标模型预测。
        father_index/candi_index: Token Recycling 根据“父 token × top-k + 子排名”
            查询邻接表时使用的索引。
        deep_split: 各树深度在扁平节点数组中的分界位置。

    TODO(优化): 当前仍有 Python 路径循环并硬编码 ``cuda:0``。可将常用模板预编译，
    或改为完全基于 ``input_ids.device`` 的向量化构造，以支持多卡和降低 CPU 开销。
    """
    candi_attention_ids = torch.zeros([input_ids.size(0), len(template)+1, len(template)+1], dtype=torch.long)
    candi_positon_ids = torch.zeros([input_ids.size(0), len(template)+1], dtype=torch.long)

    candi_attention_ids[:, :, 0] = 1
    candi_positon_ids[:, 0] = 0
    search_path = []
    search_path_set = set()
    father_index = [-1]
    candi_index = [-1]
    deep_split = []
    
    # 用路径到节点编号的映射做 O(1) 父节点查询，避免循环中反复 template.index()。
    template_to_idx = {tuple(path): idx + 1 for idx, path in enumerate(template)}
    
    for candi_i in range(len(template)):
        tree_deep = len(template[candi_i])
        tree_index = template[candi_i][-1]

        if tree_deep == 1:
            father_index.append(0)
        else:
            parent_path = tuple(template[candi_i][:-1])
            father_index.append(template_to_idx.get(parent_path, 0))
        candi_index.append(tree_index)

        candi_positon_ids[:, candi_i+1] = tree_deep

        if tree_deep != candi_positon_ids[:, candi_i]:
            deep_split.append(candi_i+1)

        cur_path = [0]
        candi_attention_ids[:, candi_i+1, candi_i+1] = 1

        for candi_j in range(candi_i):
            if template[candi_j] == template[candi_i][:len(template[candi_j])]:
                candi_attention_ids[:, candi_i+1, candi_j+1] = 1
                cur_path.append(candi_j+1)
        cur_path.append(candi_i+1)

        # 如果旧叶子现在成为新节点的祖先，它就不再是一条完整根到叶验证路径。
        # 使用集合增量维护叶路径，避免每轮重新构建所有 path。
        to_remove = []
        for sub_i in range(1, len(cur_path)):
            sub_path = tuple(cur_path[:sub_i])
            if sub_path in search_path_set:
                to_remove.append(sub_path)
        if to_remove:
            remove_set = set(to_remove)
            search_path = [p for p in search_path if tuple(p) not in remove_set]
            search_path_set.difference_update(remove_set)
        search_path.append(cur_path)
        search_path_set.add(tuple(cur_path))

    deep_split.append(len(template)+1)

    # GPU 张量要求矩形形状，因此把根到叶路径补齐到相同长度；-1 仅作占位。
    pad_len = max(len(path) for path in search_path)
    search_path = torch.tensor(
        [path + [-1] * (pad_len - len(path)) for path in search_path],
        dtype=torch.long,
        device=torch.device('cuda:0'),
    )
    father_index = torch.tensor(father_index, dtype=torch.long, device=torch.device('cuda:0'))
    candi_index = torch.tensor(candi_index, dtype=torch.long, device=torch.device('cuda:0'))
    
    return candi_attention_ids.cuda(), candi_positon_ids.cuda(), search_path, father_index, candi_index, deep_split

@torch.no_grad()
def create_template_from_candidates(candidate_pred_tokens):
    """把若干候选 token 序列转换为 ``prepare_data`` 可接受的逻辑路径。

    每个非空候选占用一个独立根分支。例如长度为 3 的第 2 个候选会产生
    ``[2]``、``[2, 0]``、``[2, 0, 0]`` 三个节点。空候选不会进入树；全部为空时
    返回最小模板 ``[[0]]``，保证后续张量构造仍然有效。

    具体例子：

    .. code-block:: text

        candidate_pred_tokens[0] = [A, B, C]
        candidate_pred_tokens[1] = [X, Y]

        new_template =
          [0]          # 候选 0 的 A
          [0, 0]       # 候选 0 的 B
          [0, 0, 0]    # 候选 0 的 C
          [1]          # 候选 1 的 X
          [1, 0]       # 候选 1 的 Y

    路径里的 ``0/1`` 在这里首先标识“第几条候选”，后续的 0 只用于延长这一条线性
    分支；它们不是 A/B/C/X/Y 的 token id。

    参数:
        candidate_pred_tokens: ``list[Tensor]``，每个一维张量是一条草稿序列。

    重要限制：这里并不是压缩后的 token trie。即便多条候选具有相同 token 前缀，
    它们仍然会创建重复节点。

    TODO(优化): 合并相同 token 前缀，构造真正的共享前缀 trie，可减少树节点数、
    Tree Attention 计算量和 KV Cache 临时写入量。
    """
    new_template = []
    
    # 每个候选的列表下标就是其逻辑根分支编号。
    for root_idx in range(len(candidate_pred_tokens)):
        seq = candidate_pred_tokens[root_idx]
        if seq.numel() == 0:
            # Skip empty candidates
            continue
        
        # 候选 token 数量等于这条链需要创建的树深度。
        candidate_depth = seq.size(0)
        
        # 为该候选依次创建从根到每一层的路径。
        for depth in range(1, candidate_depth + 1):
            if depth == 1:
                # Root node: [root_idx]
                path = [root_idx]
            else:
                # Deeper nodes: [root_idx, 0, 0, ..., 0] with (depth-1) zeros
                path = [root_idx] + [0] * (depth - 1)
            new_template.append(path)
    
    # 保底返回一个单节点分支，防止后续 max()/tensor 构造面对空列表。
    if not new_template:
        return [[0]]
    
    return new_template


@torch.no_grad()
def prepare_data_input_ids(cur_input_ids, adj_matrix, father_index, candi_index, deep_split, output_id_topk):
    """使用在线 Token Recycling 邻接表逐层填充静态回退树。

    Token Recycling 不是第二个模型。目标模型完成一次前向后，本来就已经为每个树
    节点算出了下一个 token 的 logits；``finspec_forward`` 取 top-k，并写入：

    ``adj_matrix[current_token_id] = [next_token_top1, ..., next_token_topk]``

    这里保存的是目标模型最近一次为该 token 计算出的 top-k 后继，不是出现次数统计；
    同一个 token 再次出现时对应行会被新结果覆盖。键中没有 prompt、工具名、字段名
    或该 token 所在位置，所以这是一种主动牺牲上下文以换取 O(1) 查询的近似。它并不
    比 hidden state “更底层、更懂语义”，恰恰比完整上下文表示粗糙得多。

    回退时，静态树中的路径元素表示 top-k 排名而非 token id。例如逻辑路径
    ``[0, 1]`` 表示：

    1. 从当前根 token 的邻接行选择排名 0 的后继；
    2. 再从这个后继 token 的邻接行选择排名 1 的后继。

    这样可以把此前已经计算过的多条高概率 continuation “回收”为一棵候选树。某一层
    的真实 token 依赖父层查询结果，因此必须按照 ``deep_split`` 从浅到深填充。树
    填好后仍会交给同一个目标模型并行验证，并不直接成为最终输出。

    一个简化的邻接表示例：

    .. code-block:: text

        adj_matrix[ROOT] = [A, X, ...]
        adj_matrix[A]    = [B, Y, ...]

        静态逻辑路径：
          [0]     → 从 ROOT 取排名 0，得到 A
          [1]     → 从 ROOT 取排名 1，得到 X
          [0, 0]  → 先得到 A，再从 A 取排名 0，得到 B
          [0, 1]  → 先得到 A，再从 A 取排名 1，得到 Y

        填充后的扁平树：
          [ROOT, A, X, B, Y]

    第一层必须先算出 A/X，第二层才能用 A 查询 B/Y；这就是按 ``deep_split`` 分层
    循环而不能一次性随意填满所有节点的原因。

    与普通自回归生成的区别是：普通生成每次只采用当前 top-1 token；Token Recycling
    会先用旧 top-k 结果展开多条未来路径，希望一次前向接受多个 token。若旧预测不再
    适合当前上下文，验证会在首次不匹配处停止。

    TODO(优化): 当前邻接表按完整词表分配且以单 token 为状态。可以评估稀疏存储、
    频次/时效加权，以及把工具 id、字段状态或轻量上下文哈希加入键，避免不同工具
    协议互相污染。更丰富的键会提升候选质量，但也会增加内存和查询复杂度。
    """
    # 节点 0 是当前已经确认的根 token，后续节点将按树深度依次填入。
    input_ids = torch.zeros_like(father_index, dtype=torch.long, device=cur_input_ids.device)
    input_ids[0] = cur_input_ids
    for layer in range(len(deep_split)-1):
        # 先取得这一层每个节点的父 token id。
        cur_father = input_ids[father_index[deep_split[layer]:deep_split[layer+1]]]
        # 展平后的邻接表索引等于：
        #   father_token_id * top_k + 当前逻辑路径要求的候选排名。
        cur_father_index = cur_father * output_id_topk + candi_index[deep_split[layer]:deep_split[layer+1]]
        # 查出真实后继 token，写入本层；下一层会继续把它当作 father 查询。
        input_ids[deep_split[layer]:deep_split[layer+1]] = adj_matrix.view(-1)[cur_father_index]
    return input_ids.unsqueeze(0)


@torch.no_grad()
def build_retrieval_tree_from_candidates(candidate_pred_tokens, input_ids):
    """把候选草稿填入树节点，并返回一次并行验证所需的全部输入。

    该函数分两层工作：

    1. ``create_template_from_candidates`` 只决定树的形状；
    2. 本函数把每条候选的第 ``depth`` 个真实 token 写入对应节点。

    延续 ``[A,B,C]`` 与 ``[X,Y]`` 的例子：

    .. code-block:: text

        retrieval_input_ids[0] = ROOT
        retrieval_input_ids[1] = A
        retrieval_input_ids[2] = B
        retrieval_input_ids[3] = C
        retrieval_input_ids[4] = X
        retrieval_input_ids[5] = Y

    返回的 ``input_ids`` 因而是 ``[[ROOT,A,B,C,X,Y]]``。它看起来是一个扁平序列，
    但不能按普通文本顺序理解；``attention_mask`` 和 ``search_path`` 会把它重新解释
    成 ``ROOT→A→B→C`` 与 ``ROOT→X→Y`` 两条互不干扰的分支。

    相同形状的候选（例如常见的 32/16/8/8）会复用 attention mask、position ids
    和路径索引；token 值则每轮重新填充。

    参数:
        candidate_pred_tokens: 每条候选的一维 token 张量。
        input_ids: 当前根 token，主要用于确定 device 和返回张量类型。

    返回:
        ``(tree_input_ids, tree_template, attention_mask, position_ids, search_path,
        father_index, candi_index, deep_split)``。

    TODO(优化): 当前缓存上限为 64，淘汰的是最早插入项而非真正 LRU。可增加命中率
    指标并使用有界 LRU；若实现共享前缀 trie，缓存键也需要包含压缩后的拓扑。
    """
    # Create a new template based on candidate_pred_tokens
    new_template = create_template_from_candidates(candidate_pred_tokens)
    template_key = tuple(tuple(path) for path in new_template)

    # 缓存只与树形状相关的张量；历史候选长度经常重复，因此可避免重复编译。
    cache = getattr(build_retrieval_tree_from_candidates, "_template_cache", None)
    if cache is None:
        cache = {}
        build_retrieval_tree_from_candidates._template_cache = cache

    cached = cache.get(template_key)
    if cached is not None and cached[0].device == input_ids.device:
        attention_mask, position_ids, search_path, father_index, candi_index, deep_split = cached
    else:
        # 首次遇到这种候选长度组合时编译树结构。
        attention_mask, position_ids, search_path, father_index, candi_index, deep_split = prepare_data(
            new_template, input_ids
        )
        if len(cache) >= 64:
            cache.pop(next(iter(cache)))
        cache[template_key] = (
            attention_mask,
            position_ids,
            search_path,
            father_index,
            candi_index,
            deep_split,
        )

    # 扁平数组下标与 prepare_data 的节点编号一致；0 是虚拟根占位。
    retrieval_input_ids = torch.zeros_like(father_index, dtype=torch.long, device=input_ids.device)
    retrieval_input_ids[0] = input_ids[0, 0]

    # path[0] 决定候选编号，len(path)-1 决定取该候选中的哪个 token。
    for node_idx, path in enumerate(new_template, start=1):
        if not path:
            continue
        root_idx = path[0]
        if root_idx < 0 or root_idx >= len(candidate_pred_tokens):
            continue
        seq = candidate_pred_tokens[root_idx]
        if seq.numel() == 0:
            continue
        depth = len(path)
        if depth <= seq.size(0):
            retrieval_input_ids[node_idx] = seq[depth - 1]

    input_ids = retrieval_input_ids.unsqueeze(0)
    tree_template = new_template  # Use new template for consistency
    
    return input_ids, tree_template, attention_mask, position_ids, search_path, father_index, candi_index, deep_split


@torch.no_grad()
def find_candidate_pred_tokens(
    input_ids,
    similar_outputs=None,
    retrieved_context=None,
    max_ngram_size=7,
    min_ngram_size=5,
    target_lengths=(32, 16, 8, 8), # 64 nodes
):
    """用精确 7/6/5-token 后缀匹配，从历史输出中截取 continuation 草稿。

    语义检索只负责先缩小历史范围，本函数还会执行更严格的 token 级检查：

    1. 取当前完整序列末尾 7 个 token，在“检索历史 + 当前序列”中滑窗搜索；
    2. 找不到足够候选时依次缩短为 6、5 个 token；
    3. 对每个匹配位置，截取预设的 32/16/8/8 个后续 token；
    4. 删除完全重复或互为前缀的候选，最后用空张量补齐固定分支数。

    因此，“hidden state 相似”并不会直接让历史输出被接受；只有当前 token 后缀也
    精确出现过，历史 continuation 才会成为草稿，并且仍需目标模型再次验证。

    用预约示例理解两级检索。假设后续某个相似请求开始生成：

    .. code-block:: text

        ... {"name": "ModifyRegistration",
            "parameters": {"appointment_id": "
                                                   ↑ 当前生成到这里

    语义检索已经把第 0 条历史调用放入 ``retrieved_context``。若当前结尾 7 个 token
    也精确出现在该历史输出中，本函数就可以取其后面的 token，形成可读草稿：

    .. code-block:: text

        34567890", "new_appointment_date": "2023-03-26", ...

    这并不意味着当前请求一定使用预约号 34567890。若当前上下文实际要求 99990000，
    目标模型会在第一个不同的 token 上拒绝旧值，并改走自己当前预测的 token。历史
    continuation 的作用是“猜中时批量前进”，不是把旧数据库值复制为事实。

    ``search_context`` 还会在历史之后拼接当前 ``input_ids``，所以即使
    ``output_memory`` 为空，也可能复用当前 prompt 或本次输出前面已经出现过的精确
    片段；这属于同请求的 prefix/substring recycling，不是语义历史检索命中。

    参数:
        input_ids: 当前已确认序列，形状 ``[1, current_length]``。
        similar_outputs: 兼容旧调用方式的历史记录列表。
        retrieved_context: 已在 GPU 上拼接的 top-k 历史输出。
        max_ngram_size/min_ngram_size: 后缀匹配长度范围。
        target_lengths: 每个候选希望截取的最大 continuation 长度。

    TODO(优化):
        - 保留每条历史记录的边界，避免拼接处产生并不存在的跨记录 n-gram；
        - 用 n-gram 倒排索引、suffix array/automaton 代替每轮 ``unfold`` 全量扫描；
        - 联合语义相似度、后缀长度、调用成功率和历史接受率对候选排序；
        - 明确区分历史区间和当前输入区间。历史被前置后，``match_indices`` 是相对于
          ``search_context`` 的绝对位置，而当前 self-match 边界却用 ``input_length``
          判断；应显式记录每条历史的起止偏移，避免长历史下边界语义含混。
    """
    input_length = input_ids.size(1)
    
    # 优先使用调用方已搬到当前 device 的检索上下文，避免在每轮解码重复拼接。
    if retrieved_context is not None and retrieved_context.numel() > 0:
        search_context = torch.cat([retrieved_context, input_ids], dim=1)
    else:
        search_context = input_ids
    if retrieved_context is None and similar_outputs:
        retrieved = []
        for entry in similar_outputs:
            output_ids = entry.get("output_ids") if isinstance(entry, dict) else entry
            if not output_ids:
                continue
            # 兼容 output_memory 中以 Python list 保存的 token ids。
            if isinstance(output_ids, list):
                retrieved.append(torch.tensor(output_ids, device=input_ids.device).unsqueeze(0))
            else:
                retrieved.append(output_ids.unsqueeze(0) if output_ids.dim() == 1 else output_ids)
        if retrieved:
            search_context = torch.cat(retrieved + [search_context], dim=1)

    if max_ngram_size <= 0 or max_ngram_size > input_length:
        raise ValueError("Invalid max_ngram_size")

    candidates = []
    seen_tensors = []  # 保持为 Tensor，避免频繁搬回 CPU 转 tuple。
    for ngram_size in range(max_ngram_size, min_ngram_size - 1, -1):
        if len(candidates) == len(target_lengths):
            break

        # 当前序列最后 n 个 token 就是需要在历史中查找的精确后缀。
        ngram_tensor = input_ids[0, -ngram_size:].unsqueeze(0)

        # unfold 得到所有长度为 n 的滑动窗口：[1, window_count, n]。
        windows = search_context.unfold(dimension=1, size=ngram_size, step=1)

        matches = (windows == ngram_tensor).all(dim=2)
        match_indices = matches.nonzero(as_tuple=True)[1]

        for idx in match_indices:
            if len(candidates) == len(target_lengths):
                break
            start_idx = idx + ngram_size
            end_idx = start_idx + target_lengths[len(candidates)]
            # 匹配点之后必须有足够长 continuation，并排除当前尾部与自身的平凡匹配。
            search_context_length = search_context.size(1)
            if end_idx <= search_context_length and start_idx < input_length - ngram_size:
                candidate_seq = search_context[0, start_idx:end_idx]
                # 删除完全重复和互为前缀的分支，避免浪费验证树节点。
                seq_len = candidate_seq.size(0)
                is_duplicate = False
                # Check for exact duplicates and prefixes using tensor operations
                for existing_tensor in seen_tensors:
                    existing_len = existing_tensor.size(0)
                    if existing_len == seq_len:
                        if (candidate_seq == existing_tensor).all():
                            is_duplicate = True
                            break
                    elif existing_len > seq_len:
                        # Check if candidate is prefix of existing
                        if (candidate_seq == existing_tensor[:seq_len]).all():
                            is_duplicate = True
                            break
                    # Check if existing is prefix of candidate (candidate longer)
                    elif seq_len > existing_len:
                        if (existing_tensor == candidate_seq[:existing_len]).all():
                            is_duplicate = True
                            break
                
                if is_duplicate:
                    continue
                
                # candidate_seq 是大张量的 view；clone 后保存，避免底层存储变化。
                seen_tensors.append(candidate_seq.clone())
                candidates.append(candidate_seq)

    # 固定候选数量可简化后续树构造；空张量代表该分支没有草稿。
    while len(candidates) < len(target_lengths):
        candidates.append(torch.tensor([], dtype=torch.long, device=input_ids.device))

    return candidates


@torch.no_grad()
def get_topk_similar_outputs(question_hidden_state, output_memory, top_k=3):
    """按目标模型 hidden state 的余弦相似度返回 top-k 历史调用。

    ``question_hidden_state`` 取自 prefill 段最后一个 token 的最后一层表示，形状
    通常为 ``[1, 1, hidden_dim]``。注意当前实现把原始 prompt 的最后一个 token 留作
    首轮树根，因此这里严格说是 ``verify_input_ids`` 的末 token，而不是完整
    ``inputs.input_ids`` 的末 token。因果模型在这个位置已经看过它左侧的 system/user
    上下文；若被留出的根只是 chat template 的结构性结尾，这个表示仍可作为低成本
    请求表示，但应通过实验确认与论文定义是否完全对齐。

    这里不使用额外 embedding 模型或向量数据库：目标模型的表示在 prefill 时已经
    产生，API-Bank 历史规模也足以用一次矩阵余弦相似度暴力搜索。但这不等于末 token
    表示必然优于专用 embedding，只是当前实现延迟低、无需训练且与目标模型语义空间
    对齐；是否更好仍需通过检索命中率和最终接受长度实验验证。

    返回记录包含 ``similarity``、``question_id`` 和 ``output_ids``；hidden state
    只用于选历史记录，不会直接成为生成 token。

    具体数据结构示例（相似度数字仅用于说明，不是仓库预先计算的结果）：

    .. code-block:: python

        output_memory = [
            {
                "question_id": 0,
                "question_hidden_state": h_modify,  # [1, 1, hidden_dim]
                "output_ids": ids_of_modify_registration,
            },
            {
                "question_id": 1,
                "question_hidden_state": h_query_health,
                "output_ids": ids_of_query_health_data,
            },
        ]

        # 当前也是“修改预约”请求，假设：
        cosine(current, h_modify)       = 0.92
        cosine(current, h_query_health) = 0.61

        # 返回顺序：
        [
            {"similarity": 0.92, "question_id": 0, ...},
            {"similarity": 0.61, "question_id": 1, ...},
        ]

    调用方第一次传入 ``output_memory=[]`` 时，这里直接返回空列表。等该请求完整
    结束、Agent runtime 写入 memory 后，后续请求才可能得到上面这样的结果。

    TODO(优化): 缓存归一化后的连续历史矩阵，增加最低相似度和工具类型过滤；对比
    last-token、mean pooling、加权 pooling 与专用 embedding，并在历史规模增长后
    再评估 FAISS/向量数据库等近似最近邻索引；同时比较“当前 prefill 末 token”
    与“包含保留根 token 的完整 prompt 表示”，确认检索向量的边界对齐。
    """
    if question_hidden_state is None or not output_memory:
        return []

    # 展平 [1, 1, hidden_dim] 并归一化；归一化后点积即余弦相似度。
    q_vec = question_hidden_state.to(torch.float32).view(question_hidden_state.shape[0], -1)
    q_vec = F.normalize(q_vec, dim=-1)
    
    # 过滤不完整记录，并收集为一个批次，避免逐条调用 cosine_similarity。
    valid_memories = []
    memory_states_list = []
    for memory in output_memory:
        memory_state = memory.get("question_hidden_state")
        memory_output = memory.get("output_ids")
        memory_qid = memory.get("question_id")
        if memory_state is None or memory_output is None:
            continue
        if isinstance(memory_state, torch.Tensor):
            mem_vec = memory_state.to(torch.float32).view(memory_state.shape[0], -1)
        else:
            mem_vec = torch.tensor(memory_state, dtype=torch.float32).view(1, -1)
        valid_memories.append((memory_qid, memory_output))
        memory_states_list.append(mem_vec)
    
    if not memory_states_list:
        return []

    # 所有向量必须来自同一目标模型，hidden_dim 才能一致。
    memory_states_tensor = torch.cat(memory_states_list, dim=0)
    memory_states_tensor = F.normalize(memory_states_tensor, dim=-1)
    
    # q_vec: [1, hidden_dim]；memory_states_tensor: [memory_count, hidden_dim]。
    # 广播逐元素相乘再求和，得到每条历史记录的余弦相似度。
    similarities_tensor = torch.sum(q_vec * memory_states_tensor, dim=-1)
    similarities = [(sim.item(), valid_memories[i][0], valid_memories[i][1]) 
                    for i, sim in enumerate(similarities_tensor)]
    
    if not similarities:
        return []

    # 当前为精确全排序；历史很大时可以改 topk 或 ANN，避免 O(N log N)。
    similarities.sort(key=lambda x: x[0], reverse=True)
    topk = similarities[:top_k]
    return [
        {"similarity": sim, "question_id": qid, "output_ids": output_ids}
        for sim, qid, output_ids in topk
    ]


@torch.no_grad()
def finspec_forward(
    inputs,
    output_memory,
    schema_fsm,
    model,
    tokenizer,
    max_new_tokens,
    output_id_topk=8,
    adj_matrix=None,
):
    """使用 Schema、历史检索和 Token Recycling 为一个请求生成完整响应。

    这是理解 FinSpec 解码核心的主入口，整体生命周期如下：

    1. **Prefill**：把除最后一个根 token 外的 prompt 写入可修改 KV Cache，并取得
       prefill 段末 token 的 hidden state；
    2. **一次性语义检索**：从 ``output_memory`` 选 top-k 相似历史调用；
    3. **逐轮选择草稿**：根据 SchemaFSM 状态选择 Schema、历史后缀或 Token
       Recycling；
    4. **树并行验证**：把候选包装为树，使用 4D mask 做一次目标模型前向；
    5. **接受最长前缀**：只接受从根开始连续等于目标模型 argmax 的 token；
    6. **压缩缓存**：将胜出分支 KV 搬到连续位置，丢弃其他分支；
    7. 使用第一个拒绝位置上的目标模型预测作为下一轮根 token。

    用预约示例给变量赋予直观含义：

    .. code-block:: text

        inputs.input_ids
          = [system 工具文档, user 对话, assistant generation header]
          = 形状 [1, P]

        verify_input_ids
          = 前 P-1 个 token，已经进入 KV Cache 的权威前缀

        input_ids
          = 最后 1 个 token，首轮候选树的 ROOT

        第一次循环：
          SchemaFSM(init)
            → 4 种协议前缀 × 3 个工具名 = 12 条草稿
            → build_retrieval_tree_from_candidates
            → 目标模型并行验证
            → 理想情况下连续接受 `... "ModifyRegistration`

        第二次循环：
          SchemaFSM(first_param)
            → 为 ModifyRegistration 枚举 3 个首参数名
            → 目标模型选择 `appointment_id`

        后续循环：
          SchemaFSM(remaining_param)
            → 尝试历史精确后缀 continuation
            → 字段边界且历史无命中时，Schema 补下一个参数名
            → 两者都没有时，Token Recycling 提出短草稿

    上述“理想情况下”很重要：实际每轮能接受多少 token 由目标模型 logits 决定，
    Schema 或历史候选都没有强制输出权。

    关键参数:
        inputs: tokenizer 返回的 BatchEncoding，目前实现要求 batch size 为 1。
        output_memory: 此进程内此前完成的请求，保存 hidden state 与输出 token。
        schema_fsm: 根据当前 system prompt 构造的 SchemaFSM，每个请求独立创建。
        adj_matrix: Token Recycling 的 ``[vocab_size, output_id_topk]`` 在线邻接表。

    返回:
        ``(完整 token 序列, 新生成 token 数, 解码轮数, 每轮接受长度,
        prefill 段末 token hidden state)``。

    当前验证器是贪心 ``argmax``。若支持 temperature/top-p，不能继续用简单 token
    相等判断，必须实现推测采样的接受概率和拒绝后的重采样。

    TODO(优化): 将三个草稿来源抽象为统一 provider 接口并记录每轮来源、候选数、
    接受长度和拒绝位置；进一步根据近期命中率/熵动态分配树节点预算。把 FSM 状态
    推进改成由“实际接受的结构 token”驱动，而不是在构造一次草稿后立即推进。
    """
    # 这里的 P 是 chat template 编码后的总 token 数，不是用户消息的字符数。
    # 对贯穿示例而言，P 中同时包含 system 工具文档、完整多轮 user/response 历史和
    # assistant generation header。
    input_ids = inputs.input_ids.cuda()
    accept_length_list = []
    step = 0
    start_len = input_ids.size(1)

    # 数据从 [1, P] 拆成：
    #
    #   verify_input_ids: [1, P-1]，先 prefill 并写入 KV Cache；
    #   input_ids:        [1, 1]，保留为首轮候选树的 ROOT。
    #
    # 不能把最后一个 token 既 prefill 又作为树根输入，否则它会在 KV Cache 中重复。
    # 后续每轮也遵守同一约定：verify_input_ids 是已经确认且有连续 KV 的前缀，
    # input_ids 是尚待本轮树前向处理的根和候选节点。
    verify_input_ids = input_ids[:, :-1]
    input_ids = input_ids[:, -1:]
    
    # general_template 只规定 Token Recycling 回退树的拓扑；Schema/历史草稿会根据
    # 每轮候选长度动态创建树。额外的 64 个位置为动态候选和停止边界预留空间。
    question_hidden_state = None
    question_length = verify_input_ids.size(1)
    general_template = choose_tree_template("general")
    max_cache_len = start_len + max_new_tokens + len(general_template) + 64
    (
        past_key_values,
        past_key_values_data,
        current_length_data,
    ) = initialize_past_key_values(model, max_cache_len=max_cache_len)
    stopping_criteria = StoppingCriteriaList(
        [MaxLengthCriteria(max_length=start_len + max_new_tokens)]
    )

    # Prefill 只有这一次：模型把 P-1 个权威 prompt token 的 K/V 写入可变缓存。
    # output_hidden_states=True 顺便得到检索向量，无需另跑 embedding 模型。
    initial_outputs = model(
        input_ids=verify_input_ids,
        past_key_values=past_key_values,
        output_hidden_states=True,
    )
    if initial_outputs.hidden_states is not None:
        # 取最后一层、prefill 段最后一个位置：
        #
        #   hidden_states[-1]                         → [1, P-1, hidden_dim]
        #   [:, question_length-1:question_length, :] → [1, 1, hidden_dim]
        #
        # 因果注意力使该位置聚合其左侧上下文。严格来说，它不包含被留作 ROOT 的原始
        # prompt 最后 token；这与“完整 prompt 末 token”有一位差异，后续应做消融验证。
        # detach().cpu() 是因为调用方可能跨请求保存它，不需要梯度，也不希望历史向量
        # 长期占用 GPU 显存。
        question_hidden_state = initial_outputs.hidden_states[-1][:, question_length - 1:question_length, :].detach().cpu()
    
    eos_token_values = []
    eos_token_values.extend(_normalize_token_ids(getattr(tokenizer, "eos_token_id", None)))
    eos_token_values.extend(_normalize_token_ids(getattr(model.generation_config, "eos_token_id", None)))
    eos_token_ids = torch.tensor(
        sorted(set(eos_token_values)),
        dtype=torch.long,
        device=input_ids.device,
    )
    start_header_id = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    header_token_values = [int(start_header_id)] if start_header_id is not None and start_header_id >= 0 else []
    header_token_ids = torch.tensor(
        header_token_values,
        dtype=torch.long,
        device=input_ids.device,
    )
    stop_token_ids = torch.tensor(
        sorted(set(eos_token_values + header_token_values)),
        dtype=torch.long,
        device=input_ids.device,
    )

    def has_stop(token_tensor: torch.Tensor) -> bool:
        if stop_token_ids.numel() == 0:
            return False
        return torch.isin(token_tensor, stop_token_ids).any().item()

    # 语义相似度只依赖当前请求的 prefill 表示，因此每个请求计算一次，而不是每轮
    # 解码都计算。第 0 条正式 API-Bank 数据进入这里时 output_memory=[]：
    #
    #   similar_outputs = []
    #   retrieved_context = None
    #
    # 第一次请求完成后，Agent runtime 才能保存 {question_id, hidden_state, output_ids}。若后续
    # 请求语义接近“修改预约”，它才可能检索到包含 34567890 / 2023-03-26 / Dr. Lee
    # 的历史输出。检索到只代表“允许拿来猜”，不代表这些旧值适用于当前用户。
    similar_outputs = []
    if question_hidden_state is not None and output_memory:
        similar_outputs = get_topk_similar_outputs(question_hidden_state, output_memory, top_k=10)
    retrieved_context = None
    if similar_outputs:
        retrieved_chunks = []
        for entry in similar_outputs:
            output_ids = entry.get("output_ids") if isinstance(entry, dict) else entry
            if not output_ids:
                continue
            if isinstance(output_ids, list):
                retrieved_chunks.append(torch.tensor(output_ids, device=input_ids.device).unsqueeze(0))
            else:
                retrieved_chunks.append(output_ids.unsqueeze(0) if output_ids.dim() == 1 else output_ids)
        if retrieved_chunks:
            # 从这里开始，hidden state 已经完成职责：它只决定选哪些历史记录。
            # 真正进入逐轮草稿匹配的是这些历史记录的 output token：
            #
            #   [history_0_output_ids | history_1_output_ids | ...]
            #
            # find_candidate_pred_tokens 会再要求当前末尾 7/6/5 token 精确匹配。
            # TODO(优化): 直接拼接会丢失记录边界，可能在两条历史交界处产生虚假的
            # n-gram；应保留独立序列或插入绝不会参与匹配的受保护分隔符。
            retrieved_context = torch.cat(retrieved_chunks, dim=1)

    # verify_tool_name 不是模型输出的字符串副本，而是已经完整出现在权威输出末尾的
    # 工具名 token tuple。它一旦识别为 ModifyRegistration，SchemaFSM 才能把首参数
    # 候选限制到 appointment_id/new_appointment_date/new_appointment_doctor。
    verify_tool_name = None
    # 通用回退树的拓扑与 token 值无关，一个请求内只需编译一次。
    general_tree_cache = None
    # 工具名预先放到当前 GPU，避免每轮为了比较而 cpu().tolist() 同步。
    tool_name_tensors = []
    for tool_name_tuple in schema_fsm.tokenized_schema.keys():
        tool_name_tensors.append(
            (
                len(tool_name_tuple),
                torch.tensor(tool_name_tuple, dtype=torch.long, device=input_ids.device),
                tool_name_tuple,
            )
        )
    for step in range(max_new_tokens):
        # 循环入口不变量：
        #
        #   verify_input_ids = 已被接受、KV 已压成连续布局的完整前缀；
        #   input_ids        = 当前 ROOT（第一轮是 prompt 最后 token，之后是目标模型
        #                      在上轮首个未覆盖位置给出的权威预测）。
        #
        # full_input_ids 暂时把二者拼起来，用于后缀检索和 Schema 字段边界判断；ROOT
        # 还没有写进连续 KV，真正写入发生在下面的树模型前向。
        full_input_ids = torch.cat([verify_input_ids, input_ids], dim=-1)
        # 只回看 4 个 token 是适配当前 JSON/tokenizer 的经验规则，并非通用语法状态。
        # TODO(优化): 让 FSM 消费每个已接受 token，准确跟踪嵌套 JSON 和字段边界。
        param_span = full_input_ids[0, -4:].tolist()

        if schema_fsm.state == "init" or schema_fsm.state == "first_param":
            candidate_pred_tokens = schema_fsm.find_candidate_pred_tokens(tool_name=verify_tool_name)
            # SchemaFSM 返回 Python token 列表；树构造统一接收当前 device 上的 Tensor。
            candidate_pred_tokens = [
                torch.tensor(tokens, dtype=torch.long, device=input_ids.device) 
                if tokens else torch.tensor([], dtype=torch.long, device=input_ids.device)
                for tokens in candidate_pred_tokens
            ]

            # 贯穿示例的 init 轮会得到 4 种 API-Bank 输出前缀 × 3 个工具名，共 12 条
            # token 草稿；目标模型应让 ModifyRegistration 分支得到最长连续匹配。
            #
            # first_param 轮在 verify_tool_name 已识别时得到 3 条候选，可读形式类似：
            #
            #   `", "parameters": {"appointment_id": "`
            #   `", "parameters": {"new_appointment_date": "`
            #   `", "parameters": {"new_appointment_doctor": "`
            #
            # 它们会被铺平成一棵树，而不是串行调用目标模型三次。
            input_ids, tree_template, attention_mask, position_ids, search_path, father_index, candi_index, deep_split = build_retrieval_tree_from_candidates(
                candidate_pred_tokens, input_ids
            )

            # 注意：当前实现是“草稿树一构造完就推进状态”，不是等确认整段工具名/参数名
            # 已被接受后再推进。它依赖结构草稿通常能一次接受较长前缀；若首轮很早拒绝，
            # FSM 状态可能领先于实际输出。
            # TODO(优化): 根据 best_path_input 中实际接受的语法事件推进 FSM，并允许在
            # 结构尚未完成时停留/重试当前状态。
            schema_fsm.state_transition()
        else:
            # remaining_param 阶段不是“Schema 永远第一”。代码先准备可能的字段名候选，
            # 随后仍优先检查历史 continuation；只有历史没有候选时才使用 schema_tokens。
            #
            # 例如已生成 `"appointment_id": "34567890",`，最近 4 个 token 中出现
            # separator_id，SchemaFSM 可提出 new_appointment_date 等字段名；若历史
            # 同时存在精确后缀 continuation，则历史长草稿会先进入验证树。
            schema_tokens = None
            if schema_fsm.seperator_id in param_span:
                schema_tokens = schema_fsm.find_candidate_pred_tokens(tool_name=verify_tool_name, param_span=param_span)

            candidate_pred_tokens = find_candidate_pred_tokens(
                full_input_ids,
                retrieved_context=retrieved_context,
            )
            non_empty_candidates = [c for c in candidate_pred_tokens if c.numel() > 0]
            #print("non empty candidates: \n", non_empty_candidates)

            if len(non_empty_candidates) == 0:
                if schema_tokens is not None:
                    candidate_pred_tokens = [
                        torch.tensor(tokens, dtype=torch.long, device=input_ids.device) 
                        if tokens else torch.tensor([], dtype=torch.long, device=input_ids.device)
                        for tokens in schema_tokens
                    ]
                    # 历史后缀无命中，但当前处在字段分隔处：改用剩余参数名草稿。
                    # 在预约示例中，这可以直接提出：
                    #   `"new_appointment_date": "`
                    # 或 `"new_appointment_doctor": "`
                    # Schema 只知道字段名，不会提出 2023-03-26 或 Dr. Lee。
                    input_ids, tree_template, attention_mask, position_ids, search_path, father_index, candi_index, deep_split = build_retrieval_tree_from_candidates(
                        candidate_pred_tokens, input_ids
                    )
                else:
                    # Schema 和历史后缀检索都无候选时，才回退到 Token Recycling。
                    # 它不会重新调用一个草稿模型，而是读取 adj_matrix 中缓存的目标模型
                    # top-k 后继，沿静态逻辑模板展开多条 continuation。
                    if general_tree_cache is None:
                        tree_template = general_template
                        general_tree_cache = (tree_template,) + prepare_data(tree_template, input_ids)
                    (
                        tree_template,
                        attention_mask,
                        position_ids,
                        search_path,
                        father_index,
                        candi_index,
                        deep_split,
                    ) = general_tree_cache
                    input_ids = prepare_data_input_ids(
                        input_ids, adj_matrix, father_index, candi_index, deep_split, output_id_topk
                    )
            else:
                # 找到历史 continuation：将最多四条候选链并行放入验证树。
                # 历史参数值确实可能成为草稿，但不会被盲目复制。相同的工具结构、
                # 城市、账号类型等 token 可以连续通过；日期、姓名等用户特定值只要
                # 与当前目标模型预测不一致，就会在第一个不同 token 处停止接受。
                # 例如旧历史提出 34567890，而当前用户要求 99990000：公共 JSON 与字段
                # 名可能先连续通过，但账号值从第一个不同 token 起会被拒绝。
                input_ids, tree_template, attention_mask, position_ids, search_path, father_index, candi_index, deep_split = build_retrieval_tree_from_candidates(
                    candidate_pred_tokens, input_ids
                )

        # 三种草稿最终都统一为“扁平 token + 树注意力 mask”，验证器无需知道来源。
        # prefix 区域全部可见；树内部的祖先可见关系由 attention_mask 决定。
        #
        # 若 verify_len=L、树节点数=N：
        #
        #   input_ids              [1, N]
        #   prefix 全可见 mask     [1, N, L]
        #   tree attention_mask    [1, N, N]
        #   merge_attention_mask   [1, N, L+N]
        #   model 所需 4D mask      [1, 1, N, L+N]
        #
        # 每个树节点都能看 L 个权威前缀 token，但在 N 个树节点内部只看自己的祖先。
        verify_len = verify_input_ids.size(1)
        expected_prefix_shape = (input_ids.size(0), len(tree_template) + 1, verify_len)
        if (
            not hasattr(finspec_forward, "_merge_attn_prefix")
            or tuple(finspec_forward._merge_attn_prefix.shape) != expected_prefix_shape
            or finspec_forward._merge_attn_prefix.device != input_ids.device
        ):
            finspec_forward._merge_attn_prefix = torch.ones(
                expected_prefix_shape,
                dtype=torch.bool,
                device=input_ids.device,
            )
        # 该缓存挂在函数对象上，单进程串行评测可安全复用。若 FinSpec 改成多线程或
        # 同进程多请求并发，不同 shape/device 可能互相覆盖，应改为显式、带键且有
        # 生命周期管理的请求外缓存。
        merge_attention_mask = torch.cat([finspec_forward._merge_attn_prefix, attention_mask.bool()], dim=-1)
        # position_ids 原本是树内深度 [0,1,2,...]；加 verify_len 后映射到完整序列位置
        # [L,L+1,L+2,...]。同一深度的兄弟节点使用相同位置编号。
        merge_positon_ids = position_ids + verify_len

        # 这里调用的仍然是 vanilla decode 使用的同一个目标模型。区别是：
        #
        # - vanilla 通常只输入当前一个新 token，一次前向只确定下一个 token；
        # - FinSpec/TR 输入一棵候选树，并用 4D Tree Attention mask 保证每个节点
        #   只能看到完整 prompt 和自己所在分支的祖先，不能偷看兄弟分支。
        #
        # 因此这一次模型调用可以在 GPU 上并行回答：“如果分别沿这些候选路径继续，
        # 每个位置的下一个 token 应该是什么？”候选树只是并行验证载体，模型权重、
        # 当前上下文和最终预测标准都没有改变。
        outputs = model(
            input_ids=input_ids,
            attention_mask=merge_attention_mask.unsqueeze(1),
            position_ids=merge_positon_ids,
            past_key_values=past_key_values,
        )
        
        # outputs.logits 的形状是 [1, N, vocab_size]：目标模型为树中每个节点分别预测
        # “沿该节点所在分支继续时，下一个 token 应是什么”。目标模型 argmax 始终是
        # 最终标准，草稿本身没有权力直接写入最终输出。
        model_res = torch.argmax(outputs.logits, dim=-1)
        
        # search_path 把扁平节点重新收集成若干 ROOT→叶子的矩形路径。例如：
        #
        #   search_path[0] = [0, 4, 7, 9, -1]
        #
        # 表示候选路径使用 input_ids 中的节点 0→4→7→9；-1 是 padding。PyTorch 用
        # -1 索引时会暂时取最后一个节点，所以紧接着把 padding 对应输入改成 -100，
        # 确保它不可能被当作真实匹配。
        all_input_path = input_ids[0][search_path]
        all_input_path[search_path==-1] = -100
        all_output_path = model_res[0][search_path]
        # 对每条根到叶路径，将草稿 token 与目标模型在当前上下文下的新预测错位比较：
        # 当前节点的模型预测应等于路径中的下一个草稿 token。
        #
        # 一个完整的符号例子：
        #
        #   all_input_path  = [ROOT, t1, t2, t3, t4]
        #   all_output_path = [t1,   t2, x,  t4, next]
        #
        # 比较必须错位一格：
        #
        #   草稿 input 的后继 [t1, t2, t3, t4]
        #   模型 output 当前位 [t1, t2, x,  t4]
        #   eq                [ 1,  1, 0,  1]
        #   cumprod           [ 1,  1, 0,  0]
        #
        # 即使 t4 在字面上再次相等，也不能越过 t3/x 的分叉继续接受，否则 t4 的条件
        # 上下文已经不同。故 reward=2，只连续命中 t1、t2。
        reward = torch.cumprod(all_input_path[:, 1:].eq(all_output_path[:, :-1]), dim=-1).sum(dim=-1)
        best_reward = reward.max()
        
        # 根 token 本来就是目标模型上一轮已经确定的权威 token，所以至少接受 1。
        # best_reward 是这次额外命中的草稿长度：
        #
        #   accept_len == 1：没有额外草稿命中，接近普通 decode 的单 token 推进；
        #   accept_len > 1：一次目标模型树前向接受多个 token，减少后续串行前向次数。
        #
        # 在上面的例子中：
        #
        #   best_reward = 2
        #   accept_len  = 1 + 2 = 3
        #   best_path_input = [ROOT, t1, t2]
        #
        # ROOT 也计入本轮接受长度，因为它虽然由上一轮模型决定，却直到本轮才真正输入
        # 模型并写入 KV Cache。t3 不被接受；模型在 t2 节点预测出的 x 会成为下一轮 ROOT。
        #
        # 这正是 Token Recycling/其他草稿来源可能获得延迟加速的核心；代价是本轮
        # 对多个候选节点做了比 vanilla 更多的并行计算。
        #
        # accept_len 也是判断 TR 是否“真的值得开”的关键观测量：
        # - 长期接近 1：几乎没有省下后续前向，却一直支付树验证成本；
        # - 经常明显大于 1：说明重复模式较强，批量接受可能覆盖额外成本。
        # 但它不是最终性能指标；还必须按草稿来源联合统计端到端延迟、吞吐、树节点数
        # 和 GPU 利用率。高并发下 GPU 已饱和，即使命中多个 token 也未必提高总吞吐。
        # TODO(优化): 记录 schema/retrieval/TR 来源级指标，并根据滑动窗口收益动态
        # 缩放树宽/树深；若 TR 连续低收益，则临时退回 vanilla decode。
        accept_len = 1 + best_reward
        accept_length_list.append(accept_len.item())
        best_path_index = torch.argmax(reward, dim=-1).to(torch.long)
        index_path = search_path[best_path_index][:accept_len]
        best_path_input = torch.index_select(input_ids, index=index_path, dim=1)

        # 模型已为整棵树写入临时 KV。这里只复制胜出路径到连续位置，其他分支会在
        # 下一轮被覆盖，从而把树状缓存恢复为标准自回归前缀。
        #
        # 假设本轮前已有 L 个连续 KV，胜出路径在扁平树中的节点是 [0, 4, 7]：
        #
        #   树前向临时写入的位置：L+0, L+1, ..., L+N-1
        #   选中的真实 KV：       [L+0, L+4, L+7]
        #   压缩后的连续位置：     [L,   L+1, L+2]
        #   current_length：        L+3
        #
        # 这样下一轮仍能把缓存当作普通线性自回归前缀；未胜出的兄弟分支无需清零，
        # 因为 current_length 之外的区域不属于有效缓存，之后可直接覆盖。
        tgt = past_key_values_data[..., verify_input_ids.size(1)+index_path, :]
        dst = past_key_values_data[..., verify_input_ids.size(1) : verify_input_ids.size(1) + tgt.shape[-2], :]
        dst.copy_(tgt, non_blocking=True)
        
        current_length_data.fill_(verify_input_ids.size(1) + tgt.shape[-2])
                
        # token 序列必须和刚刚压缩后的 KV 保持完全同长、同顺序。
        verify_input_ids = torch.cat([verify_input_ids, best_path_input], dim=-1)

        # 一旦已确认序列尾部完整匹配某个工具名，后续 Schema 参数候选就只来自该工具。
        verify_seq = verify_input_ids[0]  # Get the sequence tensor
        if not verify_tool_name:
            for tool_name_len, tool_name_tensor, tool_name_tuple in tool_name_tensors:
                if verify_seq.size(0) >= tool_name_len:
                    if torch.equal(verify_seq[-tool_name_len:], tool_name_tensor):
                        # 预约示例在权威输出尾部完整出现 ModifyRegistration 后才会命中。
                        # 只出现 "Modify" 等部分前缀不会设置，避免拿错工具的参数 Schema。
                        verify_tool_name = tool_name_tuple
                        break

        if has_stop(best_path_input) or stopping_criteria(verify_input_ids, None):
            break
        
        # 时间关系要分清楚：
        #
        #   本轮“当前上下文”的新 logits
        #       → 写入/覆盖 adj_matrix
        #       → 变成未来解码轮次或后续请求可读取的“旧草稿信息”
        #
        # 回收本轮目标模型已经算出的信息：
        # outputs.logits 为树中每个输入节点给出“下一个 token”的分布；取 top-k 后，
        # 把每个节点 token 映射到若干高概率后继。未来若 Schema/检索无草稿，
        # prepare_data_input_ids 会读取这些行构造通用候选树。
        #
        # 注意两个有意保留的近似：
        # 1. input token 是唯一键，本轮上下文不会写进 adj_matrix；同一 token 在别的
        #    工具/字段出现时会读取同一行。这使查询极快，也造成上下文污染。
        # 2. 一棵树里同一 token 可能出现多次。下面的高级索引赋值没有定义显式的
        #    “取最新、取最高置信度或合并”策略，不应依赖重复索引最终由哪一项覆盖。
        #    若要稳定复现，可先按 token 去重，再按置信度/时效选择一行写入。
        #
        # 这一步只缓存草稿线索，不接受任何 token；本轮/未来的最终输出仍由目标模型
        # Tree Attention 验证决定。
        #
        # 注意：这里目前把 k 硬编码为 8，而函数参数和 adj_matrix 形状由
        # output_id_topk 决定。只有 output_id_topk==8 时二者天然一致，其他配置可能
        # 产生赋值 shape 错误。
        # TODO(优化): 改用 k=output_id_topk，并校验 1 <= k <= vocab_size。
        to_update = torch.topk(outputs.logits, k=8, dim=-1)[1][0]
        adj_matrix[input_ids.squeeze(0)] = to_update
        
        # 最后一个已接受节点上的模型预测，即第一个未被草稿覆盖/匹配的权威 token；
        # 它作为下一轮树根继续生成。延续 ROOT,t1,t2,t3 的例子，这里取得的是模型在
        # t2 位置预测的 x，而不是被拒绝的草稿 t3：
        #
        #   本轮 verify_input_ids 追加 [ROOT,t1,t2]
        #   下一轮 input_ids = [x]
        input_ids = model_res[:, search_path[best_path_index][accept_len-1]].unsqueeze(-1)

        if has_stop(input_ids):
            break

        step += 1

    verify_input_ids = _truncate_generated_suffix(
        verify_input_ids=verify_input_ids,
        start_len=start_len,
        eos_token_ids=eos_token_ids,
        header_token_ids=header_token_ids,
        accept_length_list=accept_length_list,
    )

    # 调用方可从完整返回序列中切掉原 prompt，只把新生成 token 解码并写入历史。
    # question_hidden_state 与输出一起返回，是为了让“这次请求”在完成之后成为后续请求
    # 的可检索记忆；它不参与本请求已经结束后的任何 token 修正。
    return verify_input_ids, verify_input_ids.size(1) - start_len, step, accept_length_list, question_hidden_state
