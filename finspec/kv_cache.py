"""Tree Attention 并行验证使用的预分配可写 KV Cache。

先从哪里看
==========

建议按照对象的生命周期阅读，而不是从类定义第一行机械向下读：

1. 先看文件底部 ``initialize_past_key_values``，理解整块缓存如何分配，以及它返回的
   ``past_key_values``、``past_key_values_data``、``current_length_data``；
2. 再看 ``KVCache.shape``，理解“物理容量”和“当前有效长度”为什么不同；
3. 跳到 ``modeling_qwen_kv.py`` 或 ``modeling_llama_kv.py``，找到
   ``past_key_value[0].cat(key_states)`` 和
   ``past_key_value[1].cat(value_states)``；
4. 回到本文件阅读 ``KVCache.cat``，理解 prefill/候选树的 K/V 怎样原地追加；
5. 再看 ``decoding.py`` 中对 ``past_key_values_data`` 的胜出路径压缩；
6. 最后看 ``KVCache.copy``。它表达同一种压缩思想，但当前 ``decoding.py`` 为了
   一次处理所有层和 K/V，直接操作整块 ``past_key_values_data``，没有调用它。

实际调用链
==========

.. code-block:: text

    finspec_forward
        → initialize_past_key_values(model)
        → prefill model(..., past_key_values=...)
            → 每层 Attention 计算 key_states/value_states
            → KVCache.cat 原地写入 prompt K/V
        → tree model(..., past_key_values=...)
            → KVCache.cat 临时写入整棵候选树 K/V
        → decoding.py 选出最长匹配路径
        → 只把胜出节点 K/V 压到连续前缀
        → 下一轮从新的有效长度继续覆盖写入

为什么只缓存 K 和 V
===================

自注意力可简化写成 ``softmax(Q·Kᵀ)·V``。生成新 token 时，需要用“新 token 的 Q”
查询所有历史 token 的 K，再聚合其 V，因此历史 K/V 会被反复使用，值得缓存。历史 Q
只用于历史位置当时的输出；未来 token 不会再查询历史 Q，所以无需保存。

Tree Attention 与普通 KV Cache 的差别不在 K/V 数学定义，而在临时布局：普通 decode
每轮只追加一条线性路径；FinSpec 一轮会把所有候选分支的 K/V 都写入缓存，再由
attention mask 保证兄弟分支互不可见。验证结束后，缓存必须重新压成一条线性胜出路径。

贯穿示例
========

业务上仍以 ``ModifyRegistration`` 为例。prompt 完成 prefill 后，模型开始验证工具名、
``appointment_id``、日期和医生等候选树。为了让数字直观，下面不使用真实大模型配置，
而是假设一个小模型：

.. code-block:: text

    num_hidden_layers   = 2
    batch_size          = 1
    num_key_value_heads = 2
    cache_len           = 16
    head_dim            = 4

    past_key_values_data.shape
      = [layer*2, batch, kv_heads, cache_len, head_dim]
      = [4, 1, 2, 16, 4]

    第 0 片 = layer 0 key
    第 1 片 = layer 0 value
    第 2 片 = layer 1 key
    第 3 片 = layer 1 value

若简化后的 prompt 有 5 个 token，prefill 把各层 K/V 写入位置 ``0:5``，有效长度从
0 变成 5。下一轮候选树有 5 个节点时，模型暂时写入 ``5:10``，有效长度变成 10。
假设胜出路径只使用树节点 ``[0, 2, 4]``，则从绝对缓存位置 ``[5, 7, 9]`` 取出 K/V，
覆盖到连续位置 ``[5, 6, 7]``，最后把有效长度改为 8。位置 8 以后的失败分支不必清零，
因为逻辑长度之外的数据不会被读取，下一轮可以直接覆盖。

真实 Qwen/Llama 的层数、KV head 数和 head dimension 由 ``model.config`` 决定；本例
中的数字只用于展示索引变化，不能当作真实模型配置。
"""

import torch


class KVCache:
    """某一层某一种 K/V 张量的预分配可写视图。

    一个 ``KVCache`` 只代表“某一层的 Key”或“某一层的 Value”，不是整个模型缓存。
    ``data`` 的形状是 ``[batch, kv_heads, capacity, head_dim]``，它是
    ``past_key_values_data`` 某一片的视图；``current_length`` 是位于 CPU 上的零维
    ``torch.long``，表示第三维目前有多少位置有效。

    例如底层 ``data.shape=[1,2,16,4]``、``current_length=5``：

    * 物理容量仍是 16，可以继续原地写；
    * 对模型报告的逻辑 shape 是 ``[1,2,5,4]``；
    * 位置 ``5:16`` 即使存在旧数值，也不属于当前上下文。

    ``cat`` 只在预分配区域内写入，不发生普通 ``torch.cat`` 对完整历史的重新分配与
    复制；``copy`` 用于把树上的胜出节点压缩到连续前缀。
    """

    def __init__(self, data, current_length):
        """保存底层存储视图和当前有效长度；不会复制任何 K/V 数据。

        ``initialize_past_key_values`` 会为每层 K、V 分别传入不同的 ``data`` 切片和
        长度标量。修改 ``self.data`` 会直接修改共享的 ``past_key_values_data``；
        修改 ``self.current_length`` 也会反映到 ``current_length_data`` 对应元素。
        """
        self.data = data
        self.current_length = current_length

    @property
    def shape(self):
        """返回当前有效缓存的逻辑 shape，而不是预分配容量。

        Qwen/Llama 模型会读取 ``past_key_value[0].shape[-2]`` 来计算已有上下文长度和
        RoPE/attention 的序列范围。若直接返回 ``self.data.shape``，刚初始化时模型就
        会误以为容量 16 的全部位置都是真实历史。

        例：``data.shape=[1,2,16,4]``、``current_length=5`` 时返回
        ``(1,2,5,4)``。长度标量放在 CPU，因此这里的 ``.item()`` 不会为了读取一个
        GPU 标量而触发设备同步。
        """
        return (
            self.data.shape[0],
            self.data.shape[1],
            self.current_length.item(),
            self.data.shape[3],
        )

    def copy(self, indices: torch.Tensor, prev_length: int, dim: int = 2):
        """把绝对缓存位置 ``indices`` 指定的 K/V 压到 ``prev_length`` 之后。

        ``indices`` 会直接传给 ``self.data.index_select``，因此必须是底层 ``data``
        第三维的**绝对位置**，而不是从树根开始的相对节点编号。``prev_length`` 是
        进入本轮树验证前已经确认的线性前缀长度；``dim`` 在当前布局中固定为序列维 2。

        具体例子：

        .. code-block:: text

            prefill 有效长度：5
            候选树临时位置：5, 6, 7, 8, 9
            胜出相对节点：  [0, 2, 4]
            应传绝对 indices：[5, 7, 9]

            tgt = [位置5的KV, 位置7的KV, 位置9的KV]
            dst = data[:, :, 5:8, :]
            copy 后有效布局 = prompt[0:5] + winner[5,7,9]
            current_length = 8

        如果错误传入相对 ``[0,2,4]``，本实现会从 prompt 区域取 K/V，而不是从候选树
        区域取值。``index_select`` 会先生成独立的 ``tgt``，所以源位置和目标位置有
        重叠时，后续原地覆盖仍是安全的。

        当前 ``decoding.py`` 没有调用本方法，而是给相对 ``index_path`` 加上
        ``verify_input_ids.size(1)``，再一次性压缩所有层的 K/V。

        TODO(优化): 统一使用一个明确区分 relative/absolute index 的批量压缩 API，
        同时操作所有层的 K/V，避免本方法与主循环两套逻辑发生偏差。
        """
        # index_select 取出胜出路径，再原地覆盖到连续目标区间。
        tgt = self.data.index_select(dim, indices)
        dst = self.data.narrow(dim, prev_length, tgt.shape[dim])
        dst.copy_(tgt, non_blocking=True)
        self.current_length.fill_(prev_length + tgt.shape[dim])

    def cat(self, tensor: torch.Tensor, dim: int = 2):
        """把新 KV 原地追加到预分配空间，并返回当前有效区间视图。

        Attention 中新计算的 ``key_states`` 或 ``value_states`` 形状为
        ``[batch, kv_heads, new_length, head_dim]``。名称虽然是 ``cat``，实际只对空闲
        区域执行 ``narrow + copy``，不会像 ``torch.cat([old, new])`` 那样重新分配并
        复制完整历史前缀。

        仍使用容量 16 的例子：

        .. code-block:: text

            初始 current_length = 0

            prefill tensor.shape = [1,2,5,4]
              → dst = data[:,:,0:5,:]
              → current_length = 5
              → 返回 data[:,:,0:5,:]

            tree tensor.shape = [1,2,5,4]
              → dst = data[:,:,5:10,:]
              → current_length = 10
              → 返回 data[:,:,0:10,:]

        返回完整有效视图是因为当前树节点做 attention 时，既要看到之前 5 个 prompt
        token 的 K/V，也要在 Tree Attention mask 允许的范围内看到本轮祖先节点 K/V。
        兄弟分支虽然都位于 ``5:10``，但会由外部 4D attention mask 隔离。

        TODO(优化): 增加容量边界检查和明确的溢出异常；服务化批处理可替换为 paged
        KV cache，为每个请求维护独立页表和长度。当前签名提供 ``dim``，但返回时仍
        硬编码沿维度 2 做 ``narrow``；应固定并校验 ``dim==2``，或全程真正使用参数。
        """
        # self.current_length 是零维 CPU long；narrow 接受它作为当前写入起点。
        # 若 current_length + tensor.shape[dim] 超过 capacity，narrow 会在较底层报错，
        # 当前没有带请求长度/容量信息的友好异常。
        dst = self.data.narrow(dim, self.current_length, tensor.shape[dim])
        dst.copy_(tensor)
        self.current_length.add_(tensor.shape[dim])
        # 当前所有调用都使用序列维 2。返回的是 self.data 的 view，不是新缓存副本。
        return torch.narrow(self.data, 2, 0, self.current_length)


def initialize_past_key_values(model, max_cache_len=None):
    """为整个 Transformer 一次性预分配所有层的 key/value 缓存。

    这是阅读本文件的起点。它先分配一整块连续 GPU Tensor，再把不同层、不同 K/V 的
    切片包装成 ``KVCache``。与“每层各自分配很多小 Tensor”相比，整块存储便于
    ``decoding.py`` 在胜出路径确定后一次性索引和压缩所有层。

    底层张量形状为
    ``[layer*2, batch, kv_heads, cache_len, head_dim]``，第一维交替保存每层的
    key 和 value。返回的 ``past_key_values`` 是按层组织的 ``KVCache`` 视图；
    ``past_key_values_data`` 供主循环批量压缩胜出分支；``current_length_data``
    分别记录每层 K/V 的有效长度。

    ``max_cache_len`` 由 prompt、最大生成长度和临时树节点预算共同决定，避免始终按照
    模型最大上下文分配。

    小模型示例：

    .. code-block:: text

        num_hidden_layers=2, num_key_value_heads=2
        hidden_size=16, num_attention_heads=4 → head_dim=4
        cache_len=16

        past_key_values_data: [4,1,2,16,4]
          [0] → layer 0 key data
          [1] → layer 0 value data
          [2] → layer 1 key data
          [3] → layer 1 value data

        current_length_data: [0,0,0,0]

        past_key_values:
          [
            [KVCache(data[0], length[0]), KVCache(data[1], length[1])],
            [KVCache(data[2], length[2]), KVCache(data[3], length[3])],
          ]

    三个返回值不是三份缓存：``past_key_values`` 只是模型容易消费的对象视图，
    ``past_key_values_data`` 才是实际存储，``current_length_data`` 决定各视图的有效
    长度。正常前向中每层 K/V 会推进相同 token 数，所以所有长度应保持一致；
    ``decoding.py`` 压缩胜出路径时会对整个 ``current_length_data`` 执行 ``fill_``。

    显存占用可按下式估算：

    ``2 × layers × batch × kv_heads × cache_len × head_dim × dtype_bytes``。

    这里使用 ``num_key_value_heads`` 而不是 ``num_attention_heads``：GQA/MQA 模型会
    让多个 Query head 共享较少的 K/V head，缓存共享后的 K/V 即可，能显著降低显存。

    TODO(优化): 当前 batch size 固定为 1，适合论文中的单请求延迟评测。在线服务应
    使用请求级长度或 paged KV cache，并按并发量做显存预算和回收；还应在模型某层
    前向异常时回滚/重建长度，避免 K 已追加而 V 或后续层尚未追加造成长度不一致。
    """
    config = model.config
    # 论文主要评测单请求延迟，因此这里明确固定 batch size 为 1。
    batch_size = 1
    # 未提供请求级上限时，保留比模型最大上下文多 512 个临时位置，供候选树暂存。
    # decoding.py 正常会传入“prompt + max_new_tokens + 树节点余量”的请求级预算。
    # min(..., max_position_embeddings+512) 防止调用方申请无界缓存，但 cat 本身仍需要
    # 容量检查，才能在预算不足时给出清晰错误。
    if max_cache_len is None:
        cache_len = config.max_position_embeddings + 512
    else:
        cache_len = min(int(max_cache_len), int(config.max_position_embeddings + 512))

    past_key_values_data = torch.zeros(
        config.num_hidden_layers * 2,
        batch_size,
        config.num_key_value_heads,
        cache_len,
        config.hidden_size // config.num_attention_heads,
        device=model.device,
        dtype=model.dtype,
    )
    # 真实占用由模型配置和 dtype 决定。例如 float16/bfloat16 通常每元素 2 字节；
    # 预分配避免后续反复扩容，但即使实际输出很短，也会立刻占用完整 cache_len 显存。

    # 长度标量保留在 CPU，shape 查询和 fill_ 不触发额外 GPU 同步。
    current_length_data = torch.zeros(
        config.num_hidden_layers * 2, dtype=torch.long, device="cpu"
    )
    # 每层返回 [key_cache, value_cache]，两者分别指向上面连续存储的一片，并各自
    # 绑定一个长度标量。模型适配器在每层内部依次调用 key_cache.cat 和
    # value_cache.cat；它随后把局部 past_key_value 设为 None，是因为缓存已原地更新，
    # 不需要再从模型返回一份新 tuple。
    #
    # `[] * config.num_hidden_layers` 的结果仍然只是空列表；后面的 append 才真正加入
    # 每层对象。可读性上直接写 `past_key_values = []` 更清楚，但此处保留原逻辑。
    past_key_values = [] * config.num_hidden_layers
    for i in range(config.num_hidden_layers):
        past_key_values.append(
            [
                KVCache(past_key_values_data[i * 2 + j], current_length_data[i * 2 + j])
                for j in range(2)
            ]
        )
    # 返回后：
    # - 模型只接触 past_key_values 的分层 KVCache 接口；
    # - decoding.py 保留 past_key_values_data/current_length_data，用于一次性压缩
    #   所有层和 K/V 的胜出树路径。
    return past_key_values, past_key_values_data, current_length_data
