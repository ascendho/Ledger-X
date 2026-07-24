"""Schema 与历史检索均无草稿时使用的静态 Token Recycling 树。

每个列表项表示一条 top-k 排名路径，而不是实际 token id。例如 ``[0, 1]`` 表示先
采用父 token 最常见的后继，再采用该 token 第二常见的后继。树在根部较宽、深处较窄，
目的是把固定节点预算优先分配给概率更高的 continuation。

“树越大越快”并不成立：

* 增加宽度可以覆盖更多 top-k 分支，但每个兄弟节点都要参加目标模型验证；
* 增加深度可能一次接受更长前缀，但单 token 邻接关系越往后组合，受上下文丢失影响
  越明显，长路径连续命中的概率通常会下降；
* 当新增节点带来的接受长度小于额外 Tree Attention、logits 和 KV Cache 成本时，
  tokens/s 和端到端延迟都会恶化。

当前手写静态树对 JSON 标点、SQL 关键字、姓名和日期一视同仁，无法根据当前位置的不
确定性调整预算。这是实现简单、缓存拓扑方便与候选效率之间的取舍，不是理论上的最优
树。原始 Token Recycling 工作也把静态树列为可继续改进的方向。

TODO(优化): 根据近期接受率、logits 熵、草稿来源和业务字段类型动态调节宽度/深度；
JSON/SQL 固定结构可加深高概率分支，账户号、金额、日期、姓名等上下文敏感值可缩小
或关闭 TR。将手写模板迁移到可校验配置，并联合记录节点预算、接受长度、吞吐和端到端
延迟，不能只用“命中 token 更多”判断优化有效。
"""


def choose_tree_template(version):
    """按版本返回静态逻辑路径；未知版本立即报错，避免静默使用错误树。"""
    tree_template = None
    if version == "general":
        tree_template = [[0], [1], [2], [3], [4], [5], [6], [7],
                        [0,0], [0,1], [0,2], [0,3], [0,4], [0,5], [0,6], [0,7], [1,0], [1,1], [1,2], [1,3], [2,0], [2,1], [2,2], [3,0], [3,1], [4,0], [5,0], [6,0], [7,0],
                        [0,0,0], [0,0,1], [0,0,2], [0,0,3], [0,0,4], [0,0,5], [0,0,6], [0,0,7], [0,1,0], [0,1,1], [0,1,2], [0,2,0], [0,2,1], [0,3,0], [0,4,0], [0,5,0], [0,6,0], [0,7,0], [1,0,0], [1,0,1], [1,1,0], [2,0,0], [3,0,0], [4,0,0], [5,0,0],
                        [0,0,0,0], [0,0,0,1], [0,0,0,2], [0,0,0,3], [0,0,0,4], [0,0,1,0], [0,0,1,1], [0,0,2,0], [0,0,3,0], [0,0,4,0], [0,1,0,0], [0,2,0,0], [1,0,0,0], [2,0,0,0], [3,0,0,0],
                        [0,0,0,0,0], [0,0,0,0,1], [0,0,0,0,2], [0,0,0,1,0], [0,0,0,2,0], [0,0,1,0,0], [0,1,0,0,0], [1,0,0,0,0],
                        [0,0,0,0,0,0], [0,0,0,0,0,1], [0,0,0,1,0,0],
                        ]
    if tree_template is None:
        raise ValueError("Invalid version")
    return tree_template


if __name__ == "__main__":
    tree_template = choose_tree_template("general")
    
    # 独立运行本文件时输出树的根分支、深度和节点数，便于检查配置。
    root_paths_info = {}
    for path in tree_template:
        if path and len(path) > 0:
            root = path[0]
            depth = len(path)
            if root not in root_paths_info:
                root_paths_info[root] = []
            root_paths_info[root].append(depth)
    
    # 这些统计只用于开发检查，不参与正式解码。
    num_paths = len(root_paths_info)
    print(f"Number of paths: {num_paths}")
    print("Depth of each path:")
    for root in sorted(root_paths_info.keys()):
        depths = root_paths_info[root]
        max_depth = max(depths)
        min_depth = min(depths)
        num_subpaths = len(depths)
        print(f"  Path {root}: depth range [{min_depth}, {max_depth}], {num_subpaths} sub-paths")
    
    print(f"\nFull tree template ({len(tree_template)} nodes):") # (80 nodes)
    print(tree_template)
