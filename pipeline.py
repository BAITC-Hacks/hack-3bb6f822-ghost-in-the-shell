"""
HackAlem AI — Граф денег: пайплайн ролей, кластеров и приоритетов.

Запуск: python pipeline.py  (данные ожидаются в ./data/{nodes,edges,transactions}.parquet)
Выход: nodes_roles.csv, clusters.csv, top_nodes.csv (в текущей папке)

Все правила ниже — детерминированные (без ML/LLM), пороги подобраны под
объявленные в README датасета кандидаты на каждую роль. Обоснование
каждого порога — см. README.md решения.

ВАЖНО (изменение от предыдущей версии): правило transit проверяется
РАНЬШЕ consolidator/distributor. Причина: условие consolidator
(us>=3 или ideg>=3) само по себе очень широкое и ловит в том числе узлы,
которые получили от нескольких плательщиков, но тут же почти всё
передали дальше (чистый транзит с несколькими источниками). Такой узел
по смыслу ТЗ — transit ("пропускает средства дальше, не удерживая"),
а не consolidator ("аккумулирует средства"). Проверка удержания через
ratio = out/in решает эту неоднозначность объяснимо и без ML.
"""

import pandas as pd
import networkx as nx
from collections import defaultdict

RNG_SEED = 42

# ---------------------------------------------------------------- load ----
nodes = pd.read_parquet("data/nodes.parquet")
edges = pd.read_parquet("data/edges.parquet")
tx = pd.read_parquet("data/transactions.parquet")

seed_ids = set(nodes.loc[nodes.is_seed, "gid"])
depth_of = dict(zip(nodes["gid"], nodes["depth"]))

G = nx.DiGraph()
G.add_nodes_from(nodes["gid"])
for r in edges.itertuples():
    G.add_edge(r.src, r.dst, sum_kzt=r.sum_kzt, n_tx=r.n_tx, depth=r.depth)

UG = G.to_undirected()

# ------------------------------------------------- ancestor seed count ----
# Кол-во РАЗНЫХ seed-клиентов, чьи деньги (напрямую/через цепочку) доходят
# до узла. Это точный признак конвергенции — граф прослежен только вперёд
# от известных 81 seed, поэтому >1 совпадающих предков = независимые
# денежные цепочки сошлись в одной точке.
ancestors = defaultdict(set)
for s in seed_ids:
    ancestors[s] = {s}
for r in edges.sort_values("depth").itertuples():
    ancestors[r.dst] |= ancestors.get(r.src, set())
upstream_seeds = {g: len(ancestors.get(g, set())) for g in nodes["gid"]}

# ------------------------------------------------------- pass-through -----
tx["date"] = pd.to_datetime(tx["date"])
first_in = tx.groupby("dst")["date"].min()
first_out = tx.groupby("src")["date"].min()

in_amt = defaultdict(float); out_amt = defaultdict(float)
in_deg = defaultdict(int); out_deg = defaultdict(int)
in_srcs = defaultdict(set)
for r in edges.itertuples():
    out_amt[r.src] += r.sum_kzt
    in_amt[r.dst] += r.sum_kzt
    out_deg[r.src] += 1
    in_deg[r.dst] += 1
    in_srcs[r.dst].add(r.src)

# --------------------------------------------------- betweenness (bridge) -
# Для 2248 узлов / 3119 рёбер точный betweenness считается быстро (<5 мин
# требование выполняется с большим запасом).
betweenness = nx.betweenness_centrality(UG, weight=None)
bt_vals = sorted(betweenness.values(), reverse=True)
bt_p90 = bt_vals[int(len(bt_vals) * 0.10)] if bt_vals else 0  # top-10% порог

# --------------------------------------------------------- clustering -----
communities = nx.algorithms.community.louvain_communities(UG, weight="sum_kzt", seed=RNG_SEED)
cluster_of = {}
for cid, members in enumerate(communities):
    for g in members:
        cluster_of[g] = cid

# ---------------------------------------------------------- role logic ----
def classify(gid):
    ideg, odeg = in_deg.get(gid, 0), out_deg.get(gid, 0)
    ia, oa = in_amt.get(gid, 0.0), out_amt.get(gid, 0.0)
    us = upstream_seeds.get(gid, 1 if gid in seed_ids else 0)
    depth = depth_of.get(gid, 0)
    bt = betweenness.get(gid, 0.0)
    ratio = (oa / ia) if ia > 0 else None
    lag = None
    if gid in first_in.index and gid in first_out.index:
        lag = (first_out[gid] - first_in[gid]).days

    is_hub_both_ways = ideg >= 3 and odeg >= 3
    is_bridge = bt >= bt_p90 and bt > 0

    # coordinator: структурный мост + заметно и получает, и раздаёт дальше
    if is_bridge and is_hub_both_ways:
        conf = min(1.0, 0.5 + bt / (bt_vals[0] + 1e-9) * 0.3 + min(us, 5) / 10)
        ev = (f"Узел-мост (betweenness в топ-10%): принимает от {ideg} и "
              f"передаёт {odeg} контрагентам, деньги от {us} разных seed-цепочек — "
              f"кандидат в организаторы")
        return "coordinator", round(conf, 2), ev[:200]

    # transit: пришло ~ ушло, быстро — проверяем ДО consolidator/distributor,
    # т.к. это более специфичный и однозначный признак (сквозной проход денег)
    if ratio is not None and 0.8 <= ratio <= 1.2 and lag is not None and lag <= 3 and ideg >= 1 and odeg >= 1:
        conf = min(1.0, 0.6 + (3 - lag) / 10)
        ev = (f"Транзит: вход {ia:,.0f} KZT ≈ выход {oa:,.0f} KZT (ratio={ratio:.2f}), "
              f"задержка приход→уход {lag} дн., плательщиков {ideg}")
        return "transit", round(conf, 2), ev[:200]

    # consolidator: сходятся независимые seed-цепочки или явная конвергенция,
    # И деньги при этом заметно удерживаются (не чистый транзит)
    holds_money = ratio is None or ratio < 0.6
    if (us >= 3 or ideg >= 3) and holds_money:
        conf = min(1.0, 0.4 + min(us, 8) / 10 + min(ideg, 10) / 30)
        ev = (f"Получает от {ideg} разных плательщиков, из них деньги от "
              f"{us} независимых seed-цепочек; входящая сумма {ia:,.0f} KZT, "
              f"удерживает {'>40%' if ratio is None else f'{(1-ratio)*100:.0f}%'} полученного")
        return "consolidator", round(conf, 2), ev[:200]

    # distributor: веерная раздача, сам не аккумулирует
    if odeg >= 5 and ideg <= 2:
        conf = min(1.0, 0.4 + min(odeg, 30) / 40)
        ev = f"Раздаёт дальше {odeg} получателям (веер), входящих связей {ideg}, отдаёт {oa:,.0f} KZT"
        return "distributor", round(conf, 2), ev[:200]

    # terminal: деньги дошли, дальше в графе не уходят
    if odeg == 0 and ideg >= 1:
        if depth == 4:
            conf = 0.45  # неопределённость: может быть обрыв обхода на 4 колене
            ev = (f"Нет исходящих в графе, но узел на 4 колене (макс. глубина обхода) — "
                  f"возможен артефакт обрыва, а не реальный конечный получатель")
        else:
            conf = 0.85
            ev = f"Получил {ia:,.0f} KZT от {ideg} плательщиков, дальше по графу переводов не найдено"
        return "terminal", conf, ev[:200]

    # peripheral: сигналов нет
    conf = 0.5
    ev = f"in={ideg}, out={odeg} — недостаточно связей для уверенной классификации"
    return "peripheral", conf, ev[:200]

rows = []
for gid in nodes["gid"]:
    role, role_score, evidence = classify(gid)
    rows.append(dict(
        gid=gid, role=role, role_score=role_score,
        cluster_id=cluster_of.get(gid, -1), evidence=evidence,
        _ideg=in_deg.get(gid, 0), _odeg=out_deg.get(gid, 0),
        _us=upstream_seeds.get(gid, 0), _bt=betweenness.get(gid, 0.0),
        _is_seed=gid in seed_ids,
    ))
nr = pd.DataFrame(rows)

ROLE_WEIGHT = {"coordinator": 1.0, "consolidator": 0.85, "distributor": 0.6,
               "transit": 0.4, "terminal": 0.3, "peripheral": 0.05}
raw_priority = nr.apply(
    lambda r: ROLE_WEIGHT[r.role] * r.role_score
    + 0.15 * min(r._us, 10) / 10
    + 0.10 * min(r._ideg + r._odeg, 40) / 40
    + (0.05 if r._is_seed else 0.0),
    axis=1,
)
mn, mx = raw_priority.min(), raw_priority.max()
nr["priority_score"] = ((raw_priority - mn) / (mx - mn)).round(3)

nodes_roles = nr[["gid", "role", "role_score", "cluster_id", "priority_score", "evidence"]]
nodes_roles.to_csv("nodes_roles.csv", index=False)

# --------------------------------------------------------------- clusters -
crows = []
for cid, members in enumerate(communities):
    members = set(members)
    n_seed = len(members & seed_ids)
    sub_edges = edges[edges.src.isin(members) & edges.dst.isin(members)]
    sum_internal = sub_edges.sum_kzt.sum()
    top_gids = (
        nr[nr.gid.isin(members)].sort_values("priority_score", ascending=False)["gid"].head(5).tolist()
    )
    if n_seed >= 2 and sum_internal > 0:
        hyp = f"Компонент с {n_seed} исходными seed-клиентами и внутренним оборотом {sum_internal:,.0f} KZT — вероятная единая сеть/группа"
    elif len(members) <= 3:
        hyp = "Малая изолированная группа — отдельная цепочка или шум обхода"
    else:
        hyp = f"Кластер вокруг {n_seed} seed-клиента(ов), требует ручной проверки на связь с остальными"
    crows.append(dict(cluster_id=cid, n_nodes=len(members), n_seed=n_seed,
                       sum_kzt_internal=round(sum_internal, 2),
                       top_gids=";".join(map(str, top_gids)), hypothesis=hyp))
pd.DataFrame(crows).sort_values("n_nodes", ascending=False).to_csv("clusters.csv", index=False)

# -------------------------------------------------------------- top_nodes -
top = nr.sort_values("priority_score", ascending=False).head(max(20, 30)).copy()
top = top.reset_index(drop=True)
top["rank"] = top.index + 1
top_nodes = top[["rank", "gid", "role", "priority_score", "evidence"]].rename(columns={"evidence": "why"})
top_nodes.to_csv("top_nodes.csv", index=False)

print("Готово: nodes_roles.csv, clusters.csv, top_nodes.csv")
print(nr.role.value_counts())
print(f"Кластеров: {len(communities)}")
