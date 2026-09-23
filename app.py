"""
Ghost in the Shell — экран просмотра графа денежных переводов.

Запуск (после того как pipeline.py уже создал nodes_roles.csv/clusters.csv):
    streamlit run app.py

Что показывает:
- список узлов с ролью, приоритетом и обоснованием (топ-лист)
- поиск по gid: подграф узла (входящие + исходящие связи), интерактивная схема
- полная схема сети с подсветкой ролей и кластеров (для небольших подграфов)

Интерфейс НЕ пересчитывает роли — он только визуализирует то, что уже
посчитал pipeline.py. Если nodes_roles.csv нет — запустите сначала pipeline.py.
"""

import pandas as pd
import networkx as nx
import streamlit as st
from pyvis.network import Network
import streamlit.components.v1 as components
import os

st.set_page_config(page_title="Граф денег — просмотр", layout="wide")

ROLE_COLORS = {
    "coordinator": "#e74c3c",   # красный — кандидат в организаторы
    "consolidator": "#e67e22",  # оранжевый — точка накопления
    "distributor": "#f1c40f",   # жёлтый — веерная раздача
    "transit": "#3498db",       # синий — сквозной транзит
    "terminal": "#2ecc71",      # зелёный — конечная точка
    "peripheral": "#95a5a6",    # серый — периферия
}

DATA_DIR = "data"


@st.cache_data
def load_data():
    missing = [f for f in ["nodes_roles.csv", "clusters.csv"] if not os.path.exists(f)]
    if missing:
        st.error(f"Не найдены файлы: {missing}. Сначала запустите: python pipeline.py")
        st.stop()
    nr = pd.read_csv("nodes_roles.csv")
    cl = pd.read_csv("clusters.csv")
    edges = pd.read_parquet(os.path.join(DATA_DIR, "edges.parquet"))
    nodes = pd.read_parquet(os.path.join(DATA_DIR, "nodes.parquet"))
    return nr, cl, edges, nodes


nr, cl, edges, nodes = load_data()
seed_ids = set(nodes.loc[nodes.is_seed, "gid"])

st.title("Граф денег — экран просмотра")
st.caption("AML-аналитика: роли узлов, кластеры, приоритет проверки. "
           "Все выводы — гипотезы для проверки, не утверждения о виновности.")

tab_search, tab_top, tab_clusters, tab_full = st.tabs(
    ["🔍 Поиск по gid", "📋 Топ-лист приоритетов", "🧩 Кластеры", "🕸 Вся сеть"]
)

# -------------------------------------------------------- helper: draw ----
def draw_subgraph(center_gid, depth=1, height="650px"):
    """Строит подграф вокруг узла: соседи на N шагов в обе стороны."""
    G = nx.DiGraph()
    for r in edges.itertuples():
        G.add_edge(r.src, r.dst, sum_kzt=r.sum_kzt, n_tx=r.n_tx)

    if center_gid not in G:
        st.warning(f"gid {center_gid} не найден в edges (возможно, изолированный узел без связей).")
        return

    UG = G.to_undirected()
    sub_nodes = nx.single_source_shortest_path_length(UG, center_gid, cutoff=depth).keys()
    sub = G.subgraph(sub_nodes)

    net = Network(height=height, width="100%", directed=True, bgcolor="#1e1e1e", font_color="white")
    net.barnes_hut()

    role_map = dict(zip(nr.gid, nr.role))
    cluster_map = dict(zip(nr.gid, nr.cluster_id))

    for n in sub.nodes():
        role = role_map.get(n, "peripheral")
        is_center = n == center_gid
        is_seed = n in seed_ids
        label = f"{n}" + (" [SEED]" if is_seed else "")
        size = 35 if is_center else (22 if is_seed else 15)
        border = "#ffffff" if is_center else ("#f39c12" if is_seed else "#333333")
        title = f"gid={n}\nrole={role}\ncluster={cluster_map.get(n, '-')}\nseed={is_seed}"
        net.add_node(int(n), label=label, color=ROLE_COLORS.get(role, "#999"),
                     size=size, borderWidth=3 if (is_center or is_seed) else 1,
                     borderWidthSelected=4, title=title)

    for u, v, d in sub.edges(data=True):
        net.add_edge(int(u), int(v), value=max(1, d["sum_kzt"] / 1_000_000),
                     title=f"{d['sum_kzt']:,.0f} KZT, {d.get('n_tx', '?')} tx")

    net.set_options("""
    {
      "physics": {"stabilization": {"iterations": 150}},
      "edges": {"arrows": {"to": {"enabled": true}}, "color": {"color": "#555"}}
    }
    """)
    html_path = f"_graph_{center_gid}.html"
    net.save_graph(html_path)
    with open(html_path, "r", encoding="utf-8") as f:
        components.html(f.read(), height=int(height.replace("px", "")) + 20)


# ------------------------------------------------------------ tab: search -
with tab_search:
    col1, col2 = st.columns([1, 3])
    with col1:
        gid_input = st.number_input("Введите gid", min_value=0, step=1, value=int(nr.gid.iloc[0]))
        depth = st.slider("Глубина соседей (шагов)", 1, 3, 1)
        row = nr[nr.gid == gid_input]
        if not row.empty:
            r = row.iloc[0]
            st.markdown(f"**Роль:** `{r.role}`  \n**Уверенность:** {r.role_score:.2f}  \n"
                        f"**Кластер:** {r.cluster_id}  \n**Приоритет:** {r.priority_score:.3f}")
            st.markdown(f"**Обоснование:**  \n{r.evidence}")
        else:
            st.warning("gid не найден в nodes_roles.csv")
    with col2:
        draw_subgraph(int(gid_input), depth=depth)

# ---------------------------------------------------------------- tab: top-
with tab_top:
    st.subheader("Топ-лист приоритетов")
    top_n = st.slider("Показать первые N", 10, min(100, len(nr)), 30)
    show = nr.sort_values("priority_score", ascending=False).head(top_n)
    st.dataframe(
        show[["gid", "role", "role_score", "cluster_id", "priority_score", "evidence"]],
        use_container_width=True, height=500,
    )

# ----------------------------------------------------------- tab: clusters-
with tab_clusters:
    st.subheader("Кластеры")
    st.dataframe(cl.sort_values("n_nodes", ascending=False), use_container_width=True, height=400)
    cl_pick = st.selectbox("Показать кластер на схеме", cl.sort_values("n_nodes", ascending=False).cluster_id)
    members = nr[nr.cluster_id == cl_pick].gid.tolist()
    if members:
        # берём самый приоритетный узел кластера как центр отрисовки
        center = nr[nr.gid.isin(members)].sort_values("priority_score", ascending=False).gid.iloc[0]
        st.caption(f"Показан подграф вокруг самого приоритетного узла кластера (gid={center})")
        draw_subgraph(int(center), depth=2)

# ---------------------------------------------------------------- tab: full
with tab_full:
    st.subheader("Вся сеть (может быть тяжело при полном размере — используйте фильтр)")
    max_nodes = st.slider("Максимум узлов на схеме (по приоритету)", 50, min(500, len(nr)), 150)
    top_gids = set(nr.sort_values("priority_score", ascending=False).head(max_nodes).gid)
    fedges = edges[edges.src.isin(top_gids) & edges.dst.isin(top_gids)]

    net = Network(height="700px", width="100%", directed=True, bgcolor="#1e1e1e", font_color="white")
    net.barnes_hut()
    role_map = dict(zip(nr.gid, nr.role))
    for n in top_gids:
        role = role_map.get(n, "peripheral")
        is_seed = n in seed_ids
        net.add_node(int(n), label=str(n), color=ROLE_COLORS.get(role, "#999"),
                     size=20 if is_seed else 12, title=f"gid={n}\nrole={role}")
    for r in fedges.itertuples():
        net.add_edge(int(r.src), int(r.dst), value=max(1, r.sum_kzt / 1_000_000))
    net.set_options('{"physics": {"stabilization": {"iterations": 100}}}')
    net.save_graph("_full_graph.html")
    with open("_full_graph.html", "r", encoding="utf-8") as f:
        components.html(f.read(), height=720)

st.markdown("---")
st.caption("Легенда: " + "  ".join(f"🔴{'' if k!='coordinator' else ''} **{k}**" for k in ROLE_COLORS))
