#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Astar.py  ─  서울 지하철 1~5호선 A* 최단 경로 탐색기
  • CSV 폴더(기본: ./역간소요시간(수작업)/*.csv)에 주행 시간(m:ss) 데이터를 두면
    간선 가중치로 반영
  • 환승 페널티(기본 4분) 포함
  • [경로] 출력 시 구간별 소요 시간 표시
"""

import pandas as pd, glob, os, heapq
from collections import defaultdict, deque
# ─────────────────────────────────────────────────────────────
# 0.  지하철 역 리스트 (1~5호선)  ※ 분기/루프 구조 그대로
# ─────────────────────────────────────────────────────────────
# ── 1호선
line1_stations = {
    "trunk": [
        "소요산","동두천","보산","동두천중앙","지행","덕정","덕계","양주","녹양","가능","의정부",
        "회룡","망월사","도봉산","도봉","방학","창동","녹천","월계","성북","석계","신이문",
        "외대앞","회기","청량리","제기동","신설동","동묘앞","동대문","종로5가","종로3가","종각",
        "시청","서울역","남영","용산","노량진","대방","신길","영등포","신도림","구로",
        "가산디지털단지","독산","금천구청","석수","관악","안양","명학","금정","군포","당정",
        "의왕","성균관대","화서","수원","세류","병점","세마","오산대","오산","진위","송탄",
        "서정리","지제","평택","성환","직산","두정","천안","봉명","쌍용","아산","배방",
        "온양온천","신창"
    ],
    "incheon_branch": [
        "구로","구일","개봉","오류동","온수","역곡","소사","부천","중동","송내","부개",
        "부평","백운","동암","간석","주안","도화","제물포","도원","동인천","인천"
    ],
    "gwangmyeong_branch": ["금천구청","광명"],
    "seodongtan_branch": ["병점","세마","서동탄"]
}
# ── 2호선
line2_stations = {
    "main_loop": [
        "시청","을지로입구","을지로3가","을지로4가","동대문역사문화공원","신당","상왕십리",
        "왕십리","한양대","뚝섬","성수","건대입구","구의","강변","잠실나루","잠실","잠실새내",
        "종합운동장","삼성","선릉","역삼","강남","교대","서초","방배","사당","낙성대",
        "서울대입구","봉천","신림","신대방","구로디지털단지","대림","신도림","문래",
        "영등포구청","당산","합정","홍대입구","신촌","이대","아현","충정로"
    ],
    "seongsu_branch": ["성수","용답","신답","용두","신설동"],
    "sinjeong_branch": ["신도림","도림천","양천구청","신정네거리","까치산"]
}
# ── 3호선
line3_stations = [
    "대화","주엽","정발산","마두","백석","대곡","화정","원당","원흥","삼송","지축","구파발",
    "연신내","불광","녹번","홍제","무악재","독립문","경복궁","안국","종로3가","을지로3가",
    "충무로","동대입구","약수","금호","옥수","압구정","신사","잠원","고속터미널","교대",
    "남부터미널","양재","매봉","도곡","대치","학여울","대청","일원","수서","가락시장",
    "경찰병원","오금"
]
# ── 4호선
line4_stations = [
    "당고개","상계","노원","창동","쌍문","수유","미아","미아사거리","길음","성신여대입구",
    "한성대입구","혜화","동대문","동대문역사문화공원","충무로","명동","회현","서울역",
    "숙대입구","삼각지","신용산","이촌","동작","이수","사당","남태령","선바위","경마공원",
    "대공원","과천","정부과천청사","인덕원","평촌","범계","금정","산본","수리산","대야미",
    "반월","상록수","한대앞","중앙","고잔","초지","안산","신길온천","정왕","오이도"
]
# ── 5호선
line5_stations = {
    "main_line": [
        "방화","개화산","김포공항","송정","마곡","발산","우장산","화곡","까치산","신정","목동",
        "오목교","양평","영등포구청","영등포시장","신길","여의도","여의나루","마포","공덕","애오개",
        "충정로","서대문","광화문","종로3가","을지로4가","동대문역사문화공원","청구","신금호",
        "행당","왕십리","마장","답십리","장한평","군자","아차산","광나루","천호","강동",
        "길동","굽은다리","명일","고덕","상일동","강일","미사","하남풍산","하남시청","하남검단산"
    ],
    "macheon_branch": ["강동","둔촌동","올림픽공원","방이","오금","개롱","거여","마천"]
}
lines = {1: line1_stations, 2: line2_stations, 3: line3_stations, 4: line4_stations, 5: line5_stations}

# ───────────────────────────── 1. CSV → run_times ─────────────────────────────
def _parse_mmss(txt: str) -> float:
    """'m:ss' → 분(float)  ,  예외 시 None 반환"""
    try:
        m, s = map(int, txt.split(':'))
        return m + s / 60
    except Exception:
        return None

def load_run_times(folder="역간소요시간(수작업)"):
    run = {}
    for p in glob.glob(os.path.join(folder, "*.csv")):
        df = pd.read_csv(p, encoding="utf-8")
        prev = None
        for _, row in df.iterrows():
            cur = str(row["역명"]).strip()
            t_raw = str(row["시간(분)"]).strip()
            t_val = _parse_mmss(t_raw)
            if t_val is None:
                prev = cur
                continue
            if prev:
                run[(prev, cur)] = run[(cur, prev)] = t_val
            prev = cur
    return run

RUN_TIMES = load_run_times()

# ───────────────────────────── 2. 그래프 빌드 ─────────────────────────────
DEFAULT_TRAVEL = 2
TRANSFER_TIME  = 4

def build_graph(lines_dict, run_tbl):
    g = defaultdict(list)
    def add(u, v, w):
        g[u].append((w, v)); g[v].append((w, u))

    for ln, data in lines_dict.items():
        segments = data.values() if isinstance(data, dict) else [data]
        for key, seg in (data.items() if isinstance(data, dict) else [("linear", data)]):
            for a, b in zip(seg, seg[1:]):
                w = run_tbl.get((a,b), run_tbl.get((b,a), DEFAULT_TRAVEL))
                add((ln,a), (ln,b), w)
            if key.endswith("_loop"):
                w = run_tbl.get((seg[-1], seg[0]), run_tbl.get((seg[0], seg[-1]), DEFAULT_TRAVEL))
                add((ln,seg[-1]), (ln,seg[0]), w)

    station_to_lines = defaultdict(list)
    for ln, data in lines_dict.items():
        sts = (s for seg in data.values() for s in seg) if isinstance(data, dict) else data
        for st in sts:
            station_to_lines[st].append(ln)
    for st, lns in station_to_lines.items():
        for i in range(len(lns)):
            for j in range(i+1, len(lns)):
                add((lns[i],st), (lns[j],st), TRANSFER_TIME)
    return g

graph = build_graph(lines, RUN_TIMES)

# 역 이름 → 노드들
station_to_nodes = defaultdict(list)
for node in graph:
    station_to_nodes[node[1]].append(node)

# ───────────────────────────── 3. 휴리스틱 선계산 ─────────────────────────────
def precompute_hops(goal_names):
    nbr = defaultdict(set)
    for (ln,a), edges in graph.items():
        for _,(_,b) in edges:
            nbr[a].add(b); nbr[b].add(a)
    dist = {g:0 for g in goal_names}
    dq = deque(goal_names)
    while dq:
        cur = dq.popleft()
        for nxt in nbr[cur]:
            if nxt not in dist:
                dist[nxt] = dist[cur] + 1
                dq.append(nxt)
    return dist

MIN_EDGE = min(w for u in graph for w,_ in graph[u])

# ───────────────────────────── 4. A* ─────────────────────────────
def astar(start_name, goal_name):
    if start_name not in station_to_nodes or goal_name not in station_to_nodes:
        raise ValueError("역 이름이 데이터에 없습니다.")

    goals = set(station_to_nodes[goal_name])
    h_table = precompute_hops({g[1] for g in goals})
    h = lambda node: h_table.get(node[1], 0) * MIN_EDGE

    SUPER = ('S','')
    super_edges = []
    for n in station_to_nodes[start_name]:
        super_edges.append((0,n))
    graph[SUPER] = super_edges  # 한 번에 할당 → 중복 방지

    g_cost = {SUPER:0}
    parent = {SUPER:None}
    pq = []
    for w, n in super_edges:
        g_cost[n] = w
        parent[n] = SUPER
        heapq.heappush(pq, (w + h(n), w, n))

    try:
        while pq:
            f, g, u = heapq.heappop(pq)
            if u in goals:
                path = []
                cur = u
                while cur and cur != SUPER:
                    path.append(cur)
                    cur = parent[cur]
                return path[::-1], g
            if g > g_cost[u]:
                continue
            for w, v in graph[u]:
                ng = g + w
                if ng < g_cost.get(v, float('inf')):
                    g_cost[v] = ng
                    parent[v] = u
                    heapq.heappush(pq, (ng + h(v), ng, v))
        return None, float('inf')
    finally:
        graph.pop(SUPER, None)   # 슈퍼소스 깨끗이 제거 (다음 호출 대비)

# ───────────────────────────── 5. 출력 보조 ─────────────────────────────
def edge_time(u,v):
    for w,n in graph[u]:
        if n == v: return w
    return 0

def fmt_path(path):
    segs=[]
    for i,(ln,st) in enumerate(path):
        if i==0: segs.append(f"{ln}호선 {st}")
        else:    segs.append(f" --{edge_time(path[i-1],path[i]):.1f}분→ {ln}호선 {st}")
    return "".join(segs)

# ───────────────────────────── 6. CLI ─────────────────────────────
if __name__ == "__main__":
    try:
        s = input("출발역 이름: ").strip()
        g = input("도착역 이름: ").strip()
        path, total = astar(s, g)
        if path:
            print("\n[경로]\n"+fmt_path(path))
            print(f"\n[총 소요 시간] {total:.1f}분  (환승 {TRANSFER_TIME}분 포함)")
        else:
            print("❌ 경로를 찾지 못했습니다.")
    except Exception as e:
        print("⚠️ 오류:", e)