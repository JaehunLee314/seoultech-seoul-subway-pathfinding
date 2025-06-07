import pandas as pd, glob, os, heapq, math
from collections import defaultdict, deque
import time
import numpy as np

# ─────────────────────────────────────────────────────────────
# 0.  지하철 역 리스트 (1~5호선)
# ─────────────────────────────────────────────────────────────
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
line3_stations = [
    "대화","주엽","정발산","마두","백석","대곡","화정","원당","원흥","삼송","지축","구파발",
    "연신내","불광","녹번","홍제","무악재","독립문","경복궁","안국","종로3가","을지로3가",
    "충무로","동대입구","약수","금호","옥수","압구정","신사","잠원","고속터미널","교대",
    "남부터미터널","양재","매봉","도곡","대치","학여울","대청","일원","수서","가락시장",
    "경찰병원","오금"
]
line4_stations = [
    "당고개","상계","노원","창동","쌍문","수유","미아","미아사거리","길음","성신여대입구",
    "한성대입구","혜화","동대문","동대문역사문화공원","충무로","명동","회현","서울역",
    "숙대입구","삼각지","신용산","이촌","동작","이수","사당","남태령","선바위","경마공원",
    "대공원","과천","정부과천청사","인덕원","평촌","범계","금정","산본","수리산","대야미",
    "반월","상록수","한대앞","중앙","고잔","초지","안산","신길온천","정왕","오이도"
]
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
lines = {1: line1_stations, 2: line2_stations, 3: line3_stations,
         4: line4_stations, 5: line5_stations}


# ───────────────────────────── 공통 상수 ─────────────────────────────
DEFAULT_TRAVEL    = 2        # 역간 데이터 없을 때 기본 주행 시간 (분)
FALLBACK_TRANSFER = 4        # 환승 CSV에 없을 경우 사용할 기본 환승 시간 (분)
DWELL             = 0.5      # 각 정거장 도착 시 정차 시간: 0.5분 (=30초)
RUN_DIR           = "역간소요시간(수작업)"
TRANSFER_CSV_PATH = "Astar_algorithm/서울교통공사_환승역거리 소요시간 정보_20250331.csv"

###############################################################################
# 1. CSV 로드 · 그래프 빌드 · 보조 테이블                                       #
###############################################################################
def _parse_mmss(txt: str):
    try:
        m, s = map(int, txt.split(":"))
        return m + s / 60
    except Exception:
        return None


def load_run_times(folder=RUN_DIR) -> dict[tuple[str, str], float]:
    run = {}
    for p in glob.glob(os.path.join(folder, "*.csv")):
        df = pd.read_csv(p, encoding="utf-8")
        prev = None
        for _, row in df.iterrows():
            cur = str(row["역명"]).strip()
            t = _parse_mmss(str(row["시간(분)"]).strip())
            if t is None:
                prev = cur
                continue
            if prev:
                run[(prev, cur)] = run[(cur, prev)] = t
            prev = cur
    return run


def load_transfer_times(csv_path=TRANSFER_CSV_PATH):
    tbl = {}
    if not os.path.exists(csv_path):
        return tbl
    df = pd.read_csv(csv_path, encoding="cp949")
    for _, row in df.iterrows():
        ln1 = int(row["호선"])
        st = str(row["환승역명"]).strip()
        ln2_str = "".join(filter(str.isdigit, str(row["환승노선"])))
        if not ln2_str:
            continue
        ln2 = int(ln2_str)
        t = _parse_mmss(str(row["환승소요시간"]).strip())
        if t is None:
            continue
        tbl[((ln1, st), (ln2, st))] = tbl[((ln2, st), (ln1, st))] = t
    return tbl


def load_coords(folder_path="Astar_algorithm/station_location"):
    coords = {}
    for file in glob.glob(os.path.join(folder_path, "*.csv")):
        try:
            df = pd.read_csv(file, encoding="utf-8")
        except Exception as e:
            print(f"⚠️ CSV 파일 {file} 열기 실패: {e}")
            continue

        for _, row in df.iterrows():
            try:
                line_str = str(row["line"]).strip()
                if not line_str.endswith("호선"):
                    continue
                ln = int(line_str.replace("호선", ""))
                station = str(row["station_name"]).strip()
                lat = float(row["latitude"])
                lon = float(row["longitude"])
                coords[(ln, station)] = (lat, lon)
            except Exception:
                continue  # 형식이 맞지 않는 row는 스킵
    return coords



def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))  # km


def build_graph(lines_dict, run_tbl, transfer_tbl):
    g = defaultdict(list)

    def add(u, v, w):
        g[u].append((w, v))
        g[v].append((w, u))

    # 주행 간선
    for ln, data in lines_dict.items():
        segments = data.values() if isinstance(data, dict) else [data]
        for key, seg in (data.items() if isinstance(data, dict)
                         else [("linear", data)]):
            for a, b in zip(seg, seg[1:]):
                base = run_tbl.get((a, b), run_tbl.get((b, a), DEFAULT_TRAVEL))
                add((ln, a), (ln, b), base + DWELL)
            if key.endswith("_loop"):
                a, b = seg[-1], seg[0]
                base = run_tbl.get((a, b), run_tbl.get((b, a), DEFAULT_TRAVEL))
                add((ln, a), (ln, b), base + DWELL)

    # 환승 간선
    st_to_lines = defaultdict(list)
    for ln, data in lines_dict.items():
        sts = (s for seg in data.values() for s in seg) \
            if isinstance(data, dict) else data
        for st in sts:
            st_to_lines[st].append(ln)
    for st, lns in st_to_lines.items():
        for i in range(len(lns)):
            for j in range(i + 1, len(lns)):
                u, v = (lns[i], st), (lns[j], st)
                w = transfer_tbl.get((u, v), FALLBACK_TRANSFER)
                add(u, v, w)
    return g


RUN_TIMES = load_run_times()
TRANSFER_TIMES = load_transfer_times()
COORDS = load_coords()
graph = build_graph(lines, RUN_TIMES, TRANSFER_TIMES)

station_to_nodes = defaultdict(list)
for n in graph:
    station_to_nodes[n[1]].append(n)

MIN_EDGE = min(w for u in graph for w, _ in graph[u])

###############################################################################
# 2.  휴리스틱 · 중간 허브 선정                                               #
###############################################################################
def heuristic(u: tuple[int, str], v: tuple[int, str]) -> float:
    if u not in COORDS or v not in COORDS:
        return MIN_EDGE
    lat1, lon1 = COORDS[u]
    lat2, lon2 = COORDS[v]
    return haversine(lat1, lon1, lat2, lon2) / 30 * 60  # 30 km/h → 분

def select_mid_nodes(start: str, goal: str) -> list[tuple[int, str]]:
    """출발·도착의 중간 지점 좌표에서 가장 가까운 역의 노드들을 반환 (좌표 없는 경우 근처 역 fallback)"""
    s_coords = [COORDS[n] for n in station_to_nodes[start] if n in COORDS]
    g_coords = [COORDS[n] for n in station_to_nodes[goal] if n in COORDS]

    if not s_coords or not g_coords:
        raise ValueError("출발 또는 도착역의 좌표 정보가 없습니다.")

    s_xy = np.mean(s_coords, axis=0)
    g_xy = np.mean(g_coords, axis=0)
    mid_lat, mid_lon = (s_xy[0] + g_xy[0]) / 2, (s_xy[1] + g_xy[1]) / 2

    # 중간점과 가장 가까운 좌표 보유 노드를 찾는다
    best, best_node = float('inf'), None
    for node in graph:
        if node in COORDS:
            d = haversine(mid_lat, mid_lon, *COORDS[node])
            if d < best:
                best = d
                best_node = node

    # 선택된 노드의 역 이름과 연결된 모든 노드를 반환
    if best_node:
        return station_to_nodes[best_node[1]]
    else:
        raise ValueError("허브 후보 노드에 대한 좌표 정보를 찾을 수 없습니다.")


###############################################################################
# 3.  다중-소스 Dijkstra (허브용)                                             #
###############################################################################
def dijkstra_multi(src_nodes):
    dist = {n: 0.0 for n in src_nodes}
    parent = {n: None for n in src_nodes}
    pq = [(0.0, n) for n in src_nodes]
    heapq.heapify(pq)

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for w, v in graph[u]:
            nd = d + w
            if nd < dist.get(v, float('inf')):
                dist[v] = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))
    return dist, parent

###############################################################################
# 4.  세 방향 동시 탐색 (F-A*, B-A*, H-Dijkstra)                              #
###############################################################################
def tri_concurrent_path(start: str, goal: str):
    if start not in station_to_nodes or goal not in station_to_nodes:
        raise ValueError("역 이름이 데이터에 없습니다.")

    # 허브-Dijkstra 준비
    hub_src = select_mid_nodes(start, goal)
    pq_h = [(0.0, n) for n in hub_src]
    g_h = {n: 0.0 for n in hub_src}
    parent_h = {n: None for n in hub_src}

    # forward / backward A*
    starts, goals = station_to_nodes[start], station_to_nodes[goal]
    pq_f, pq_b = [], []
    g_f, g_b = {}, {}
    parent_f, parent_b = {}, {}

    for n in starts:
        g_f[n] = 0.0
        parent_f[n] = None
        heapq.heappush(pq_f, (heuristic(n, goals[0]), 0.0, n))
    for n in goals:
        g_b[n] = 0.0
        parent_b[n] = None
        heapq.heappush(pq_b, (heuristic(n, starts[0]), 0.0, n))

    # 노드 소속 집합
    owner = defaultdict(set)
    for n in hub_src:
        owner[n].add('H')

    meet = None
    best_cost = float('inf')

    # ---------- 메인 루프 ----------
    while pq_f or pq_b or pq_h:
        # helper
        def step(label):
            nonlocal meet, best_cost
            if label == 'F' and pq_f:
                f, gcur, u = heapq.heappop(pq_f)
                if gcur != g_f[u]:
                    return
            elif label == 'B' and pq_b:
                f, gcur, u = heapq.heappop(pq_b)
                if gcur != g_b[u]:
                    return
            elif label == 'H' and pq_h:
                gcur, u = heapq.heappop(pq_h)
                if gcur != g_h[u]:
                    return
            else:
                return  # 해당 큐 비어있음

            owner[u].add(label)

            # 만남 판정
            if owner[u] == {'F', 'B'} or owner[u] == {'F', 'B', 'H'}:
                meet = u
                best_cost = g_f.get(u, 0) + g_b.get(u, 0)
                raise StopIteration

            # relax
            for w, v in graph[u]:
                ng = gcur + w
                if label == 'F':
                    if ng < g_f.get(v, float('inf')):
                        g_f[v] = ng
                        parent_f[v] = u
                        heapq.heappush(pq_f,
                                       (ng + heuristic(v, goals[0]), ng, v))
                elif label == 'B':
                    if ng < g_b.get(v, float('inf')):
                        g_b[v] = ng
                        parent_b[v] = u
                        heapq.heappush(pq_b,
                                       (ng + heuristic(v, starts[0]), ng, v))
                else:  # 'H'
                    if ng < g_h.get(v, float('inf')):
                        g_h[v] = ng
                        parent_h[v] = u
                        heapq.heappush(pq_h, (ng, v))
                        owner[v].add('H')  # ✅ 이 줄을 여기에 추가

        try:
            step('F')
            step('B')
            step('H')
        except StopIteration:
            break

    if meet is None:
        return None, float('inf')

    # ---------- 경로 복원 ----------
    def back_chain(pmap, node):
        out = []
        while node is not None:
            out.append(node)
            node = pmap.get(node)
        return out[::-1]

    path_f = back_chain(parent_f, meet)
    path_b = back_chain(parent_b, parent_b.get(meet)) if meet in parent_b else []

    return path_f + path_b, best_cost


# ───────────────────────────── 출력 보조 ─────────────────────────────
def edge_time(u,v):
    for w,n in graph[u]:
        if n==v: return w
    return 0

def fmt_path(path:list[tuple[int,str]]):
    seg=[]
    for i,(ln,st) in enumerate(path):
        if i==0: seg.append(f"{ln}호선 {st}")
        else:
            seg.append(f" --{edge_time(path[i-1],(ln,st)):.1f}분→ {ln}호선 {st}")
    return ''.join(seg)

def analysing_alogorithm(s, g, repeats=4000):
    timelist = []
    summary = {"total_time":0,
               "best_time" :0,
               "worst_time":0,
               "mean_time" :0,
               "std_time"  :0
                }
    for i in range(4000):
        t0, t1 = 0, 0
        t0 = time.perf_counter()
        path, total = tri_concurrent_path(s, g)
        t1 = time.perf_counter()
        elapsed_time = t1-t0
        timelist.append(elapsed_time)

    summary["total_time"] = sum(timelist)
    summary["best_time"] = min(timelist)
    summary["worst_time"] = max(timelist)
    summary["mean_time"] = np.mean(timelist)
    summary["std_time"] = np.std(timelist)

    return summary

# ───────────────────────────── 6. CLI ─────────────────────────────
if __name__ == "__main__":
    try:
        s = input("출발역 이름: ").strip()
        g = input("도착역 이름: ").strip()
        t0 = time.perf_counter()
        path, total = tri_concurrent_path(s, g)
        t1 = time.perf_counter()

        if path:
            print("\n[경로]\n" + fmt_path(path))
            print(f"\n[총 소요 시간] {total:.1f}분  (정차 30초 포함)")
            print(f"[연산 시간] {t1 - t0:.6f}초")
            print("4000회 실행 결과")
            summary = analysing_alogorithm(s, g, 4000)
            print(f"  총합      : {summary['total_time']:.6f}초")
            print(f"  평균      : {summary['mean_time']:.6f}초")
            print(f"  표준편차  : {summary['std_time']:.6f}초")
            print(f"  bestcase : {summary['best_time']:.6f}초")
            print(f"  worstcase : {summary['worst_time']:.6f}초")
            
        else:
            print("❌ 경로를 찾지 못했습니다.")
            print(f"[연산 시간] {t1 - t0:.6f}초")
    except Exception as e:
        print("⚠️ 오류:", e)