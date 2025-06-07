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
    "진접광릉숲", "오남역", "풍양역", "별내별가람역", "당고개","상계","노원","창동","쌍문","수유","미아","미아사거리","길음","성신여대입구",
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
DEFAULT_TRAVEL    = 2        # 기본 주행 시간 (분)
FALLBACK_TRANSFER = 4        # 기본 환승 시간 (분)
DWELL             = 0.5      # 정차 시간 (분)
RUN_DIR           = "dataset/역간소요시간(수작업)"
TRANSFER_CSV_PATH = "dataset/서울교통공사_환승역거리 소요시간 정보_20250331.csv"
DISTANCE_CSV_PATH = "dataset/국가철도공단_코레일 역간거리_20240430.csv"

# ───────────────────────────── 헬퍼 함수 ─────────────────────────────
def _parse_mmss(txt: str) -> float:
    try:
        m, s = map(int, txt.split(':'))
        return m + s / 60
    except:
        return None

# 수작업 주행 시간 로드
def load_run_times(folder="dataset/역간소요시간(수작업)"):
    """
    폴더 내 CSV 모두 읽어, 인접역 간 주행 시간을 분 단위 dict 로 생성.
    CSV 내부의 '시간(분)' 컬럼 이름에 공백 등이 있을 수 있으므로
    컬럼 이름을 strip() 해서 접근합니다.
    """
    run = {}
    for p in glob.glob(os.path.join(folder, "*.csv")):
        df = pd.read_csv(p, encoding="utf-8")
        # 컬럼 이름 공백 제거
        df.columns = [c.strip() for c in df.columns]
        prev = None
        for _, row in df.iterrows():
            cur = str(row["역명"]).strip()
            # '시간(분)' 컬럼 접근
            t_val = _parse_mmss(str(row.get("시간(분)", "")).strip())
            if t_val is None:
                prev = cur
                continue
            if prev:
                run[(prev, cur)] = run[(cur, prev)] = t_val
            prev = cur
    return run

# 환승 시간 로드
def load_transfer_times(csv_path=TRANSFER_CSV_PATH):
    tbl = {}
    df = pd.read_csv(csv_path, encoding="cp949")
    for _, row in df.iterrows():
        ln1 = int(row["호선"])
        st  = str(row["환승역명"]).strip()
        digits = ''.join(filter(str.isdigit, str(row["환승노선"])))
        if not digits: continue
        ln2 = int(digits)
        t = _parse_mmss(str(row["환승소요시간"]).strip())
        if t is None: continue
        tbl[((ln1, st), (ln2, st))] = tbl[((ln2, st), (ln1, st))] = t
    return tbl

def load_distance_times(csv_path=DISTANCE_CSV_PATH):
    """
    CSV 컬럼: 철도운영기관명, 선명, 역명, 역간거리(km)
    → 원래의 km 값 자체를 dist_tbl[(A,B)]에 저장
    """
    df = pd.read_csv(csv_path, encoding="cp949")
    # '코레일'만 필터링 (칼럼이 있다면)
    if "철도운영기관명" in df.columns:
        df = df[df["철도운영기관명"].str.contains("코레일", na=False)]

    dist_tbl = {}
    # 선명별로 순차 처리
    for _, grp in df.groupby("선명"):
        stations = grp["역명"].tolist()
        kms      = grp["역간거리"].tolist()
        # 인접역 간 km 그대로 저장
        for a, b, km in zip(stations, stations[1:], kms[1:]):
            dist_tbl[(a, b)] = dist_tbl[(b, a)] = float(km)

    return dist_tbl


# ───────────────────────────── 1. CSV 로드 및 그래프 빌드 ─────────────────────────────
RUN_TIMES      = load_run_times()
TRANSFER_TIMES = load_transfer_times()
DISTANCE_RUN   = load_distance_times()

def build_graph(lines_dict, run_tbl, transfer_tbl):
    g = defaultdict(list)
    def add(u, v, w):
        g[u].append((w, v)); g[v].append((w, u))
    # 주행 간선
    for ln, data in lines_dict.items():
        segments = data.values() if isinstance(data, dict) else [data]
        for key, seg in (data.items() if isinstance(data, dict) else [(None, data)]):
            for a, b in zip(seg, seg[1:]):
                base = run_tbl.get((a, b), run_tbl.get((b, a), DEFAULT_TRAVEL))
                add((ln, a), (ln, b), base + DWELL)
            if key and key.endswith("_loop"):
                a, b = seg[-1], seg[0]
                base = run_tbl.get((a, b), run_tbl.get((b, a), DEFAULT_TRAVEL))
                add((ln, a), (ln, b), base + DWELL)
    # 환승 간선
    st2ln = defaultdict(list)
    for ln, data in lines_dict.items():
        sts = (s for seg in data.values() for s in seg) if isinstance(data, dict) else data
        for st in sts:
            st2ln[st].append(ln)
    for st, lns in st2ln.items():
        for i in range(len(lns)):
            for j in range(i+1, len(lns)):
                u, v = (lns[i], st), (lns[j], st)
                w     = transfer_tbl.get((u, v), FALLBACK_TRANSFER)
                add(u, v, w)
    return g

# 거리 기반 그래프: dwell 없이, transfer weight = 0

def build_distance_graph(lines_dict, dist_tbl):
    """
    순수 역간거리(km)만을 간선 가중치로 쓰는 그래프 생성.
    환승 간선은 비용 0으로 처리합니다.
    """
    g = defaultdict(list)
    def add(u, v, w):
        g[u].append((w, v))
        g[v].append((w, u))

    # 1) 인접역 간 거리(km) 간선
    for ln, data in lines_dict.items():
        segments = data.values() if isinstance(data, dict) else [data]
        for seg in (data.values() if isinstance(data, dict) else [data]):
            for a, b in zip(seg, seg[1:]):
                w = dist_tbl.get((a, b), 0.0)
                add((ln, a), (ln, b), w)

    # 2) 환승 간선: 비용 0
    station_to_lines = defaultdict(list)
    for ln, data in lines_dict.items():
        sts = (s for seg in (data.values() if isinstance(data, dict) else [data]) for s in seg)
        for st in sts:
            station_to_lines[st].append(ln)

    for st, lns in station_to_lines.items():
        for i in range(len(lns)):
            for j in range(i + 1, len(lns)):
                add((lns[i], st), (lns[j], st), 0.0)

    return g

# 그래프 생성
graph          = build_graph(lines, RUN_TIMES, TRANSFER_TIMES)
graph_distance = build_distance_graph(lines, DISTANCE_RUN)
station_to_nodes = defaultdict(list)
for node in graph:
    station_to_nodes[node[1]].append(node)

# ───────────────────────────── 2. Dijkstra helpers ─────────────────────────────

def dijkstra_multi(graph_obj, src_nodes):
    dist = {n: 0.0 for n in src_nodes}
    pq = [(0.0, n) for n in src_nodes]
    heapq.heapify(pq)
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]: continue
        for w, v in graph_obj[u]:
            nd = d + w
            if nd < dist.get(v, float('inf')):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist

# ───────────────────────────── 3. A* 탐색 (거리 기반 휴리스틱) ─────────────────────────────

def astar(start_name, goal_name):
    if start_name not in station_to_nodes or goal_name not in station_to_nodes:
        raise ValueError("역 이름이 데이터에 없습니다.")
    starts = station_to_nodes[start_name]
    goals  = station_to_nodes[goal_name]

    # 3-1) distance_graph 에서 목표역으로 multi-source Dijkstra → heuristic table
    dist_goal = dijkstra_multi(graph_distance, goals)
    def h(u):
        return dist_goal.get(u, 0.0)

    # 3-2) 실제 탐색용 graph with run_times + dwell + transfer
    SUPER = ('S', '')
    graph[SUPER] = [(0.0, n) for n in starts]
    g_cost = {SUPER: 0.0}
    parent = {SUPER: None}
    pq = []
    for w, n in graph[SUPER]:
        g_cost[n] = w; parent[n] = SUPER
        heapq.heappush(pq, (w + h(n), w, n))
    try:
        while pq:
            f, g_acc, u = heapq.heappop(pq)
            if u in goals:
                path = []
                while u and u != SUPER:
                    path.append(u); u = parent[u]
                return path[::-1], g_acc
            if g_acc > g_cost[u]: continue
            for w, v in graph[u]:
                ng = g_acc + w
                if ng < g_cost.get(v, float('inf')):
                    g_cost[v] = ng; parent[v] = u
                    heapq.heappush(pq, (ng + h(v), ng, v))
        return None, float('inf')
    finally:
        graph.pop(SUPER, None)

# ───────────────────────────── 5. 출력 보조 ─────────────────────────────
def edge_time(u, v):
    for w, n in graph[u]:
        if n == v:
            return w
    return 0


def fmt_path(path):
    segs = []
    for i, (ln, st) in enumerate(path):
        if i == 0:
            segs.append(f"{ln}호선 {st}")
        else:
            prev = path[i - 1]
            t = edge_time(prev, (ln, st))
            segs.append(f" --{t:.1f}분→ {ln}호선 {st}")
    return "".join(segs)


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
        path, total = astar(s, g)
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
        path, total = astar(s, g)
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