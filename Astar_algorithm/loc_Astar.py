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
TRANSFER_CSV_PATH = "Astar_algorithm/서울교통공사_환승역거리 소요시간 정보_20250331.csv"
COORD_CSV_PATH    = "Astar_algorithm/station_location/subway_1to5_master.csv"


# ───────────────────────────── 헬퍼 함수 ─────────────────────────────
def _parse_mmss(txt: str) -> float:
    """'m:ss' → 분(float), 파싱 실패 시 None 반환"""
    try:
        m, s = map(int, txt.split(':'))
        return m + s / 60
    except Exception:
        return None


def load_run_times(folder="역간소요시간(수작업)"):
    """
    CSV 폴더 내 모든 파일을 읽어, 인접역 간 주행 시간을 분(float)으로 반환하는 dict
    키: (역A, 역B), 값: float(분)
    """
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


def load_transfer_times(csv_path=TRANSFER_CSV_PATH):
    """
    환승 CSV를 읽어서, ((ln1, station), (ln2, station)) 쌍에 대한 환승 시간을 분(float)으로 반환하는 dict 생성
    CSV 컬럼: 호선 (int), 환승역명 (str), 환승노선 (str), 환승소요시간 (m:ss)
    """
    tbl = {}
    try:
        df = pd.read_csv(csv_path, encoding="cp949")
    except Exception as e:
        raise FileNotFoundError(f"환승 CSV 파일을 읽을 수 없습니다: {e}")

    for _, row in df.iterrows():
        ln1 = int(row["호선"])
        station = str(row["환승역명"]).strip()
        transfer_line_str = str(row["환승노선"]).strip()
        # '4호선' → '4'
        digits = "".join(filter(str.isdigit, transfer_line_str))
        if not digits:
            continue
        ln2 = int(digits)
        t_val = _parse_mmss(str(row["환승소요시간"]).strip())
        if t_val is None:
            continue
        tbl[((ln1, station), (ln2, station))] = t_val
        tbl[((ln2, station), (ln1, station))] = t_val
    return tbl


# ───────────────────────────── 0-5호선 전체 좌표 로딩 ────────────────────────────
def load_coords(csv_path=COORD_CSV_PATH):
    """
    subway_1to5_master.csv를 읽어, (호선_int, 역명) → (위도, 경도) 딕셔너리 반환
    CSV 컬럼: ['line','station_name','latitude','longitude']
    """
    coords = {}
    df = pd.read_csv(csv_path, encoding="utf-8")
    for _, row in df.iterrows():
        # CSV의 'line' 컬럼 값 예: '1호선', '2호선' 등
        line_str = str(row["line"]).strip()
        if not line_str.endswith("호선"):
            continue
        line_int = int(line_str.replace("호선", ""))
        station = str(row["station_name"]).strip()
        lat = float(row["latitude"])
        lng = float(row["longitude"])
        coords[(line_int, station)] = (lat, lng)
    return coords

# 하버사인 공식 (두 위·경도 간 직선 거리, 단위: km)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # 지구 반지름 (km)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c  # 결과: 킬로미터 단위 직선 거리

# 좌표 딕셔너리 전역 생성
COORDS = load_coords()


# ───────────────────────────── 1. CSV → 데이터 테이블 ─────────────────────────────
RUN_TIMES      = load_run_times()
TRANSFER_TIMES = load_transfer_times()


# ───────────────────────────── 2. 그래프 빌드 ─────────────────────────────
def build_graph(lines_dict, run_tbl, transfer_tbl):
    """
    lines_dict: {호선번호: [역목록] 또는 {key: [역목록], …}} 형식
    run_tbl:   인접역 주행 시간 딕셔너리
    transfer_tbl: 환승 시간 딕셔너리
    → (노드=(호선, 역이름), 가중치, 노드)형태로 양방향 그래프 구성
      • 주행 간선: run_tbl[(A,B)] + DWELL
      • 환승 간선: transfer_tbl[(…)], 없으면 FALLBACK_TRANSFER
    """
    g = defaultdict(list)

    def add(u, v, w):
        g[u].append((w, v))
        g[v].append((w, u))

    # 2-1) 노선별 인접역(주행) 간선 추가 (정차시간 DWELL 포함)
    for ln, data in lines_dict.items():
        segments = data.values() if isinstance(data, dict) else [data]
        for key, seg in (data.items() if isinstance(data, dict) else [("linear", data)]):
            for a, b in zip(seg, seg[1:]):
                base_time = run_tbl.get((a, b), run_tbl.get((b, a), DEFAULT_TRAVEL))
                w = base_time + DWELL
                add((ln, a), (ln, b), w)
            # 루프(순환선) 구간
            if key.endswith("_loop"):
                a, b = seg[-1], seg[0]
                base_time = run_tbl.get((a, b), run_tbl.get((b, a), DEFAULT_TRAVEL))
                w = base_time + DWELL
                add((ln, a), (ln, b), w)

    # 2-2) 역 이름 기준, 속한 노선 목록 생성
    station_to_lines = defaultdict(list)
    for ln, data in lines_dict.items():
        sts = (s for seg in data.values() for s in seg) if isinstance(data, dict) else data
        for st in sts:
            station_to_lines[st].append(ln)

    # 2-3) 환승 간선 추가 (CSV 기준, 없으면 FALLBACK_TRANSFER)
    #     환승 간선에는 별도의 “정차 시간”을 추가하지 않음 (이미 주행 간선에 DWELL이 포함돼 있음)
    for st, lns in station_to_lines.items():
        for i in range(len(lns)):
            for j in range(i + 1, len(lns)):
                ln_i = lns[i]
                ln_j = lns[j]
                u = (ln_i, st)
                v = (ln_j, st)
                w = transfer_tbl.get((u, v), FALLBACK_TRANSFER)
                add(u, v, w)

    return g


# 최종 그래프 생성
graph = build_graph(lines, RUN_TIMES, TRANSFER_TIMES)

# 역 이름 → 해당 노드 리스트 (ex: "서울역" → [(1,"서울역"), (4,"서울역")])
station_to_nodes = defaultdict(list)
for node in graph:
    station_to_nodes[node[1]].append(node)


# ───────────────────────────── 3. 휴리스틱 선계산 (홉 수 기반) ─────────────────────────────
def precompute_hops(goal_names):
    nbr = defaultdict(set)
    for (ln, a), edges in graph.items():
        for _, (_, b) in edges:
            nbr[a].add(b)
            nbr[b].add(a)
    dist = {g: 0 for g in goal_names}
    dq = deque(goal_names)
    while dq:
        cur = dq.popleft()
        for nxt in nbr[cur]:
            if nxt not in dist:
                dist[nxt] = dist[cur] + 1
                dq.append(nxt)
    return dist

MIN_EDGE = min(w for u in graph for w, _ in graph[u])


# ───────────────────────────── 4. A* 탐색 ─────────────────────────────
def astar(start_name, goal_name):
    if start_name not in station_to_nodes or goal_name not in station_to_nodes:
        raise ValueError("역 이름이 데이터에 없습니다.")

    # 목표 노드 집합
    goals = set(station_to_nodes[goal_name])

    # (1) 홉 수 기반 테이블: 후순위 휴리스틱용
    h_table = precompute_hops({g[1] for g in goals})

    # (2) 위경도 기반 휴리스틱 함수
    def h(node):
        """
        node: (호선_int, 역명)
        goals 중 임의의 하나(역명 동일)에 대한 좌표를 사용해 직선 거리 계산
        """
        ln_u, st_u = node
        # 해당 노드 좌표가 없으면, 홉 수 기반으로 대체
        if (ln_u, st_u) not in COORDS:
            # 홉 수 기반: 남은 홉 수 × 최소 가중치
            return h_table.get(st_u, 0) * MIN_EDGE

        # 목표 노드들 중, 같은 역명(goal_name)만 골라서 좌표 비교
        # (역명은 unique하다고 가정하므로, 사실 goals 안에는 같은 역명 노드가 한두 개)
        # 두 노드(ln_v, st_v) 모두의 좌표를 구해 최소 직선거리를 선택
        best_dist = float('inf')
        lat1, lon1 = COORDS[(ln_u, st_u)]
        for ln_v, st_v in goals:
            if st_v != st_u and (ln_v, st_v) in COORDS:
                lat2, lon2 = COORDS[(ln_v, st_v)]
                d_km = haversine(lat1, lon1, lat2, lon2)
                best_dist = min(best_dist, d_km)
            elif st_v == st_u:
                # 이미 같은 역명(환승 시), 거리 0
                best_dist = 0
                break

        # 최대 지하철 속도를 30 km/h라 가정 → 시간(분) = (거리(km) / 30) * 60
        # 만약 best_dist가 inf라면(좌표 누락 등), 홉 수 기반으로 대체
        if best_dist == float('inf'):
            return h_table.get(st_u, 0) * MIN_EDGE
        return best_dist / 30 * 60

    # 슈퍼 소스 생성
    SUPER = ('S', '')
    super_edges = []
    for n in station_to_nodes[start_name]:
        super_edges.append((0, n))
    graph[SUPER] = super_edges

    g_cost = {SUPER: 0}
    parent = {SUPER: None}
    pq = []
    for w, n in super_edges:
        g_cost[n] = w
        parent[n] = SUPER
        heapq.heappush(pq, (w + h(n), w, n))

    try:
        while pq:
            f, g_acc, u = heapq.heappop(pq)
            if u in goals:
                path = []
                cur = u
                while cur and cur != SUPER:
                    path.append(cur)
                    cur = parent[cur]
                return path[::-1], g_acc
            if g_acc > g_cost[u]:
                continue
            for w, v in graph[u]:
                ng = g_acc + w
                if ng < g_cost.get(v, float('inf')):
                    g_cost[v] = ng
                    parent[v] = u
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