import pandas as pd, glob, os, heapq, time
import numpy as np
from collections import defaultdict, deque
import random

# 1. Python random
random.seed(42)
# 2. Numpy random
np.random.seed(42)
# 3. (선택) hash 랜덤성 고정 (Python >= 3.3 이상)
os.environ["PYTHONHASHSEED"] = "42"

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
DEFAULT_TRAVEL    = 2        # 역간 데이터 없을 때 기본 주행 시간 (분)
FALLBACK_TRANSFER = 4        # 환승 CSV에 없을 경우 기본 환승 시간 (분)
DWELL             = 0.5      # 정차 시간 0.5분 (=30초)
TRANSFER_CSV_PATH = "dataset/서울교통공사_환승역거리 소요시간 정보_20250331.csv"

# ───────────────────────────── 헬퍼 함수 ─────────────────────────────
def _parse_mmss(txt: str) -> float:
    """'m:ss' → 분(float). 파싱 실패 시 None."""
    try:
        m, s = map(int, txt.split(':'))
        return m + s / 60
    except Exception:
        return None


def load_run_times(folder="dataset/역간소요시간(수작업)"):
    """
    폴더 내 CSV 모두 읽어, 인접역 간 주행 시간을 분 단위 dict 로 생성.
    키: (역A, 역B), 값: float 분
    """
    run = {}
    for p in glob.glob(os.path.join(folder, "*.csv")):
        df = pd.read_csv(p, encoding="utf-8")
        prev = None
        for _, row in df.iterrows():
            cur = str(row["역명"]).strip()
            t_val = _parse_mmss(str(row["시간(분)"]).strip())
            if t_val is None:
                prev = cur
                continue
            if prev:
                run[(prev, cur)] = run[(cur, prev)] = t_val
            prev = cur
    return run


def load_transfer_times(csv_path=TRANSFER_CSV_PATH):
    """
    환승 CSV → ((ln1, station), (ln2, station)) ↔ 환승 시간(분) dict
    """
    tbl = {}
    df = pd.read_csv(csv_path, encoding="cp949")

    for _, row in df.iterrows():
        ln1 = int(row["호선"])
        st  = str(row["환승역명"]).strip()

        # '2호선' 같은 문자열에서 숫자만 추출
        digits = "".join(filter(str.isdigit, str(row["환승노선"])))
        if not digits:           # 숫자가 없으면 건너뜀
            continue
        ln2 = int(digits)

        t_val = _parse_mmss(str(row["환승소요시간"]).strip())
        if t_val is None:
            continue

        tbl[((ln1, st), (ln2, st))] = t_val
        tbl[((ln2, st), (ln1, st))] = t_val

    return tbl

# ───────────────────────────── 1. CSV → 데이터 테이블 ─────────────────────────────
RUN_TIMES      = load_run_times()
TRANSFER_TIMES = load_transfer_times()

# ───────────────────────────── 2. 그래프 빌드 ─────────────────────────────
def build_graph(lines_dict, run_tbl, transfer_tbl):
    """
    역·노선 데이터를 그래프로 변환.
    간선 가중치:
      • 주행 : run_tbl + DWELL
      • 환승 : transfer_tbl, 없으면 FALLBACK_TRANSFER
    """
    g = defaultdict(list)

    def add(u, v, w):
        g[u].append((w, v))
        g[v].append((w, u))

    # 2-1) 노선별 인접역(주행) 간선
    for ln, data in lines_dict.items():
        segments = data.values() if isinstance(data, dict) else [data]
        for key, seg in (data.items() if isinstance(data, dict) else [("linear", data)]):
            for a, b in zip(seg, seg[1:]):
                base = run_tbl.get((a, b), run_tbl.get((b, a), DEFAULT_TRAVEL))
                add((ln, a), (ln, b), base + DWELL)
            # 순환 구간
            if key.endswith("_loop"):
                a, b = seg[-1], seg[0]
                base = run_tbl.get((a, b), run_tbl.get((b, a), DEFAULT_TRAVEL))
                add((ln, a), (ln, b), base + DWELL)

    # 2-2) 역 → 속한 노선 목록
    st_to_lines = defaultdict(list)
    for ln, data in lines_dict.items():
        sts = (s for seg in data.values() for s in seg) if isinstance(data, dict) else data
        for st in sts:
            st_to_lines[st].append(ln)

    # 2-3) 환승 간선
    for st, lns in st_to_lines.items():
        for i in range(len(lns)):
            for j in range(i + 1, len(lns)):
                u, v = (lns[i], st), (lns[j], st)
                w    = transfer_tbl.get((u, v), FALLBACK_TRANSFER)
                add(u, v, w)

    return g


# 그래프 및 보조 인덱스
graph = build_graph(lines, RUN_TIMES, TRANSFER_TIMES)
station_to_nodes = defaultdict(list)
for node in graph:
    station_to_nodes[node[1]].append(node)

# ───────────────────────────── 3. 휴리스틱 선계산 (홉 수 기반) ─────────────────────────────
def precompute_hops(goal_names):
    """각 역까지 최소 '홉 수'를 BFS로 선계산."""
    nbr = defaultdict(set)
    for (ln, a), edges in graph.items():
        for _, (_, b) in edges:
            nbr[a].add(b)
            nbr[b].add(a)

    dist = {g: 0 for g in goal_names}
    dq   = deque(goal_names)
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

    # 목표 노드(동일 역명) 집합
    goals = set(station_to_nodes[goal_name])

    # (1) 홉 기반 테이블
    h_table = precompute_hops({g[1] for g in goals})

    # (2) 휴리스틱: 남은 홉 × 최소 간선 시간
    def h(node):
        _, st = node
        return h_table.get(st, 0) * MIN_EDGE

    # 슈퍼 소스 생성
    SUPER = ('S', '')
    graph[SUPER] = [(0, n) for n in station_to_nodes[start_name]]

    g_cost = {SUPER: 0}
    parent = {SUPER: None}
    pq = []
    for w, n in graph[SUPER]:
        g_cost[n] = w
        parent[n] = SUPER
        heapq.heappush(pq, (w + h(n), w, n))

    try:
        while pq:
            f, g_acc, u = heapq.heappop(pq)
            if u in goals:
                path = []
                while u and u != SUPER:
                    path.append(u)
                    u = parent[u]
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
            segs.append(f" --{edge_time(prev, (ln, st)):.1f}분→ {ln}호선 {st}")
    return "".join(segs)

def generate_station_list():
    all_stations = set()

    def flatten_station_dict(d):
        for v in d.values():
            all_stations.update(v)

    flatten_station_dict(line1_stations)
    flatten_station_dict(line2_stations)
    all_stations.update(line3_stations)
    all_stations.update(line4_stations)
    flatten_station_dict(line5_stations)

    # ✅ 항상 같은 순서로 정렬
    return sorted(list(all_stations))  # ← 핵심!


def analyse_1000_random_astar(all_stations, sample_count=100, repeats=30,
                              detail_csv="astar_runs.csv", summary_csv="astar_summary.csv"):
    random.seed(42)
    # ❶ 샘플 쌍 생성
    pairs = []
    seen = set()
    while len(pairs) < sample_count:
        s, g = random.sample(all_stations, 2)
        if (s, g) not in seen:
            pairs.append((s, g))
            seen.add((s, g))

    detail_records = []
    summary_records = []

    # ❷ 각 샘플별 반복 실행 및 summary 출력
    for i, (s, g) in enumerate(pairs, 1):
        times = []
        success = 0

        for r in range(1, repeats + 1):
            try:
                t0 = time.perf_counter()
                path, total = astar(s, g)
                t1 = time.perf_counter()
                if path:
                    elapsed = t1 - t0
                    times.append(elapsed)
                    success += 1
                    fmt = fmt_path(path)
                    detail_records.append({
                        "start": s, "goal": g, "run": r,
                        "elapsed_sec": round(elapsed, 6),
                        "total_min": round(total, 1),
                        "path": fmt
                    })
            except Exception:
                # 실패한 run은 건너뜁니다
                pass

        # 샘플별 summary 계산
        mean_sec  = float(np.mean(times)) if times else 0.0
        std_sec   = float(np.std(times))  if times else 0.0
        best_sec  = float(min(times))     if times else 0.0
        worst_sec = float(max(times))     if times else 0.0

        # ❸ 샘플별 summary 출력
        print(f"--- Sample {i}/{sample_count}: {s} → {g} 요약 ---")
        print(f"  시도 횟수  : {repeats}")
        print(f"  성공 횟수  : {success}")
        print(f"  평균 시간  : {mean_sec:.6f}초")
        print(f"  표준편차   : {std_sec:.6f}초")
        print(f"  최단 시간  : {best_sec:.6f}초")
        print(f"  최장 시간  : {worst_sec:.6f}초\n")

        summary_records.append({
            "start": s,
            "goal": g,
            "runs_attempted": repeats,
            "runs_successful": success,
            "mean_sec": mean_sec,
            "std_sec": std_sec,
            "best_sec": best_sec,
            "worst_sec": worst_sec
        })

    # ❹ CSV로 저장
    pd.DataFrame(detail_records).to_csv(detail_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(summary_records).to_csv(summary_csv, index=False, encoding="utf-8-sig")

    return summary_records


if __name__ == "__main__":
    random.seed(42)

    all_stations = generate_station_list()

    print("🚀 1000개 랜덤 출발-도착 쌍에 대해 A* 탐색 시작")
    # per‐sample summary 리스트를 받습니다.
    summary_records = analyse_1000_random_astar(all_stations, sample_count=100)