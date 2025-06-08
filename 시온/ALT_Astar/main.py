"""
A* 최단시간 탐색 (서울 지하철 1~5호선)
- 휴리스틱: ALT (Landmarks + Triangle inequality)

사용법 (CLI)
$ python astar_alt_full.py
출발역 이름: 서울역
도착역 이름: 잠실
... 결과 출력 ...

Notes
-----
* 첫 실행 시 12개 랜드마크에서 모든 노드까지의 최단거리(시간)를 미리 계산합니다.
  약 1~2초(1만 노드 미만) 정도 소요되며, 이후 쿼리에서는 메모리 lookup 만 수행합니다.
* landmark_dist.pkl 파일이 존재하면 전처리를 건너띕니다.
"""

from __future__ import annotations
import os, pickle, math, heapq, glob, time
from collections import defaultdict, deque
from typing import Dict, Tuple, List
import pandas as pd
import numpy as np
import random

# ───────────────────────────── 0.  노선 정의 (1~5호선) ─────────────────────────────
#  ▶ 기존 ALT_Astar.py 의 리스트 그대로 복붙했습니다.
#  ▶ …(길어서 생략) 목록은 동일하므로 본 파일만으로 독립 실행 가능합니다.
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

lines = {1: line1_stations, 2: line2_stations, 3: line3_stations, 4: line4_stations, 5: line5_stations}

# ───────────────────────────── 1. CSV 로드 & 상수 ─────────────────────────────
DEFAULT_TRAVEL    = 2.0   # 역간 주행 데이터 없을 때 (분)
DWELL             = 0.5   # 정차 (분)
FALLBACK_TRANSFER = 4.0   # 환승 기본 (분)

RUN_TIME_DIR      = "dataset/역간소요시간(수작업)"
TRANSFER_CSV_PATH = "dataset/서울교통공사_환승역거리 소요시간 정보_20250331.csv"
COORD_CSV_PATH    = "dataset/station_location"
LM_CACHE_PATH     = "landmark_dist.pkl"

# ───────────────────────────── 2. 유틸 함수 ─────────────────────────────

def _parse_mmss(txt: str) -> float | None:
    try:
        m, s = map(int, txt.split(":"))
        return m + s/60
    except Exception:
        return None

def load_run_times(folder: str = RUN_TIME_DIR) -> Dict[Tuple[str,str], float]:
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
                run[(prev,cur)] = run[(cur,prev)] = t_val
            prev = cur
    return run

def load_transfer_times(csv_path: str = TRANSFER_CSV_PATH):
    tbl = {}
    if not os.path.exists(csv_path):
        return tbl
    df = pd.read_csv(csv_path, encoding="cp949")
    for _, row in df.iterrows():
        ln1 = int(row["호선"])              # 예: 2
        station = str(row["환승역명"]).strip()
        digits = "".join(filter(str.isdigit, str(row["환승노선"])) )
        if not digits:
            continue
        ln2 = int(digits)
        t_val = _parse_mmss(str(row["환승소요시간"]).strip())
        if t_val is None:
            continue
        tbl[((ln1,station),(ln2,station))] = tbl[((ln2,station),(ln1,station))] = t_val
    return tbl

def load_coords(folder_path=COORD_CSV_PATH):
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
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R*c

# ───────────────────────────── 3. 그래프 생성 ─────────────────────────────

def build_graph(lines_dict, run_tbl, transfer_tbl):
    g = defaultdict(list)
    def add(u,v,w):
        g[u].append((w,v)); g[v].append((w,u))

    for ln, data in lines_dict.items():
        segments = data.values() if isinstance(data,dict) else [data]
        for key, seg in (data.items() if isinstance(data,dict) else [("linear",data)]):
            for a,b in zip(seg, seg[1:]):
                w = run_tbl.get((a,b), run_tbl.get((b,a), DEFAULT_TRAVEL)) + DWELL
                add((ln,a),(ln,b),w)
            if key.endswith("_loop"):
                a,b = seg[-1], seg[0]
                w = run_tbl.get((a,b), run_tbl.get((b,a), DEFAULT_TRAVEL)) + DWELL
                add((ln,a),(ln,b),w)

    # 환승 간선
    station_to_lines = defaultdict(list)
    for ln, data in lines_dict.items():
        sts = (s for seg in data.values() for s in seg) if isinstance(data,dict) else data
        for st in sts:
            station_to_lines[st].append(ln)
    for st, lns in station_to_lines.items():
        for i in range(len(lns)):
            for j in range(i+1,len(lns)):
                u,v = (lns[i],st),(lns[j],st)
                w = transfer_tbl.get((u,v), FALLBACK_TRANSFER)
                add(u,v,w)
    return g

# ───────────────────────────── 4. Dijkstra (1 → All) ─────────────────────────────

def dijkstra(graph:dict, src) -> Dict[Tuple[int,str], float]:
    dist = {src:0.0}
    pq = [(0.0,src)]
    while pq:
        d,u = heapq.heappop(pq)
        if d>dist[u]:
            continue
        for w,v in graph[u]:
            nd = d+w
            if nd < dist.get(v,float('inf')):
                dist[v]=nd
                heapq.heappush(pq,(nd,v))
    return dist

# ───────────────────────────── 5. Landmark 선택 ─────────────────────────────

def choose_landmarks(graph:dict, k:int=12):
    # 간단 전략: (1) 노드 degree 높은 순, (2) 서로 먼 노드 우선
    degrees = {n: len(graph[n]) for n in graph}
    sorted_nodes = sorted(degrees, key=degrees.get, reverse=True)
    landmarks=[]
    coord = COORDS
    for n in sorted_nodes:
        if n not in coord:  # 좌표 없으면 스킵 (거리 평가 불가)
            continue
        lat1,lon1 = coord[n]
        # 거리가 일정 이상 떨어진 노드만
        if all(haversine(lat1,lon1,*coord[lm])>10 for lm in landmarks):
            landmarks.append(n)
        if len(landmarks)>=k:
            break
    # 만약 부족하면 degree순으로 채움
    for n in sorted_nodes:
        if len(landmarks)>=k:
            break
        if n not in landmarks and n in coord:
            landmarks.append(n)
    return landmarks

# ───────────────────────────── 6. 메인 (전처리 & A*) ─────────────────────────────

RUN_TIMES      = load_run_times()
TRANSFER_TIMES = load_transfer_times()
COORDS         = load_coords()

graph = build_graph(lines, RUN_TIMES, TRANSFER_TIMES)
station_to_nodes = defaultdict(list)
for node in graph:
    station_to_nodes[node[1]].append(node)

# 6-1) Landmark distance table 로드 or 계산
if os.path.exists(LM_CACHE_PATH):
    with open(LM_CACHE_PATH,"rb") as f:
        LANDMARKS, LM_DIST = pickle.load(f)
else:
    LANDMARKS = choose_landmarks(graph, k=12)
    LM_DIST = {L: dijkstra(graph, L) for L in LANDMARKS}
    with open(LM_CACHE_PATH,"wb") as f:
        pickle.dump((LANDMARKS,LM_DIST), f)

# 6-2) ALT 휴리스틱

def h_alt(node, goals:List[Tuple[int,str]]):
    # goals 는 (호선, 역) 리스트 (같은 역명 여러 노선 가능)
    best = 0.0
    for L in LANDMARKS:
        dL_node = LM_DIST[L].get(node, float('inf'))
        if dL_node==float('inf'):
            continue
        dL_goal_max = -1
        for g in goals:
            d = LM_DIST[L].get(g, float('inf'))
            if d!=float('inf'):
                dL_goal_max = max(dL_goal_max, abs(d - dL_node))
        if dL_goal_max>=0:
            best = max(best, dL_goal_max)
    # ALT가 계산 불가능한 rare 케이스(랜드마크와 단절) → 0
    return best if best>0 else 0.0

# 6-3) A* search (ALT)

def astar(start_name:str, goal_name:str):
    if start_name not in station_to_nodes or goal_name not in station_to_nodes:
        raise ValueError("역 이름이 데이터에 없습니다.")
    goals = station_to_nodes[goal_name]

    SUPER = ('S','')
    graph[SUPER] = [(0,n) for n in station_to_nodes[start_name]]
    g_cost = {SUPER:0.0}
    parent = {SUPER:None}
    pq=[]
    for _,n in graph[SUPER]:
        g_cost[n]=0.0
        parent[n]=SUPER
        heapq.heappush(pq,(h_alt(n,goals),0.0,n))

    while pq:
        f,g,u = heapq.heappop(pq)
        if u in goals:
            path=[]
            cur=u
            while cur and cur!=SUPER:
                path.append(cur)
                cur=parent[cur]
            graph.pop(SUPER,None)
            return path[::-1], g
        if g>g_cost[u]:
            continue
        for w,v in graph[u]:
            ng=g+w
            if ng < g_cost.get(v,float('inf')):
                g_cost[v]=ng
                parent[v]=u
                heapq.heappush(pq,(ng+h_alt(v,goals), ng, v))
    graph.pop(SUPER,None)
    return None, float('inf')

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

import time, random
import pandas as pd
import numpy as np

def analyse_1000_random_astar(all_stations, sample_count=100, repeats=30,
                              detail_csv="astar_runs.csv",
                              summary_csv="astar_summary.csv",
                              global_csv="global_summary.csv"):
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
                pass

        mean_sec  = float(np.mean(times)) if times else 0.0
        std_sec   = float(np.std(times))  if times else 0.0
        best_sec  = float(min(times))     if times else 0.0
        worst_sec = float(max(times))     if times else 0.0

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

    # ❸ CSV 저장
    pd.DataFrame(detail_records).to_csv(detail_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(summary_records).to_csv(summary_csv, index=False, encoding="utf-8-sig")

    # ❹ 글로벌 요약 계산
    all_times = [rec["elapsed_sec"] for rec in detail_records]
    global_summary = {
        "total_runs":             len(all_times),
        "total_successful_runs":  sum(1 for rec in detail_records),
        "total_time_sec":         float(sum(all_times)),
        "mean_time_sec":          float(np.mean(all_times)) if all_times else 0.0,
        "std_time_sec":           float(np.std(all_times))  if all_times else 0.0,
        "best_time_sec":          float(min(all_times))     if all_times else 0.0,
        "worst_time_sec":         float(max(all_times))     if all_times else 0.0
    }

    # ❺ 글로벌 요약 출력
    print("=== 전체(Global) 요약 ===")
    print(f"  시도 총횟수      : {global_summary['total_runs']}")
    print(f"  성공 총횟수      : {global_summary['total_successful_runs']}")
    print(f"  전체 소요 시간  : {global_summary['total_time_sec']:.6f}초")
    print(f"  전체 평균 시간  : {global_summary['mean_time_sec']:.6f}초")
    print(f"  전체 표준편차    : {global_summary['std_time_sec']:.6f}초")
    print(f"  전체 최단 시간  : {global_summary['best_time_sec']:.6f}초")
    print(f"  전체 최장 시간  : {global_summary['worst_time_sec']:.6f}초\n")

    # ❻ 글로벌 요약 CSV 저장
    pd.DataFrame([global_summary]).to_csv(global_csv, index=False, encoding="utf-8-sig")

    return summary_records, global_summary

if __name__ == "__main__":
    random.seed(42)

    all_stations = generate_station_list()

    print("🚀 1000개 랜덤 출발-도착 쌍에 대해 A* 탐색 시작")
    # per‐sample summary 리스트를 받습니다.
    summary_records = analyse_1000_random_astar(all_stations, sample_count=100)
