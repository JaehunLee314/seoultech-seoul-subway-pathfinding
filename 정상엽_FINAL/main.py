import pandas as pd, glob, os, heapq, time
import numpy as np
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor
import pickle
from pathlib import Path
import shutil

# 캐싱을 위한 전역 변수
CACHE_DIR = Path("path_cache")
CACHE_DIR.mkdir(exist_ok=True)
path_cache = {}  # 이 줄이 있는지 확인

def clear_cache():
    """캐시 완전 초기화"""
    global path_cache
    path_cache.clear()  # 메모리 캐시 초기화
    shutil.rmtree(CACHE_DIR, ignore_errors=True)  # 파일 캐시 삭제
    CACHE_DIR.mkdir(exist_ok=True)  # 캐시 디렉토리 재생성
    print("🗑️ 캐시가 완전히 초기화되었습니다.")

# ─────────────────────────────────────────────────────────────
# 지하철 역 리스트 (1~5호선) - 기존 유지
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
FALLBACK_TRANSFER = 4        # 환승 CSV에 없을 경우 기본 환승 시간 (분)
DWELL             = 0.5      # 정차 시간 0.5분 (=30초)
TRANSFER_CSV_PATH = "서울교통공사_환승역거리 소요시간 정보_20250331.csv"

# 캐싱을 위한 전역 변수
CACHE_DIR = Path("path_cache")
CACHE_DIR.mkdir(exist_ok=True)
path_cache = {}

# ───────────────────────────── 헬퍼 함수 ─────────────────────────────
def _parse_mmss(txt: str) -> float:
    """'m:ss' → 분(float). 파싱 실패 시 None."""
    try:
        m, s = map(int, txt.split(':'))
        return m + s / 60
    except Exception:
        return None

def load_run_times(folder="역간소요시간(수작업)"):
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

# ─────────────────────────────  CSV → 데이터 테이블 ─────────────────────────────
RUN_TIMES      = load_run_times()
TRANSFER_TIMES = load_transfer_times()

# ───────────────────────────── 그래프 빌드 ─────────────────────────────
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

# ───────────────────────────── 휴리스틱 선계산 (홉 수 기반) ─────────────────────────────
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

# ─────────────────────────────────────────────────────────────
# 기본 Dijkstra 알고리즘
# ─────────────────────────────────────────────────────────────
def dijkstra(start_name, goal_name):
    """기본 Dijkstra 알고리즘 구현"""
    if start_name not in station_to_nodes or goal_name not in station_to_nodes:
        raise ValueError("역 이름이 데이터에 없습니다.")
    
    goals = set(station_to_nodes[goal_name])
    
    # 슈퍼 소스 생성
    SUPER = ('S', '')
    graph[SUPER] = [(0, n) for n in station_to_nodes[start_name]]
    
    distances = {SUPER: 0}
    parent = {SUPER: None}
    pq = [(0, SUPER)]
    visited = set()
    
    try:
        while pq:
            current_dist, current = heapq.heappop(pq)
            
            if current in visited:
                continue
                
            visited.add(current)
            
            if current in goals:
                path = []
                while current and current != SUPER:
                    path.append(current)
                    current = parent[current]
                return path[::-1], current_dist
            
            for weight, neighbor in graph[current]:
                if neighbor not in visited:
                    new_dist = current_dist + weight
                    if new_dist < distances.get(neighbor, float('inf')):
                        distances[neighbor] = new_dist
                        parent[neighbor] = current
                        heapq.heappush(pq, (new_dist, neighbor))
        
        return None, float('inf')
    finally:
        graph.pop(SUPER, None)

# ─────────────────────────────────────────────────────────────
# 병렬 양방향 탐색을 위한 결과 클래스
# ─────────────────────────────────────────────────────────────
class BidirectionalResult:
    def __init__(self):
        self.forward_distances = {}
        self.backward_distances = {}
        self.forward_parent = {}
        self.backward_parent = {}
        self.meeting_point = None
        self.total_distance = float('inf')
        self.lock = threading.Lock()

# ─────────────────────────────────────────────────────────────
# 병렬 양방향 Dijkstra
# ─────────────────────────────────────────────────────────────
def parallel_bidirectional_dijkstra(start_name, goal_name):
    if start_name not in station_to_nodes or goal_name not in station_to_nodes:
        raise ValueError("역 이름이 데이터에 없습니다.")
    
    result = BidirectionalResult()
    
    def forward_search():
        start_nodes = station_to_nodes[start_name]
        SUPER_F = ('SF', '')
        graph[SUPER_F] = [(0, n) for n in start_nodes]
        
        distances = {SUPER_F: 0}
        parent = {SUPER_F: None}
        pq = [(0, SUPER_F)]
        visited = set()
        
        try:
            while pq:
                current_dist, current = heapq.heappop(pq)
                
                if current in visited:
                    continue
                
                visited.add(current)
                
                with result.lock:
                    result.forward_distances[current] = current_dist
                    result.forward_parent[current] = parent.get(current)
                    
                    # 만남 지점 확인
                    if current in result.backward_distances:
                        total = current_dist + result.backward_distances[current]
                        if total < result.total_distance:
                            result.total_distance = total
                            result.meeting_point = current
                
                for weight, neighbor in graph[current]:
                    if neighbor not in visited:
                        new_dist = current_dist + weight
                        if new_dist < distances.get(neighbor, float('inf')):
                            distances[neighbor] = new_dist
                            parent[neighbor] = current
                            heapq.heappush(pq, (new_dist, neighbor))
        finally:
            graph.pop(SUPER_F, None)
    
    def backward_search():
        goal_nodes = station_to_nodes[goal_name]
        
        # 역방향 그래프 생성
        reverse_graph = defaultdict(list)
        for u, edges in graph.items():
            for w, v in edges:
                reverse_graph[v].append((w, u))
        
        SUPER_B = ('SB', '')
        reverse_graph[SUPER_B] = [(0, n) for n in goal_nodes]
        
        distances = {SUPER_B: 0}
        parent = {SUPER_B: None}
        pq = [(0, SUPER_B)]
        visited = set()
        
        try:
            while pq:
                current_dist, current = heapq.heappop(pq)
                
                if current in visited:
                    continue
                
                visited.add(current)
                
                with result.lock:
                    result.backward_distances[current] = current_dist
                    result.backward_parent[current] = parent.get(current)
                    
                    # 만남 지점 확인
                    if current in result.forward_distances:
                        total = current_dist + result.forward_distances[current]
                        if total < result.total_distance:
                            result.total_distance = total
                            result.meeting_point = current
                
                for weight, neighbor in reverse_graph[current]:
                    if neighbor not in visited:
                        new_dist = current_dist + weight
                        if new_dist < distances.get(neighbor, float('inf')):
                            distances[neighbor] = new_dist
                            parent[neighbor] = current
                            heapq.heappush(pq, (new_dist, neighbor))
        finally:
            reverse_graph.pop(SUPER_B, None)
    
    # 병렬 실행
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_forward = executor.submit(forward_search)
        future_backward = executor.submit(backward_search)
        
        future_forward.result()
        future_backward.result()
    
    if result.meeting_point is None:
        return None, float('inf')
    
    # 경로 재구성
    path = reconstruct_bidirectional_path(result, start_name, goal_name)
    return path, result.total_distance

# ─────────────────────────────────────────────────────────────
# 병렬 양방향 A*
# ─────────────────────────────────────────────────────────────
def parallel_bidirectional_astar(start_name, goal_name):
    if start_name not in station_to_nodes or goal_name not in station_to_nodes:
        raise ValueError("역 이름이 데이터에 없습니다.")
    
    result = BidirectionalResult()
    
    # 휴리스틱 함수들
    h_forward = precompute_hops([goal_name])
    h_backward = precompute_hops([start_name])
    
    def forward_astar():
        start_nodes = station_to_nodes[start_name]
        SUPER_F = ('SF', '')
        graph[SUPER_F] = [(0, n) for n in start_nodes]
        
        g_cost = {SUPER_F: 0}
        parent = {SUPER_F: None}
        pq = []
        
        for w, n in graph[SUPER_F]:
            g_cost[n] = w
            parent[n] = SUPER_F
            h_val = h_forward.get(n[1], 0) * MIN_EDGE
            heapq.heappush(pq, (w + h_val, w, n))
        
        try:
            while pq:
                f, g_acc, current = heapq.heappop(pq)
                
                if g_acc > g_cost.get(current, float('inf')):
                    continue
                
                with result.lock:
                    result.forward_distances[current] = g_acc
                    result.forward_parent[current] = parent.get(current)
                    
                    # 만남 지점 확인
                    if current in result.backward_distances:
                        total = g_acc + result.backward_distances[current]
                        if total < result.total_distance:
                            result.total_distance = total
                            result.meeting_point = current
                
                for weight, neighbor in graph[current]:
                    new_g = g_acc + weight
                    if new_g < g_cost.get(neighbor, float('inf')):
                        g_cost[neighbor] = new_g
                        parent[neighbor] = current
                        h_val = h_forward.get(neighbor[1], 0) * MIN_EDGE
                        heapq.heappush(pq, (new_g + h_val, new_g, neighbor))
        finally:
            graph.pop(SUPER_F, None)
    
    def backward_astar():
        goal_nodes = station_to_nodes[goal_name]
        
        # 역방향 그래프 생성
        reverse_graph = defaultdict(list)
        for u, edges in graph.items():
            for w, v in edges:
                reverse_graph[v].append((w, u))
        
        SUPER_B = ('SB', '')
        reverse_graph[SUPER_B] = [(0, n) for n in goal_nodes]
        
        g_cost = {SUPER_B: 0}
        parent = {SUPER_B: None}
        pq = []
        
        for w, n in reverse_graph[SUPER_B]:
            g_cost[n] = w
            parent[n] = SUPER_B
            h_val = h_backward.get(n[1], 0) * MIN_EDGE
            heapq.heappush(pq, (w + h_val, w, n))
        
        try:
            while pq:
                f, g_acc, current = heapq.heappop(pq)
                
                if g_acc > g_cost.get(current, float('inf')):
                    continue
                
                with result.lock:
                    result.backward_distances[current] = g_acc
                    result.backward_parent[current] = parent.get(current)
                    
                    # 만남 지점 확인
                    if current in result.forward_distances:
                        total = g_acc + result.forward_distances[current]
                        if total < result.total_distance:
                            result.total_distance = total
                            result.meeting_point = current
                
                for weight, neighbor in reverse_graph[current]:
                    new_g = g_acc + weight
                    if new_g < g_cost.get(neighbor, float('inf')):
                        g_cost[neighbor] = new_g
                        parent[neighbor] = current
                        h_val = h_backward.get(neighbor[1], 0) * MIN_EDGE
                        heapq.heappush(pq, (new_g + h_val, new_g, neighbor))
        finally:
            reverse_graph.pop(SUPER_B, None)
    
    # 병렬 실행
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_forward = executor.submit(forward_astar)
        future_backward = executor.submit(backward_astar)
        
        future_forward.result()
        future_backward.result()
    
    if result.meeting_point is None:
        return None, float('inf')
    
    # 경로 재구성
    path = reconstruct_bidirectional_path(result, start_name, goal_name)
    return path, result.total_distance

# ─────────────────────────────────────────────────────────────
# 캐싱 시스템
# ─────────────────────────────────────────────────────────────
def get_cache_key(start, goal, algorithm):
    return f"{algorithm}_{start}_{goal}"

def save_to_cache(start, goal, algorithm, path, distance):
    key = get_cache_key(start, goal, algorithm)
    path_cache[key] = (path, distance)
    
    cache_file = CACHE_DIR / f"{key}.pkl"
    with open(cache_file, 'wb') as f:
        pickle.dump((path, distance), f)

def load_from_cache(start, goal, algorithm):
    key = get_cache_key(start, goal, algorithm)
    
    if key in path_cache:
        return path_cache[key]
    
    cache_file = CACHE_DIR / f"{key}.pkl"
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            result = pickle.load(f)
            path_cache[key] = result
            return result
    
    return None

def cached_parallel_bidirectional_dijkstra(start_name, goal_name):
    cached_result = load_from_cache(start_name, goal_name, "parallel_bidirectional_dijkstra")
    if cached_result:
        return cached_result
    
    path, distance = parallel_bidirectional_dijkstra(start_name, goal_name)
    if path:
        save_to_cache(start_name, goal_name, "parallel_bidirectional_dijkstra", path, distance)
    
    return path, distance

def cached_parallel_bidirectional_astar(start_name, goal_name):
    cached_result = load_from_cache(start_name, goal_name, "parallel_bidirectional_astar")
    if cached_result:
        return cached_result
    
    path, distance = parallel_bidirectional_astar(start_name, goal_name)
    if path:
        save_to_cache(start_name, goal_name, "parallel_bidirectional_astar", path, distance)
    
    return path, distance

def cached_dijkstra(start_name, goal_name):
    cached_result = load_from_cache(start_name, goal_name, "dijkstra")
    if cached_result:
        return cached_result
    
    path, distance = dijkstra(start_name, goal_name)
    if path:
        save_to_cache(start_name, goal_name, "dijkstra", path, distance)
    
    return path, distance

def cached_astar(start_name, goal_name):
    cached_result = load_from_cache(start_name, goal_name, "astar")
    if cached_result:
        return cached_result
    
    path, distance = astar(start_name, goal_name)
    if path:
        save_to_cache(start_name, goal_name, "astar", path, distance)
    
    return path, distance

# ─────────────────────────────────────────────────────────────
# 기존 A* 알고리즘 (유지)
# ─────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────
# 13-2. 완전히 캐시 없는 Hub-based 라우팅
# ─────────────────────────────────────────────────────────────

def get_station_lines(station_name):
    lines_set = set()
    for line_num, line_data in lines.items():
        if isinstance(line_data, dict):
            for segment in line_data.values():
                if station_name in segment:
                    lines_set.add(line_num)
        else:
            if station_name in line_data:
                lines_set.add(line_num)
    return lines_set

def select_hubs_by_connectivity(start_name, goal_name):
    # 출발지와 목적지의 노선 정보 분석
    start_lines = get_station_lines(start_name)
    goal_lines = get_station_lines(goal_name)
    
    candidate_hubs = []
    for hub in HUB_STATIONS:
        if hub == start_name or hub == goal_name:
            continue
            
        hub_lines = get_station_lines(hub)
        
        # 연결성 점수 계산
        score = 0
        if hub_lines & start_lines:  # 출발지와 같은 노선
            score += 3
        if hub_lines & goal_lines:   # 목적지와 같은 노선
            score += 3
        score += len(hub_lines)      # 허브의 노선 수
        
        candidate_hubs.append((hub, score))
    
    # 점수 순으로 정렬하여 상위 허브들 반환
    candidate_hubs.sort(key=lambda x: x[1], reverse=True)
    return [hub for hub, score in candidate_hubs[:5]]

def pure_hub_based_routing(start_name, goal_name):
    # 모든 계산을 실시간으로 수행 (캐시 사용 안함)
    direct_path, direct_distance = dijkstra(start_name, goal_name)
    
    # 허브 선택 (거리 기반이 아닌 노선 연결성 기반)
    selected_hubs = select_hubs_by_connectivity(start_name, goal_name)
    
    best_path = direct_path
    best_distance = direct_distance
    best_route_info = "직접 경로"
    
    # 선택된 허브들을 통한 경로 탐색 (실시간 계산)
    for hub in selected_hubs[:3]:  # 상위 3개 허브만 검사
        # 실시간 계산: 출발지 → 허브 → 목적지
        path1, dist1 = dijkstra(start_name, hub)
        path2, dist2 = dijkstra(hub, goal_name)
        
        if path1 and path2:
            total_distance = dist1 + dist2
            if total_distance < best_distance:
                # 경로 병합 (중복 허브 제거)
                combined_path = path1 + path2[1:]
                
                # 목적지가 포함되어 있는지 확인
                if not combined_path or combined_path[-1][1] != goal_name:
                    goal_nodes = station_to_nodes[goal_name]
                    if goal_nodes:
                        combined_path.append(goal_nodes[0])
                
                best_path = combined_path
                best_distance = total_distance
                best_route_info = f"{hub} 허브 경유 (실시간 계산)"
    
    return best_path, best_distance, best_route_info

def pure_hub_based_routing_wrapper(s, g):
    path, distance, route_info = pure_hub_based_routing(s, g)
    return path, distance


# ─────────────────────────────────────────────────────────────
# 보조 함수
# ─────────────────────────────────────────────────────────────
def reconstruct_bidirectional_path(result, start_name, goal_name):
    if not result.meeting_point:
        return None
    
    # 전진 경로
    forward_path = []
    current = result.meeting_point
    while current and current[0] != 'SF':
        forward_path.append(current)
        current = result.forward_parent.get(current)
    forward_path.reverse()
    
    # 후진 경로
    backward_path = []
    current = result.backward_parent.get(result.meeting_point)
    while current and current[0] != 'SB':
        backward_path.append(current)
        current = result.backward_parent.get(current)
    
    return forward_path + backward_path

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

# ─────────────────────────────────────────────────────────────
# Hub-based 경로 탐색 알고리즘
# ─────────────────────────────────────────────────────────────

HUB_STATIONS = {
    "강남", "홍대입구", "신촌", "명동", "종로3가", "을지로3가", "충무로",
    "동대문역사문화공원", "신도림", "사당", "교대", "왕십리", "건대입구",
    "잠실", "영등포구청", "신림", "구로", "서울역", "용산", "청량리"
}

# ALGORITHMS 딕셔너리
ALGORITHMS = {
    "dijkstra": dijkstra,
    "astar": astar,
    "cached_dijkstra": cached_dijkstra,
    "cached_astar": cached_astar,
    "parallel_bidirectional_dijkstra": parallel_bidirectional_dijkstra,
    "parallel_bidirectional_astar": parallel_bidirectional_astar,
    "cached_parallel_bidirectional_dijkstra": cached_parallel_bidirectional_dijkstra,
    "cached_parallel_bidirectional_astar": cached_parallel_bidirectional_astar,
    "pure_hub_based_routing": pure_hub_based_routing_wrapper,

}

def run_algorithm(algorithm_name, start, goal):
    if algorithm_name not in ALGORITHMS:
        raise ValueError(f"알 수 없는 알고리즘: {algorithm_name}")
    
    algorithm = ALGORITHMS[algorithm_name]
    start_time = time.perf_counter()
    path, distance = algorithm(start, goal)
    end_time = time.perf_counter()
    
    return {
        "algorithm": algorithm_name,
        "path": path,
        "distance": distance,
        "execution_time": end_time - start_time
    }

def compare_algorithms(start, goal, repeats=4000):
    results = {}
    
    for algo_name in ALGORITHMS.keys():
        times = []
        path_result = None
        distance_result = None
        
        for _ in range(repeats):
            try:
                result = run_algorithm(algo_name, start, goal)
                times.append(result["execution_time"])
                if path_result is None:
                    path_result = result["path"]
                    distance_result = result["distance"]
            except Exception as e:
                print(f" {algo_name} 실행 중 오류: {e}")
                continue
        
        if times:
            results[algo_name] = {
                "mean_time": np.mean(times),
                "std_time": np.std(times),
                "min_time": np.min(times),
                "max_time": np.max(times),
                "total_time": sum(times),  
                "path": path_result,
                "distance": distance_result
            }
    
    return results

def clean_path_display(path):
    if not path:
        return []
    
    clean_stations = []
    prev_station = None
    
    for line, station in path:
        if station != prev_station:
            clean_stations.append(station)
            prev_station = station
    
    return clean_stations

# ─────────────────────────────────────────────────────────────
# 14. 성능 테스트 코드
# ─────────────────────────────────────────────────────────────
import random
import csv

def remove_duplicate_stations(path):
    if not path:
        return []
    
    clean_path = [path[0]]
    for station in path[1:]:
        # 역명만 비교 (노선 정보 제외)
        if station[1] != clean_path[-1][1]:
            clean_path.append(station)
    return clean_path

def run_algorithm_n_times(algorithm_func, start, goal, n=30):
    total_path = None
    total_distance = 0
    total_exec_time = 0
    successful_runs = 0
    
    for _ in range(n):
        try:
            start_time = time.perf_counter()
            result = algorithm_func(start, goal)
            end_time = time.perf_counter()
            
            if isinstance(result, tuple):
                if len(result) == 2:
                    path, distance = result
                elif len(result) == 3:
                    path, distance, _ = result  
                else:
                    continue
            else:
                continue
            
            if path:  
                clean_path = remove_duplicate_stations(path)
                total_exec_time += (end_time - start_time)
                total_path = clean_path
                total_distance = distance
                successful_runs += 1
        except Exception as e:
            print(f" 오류 발생 ({start} → {goal}): {e}")
            continue
    
    if successful_runs > 0:
        avg_exec_time = total_exec_time / successful_runs
        return total_path, total_distance, avg_exec_time
    else:
        return None, float('inf'), float('inf')

def generate_random_pairs(seed=42, num_pairs=100):
    random.seed(seed)
    
    all_stations = list(station_to_nodes.keys())
    
    random_pairs = []
    attempts = 0
    max_attempts = num_pairs * 3 
    
    while len(random_pairs) < num_pairs and attempts < max_attempts:
        start = random.choice(all_stations)
        goal = random.choice(all_stations)
        
        if start != goal and (start, goal) not in random_pairs:
            random_pairs.append((start, goal))
        
        attempts += 1
    
    return random_pairs

def save_results_to_csv(results, filename="performance_results.csv"):
    with open(filename, mode='w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        # 헤더 작성
        writer.writerow([
            "algorithm", "start", "goal", "path", "stations_count", 
            "travel_time_minutes", "avg_execution_time_seconds"
        ])
        
        # 데이터 작성
        for result in results:
            path_str = " → ".join(result['path']) if result['path'] else "경로 없음"
            writer.writerow([
                result['algorithm'],
                result['start'],
                result['goal'],
                path_str,
                len(result['path']) if result['path'] else 0,
                result['travel_time'],
                result['avg_exec_time']
            ])

def run_performance_test(algorithms_to_test=None, num_pairs=100, runs_per_pair=30, seed=42):
    print(" 성능 테스트 시작...")
    
    # 테스트할 알고리즘 선택
    if algorithms_to_test is None:
        algorithms_to_test = [
            "dijkstra",
            "astar", 
            "cached_dijkstra",
            "cached_astar",
            "parallel_bidirectional_dijkstra",
            "parallel_bidirectional_astar", 
            "cached_parallel_bidirectional_dijkstra",
            "cached_parallel_bidirectional_astar",
            "pure_hub_based_routing"
]
    
    # 무작위 시점-종점 쌍 생성
    print(f" {num_pairs}개 무작위 경로 생성 중... (Seed: {seed})")
    random_pairs = generate_random_pairs(seed=seed, num_pairs=num_pairs)
    print(f" {len(random_pairs)}개 경로 생성 완료")
    
    all_results = []
    
    # 각 알고리즘별 테스트
    for algo_name in algorithms_to_test:
        if algo_name not in ALGORITHMS:
            print(f" 알고리즘 '{algo_name}' 을 찾을 수 없습니다.")
            continue
            
        print(f"\n🔄 {algo_name.upper()} 테스트 중...")
        algorithm_func = ALGORITHMS[algo_name]
        
        # 각 시점-종점 쌍에 대해 테스트
        for i, (start, goal) in enumerate(random_pairs):
            if (i + 1) % 20 == 0:  # 진행상황 표시
                print(f"   진행률: {i + 1}/{len(random_pairs)} ({(i + 1) / len(random_pairs) * 100:.1f}%)")
            
            # n회 실행하여 평균 계산
            path, travel_time, avg_exec_time = run_algorithm_n_times(
                algorithm_func, start, goal, n=runs_per_pair
            )
            
            # 결과 저장
            result = {
                'algorithm': algo_name,
                'start': start,
                'goal': goal,
                'path': [station for _, station in path] if path else [],
                'travel_time': travel_time if travel_time != float('inf') else -1,
                'avg_exec_time': avg_exec_time if avg_exec_time != float('inf') else -1
            }
            all_results.append(result)
        
        print(f" {algo_name.upper()} 테스트 완료")
    
    # CSV 파일로 저장
    filename = f"performance_test_results_seed{seed}.csv"
    save_results_to_csv(all_results, filename)
    print(f"\n 결과가 '{filename}' 파일로 저장되었습니다.")
    
    # 간단한 통계 출력
    print_performance_summary(all_results, algorithms_to_test)
    
    return all_results

def print_performance_summary(results, algorithms_to_test):
    print("\n 성능 테스트 결과 요약")
    print("=" * 60)
    
    for algo_name in algorithms_to_test:
        algo_results = [r for r in results if r['algorithm'] == algo_name]
        
        if not algo_results:
            continue
            
        # 성공한 경로만 필터링
        successful_results = [r for r in algo_results if r['avg_exec_time'] > 0]
        
        if successful_results:
            avg_exec_time = sum(r['avg_exec_time'] for r in successful_results) / len(successful_results)
            avg_travel_time = sum(r['travel_time'] for r in successful_results if r['travel_time'] > 0) / len([r for r in successful_results if r['travel_time'] > 0])
            success_rate = len(successful_results) / len(algo_results) * 100
            
            print(f"\n🔹 {algo_name.upper()}")
            print(f"   성공률: {success_rate:.1f}%")
            print(f"   평균 실행시간: {avg_exec_time:.6f}초")
            print(f"   평균 이동시간: {avg_travel_time:.1f}분")
        else:
            print(f"\n🔹 {algo_name.upper()}: 성공한 경로 없음")

def run_performance_test_cli():
    print(" 성능 테스트 모드")
    print("=" * 40)
    
    try:
        num_pairs = int(input("테스트할 경로 수 (기본값: 100): ") or "100")
        runs_per_pair = int(input("경로당 실행 횟수 (기본값: 30): ") or "30")
        seed = int(input("랜덤 시드 (기본값: 42): ") or "42")
    except ValueError:
        print(" 잘못된 입력입니다. 기본값을 사용합니다.")
        num_pairs, runs_per_pair, seed = 100, 30, 42
    
    results = run_performance_test(
        num_pairs=num_pairs,
        runs_per_pair=runs_per_pair,
        seed=seed
    )
    
    return results

# ─────────────────────────────────────────────────────────────
# 메인함수
# ─────────────────────────────────────────────────────────────
# CLI 인터페이스
if __name__ == "__main__":
    try:
        clear_cache()
        
        print(" 지하철 경로 탐색 시스템")
        print("=" * 40)
        print("1. 단일 경로 탐색")
        print("2. 성능 테스트")
        
        mode = input("모드를 선택하세요 (1 또는 2): ").strip()
        
        if mode == "2":
            run_performance_test_cli()
        else:
            s = input("출발역 이름: ").strip()
            g = input("도착역 이름: ").strip()
            
            print("\n 경로 상세 비교")
            print("=" * 50)
            
            # Dijkstra 경로
            path1, distance1 = dijkstra(s, g)
            clean_path1 = remove_duplicate_stations(path1)
            print(f" Dijkstra 경로: {[station for _, station in clean_path1]}")
            print(f"   총 소요시간: {distance1:.1f}분, 경로 길이: {len(clean_path1)}개 역")
            
            # 순수 Hub 기반 경로
            path2, distance2 = ALGORITHMS["pure_hub_based_routing"](s, g)
            clean_path2 = remove_duplicate_stations(path2)
            print(f" 순수 Hub 기반 경로: {[station for _, station in clean_path2]}")
            print(f"   총 소요시간: {distance2:.1f}분, 경로 길이: {len(clean_path2)}개 역")
            
            # 차이점 분석
            if clean_path1 and clean_path2:
                time_diff = distance1 - distance2
                station_diff = len(clean_path1) - len(clean_path2)
                print(f"\n 차이점:")
                print(f"   시간 차이: {time_diff:.1f}분 ({'Hub 기반이 더 빠름' if time_diff > 0 else 'Dijkstra가 더 빠름'})")
                print(f"   역 개수 차이: {station_diff}개 역")
            
            print("\n 알고리즘 성능 비교 결과 (4000회 실행)")
            print("=" * 80)
            
            comparison_results = compare_algorithms(s, g, repeats=4000)
            
            for algo_name, stats in comparison_results.items():
                print(f"\n **{algo_name.upper()}**")
                print(f"   평균 실행시간: {stats['mean_time']:.6f}초")
                print(f"   표준편차:     {stats['std_time']:.6f}초")
                print(f"   최단시간:     {stats['min_time']:.6f}초")
                print(f"   최장시간:     {stats['max_time']:.6f}초")
                print(f"   총합 시간:     {stats['total_time']:.6f}초")
                if stats['path']:
                    clean_path = remove_duplicate_stations(stats['path'])
                    print(f"   총 소요시간:   {stats['distance']:.1f}분")
                    print(f"   경로 길이:     {len(clean_path)}개 역")
            
            if comparison_results:
                fastest = min(comparison_results.items(), key=lambda x: x[1]['mean_time'])
                print(f"\n **가장 빠른 알고리즘**: {fastest[0]} ({fastest[1]['mean_time']:.6f}초)")
                print(f" **총합 시간이 가장 짧은 알고리즘**: {fastest[0]} ({fastest[1]['total_time']:.6f}초)")
                
                if fastest[1]['path']:
                    clean_path = remove_duplicate_stations(fastest[1]['path'])
                    print(f"\n **최적 경로**:")
                    print(fmt_path(clean_path))
        
    except Exception as e:
        print(" 오류:", e)