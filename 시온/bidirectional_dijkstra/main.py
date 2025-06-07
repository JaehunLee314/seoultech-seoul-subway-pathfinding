import pandas as pd, glob, os, heapq, math
from collections import defaultdict
import time
import numpy as np
import random

# ─────────────────────────────────────────────────────────────
# 0.  지하철 노선별 역 리스트 (1~5호선)
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
lines = {1: line1_stations, 2: line2_stations,
         3: line3_stations, 4: line4_stations, 5: line5_stations}

# ─────────────────────────────────────────────────────────────
# 상수
# ─────────────────────────────────────────────────────────────
DEFAULT_TRAVEL    = 2        # 주행 시간 미등록 시 기본값 (분)
FALLBACK_TRANSFER = 4        # 환승 시간 미등록 시 기본값 (분)
DWELL             = 0.5      # 정차 시간 (분)
RUN_DIR           = "dataset/역간소요시간(수작업)"
TRANSFER_CSV_PATH = "dataset/서울교통공사_환승역거리 소요시간 정보_20250331.csv"
COORD_CSV_PATH    = "dataset/station_location"

# ─────────────────────────────────────────────────────────────
# CSV 로딩
# ─────────────────────────────────────────────────────────────
def _parse_mmss(txt: str):
    try:
        m, s = map(int, txt.split(':'))
        return m + s/60
    except:
        return None

# 주행 시간
RUN_TIMES = {}
for p in glob.glob(os.path.join(RUN_DIR, '*.csv')):
    df = pd.read_csv(p, encoding='utf-8')
    df.columns = [c.strip() for c in df.columns]
    prev = None
    for _, row in df.iterrows():
        cur = row['역명'].strip()
        t = _parse_mmss(str(row['시간(분)']).strip())
        if t is None:
            prev = cur; continue
        if prev:
            RUN_TIMES[(prev, cur)] = RUN_TIMES[(cur, prev)] = t
        prev = cur

# 환승 시간
TRANSFER_TIMES = {}
if os.path.exists(TRANSFER_CSV_PATH):
    df_t = pd.read_csv(TRANSFER_CSV_PATH, encoding='cp949')
    for _, row in df_t.iterrows():
        ln1 = int(row['호선']); st = row['환승역명'].strip()
        digits = ''.join(filter(str.isdigit, str(row['환승노선'])))
        if not digits: continue
        ln2 = int(digits)
        t = _parse_mmss(str(row['환승소요시간']).strip())
        if t is None: continue
        TRANSFER_TIMES[((ln1, st), (ln2, st))] = t
        TRANSFER_TIMES[((ln2, st), (ln1, st))] = t

def load_coords(folder_path):
    """
    지정한 폴더 내 모든 CSV 파일을 읽어,
    (호선_int, 역명) → (위도, 경도) 딕셔너리 생성
    CSV는 ['line','station_name','latitude','longitude'] 컬럼을 가져야 합니다.
    """
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
                # 형 변환 오류나 컬럼 누락 시 스킵
                continue

    return coords

COORDS = {}
COORDS = load_coords(COORD_CSV_PATH)

# ─────────────────────────────────────────────────────────────
# 그래프 구성
# ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
# 그래프 구성 (수정 버전)
# ─────────────────────────────────────────────────────────────
graph = defaultdict(list)
def add_edge(u, v, w):
    graph[u].append((w, v))
    graph[v].append((w, u))

# 1) 주행 간선 추가
for ln, data in lines.items():
    # dict이면 각 branch 의 list, 아니면 전체 list 하나
    segments = data.values() if isinstance(data, dict) else [data]
    for seg in segments:
        # seg 은 반드시 list 이므로 seg[1:] 가능
        for a, b in zip(seg, seg[1:]):
            w = RUN_TIMES.get((a, b),
                              RUN_TIMES.get((b, a), DEFAULT_TRAVEL)) + DWELL
            add_edge((ln, a), (ln, b), w)

# 2) 순환선(loop) 처리 (dict key 가 '_loop' 로 끝날 때만)
for ln, data in lines.items():
    if isinstance(data, dict):
        for key, seg in data.items():
            if key.endswith('_loop'):
                a, b = seg[-1], seg[0]
                w = RUN_TIMES.get((a, b),
                                  RUN_TIMES.get((b, a), DEFAULT_TRAVEL)) + DWELL
                add_edge((ln, a), (ln, b), w)

# 3) 환승 간선 추가
st_to_lines = defaultdict(list)
for ln, data in lines.items():
    segments = data.values() if isinstance(data, dict) else [data]
    for seg in segments:
        for st in seg:
            st_to_lines[st].append(ln)

for st, lns in st_to_lines.items():
    for i in range(len(lns)):
        for j in range(i+1, len(lns)):
            u, v = (lns[i], st), (lns[j], st)
            w = TRANSFER_TIMES.get((u, v), FALLBACK_TRANSFER)
            add_edge(u, v, w)

# 역명→노드
station_to_nodes = defaultdict(list)
for node in graph:
    station_to_nodes[node[1]].append(node)

# ─────────────────────────────────────────────────────────────
# Bidirectional Dijkstra
# ─────────────────────────────────────────────────────────────
def bidir_dijkstra(start: str, goal: str):
    if start not in station_to_nodes or goal not in station_to_nodes:
        raise ValueError('역 이름 오류')
    starts = station_to_nodes[start]
    goals  = station_to_nodes[goal]

    pq_f = [(0, n) for n in starts]
    pq_b = [(0, n) for n in goals]
    heapq.heapify(pq_f); heapq.heapify(pq_b)

    dist_f = {n: 0 for n in starts}
    dist_b = {n: 0 for n in goals}
    parent_f = {n: None for n in starts}
    parent_b = {n: None for n in goals}

    best = float('inf'); meet = None

    while pq_f and pq_b:
        # 어느 방향 확장할지 결정 (g 값 작은 쪽)
        if pq_f[0][0] <= pq_b[0][0]:
            d, u = heapq.heappop(pq_f)
            if d > dist_f[u]: continue
            # 만남
            if u in dist_b and d + dist_b[u] < best:
                best, meet = d + dist_b[u], u
            # 확장
            if d < best:
                for w, v in graph[u]:
                    nd = d + w
                    if nd < dist_f.get(v, float('inf')):
                        dist_f[v] = nd; parent_f[v] = u
                        heapq.heappush(pq_f, (nd, v))
        else:
            d, u = heapq.heappop(pq_b)
            if d > dist_b[u]: continue
            if u in dist_f and d + dist_f[u] < best:
                best, meet = d + dist_f[u], u
            if d < best:
                for w, v in graph[u]:
                    nd = d + w
                    if nd < dist_b.get(v, float('inf')):
                        dist_b[v] = nd; parent_b[v] = u
                        heapq.heappush(pq_b, (nd, v))
        # 종료조건
        if meet and pq_f and pq_b and pq_f[0][0] + pq_b[0][0] >= best:
            break

    if not meet:
        return None, float('inf')
    # 복원
    path = []
    cur = meet
    while cur:
        path.append(cur); cur = parent_f[cur]
    path = path[::-1]
    cur = parent_b[meet]
    while cur:
        path.append(cur); cur = parent_b[cur]
    return path, best

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


def analyse_1000_random_astar(all_stations, sample_count=1000, output_csv="astar_results.csv"):
    random.seed(42)  # ❶ seed 고정 (반드시 함수 시작 시점에서)
    
    pairs = []
    seen = set()

    while len(pairs) < sample_count:
        s, g = random.sample(all_stations, 2)
        if s != g and (s, g) not in seen:
            pairs.append((s, g))  # ❷ 순서 있는 리스트 사용
            seen.add((s, g))

    records = []
    timelist = []
    success_count = 0

    for i, (s, g) in enumerate(pairs, 1):
        try:
            t0 = time.perf_counter()
            path, total = bidir_dijkstra(s, g)
            t1 = time.perf_counter()

            if path:
                elapsed = t1 - t0
                timelist.append(elapsed)
                success_count += 1
                formatted_path = fmt_path(path)

                print(f"[{i}] {s} → {g}")
                print(f"    총 소요 시간: {total:.1f}분")
                print(f"    경로: {formatted_path}\n")

                records.append({
                    "start": s,
                    "goal": g,
                    "elapsed_sec": round(elapsed, 6),
                    "total_minutes": round(total, 1),
                    "path": formatted_path
                })

        except Exception as e:
            print(f"[{i}] {s} → {g} ❌ 오류 발생: {e}\n")

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')

    summary = {
        "count": success_count,
        "total_time": sum(timelist),
        "mean_time": np.mean(timelist) if timelist else 0,
        "std_time": np.std(timelist) if timelist else 0,
        "best_time": min(timelist) if timelist else 0,
        "worst_time": max(timelist) if timelist else 0
    }

    return summary

# ───────────────────────────── CLI ─────────────────────────────
if __name__ == "__main__":
    random.seed(42)

    all_stations = generate_station_list()

    print("🚀 1000개 랜덤 출발-도착 쌍에 대해 A* 탐색 시작")
    summary = analyse_1000_random_astar(all_stations, sample_count=1000, output_csv="astar_1000_result.csv")

    print(f"\n📊 [요약] 성공: {summary['count']} / 1000")
    print(f"  총합      : {summary['total_time']:.6f}초")
    print(f"  평균      : {summary['mean_time']:.6f}초")
    print(f"  표준편차  : {summary['std_time']:.6f}초")
    print(f"  bestcase  : {summary['best_time']:.6f}초")
    print(f"  worstcase : {summary['worst_time']:.6f}초")