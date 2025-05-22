import heapq
from collections import defaultdict

# 1) 1호선
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
    "incheon_branch": [ # 구로 : 분기점 연결 위해 포함
        "구로","구일","개봉","오류동","온수","역곡","소사","부천","중동","송내","부개",
        "부평","백운","동암","간석","주안","도화","제물포","도원","동인천","인천"
    ],
    "gwangmyeong_branch": ["금천구청","광명"], # 금청구청 : 분기점 연결 위해 포함
    "seodongtan_branch": ["병점","세마","서동탄"] # 병점 : 분기점 연결 위해 포함
}

# 2) 2호선
line2_stations = {
    "main_loop": [
        "시청","을지로입구","을지로3가","을지로4가","동대문역사문화공원","신당","상왕십리",
        "왕십리","한양대","뚝섬","성수","건대입구","구의","강변","잠실나루","잠실","잠실새내",
        "종합운동장","삼성","선릉","역삼","강남","교대","서초","방배","사당","낙성대",
        "서울대입구","봉천","신림","신대방","구로디지털단지","대림","신도림","문래",
        "영등포구청","당산","합정","홍대입구","신촌","이대","아현","충정로"
    ],
    "seongsu_branch": ["성수","용답","신답","용두","신설동"],
    "sinjeong_branch": ["신도림","도림천","양천구청","신정네거리","까치산"],
}

# 3) 3호선
line3_stations = [
    "대화","주엽","정발산","마두","백석","대곡","화정","원당","원흥","삼송","지축","구파발",
    "연신내","불광","녹번","홍제","무악재","독립문","경복궁","안국","종로3가","을지로3가",
    "충무로","동대입구","약수","금호","옥수","압구정","신사","잠원","고속터미널","교대",
    "남부터미널","양재","매봉","도곡","대치","학여울","대청","일원","수서","가락시장",
    "경찰병원","오금"
]

# 4) 4호선
line4_stations = [
    "당고개","상계","노원","창동","쌍문","수유","미아","미아사거리","길음","성신여대입구",
    "한성대입구","혜화","동대문","동대문역사문화공원","충무로","명동","회현","서울역",
    "숙대입구","삼각지","신용산","이촌","동작","이수","사당","남태령","선바위","경마공원",
    "대공원","과천","정부과천청사","인덕원","평촌","범계","금정","산본","수리산","대야미",
    "반월","상록수","한대앞","중앙","고잔","초지","안산","신길온천","정왕","오이도"
]

# 5) 5호선
line5_stations = {
    "main_line": [
        "방화","개화산","김포공항","송정","마곡","발산","우장산","화곡","까치산","신정","목동",
        "오목교","양평","영등포구청","영등포시장","신길","여의도","여의나루","마포","공덕","애오개",
        "충정로","서대문","광화문","종로3가","을지로4가","동대문역사문화공원","청구","신금호",
        "행당","왕십리","마장","답십리","장한평","군자","아차산","광나루","천호","강동",
        "길동","굽은다리","명일","고덕","상일동", "강일", "미사", "하남풍산", "하남시청", "하남검단산"
    ],
    "macheon_branch": ["강동","둔촌동","올림픽공원","방이","오금","개롱","거여","마천"]
}

# -----------------------------
# 1. 2~5호선(단선) 리스트는 그대로
# -----------------------------
lines = {
    1: line1_stations,
    2: line2_stations,
    3: line3_stations,
    4: line4_stations,
    5: line5_stations
}

# -----------------------------
# 2. 그래프 구축 로직 변경
# -----------------------------
from collections import defaultdict
import heapq
DEFAULT_TRAVEL = 2      # 인접 역 이동 시간(분) 데이터 수집 후 추가 편집 필요
TRANSFER_TIME  = 2      # 환승 시간(분)

def build_graph(lines_dict):
    graph = defaultdict(list)

    def add_edge(a, b, w):
        graph[a].append((w, b))
        graph[b].append((w, a))

    for ln, data in lines_dict.items():
        # (A) dict = 여러 구간   (B) list = 단선
        segments = data.values() if isinstance(data, dict) else [data]
        for key, seg in (data.items() if isinstance(data, dict) else [("linear", data)]):
            # 인접 역 연결
            for s1, s2 in zip(seg, seg[1:]):
                add_edge((ln, s1), (ln, s2), DEFAULT_TRAVEL)
            # loop이면 마지막-첫 역도 추가
            if key.endswith("_loop"):
                add_edge((ln, seg[-1]), (ln, seg[0]), DEFAULT_TRAVEL)

    # 환승 간선
    station_to_lines = defaultdict(list)
    for ln, data in lines_dict.items():
        stations_iter = (
            (st for seg in data.values() for st in seg)
            if isinstance(data, dict)
            else data
        )
        for st in stations_iter:
            station_to_lines[st].append(ln)

    for st, lns in station_to_lines.items():
        if len(lns) > 1:
            for i in range(len(lns)):
                for j in range(i + 1, len(lns)):
                    add_edge((lns[i], st), (lns[j], st), TRANSFER_TIME)

    return graph

# -------------------------------------------------------------
# constants & helpers
# -------------------------------------------------------------
def parse_node(token: str):
    """
    '2,강남' → (2, '강남')
    공백 자동 제거, 한글·영문 역명 모두 OK
    """
    line_str, station = map(str.strip, token.split(','))
    return int(line_str), station
# -------------------------------------------------------------
# 3. 준비: build_graph(lines) 는 그대로
# -------------------------------------------------------------
graph = build_graph(lines)

# 🔑  station → [(line, station)]   역 이름을 노드 리스트로 매핑
station_to_nodes = defaultdict(list)
for node in graph.keys():                 # graph 에 존재하는 모든 노드
    ln, st = node
    station_to_nodes[st].append(node)

# -------------------------------------------------------------
# 4. 다중 출발/도착 지원 Dijkstra
# -------------------------------------------------------------
def dijkstra_multi(graph, start_name, goal_name):
    if start_name not in station_to_nodes:
        raise ValueError(f"출발역 '{start_name}' 이(가) 노선 데이터에 없습니다.")
    if goal_name   not in station_to_nodes:
        raise ValueError(f"도착역 '{goal_name}' 이(가) 노선 데이터에 없습니다.")

    # (1) 슈퍼 소스 추가: 노드 ID는 ('S', '') 로 임시 사용
    SUPER = ('S', '')
    for n in station_to_nodes[start_name]:
        graph[SUPER].append((0, n))   # 가중치 0 간선

    # (2) 목적지 후보 세트
    goals = set(station_to_nodes[goal_name])

    # (3) Dijkstra
    dist   = {SUPER: 0}
    parent = {SUPER: None}
    pq     = [(0, SUPER)]

    final_goal = None
    while pq:
        cost, u = heapq.heappop(pq)
        if u in goals:               # 가장 먼저 꺼낸 목적지 == 최단
            final_goal = u
            break
        if cost > dist[u]:
            continue
        for w, v in graph[u]:
            nc = cost + w
            if v not in dist or nc < dist[v]:
                dist[v]   = nc
                parent[v] = u
                heapq.heappush(pq, (nc, v))

    if final_goal is None:
        return None, float('inf')

    # (4) 경로 복원 (슈퍼 소스 제거)
    path = []
    cur = final_goal
    while cur and cur != SUPER:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path, dist[final_goal]

# -------------------------------------------------------------
# 5. CLI  –  역 이름만 입력
# -------------------------------------------------------------
if __name__ == "__main__":
    try:
        s_name = input("출발역 이름: ").strip()
        g_name = input("도착역 이름: ").strip()

        path, total = dijkstra_multi(graph, s_name, g_name)
        if path is None:
            print("❌ 경로를 찾지 못했습니다.")
        else:
            txt = " → ".join([f"{ln}호선 {st}" for ln, st in path])
            print(f"\n[경로] {txt}")
            print(f"[소요 시간] {total}분  (역 간 {DEFAULT_TRAVEL}분 / 환승 {TRANSFER_TIME}분 기준)")

    except Exception as e:
        print("⚠️ 오류:", e)
