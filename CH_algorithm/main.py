# ----- Libraries -----
import glob, os, heapq, time, random
from collections import defaultdict, deque
from typing import List, Tuple, Dict, Set, Optional
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm

# Configure matplotlib to use AppleGothic font for Korean text
plt.rcParams['font.family'] = 'AppleGothic'
# Disable the minus sign issue when using Korean fonts
plt.rcParams['axes.unicode_minus'] = False

# ----- Constants -----
DEFAULT_TRAVEL    = 2        # 역간 데이터 없을 때 기본 주행 시간 (분)
FALLBACK_TRANSFER = 4        # 환승 CSV에 없을 경우 기본 환승 시간 (분)
DWELL             = 0.5      # 정차 시간 0.5분 (=30초)
TRAVEL_CSV_FOLDER = "CH_algorithm/dataset/역간소요시간(수작업)"
TRANSFER_CSV_PATH = "CH_algorithm/dataset/서울교통공사_환승역거리 소요시간 정보_20250331.csv"


# ----- Subway Station Lists -----
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
    "진접", "오남", "별가람", "당고개","상계","노원","창동","쌍문","수유","미아","미아사거리","길음","성신여대입구",
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

# ----- Load subway data and create subway graph -----
def _parse_minutes_seconds(text: str) -> float:
    """
    Parses a string in the format "MM:SS" and returns the total time in minutes as a float.
    """
    try:
        m, s = map(int, text.split(':'))
        return m + s / 60
    except Exception:
        return None
    
def _load_inter_station_travel_times():
    """
    Loads inter-station travel times from CSV files in the specified folder.
    Returns a dictionary of {(station1, station2): travel_time_in_minutes}.
    """
    run = {}
    for p in glob.glob(os.path.join(TRAVEL_CSV_FOLDER, "*.csv")):
        df = pd.read_csv(p, encoding="utf-8")
        prev = None
        for _, row in df.iterrows():
            cur = str(row["역명"]).strip()
            t_val = _parse_minutes_seconds(str(row["시간(분)"]).strip())
            if t_val is None:
                prev = cur
                continue
            if prev:
                run[(prev, cur)] = run[(cur, prev)] = t_val
            prev = cur
    return run

def _load_transfer_times():
    """
    Loads transfer times from the transfer CSV file.
    Returns a dictionary of {((line1, station), (line2, station)): transfer_time_in_minutes}.
    """
    tbl = {}
    df = pd.read_csv(TRANSFER_CSV_PATH, encoding="cp949")

    for _, row in df.iterrows():
        ln1 = int(row["호선"])
        st  = str(row["환승역명"]).strip()

        # '2호선' 같은 문자열에서 숫자만 추출
        digits = "".join(filter(str.isdigit, str(row["환승노선"])))
        if not digits:           # 숫자가 없으면 건너뜀
            continue
        ln2 = int(digits)

        t_val = _parse_minutes_seconds(str(row["환승소요시간"]).strip())
        if t_val is None:
            continue

        tbl[((ln1, st), (ln2, st))] = t_val
        tbl[((ln2, st), (ln1, st))] = t_val

    return tbl

def get_subway_graph() -> Tuple[List[str], List[Tuple[str, str, float]]]:
    """
    Constructs a subway graph with travel times between stations.
    Returns a tuple containing:
        - A list of all stations in the subway system.
        - A list of tuples representing edges in the graph, where each tuple is
          (station1, station2, travel_time_in_minutes).
    Each station is encoded as f{line_num}_{station_name} to ensure uniqueness.
    """
    inter_station_times = _load_inter_station_travel_times()
    transfer_times = _load_transfer_times()

    stations = set()
    edges = []

    for line_num, line_data in lines.items():
        for key, segment in (line_data.items() if isinstance(line_data, dict) else [("linear", line_data)]):
            for a, b in zip(segment, segment[1:]):
                station_a = f"{line_num}_{a}"
                station_b = f"{line_num}_{b}"
                stations.add(station_a)
                stations.add(station_b)

                travel_time = inter_station_times.get((a, b), inter_station_times.get((b, a), DEFAULT_TRAVEL))
                edges.append((station_a, station_b, travel_time + DWELL))

            # Handle circular segments (line 2)
            if key.endswith("_loop"):
                a, b = segment[-1], segment[0]
                station_a = f"{line_num}_{a}"
                station_b = f"{line_num}_{b}"
                travel_time = inter_station_times.get((a, b), inter_station_times.get((b, a), DEFAULT_TRAVEL))
                edges.append((station_a, station_b, travel_time + DWELL))

    # Add transfer edges
    # For the stations that's in multiple lines, add transfer edges
    station_to_lines = defaultdict(list)
    for line_num, line_data in lines.items():
        stations_in_line = (s for seg in line_data.values() for s in seg) if isinstance(line_data, dict) else line_data
        for station in stations_in_line:
            if not line_num in station_to_lines[station]: # Avoid duplicates
                station_to_lines[station].append(line_num)

    for station, line_nums in station_to_lines.items():
        if len(line_nums) < 2:
            continue
        
        for i in range(len(line_nums)):
            print(f"Processing transfers for {station} on lines {line_nums}")
            for j in range(i + 1, len(line_nums)):
                u = f"{line_nums[i]}_{station}"
                v = f"{line_nums[j]}_{station}"
                travel_time = transfer_times.get(
                    ((line_nums[i], station), (line_nums[j], station)),
                    FALLBACK_TRANSFER
                )
                edges.append((u, v, travel_time))

    return list(stations), edges


'''
# Debug the subway graph creation
if __name__ == "__main__":
    stations, edges = get_subway_graph()
    print(f"Total stations: {len(stations)}")
    print(f"Total edges: {len(edges)}")
    
    # Create a NetworkX graph for visualization
    G = nx.Graph()
    G.add_nodes_from(stations)
    G.add_weighted_edges_from(edges)
    
    # Draw the graph with better layout for Korean text
    plt.figure()
    
    # You might want to use a different layout for better visualization
    pos = nx.kamada_kawai_layout(G)
    
    # Draw nodes and edges separately for more control
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_nodes(G, pos, node_size=100, node_color='lightblue')
    nx.draw_networkx_labels(G, pos, font_size=8, font_family='AppleGothic')

    # Edge labels for travel times
    edge_labels = {(u, v): f"{w:.1f}분" for u, v, w in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_family='AppleGothic')
    
    plt.title("서울 지하철 노선도", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
'''

class ContractionHierarchy:
    def __init__(self, nodes: List[str], edges: List[Tuple[str, str, float]]):
        """
        Initialize the Contraction Hierarchy with nodes and edges.
        
        Args:
            nodes: List of node identifiers
            edges: List of (from_node, to_node, weight) tuples
        """
        self.nodes = set(nodes)
        self.original_graph = defaultdict(list)
        self.contracted_graph = defaultdict(list)
        
        # Build adjacency list for original graph (bidirectional)
        for u, v, w in edges:
            self.original_graph[u].append((v, w))
            self.original_graph[v].append((u, w))
            self.contracted_graph[u].append((v, w))
            self.contracted_graph[v].append((u, w))
        
        # Node ordering and level (importance)
        self.node_level = {}
        self.contracted_nodes = set()
        
        # Store shortcuts for path reconstruction
        self.shortcuts = {}  # (u, v) -> middle_node
        
    def _edge_difference(self, node: str) -> int:
        """
        Calculate the edge difference if we contract this node.
        This is the number of shortcuts created minus edges removed.
        """
        neighbors = [(v, w) for v, w in self.contracted_graph[node] 
                    if v not in self.contracted_nodes]
        
        shortcuts_needed = 0
        
        # Check each pair of neighbors
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                u, dist_u = neighbors[i]
                v, dist_v = neighbors[j]
                
                # Check if shortest path from u to v goes through node
                direct_dist = self._witness_search(u, v, node, dist_u + dist_v)
                
                if direct_dist > dist_u + dist_v:
                    shortcuts_needed += 1
        
        # Edge difference = shortcuts added - edges removed
        return shortcuts_needed - len(neighbors)
    
    def _witness_search(self, start: str, target: str, excluded: str, 
                       max_dist: float) -> float:
        """
        Find shortest path from start to target excluding the given node.
        Returns infinity if no path exists or if shortest path > max_dist.
        """
        if start == target:
            return 0
            
        distances = {start: 0}
        heap = [(0, start)]
        
        while heap:
            dist, u = heapq.heappop(heap)
            
            if dist > max_dist:
                return float('inf')
                
            if u == target:
                return dist
                
            if dist > distances.get(u, float('inf')):
                continue
                
            for v, w in self.contracted_graph[u]:
                if v == excluded or v in self.contracted_nodes:
                    continue
                    
                new_dist = dist + w
                
                if new_dist < distances.get(v, float('inf')):
                    distances[v] = new_dist
                    heapq.heappush(heap, (new_dist, v))
        
        return float('inf')
    
    def _contract_node(self, node: str):
        """Contract a single node and add necessary shortcuts."""
        neighbors = [(v, w) for v, w in self.contracted_graph[node] 
                    if v not in self.contracted_nodes]
        
        # Add shortcuts between all pairs of neighbors if needed
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                u, dist_u = neighbors[i]
                v, dist_v = neighbors[j]
                shortcut_dist = dist_u + dist_v
                
                # Check if we need a shortcut
                witness_dist = self._witness_search(u, v, node, shortcut_dist)
                
                if witness_dist > shortcut_dist:
                    # Add shortcut
                    self.contracted_graph[u].append((v, shortcut_dist))
                    self.contracted_graph[v].append((u, shortcut_dist))
                    
                    # Store shortcut info for path reconstruction
                    self.shortcuts[(u, v)] = node
                    self.shortcuts[(v, u)] = node
        
        self.contracted_nodes.add(node)
    
    def preprocess(self):
        """
        Perform the preprocessing phase using a simple bottom-up heuristic.
        """
        print("Starting preprocessing...")
        
        # Contract nodes in order of edge difference (greedy approach)
        remaining_nodes = list(self.nodes)
        level = 0
        
        while remaining_nodes:
            # Find node with minimum edge difference
            best_node = None
            best_score = float('inf')
            
            for node in remaining_nodes:
                if node in self.contracted_nodes:
                    continue
                    
                score = self._edge_difference(node)
                
                # Add small random factor to break ties
                score += len([v for v, _ in self.contracted_graph[node] 
                            if v in self.contracted_nodes]) * 0.1
                
                if score < best_score:
                    best_score = score
                    best_node = node
            
            if best_node is None:
                break
                
            # Contract the best node
            self._contract_node(best_node)
            self.node_level[best_node] = level
            remaining_nodes.remove(best_node)
            level += 1
            
            if level % 100 == 0:
                print(f"Contracted {level} nodes...")
        
        print(f"Preprocessing complete. Contracted {len(self.contracted_nodes)} nodes.")
    
    def _bidirectional_dijkstra(self, start: str, target: str) -> Tuple[float, List[str]]:
        """
        Perform bidirectional Dijkstra search using the contracted graph.
        """
        if start not in self.nodes or target not in self.nodes:
            return float('inf'), []
            
        # Forward search from start
        forward_dist = {start: 0}
        forward_heap = [(0, start)]
        forward_parent = {start: None}
        forward_visited = set()
        
        # Backward search from target
        backward_dist = {target: 0}
        backward_heap = [(0, target)]
        backward_parent = {target: None}
        backward_visited = set()
        
        best_distance = float('inf')
        meeting_node = None
        
        while forward_heap or backward_heap:
            # Forward search step
            if forward_heap:
                dist, u = heapq.heappop(forward_heap)
                
                if u in forward_visited:
                    continue
                    
                forward_visited.add(u)
                
                # Check if we've found a better path
                if u in backward_dist:
                    total_dist = dist + backward_dist[u]
                    if total_dist < best_distance:
                        best_distance = total_dist
                        meeting_node = u
                
                # Only go upward in hierarchy
                u_level = self.node_level.get(u, 0)
                
                for v, w in self.contracted_graph[u]:
                    v_level = self.node_level.get(v, 0)
                    
                    # Only consider edges going up in hierarchy
                    if v_level >= u_level:
                        new_dist = dist + w
                        
                        if new_dist < forward_dist.get(v, float('inf')):
                            forward_dist[v] = new_dist
                            forward_parent[v] = u
                            heapq.heappush(forward_heap, (new_dist, v))
            
            # Backward search step
            if backward_heap:
                dist, u = heapq.heappop(backward_heap)
                
                if u in backward_visited:
                    continue
                    
                backward_visited.add(u)
                
                # Check if we've found a better path
                if u in forward_dist:
                    total_dist = forward_dist[u] + dist
                    if total_dist < best_distance:
                        best_distance = total_dist
                        meeting_node = u
                
                # Only go upward in hierarchy
                u_level = self.node_level.get(u, 0)
                
                for v, w in self.contracted_graph[u]:
                    v_level = self.node_level.get(v, 0)
                    
                    # Only consider edges going up in hierarchy
                    if v_level >= u_level:
                        new_dist = dist + w
                        
                        if new_dist < backward_dist.get(v, float('inf')):
                            backward_dist[v] = new_dist
                            backward_parent[v] = u
                            heapq.heappush(backward_heap, (new_dist, v))
        
        if meeting_node is None:
            return float('inf'), []
        
        # Reconstruct path
        path = self._reconstruct_path(start, target, meeting_node, 
                                    forward_parent, backward_parent)
        
        return best_distance, path
    
    def _reconstruct_path(self, start: str, target: str, meeting: str,
                         forward_parent: Dict[str, str], 
                         backward_parent: Dict[str, str]) -> List[str]:
        """Reconstruct the shortest path."""
        # Build forward path
        forward_path = []
        node = meeting
        while node is not None:
            forward_path.append(node)
            node = forward_parent.get(node)
        forward_path.reverse()
        
        # Build backward path
        backward_path = []
        node = backward_parent.get(meeting)
        while node is not None:
            backward_path.append(node)
            node = backward_parent.get(node)
        
        # Combine paths
        full_path = forward_path + backward_path
        
        # Unpack shortcuts
        unpacked_path = []
        for i in range(len(full_path) - 1):
            u, v = full_path[i], full_path[i + 1]
            unpacked_path.extend(self._unpack_shortcut(u, v))
        unpacked_path.append(full_path[-1])
        
        return unpacked_path
    
    def _unpack_shortcut(self, u: str, v: str) -> List[str]:
        """Recursively unpack a shortcut to get the original path."""
        if (u, v) not in self.shortcuts:
            return [u]
        
        middle = self.shortcuts[(u, v)]
        left_path = self._unpack_shortcut(u, middle)
        right_path = self._unpack_shortcut(middle, v)
        
        return left_path + right_path
    
    def shortest_path(self, start: str, target: str) -> Tuple[float, List[str]]:
        """
        Find the shortest path from start to target.
        
        Returns:
            Tuple of (distance, path) where path is a list of nodes
        """
        return self._bidirectional_dijkstra(start, target)


def plot_ch_shortcuts(ch):
    """
    Visualize the shortcuts in the Contraction Hierarchy.
    """
    G = nx.Graph()
    
    # Add nodes
    G.add_nodes_from(ch.nodes)
    
    # Add edges from contracted graph
    for u, neighbors in ch.contracted_graph.items():
        for v, w in neighbors:
            if u < v:  # Avoid duplicate edges
                G.add_edge(u, v, weight=w)
    
    # Draw the graph
    pos = nx.spring_layout(G)
    # nx.draw(G, pos, with_labels=True, node_size=50, font_size=8)
    nx.draw_networkx_nodes(G, pos, node_size=20, node_color='lightblue')
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=6, font_family='AppleGothic')
    
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
    
    plt.title("Contraction Hierarchy Shortcuts")
    plt.show()


if __name__ == "__main__":
    # Load subway graph
    stations, edges = get_subway_graph()
    
    # Initialize Contraction Hierarchy
    ch = ContractionHierarchy(stations, edges)
    
    # Preprocess the graph
    start_time = time.perf_counter()
    ch.preprocess()
    end_time = time.perf_counter()

    # Plot the shortcuts in the Contraction Hierarchy
    # plot_ch_shortcuts(ch)
    
    print(f"Preprocessing completed in {end_time - start_time:.2f} seconds.")
    
    # Example usage: Find shortest path from "1_서울역" to "2_강남"
    start_time = time.perf_counter()
    distance, path = ch.shortest_path("1_서울역", "4_노원")
    end_time = time.perf_counter()
    print(f"Shortest path: {path} with distance {distance:.2f} minutes")
    print(f"Query completed in {end_time - start_time:.6f} seconds.")

    # Randomized test
    start_end_pair_sample = [random.sample(stations, 2) for _ in range(100)]
    test_results  = []
    for start, target in tqdm(start_end_pair_sample, desc="Running randomized tests", total=len(start_end_pair_sample)):
        # Run 30 times
        avg_time = 0
        for _ in range(30):
            start_time = time.perf_counter()
            distance, path = ch.shortest_path(start, target)
            end_time = time.perf_counter()
            avg_time += (end_time - start_time)
        avg_time /= 30
        test_results.append({
            'start': start,
            'target': target,
            'distance': distance,
            'path': path,
            'avg_time': avg_time
        })
    
    # Convert results to DataFrame for better visualization
    df_results = pd.DataFrame(test_results)
    # Save results to CSV
    df_results.to_csv("CH_algorithm/shortest_path_results.csv", index=False, encoding='utf-8-sig')

    # Print statistics
    print("\nRandomized Test Results:")
    print(df_results.describe())

    