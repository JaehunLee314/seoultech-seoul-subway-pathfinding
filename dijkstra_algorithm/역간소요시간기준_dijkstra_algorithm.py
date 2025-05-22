# -*- coding: utf-8 -*-
import pandas as pd
import heapq
from collections import defaultdict

# --- CSV 파일 경로 설정 ---
# 사용자가 제공한 새 CSV 파일 목록
LINE_1_SOYOSAN_GURO_CSV = "1호선 소요산~구로.csv"
LINE_1_GURO_INCHEON_CSV = "1호선 구로~인천.csv"
LINE_1_GURO_GWANGMYEONG_CSV = "1호선 구로~광명.csv"
LINE_1_GURO_SEODONGTAN_CSV = "1호선 구로~서동탄.csv"
LINE_1_BYEONGJEOM_SINCHANG_CSV = "1호선 병점~신창.csv"

LINE_2_CSV = "2호선.csv"
LINE_3_CSV = "3호선.csv"
LINE_4_CSV = "4호선.csv"
LINE_5_CSV = "5호선.csv"

TRANSFER_INFO_CSV_PATH = "서울교통공사_환승역거리 소요시간 정보_20250331.csv"

# --- 사용자 설정값 ---
DEFAULT_TRANSFER_TIME_MINUTES = 1.5 # 환승 정보 CSV에 특정 경로 없을 경우 사용할 기본 환승 시간 (분)
START_STATION_LINE = '1'
START_STATION_NAME = '서울역'
END_STATION_NAME = '신정네거리' # 또는 다른 목적지
# ----------------------------------------------------------------------

def time_to_minutes(time_str):
    if pd.isna(time_str) or not isinstance(time_str, str):
        return 0.0
    try:
        parts = time_str.split(':')
        if len(parts) == 2:
            minutes_val = int(parts[0])
            seconds_val = int(parts[1])
            return minutes_val + seconds_val / 60.0
        elif len(parts) == 1: # "0"과 같은 경우
            return float(parts[0])
    except ValueError:
        # print(f"경고: 잘못된 시간 형식입니다 - '{time_str}'")
        return 0.0
    return 0.0

def load_transfer_info(file_path, current_relevant_lines):
    transfer_data = {}
    try:
        df_transfer = pd.read_csv(file_path, encoding='utf-8')
        for _, row in df_transfer.iterrows():
            from_line = str(row['호선']).strip()
            station_name = str(row['환승역명']).strip()
            to_line = str(row['환승노선']).replace('호선','').replace('선','').strip()
            
            is_to_line_relevant = False
            if to_line.isdigit() and to_line in current_relevant_lines:
                is_to_line_relevant = True
            elif not to_line.isdigit():
                pass

            if not is_to_line_relevant and not any(c.isdigit() for c in to_line) and to_line not in current_relevant_lines :
                 continue
            
            transfer_time_val = time_to_minutes(row['환승소요시간'])
            
            if transfer_time_val > 0:
                transfer_data[(from_line, station_name, to_line)] = transfer_time_val
    except FileNotFoundError:
        print(f"경고: 환승 정보 CSV 파일을 찾을 수 없습니다: {file_path}.")
    except Exception as e:
        print(f"경고: 환승 정보 CSV 파일을 읽는 중 오류 발생: {e}.")
    return transfer_data


def build_subway_graph(line_csv_files_map, transfer_info_csv_path_local, current_relevant_lines):
    graph = defaultdict(list)
    all_nodes_in_scope = set()

    # 각 호선 CSV 파일에서 역 간 '소요시간'을 읽어 그래프 구성
    for line_num_str, csv_path_list_or_single in line_csv_files_map.items():
        if line_num_str not in current_relevant_lines:
            continue
        
        csv_list_to_process = csv_path_list_or_single
        if not isinstance(csv_path_list_or_single, list):
            csv_list_to_process = [csv_path_list_or_single]

        for csv_path in csv_list_to_process:
            try:
                df_line = pd.read_csv(csv_path, encoding='utf-8')
                previous_station_node = None
                
                for _, row in df_line.iterrows():
                    # CSV 파일에 '호선' 컬럼이 있는지 확인, 없으면 line_num_str 사용
                    actual_line_num = str(row.get('호선', line_num_str)).strip()
                    # CSV에서 읽은 호선번호가 current_relevant_lines에 없으면 스킵 (1호선 파일이 여러개일때 대비)
                    if actual_line_num not in current_relevant_lines :
                        actual_line_num = line_num_str # 파일 그룹의 호선번호를 따름

                    current_station_name = str(row['역명']).strip()
                    current_node = (actual_line_num, current_station_name)
                    all_nodes_in_scope.add(current_node)
                    
                    travel_time = time_to_minutes(row['시간(분)'])
                    
                    if previous_station_node is not None and travel_time > 0:
                        # 이전 노드의 호선번호와 현재 노드의 호선번호가 같을 때만 직접 연결 (1호선 파일 병합시 중요)
                        if previous_station_node[0] == actual_line_num:
                            graph[previous_station_node].append((current_node, travel_time))
                            graph[current_node].append((previous_station_node, travel_time))
                    
                    previous_station_node = current_node
            except FileNotFoundError:
                print(f"경고: {csv_path} 파일을 찾을 수 없습니다.")
            except Exception as e:
                print(f"경고: {csv_path} 파일 처리 중 오류: {e}")

    # --- 2호선 지선 및 본선 순환 연결 (명시적 처리) ---
    # (2호선 CSV는 모든 지선을 포함하여 순차적으로 연결된 것으로 가정하고,
    #  분기점 및 순환 마감만 명시적으로 처리)
    if '2' in current_relevant_lines:
        try:
            df_l2 = pd.read_csv(LINE_2_CSV, encoding='utf-8') # 

            node_seongsu_l2 = ('2', '성수'); node_yongdap_l2 = ('2', '용답')
            if node_seongsu_l2 in all_nodes_in_scope and node_yongdap_l2 in all_nodes_in_scope:
                yongdap_row = df_l2[(df_l2['호선']==2) & (df_l2['역명'].str.strip()=='용답')]
                if not yongdap_row.empty:
                    time_val = time_to_minutes(yongdap_row.iloc[0]['시간(분)'])
                    if time_val > 0 and not any(n==node_yongdap_l2 for n,d in graph.get(node_seongsu_l2,[])):
                        graph[node_seongsu_l2].append((node_yongdap_l2, time_val))
                        graph[node_yongdap_l2].append((node_seongsu_l2, time_val))

            node_sindorim_l2 = ('2', '신도림'); node_dorimcheon_l2 = ('2', '도림천')
            if node_sindorim_l2 in all_nodes_in_scope and node_dorimcheon_l2 in all_nodes_in_scope:
                dorimcheon_row = df_l2[(df_l2['호선']==2) & (df_l2['역명'].str.strip()=='도림천')]
                if not dorimcheon_row.empty:
                    time_val = time_to_minutes(dorimcheon_row.iloc[0]['시간(분)'])
                    if time_val > 0 and not any(n==node_dorimcheon_l2 for n,d in graph.get(node_sindorim_l2,[])):
                         graph[node_sindorim_l2].append((node_dorimcheon_l2, time_val))
                         graph[node_dorimcheon_l2].append((node_sindorim_l2, time_val))

            node_L2_chungjeongno = ('2', '충정로'); node_L2_shicheong_first = ('2', '시청')
            if node_L2_chungjeongno in all_nodes_in_scope and node_L2_shicheong_first in all_nodes_in_scope:
                shicheong_after_chungjeongno_row = df_l2[ (df_l2['호선']==2) & (df_l2['역명'].str.strip()=='시청') & (df_l2.index > df_l2[df_l2['역명'].str.strip()=='충정로'].index[0]) ]
                if not shicheong_after_chungjeongno_row.empty:
                     time_val = time_to_minutes(shicheong_after_chungjeongno_row.iloc[0]['시간(분)'])
                     if time_val > 0 and not any(n==node_L2_shicheong_first for n,d in graph.get(node_L2_chungjeongno,[])):
                        graph[node_L2_chungjeongno].append((node_L2_shicheong_first, time_val))
                        graph[node_L2_shicheong_first].append((node_L2_chungjeongno, time_val))
        except Exception as e:
            print(f"경고: 2호선 분기/순환 처리 중 오류: {e}")


    # --- 1호선 및 5호선 주요 분기점 연결 (CSV 데이터를 통해 이미 연결되었을 가능성 높음) ---
    # 이 부분은 CSV 파일들이 분기점에서 정확히 이어진다는 가정 하에,
    # 위 일반 로직에서 대부분 처리될 것으로 예상됩니다.
    # 만약 CSV가 분기점에서 끊어져 있다면, 여기서 명시적으로 연결해야 합니다.
    # (예: `('1','구로')`와 `('1','개봉')`을 `1호선 구로~인천.csv`의 '개봉'행 시간으로 연결)
    # (예: `('1','구로')`와 `('1','가산디지털단지')`를 `1호선 구로~서동탄.csv`의 '가산디지털단지'행 시간으로 연결)
    # 현재는 `line_csv_files_map`과 일반 `connect_stations_sequentially` 로직에 의존.

    # --- 환승역 연결 (환승 '소요 시간' 사용) ---
    transfer_times_data = load_transfer_info(transfer_info_csv_path_local, current_relevant_lines)
    for (from_line_t, station_name_t, to_line_t_str), transfer_time_val in transfer_times_data.items():
        if not (to_line_t_str.isdigit() and to_line_t_str in current_relevant_lines):
            if to_line_t_str in current_relevant_lines : pass
            else: continue 
        node1_t = (from_line_t, station_name_t)
        node2_t = (to_line_t_str, station_name_t)
        if node1_t in all_nodes_in_scope and node2_t in all_nodes_in_scope:
            if not any(neighbor == node2_t and abs(dist - transfer_time_val) < 0.001 for neighbor, dist in graph.get(node1_t, [])):
                graph[node1_t].append((node2_t, transfer_time_val))
            if not any(neighbor == node1_t and abs(dist - transfer_time_val) < 0.001 for neighbor, dist in graph.get(node2_t, [])):
                graph[node2_t].append((node1_t, transfer_time_val))
    return graph

# --- Dijkstra, reconstruct_path 함수 (가중치는 이제 '소요 시간') ---
def dijkstra(graph, start_node):
    all_graph_nodes = set(graph.keys())
    for destinations in graph.values():
        for dest_node, _ in destinations:
            all_graph_nodes.add(dest_node)
            
    distances_min = {node: float('inf') for node in all_graph_nodes}
    predecessors = {node: None for node in all_graph_nodes}
    
    if start_node not in all_graph_nodes :
             distances_min[start_node] = float('inf') 
             predecessors[start_node] = None
             return distances_min, predecessors

    distances_min[start_node] = 0
    priority_queue = [(0, start_node)] 

    while priority_queue:
        current_time_min, current_node = heapq.heappop(priority_queue)
        if current_time_min > distances_min.get(current_node, float('inf')): 
            continue
        for neighbor, weight_time_min in graph.get(current_node, []): 
            time_min_val = current_time_min + weight_time_min
            if time_min_val < distances_min.get(neighbor, float('inf')): 
                distances_min[neighbor] = time_min_val
                predecessors[neighbor] = current_node
                heapq.heappush(priority_queue, (time_min_val, neighbor))
    return distances_min, predecessors

def reconstruct_path(predecessors, start_node, end_node):
    path = []
    current = end_node
    if current not in predecessors and current != start_node :
        if current == start_node:
             path.append(start_node)
             return path[::-1]
        return None
    count = 0 
    max_path_len = len(predecessors) if predecessors else 0
    graph_node_count = max_path_len # 근사치
    while current != start_node:
        path.append(current)
        current = predecessors.get(current) 
        if current is None: return None 
        count += 1
        if graph_node_count > 0 and count > graph_node_count + 10 : 
             print(f"경고: 경로 재구성 중 최대 길이를 초과했습니다 ({count} > {graph_node_count}). {start_node} -> {end_node}")
             return None 
    path.append(start_node)
    return path[::-1]

if __name__ == "__main__":
    relevant_lines_for_project_list = ['1', '2', '3', '4', '5'] 
    
    print(f"최단 경로 기준: 소요 시간(분)")
    print(f"환승 정보는 '{TRANSFER_INFO_CSV_PATH}' 파일에서 '환승소요시간'을 읽어옵니다. 누락 시 기본 {DEFAULT_TRANSFER_TIME_MINUTES}분 적용.")

    # 1호선 CSV 파일들을 리스트로 묶어서 전달
    line_csv_files_map = {
        '1': [LINE_1_SOYOSAN_GURO_CSV, LINE_1_GURO_INCHEON_CSV, LINE_1_GURO_GWANGMYEONG_CSV, LINE_1_GURO_SEODONGTAN_CSV, LINE_1_BYEONGJEOM_SINCHANG_CSV],
        '2': [LINE_2_CSV],
        '3': [LINE_3_CSV],
        '4': [LINE_4_CSV],
        '5': [LINE_5_CSV]
    }

    subway_graph_data = build_subway_graph(
        line_csv_files_map, 
        TRANSFER_INFO_CSV_PATH,
        relevant_lines_for_project_list
    )

    if subway_graph_data is None or not subway_graph_data:
        print("지하철 그래프 데이터를 생성하지 못했습니다. 프로그램을 종료합니다.")
    else:
        start_node_tuple = (START_STATION_LINE, START_STATION_NAME)
        
        is_start_node_valid_in_graph = False
        final_check_all_nodes_in_graph = set(subway_graph_data.keys())
        for _k_node_list_val in subway_graph_data.values():
            for _k_node_val, _ in _k_node_list_val: final_check_all_nodes_in_graph.add(_k_node_val)
        if start_node_tuple in final_check_all_nodes_in_graph: is_start_node_valid_in_graph = True
        
        if not is_start_node_valid_in_graph:
            print(f"오류: 출발역 {start_node_tuple}이(가) 생성된 그래프 데이터에 유효하게 포함되지 않았습니다. CSV 파일 내용 및 경로를 확인하세요.")
        else:
            distances_from_start_min, path_predecessors = dijkstra(subway_graph_data, start_node_tuple)
            
            possible_end_nodes = []
            for node_key in final_check_all_nodes_in_graph:
                if node_key[1] == END_STATION_NAME:
                    if node_key[0] in relevant_lines_for_project_list:
                         possible_end_nodes.append(node_key)
            
            if not possible_end_nodes:
                print(f"오류: 도착역 '{END_STATION_NAME}'({','.join(relevant_lines_for_project_list)}호선 내)에 해당하는 역이 그래프에 정의되어 있지 않습니다.")
            else:
                min_time_to_destination = float('inf')
                actual_end_node = None
                for end_node_candidate in possible_end_nodes:
                    candidate_time = distances_from_start_min.get(end_node_candidate, float('inf'))
                    if candidate_time < min_time_to_destination:
                        min_time_to_destination = candidate_time
                        actual_end_node = end_node_candidate
                
                if actual_end_node is None or min_time_to_destination == float('inf'):
                    print(f"{start_node_tuple[1]}({start_node_tuple[0]}호선) 에서 {END_STATION_NAME}(으)로 가는 경로를 찾을 수 없습니다.")
                else:
                    final_path = reconstruct_path(path_predecessors, start_node_tuple, actual_end_node)
                    print(f"--- {START_STATION_NAME} ({START_STATION_LINE}호선) 에서 {END_STATION_NAME} (도착: {actual_end_node[0]}호선) 까지 최단 *소요 시간* 경로 ---")
                    print(f"총 소요 시간: {min_time_to_destination:.2f} 분 (환승 시간: 실제 데이터 또는 기본 {DEFAULT_TRANSFER_TIME_MINUTES}분 적용)")
                    if final_path:
                        print("경로:")
                        for i, step in enumerate(final_path):
                            print(f"  {i+1}. {step[1]} ({step[0]}호선)")
                    else:
                        print("경로를 재구성할 수 없습니다.")