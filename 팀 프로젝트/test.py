# -*- coding: utf-8 -*-
import pandas as pd
import heapq
from collections import defaultdict

# --- CSV 파일 경로 설정 ---
STATION_TRAVEL_INFO_SEOUL_METRO_CSV_PATH = "서울교통공사_역간거리.csv" # 
STATION_TRAVEL_INFO_KORAIL_CSV_PATH = "국가철도공단_코레일 역간거리_20240430.csv" # 
TRANSFER_INFO_CSV_PATH = "서울교통공사_환승역거리 소요시간 정보_20250331.csv" # 

# --- 사용자 설정값 ---
DEFAULT_TRANSFER_DISTANCE_KM = 0.150
START_STATION_LINE = '1'
START_STATION_NAME = '서울역'
END_STATION_NAME = '까치산'

# --- 분기점 실제 선로 이동 거리 (km) ---
DISTANCE_SINDORIM_DORIMCHEON_KM = 1.0
DISTANCE_SEONGSU_YONGDAP_KM = 2.3
DISTANCE_GURO_GASAN_KM = 2.0
DISTANCE_GEUMCHEON_GWANGMYEONG_KM = 2.2 # 실제 코레일 CSV 상 금천구청-광명은 4.8km. 확인 필요.
DISTANCE_GURO_GAEBONG_KM = 1.0      # (구로-구일 1.0) + (구일-개봉 1.0) = 2.0km 이거나, project_stations 순서에 따라 처리됨. 여기서는 단일 구간 가정.
DISTANCE_BYEONGJEOM_SEODONGTAN_KM = 2.2 # 코레일CSV '서동탄' 행 '역간거리' 값
DISTANCE_BYEONGJEOM_SEMA_KM = 2.7       # 코레일CSV '세마' 행 '역간거리' 값 (병점-세마 직접 연결 가정 시)
DISTANCE_GANGDONG_GILDONG_KM = 0.9    # 서울교통공사CSV '길동' 행 '거리(km)' 값
DISTANCE_GANGDONG_DUNCHON_KM = 1.2  # 서울교통공사CSV '둔촌동' 행 '거리(km)' 값 (단, CSV상 이전역이 강동인지 확인필요)
# ----------------------------------------------------------------------

relevant_lines_for_project_list_global = []

def load_transfer_distances_data(file_path, current_relevant_lines):
    transfer_data = {}
    try:
        df_transfer = pd.read_csv(file_path, encoding='cp949') # 
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
            
            try:
                distance_m_str = str(row['환승거리']).replace('m','').replace('미터','') # 
                distance_m = pd.to_numeric(distance_m_str, errors='coerce')
            except Exception:
                distance_m = pd.NA

            if pd.notna(distance_m) and distance_m >= 0:
                distance_km = distance_m / 1000.0
                
                # --- 신도림역 1호선-2호선 환승 거리 특별 처리 ---
                if station_name == '신도림' and \
                   ((from_line == '1' and to_line == '2') or \
                    (from_line == '2' and to_line == '1')):
                    # 사용자 확인 정보: 신도림 1-2호선 환승은 도보 약 2분.
                    # 2분에 해당하는 환승 거리로 0.15km (150m)를 가정하여 적용.
                    # (참고: 환승정보CSV 에서 행 33번 ("2","신도림","1호선",81,"01:08")는 0.081km / 1분 8초 임)
                    user_defined_sindorim_transfer_km = 0.150 # 예: 2분 도보거리 150m로 가정
                    transfer_data[(from_line, station_name, to_line)] = user_defined_sindorim_transfer_km
                    # print(f"Debug: Applied user-defined transfer distance for Sindorim 1-2: {user_defined_sindorim_transfer_km} km")
                else:
                    transfer_data[(from_line, station_name, to_line)] = distance_km
                # --- 특별 처리 종료 ---
    except FileNotFoundError:
        print(f"경고: 환승 정보 CSV 파일을 찾을 수 없습니다: {file_path}.")
    except Exception as e:
        print(f"경고: 환승 정보 CSV 파일을 읽는 중 오류 발생: {e}.")
    return transfer_data

def build_subway_graph_by_distance(sm_station_csv_path, korail_station_csv_path, transfer_info_csv_path_param, current_relevant_lines):
    station_name_to_row_sm = {}
    station_name_to_row_korail = {}

    try:
        df_sm = pd.read_csv(sm_station_csv_path, encoding='cp949') # 
        df_sm['호선'] = df_sm['호선'].astype(str)
        df_sm_filtered = df_sm[df_sm['호선'].isin(current_relevant_lines)].copy()
        df_sm_filtered.loc[:, '거리(km)_numeric'] = pd.to_numeric(df_sm_filtered['거리(km)'], errors='coerce').fillna(0.0)
        for line_num, group in df_sm_filtered.groupby('호선'):
            station_name_to_row_sm[line_num] = {str(row['역명']).strip(): row for _, row in group.iterrows()}
    except FileNotFoundError:
        print(f"오류: 서울교통공사 역간 거리 CSV 파일을 찾을 수 없습니다: {sm_station_csv_path}")
        return None
    except Exception as e:
        print(f"오류: 서울교통공사 역간 거리 CSV 처리 중 오류: {e}")
        return None

    try:
        df_korail = pd.read_csv(korail_station_csv_path, encoding='cp949') # 
        def map_korail_line(line_name_korail):
            if any(kw in line_name_korail for kw in ['1호선', '경부선', '경인선', '장항선']): return '1'
            if any(kw in line_name_korail for kw in ['3호선', '일산선']): return '3'
            if any(kw in line_name_korail for kw in ['4호선', '과천선', '안산선']): return '4'
            return None
        df_korail['호선'] = df_korail['선명'].apply(map_korail_line)
        df_korail_filtered = df_korail[df_korail['호선'].isin(current_relevant_lines)].copy()
        df_korail_filtered.loc[:, '역간거리_numeric'] = pd.to_numeric(df_korail_filtered['역간거리'], errors='coerce').fillna(0.0)
        for line_num, group in df_korail_filtered.groupby('호선'):
            if line_num is None: continue
            if line_num not in station_name_to_row_korail: station_name_to_row_korail[line_num] = {}
            for _, row in group.iterrows():
                station_name = str(row['역명']).strip()
                if not (line_num in station_name_to_row_sm and station_name in station_name_to_row_sm[line_num]):
                    station_name_to_row_korail[line_num][station_name] = row
    except FileNotFoundError:
        print(f"오류: 코레일 역간 거리 CSV 파일을 찾을 수 없습니다: {korail_station_csv_path}")
        return None
    except Exception as e:
        print(f"오류: 코레일 역간 거리 CSV 처리 중 오류: {e}")
        return None

    transfer_distances_data = load_transfer_distances_data(transfer_info_csv_path_param, current_relevant_lines)

    graph = defaultdict(list)
    all_nodes_in_scope = set()

    project_stations = {
        '1': ['소요산', '동두천', '보산', '동두천중앙', '지행', '덕정', '덕계', '양주', '녹양', '가능', '의정부', '회룡', '망월사', '도봉산', '도봉', '방학', '창동', '녹천', '월계', '광운대', '석계', '신이문', '외대앞', '회기', '청량리', '제기동', '신설동', '동묘앞', '동대문', '종로5가', '종로3가', '종각', '시청', '서울역', '남영', '용산', '노량진', '대방', '신길', '영등포', '신도림', '구로', '구일', '개봉', '오류동', '온수', '역곡', '소사', '부천', '중동', '송내', '부개', '부평', '백운', '동암', '간석', '주안', '도화', '제물포', '도원', '동인천', '인천', '가산디지털단지', '독산', '금천구청', '광명', '석수', '관악', '안양', '명학', '금정', '군포', '당정', '의왕', '성균관대', '화서', '수원', '세류', '병점', '서동탄', '세마', '오산대', '오산', '진위', '송탄', '서정리', '지제', '평택', '성환', '직산', '두정', '천안', '봉명', '쌍용', '아산', '배방', '온양온천', '신창'],
        '2_main': ['시청', '을지로입구', '을지로3가', '을지로4가', '동대문역사문화공원', '신당', '상왕십리', '왕십리', '한양대', '뚝섬', '성수', '건대입구', '구의', '강변', '잠실나루', '잠실', '잠실새내', '종합운동장', '삼성', '선릉', '역삼', '강남', '교대', '서초', '방배', '사당', '낙성대', '서울대입구', '봉천', '신림', '신대방', '구로디지털단지', '대림', '신도림', '문래', '영등포구청', '당산', '합정', '홍대입구', '신촌', '이대', '아현', '충정로'],
        '2_seongsu_branch': ['용답', '신답', '용두', '신설동'],
        '2_sinjeong_branch': ['도림천', '양천구청', '신정네거리', '까치산'],
        '3': ['대화', '주엽', '정발산', '마두', '백석', '대곡', '화정', '원당', '원흥','삼송', '지축', '구파발', '연신내', '불광', '녹번', '홍제', '무악재', '독립문', '경복궁', '안국', '종로3가', '을지로3가', '충무로', '동대입구', '약수', '금호', '옥수', '압구정', '신사', '잠원', '고속터미널', '교대', '남부터미널', '양재', '매봉', '도곡', '대치', '학여울', '대청', '일원', '수서', '가락시장', '경찰병원', '오금'],
        '4': ['진접', '오남', '별내별가람', '당고개', '상계', '노원', '창동', '쌍문', '수유', '미아', '미아사거리', '길음', '성신여대입구', '한성대입구', '혜화', '동대문', '동대문역사문화공원', '충무로', '명동', '회현', '서울역', '숙대입구', '삼각지', '신용산', '이촌', '동작', '총신대입구(이수)', '사당', '남태령', '선바위', '경마공원', '대공원', '과천', '정부과천청사', '인덕원', '평촌', '범계', '금정', '산본', '수리산', '대야미', '반월', '상록수', '한대앞', '중앙', '고잔', '초지', '안산', '신길온천', '정왕', '오이도'],
        '5_main_to_sangil': ['방화', '개화산', '김포공항', '송정', '마곡', '발산', '우장산', '화곡', '까치산', '신정', '목동', '오목교', '양평', '영등포구청', '영등포시장', '신길', '여의도', '여의나루', '마포', '공덕', '애오개', '충정로', '서대문', '광화문', '종로3가', '을지로4가', '동대문역사문화공원', '청구', '신금호', '행당', '왕십리', '마장', '답십리', '장한평', '군자', '아차산', '광나루', '천호', '강동', '길동', '굽은다리', '명일', '고덕', '상일동'],
        '5_gangdong_to_macheon': ['둔촌동', '올림픽공원', '방이', '오금', '개롱', '거여', '마천'] 
    }
    project_stations['1'] = [s.replace('평택지제','지제').replace('쌍용(나사렛대)','쌍용').replace('신창(순천향대)','신창').replace('성북','광운대') for s in project_stations['1']]
    project_stations['4'] = [s.replace('미아삼거리','미아사거리').replace('이수','총신대입구(이수)') for s in project_stations['4']]
    project_stations['2_main'] = [s.replace('신천','잠실새내') for s in project_stations['2_main']] # 2호선 본선에만 적용

    for line_key, station_name_list_segment in project_stations.items():
        line_num_str = line_key.split('_')[0] 
        if line_num_str not in current_relevant_lines:
            continue

        previous_station_name_in_segment = None
        for station_name in station_name_list_segment:
            current_node = (line_num_str, station_name)
            all_nodes_in_scope.add(current_node)
            
            if previous_station_name_in_segment:
                distance_km = 0
                # 1. 서울교통공사 데이터에서 거리 찾기
                sm_curr_station_info = station_name_to_row_sm.get(line_num_str, {}).get(station_name)
                if sm_curr_station_info is not None:
                    # CSV의 '거리(km)'는 해당 역과 그 CSV상 이전 역과의 거리임
                    # project_stations 리스트상의 이전 역과 SM CSV상의 이전 역이 일치하는지 확인하는 것이 가장 정확하나,
                    # 여기서는 현재 역의 SM CSV row에 있는 '거리(km)'를 사용.
                    # 단, SM CSV에서 첫번째 역이거나 누계가 0인경우 이 거리는 0이므로 사용 안함.
                    is_sm_first_in_csv_segment = (sm_curr_station_info.get('누계(km)') == 0.0 or sm_curr_station_info.get('시간(분)') == '0:00')
                    if not is_sm_first_in_csv_segment:
                         distance_km = sm_curr_station_info.get('거리(km)_numeric', 0)
                
                # 2. 코레일 데이터에서 거리 찾기 (SM에서 못찾았거나 0인 경우)
                if distance_km == 0:
                    korail_curr_station_info = station_name_to_row_korail.get(line_num_str, {}).get(station_name)
                    if korail_curr_station_info is not None:
                        # 코레일 CSV에서 현재 역의 '역간거리'는 그 CSV상 이전 역과의 거리임.
                        # 코레일 CSV에서 첫번째 역은 '역간거리'가 0.
                        is_korail_first_in_csv_segment = (korail_curr_station_info.get('역간거리_numeric', 0) == 0 and 
                                                       station_name_list_segment.index(station_name) == 0 and # 프로젝트 세그먼트의 첫 역이면서
                                                       korail_curr_station_info.get('역간거리_numeric', -1) == 0) # 코레일CSV에서도 역간거리가 0이면 진짜 첫 역

                        if not is_korail_first_in_csv_segment:
                            distance_km = korail_curr_station_info.get('역간거리_numeric', 0)
                
                if distance_km > 0:
                    graph[(line_num_str, previous_station_name_in_segment)].append((current_node, distance_km))
                    graph[current_node].append(((line_num_str, previous_station_name_in_segment), distance_km))
            previous_station_name_in_segment = station_name
            
    # --- 특정 분기점 명시적 연결 (거리 기반) ---
    # (이전과 동일하게 유지, 이 연결들은 위 순차연결에서 빠졌을 경우를 대비한 보강 또는 명시적 지정)
    node_sindorim_l2 = ('2', '신도림'); node_dorimcheon_l2 = ('2', '도림천')
    if node_sindorim_l2 in all_nodes_in_scope and node_dorimcheon_l2 in all_nodes_in_scope:
        if not any(n == node_dorimcheon_l2 and abs(d - DISTANCE_SINDORIM_DORIMCHEON_KM) < 0.001 for n,d in graph.get(node_sindorim_l2,[])):
            graph[node_sindorim_l2].append((node_dorimcheon_l2, DISTANCE_SINDORIM_DORIMCHEON_KM))
            graph[node_dorimcheon_l2].append((node_sindorim_l2, DISTANCE_SINDORIM_DORIMCHEON_KM))
    node_seongsu_l2 = ('2', '성수'); node_yongdap_l2 = ('2', '용답')
    if node_seongsu_l2 in all_nodes_in_scope and node_yongdap_l2 in all_nodes_in_scope:
        if not any(n == node_yongdap_l2 and abs(d - DISTANCE_SEONGSU_YONGDAP_KM) < 0.001 for n,d in graph.get(node_seongsu_l2,[])):
            graph[node_seongsu_l2].append((node_yongdap_l2, DISTANCE_SEONGSU_YONGDAP_KM))
            graph[node_yongdap_l2].append((node_seongsu_l2, DISTANCE_SEONGSU_YONGDAP_KM))

    node_guro_l1 = ('1', '구로'); node_gaebong_l1 = ('1', '개봉'); node_gasan_l1 = ('1', '가산디지털단지')
    if node_guro_l1 in all_nodes_in_scope:
        if node_gaebong_l1 in all_nodes_in_scope and not any(n==node_gaebong_l1 for n,d in graph.get(node_guro_l1,[])):
            graph[node_guro_l1].append((node_gaebong_l1, DISTANCE_GURO_GAEBONG_KM))
            graph[node_gaebong_l1].append((node_guro_l1, DISTANCE_GURO_GAEBONG_KM))
        if node_gasan_l1 in all_nodes_in_scope and not any(n==node_gasan_l1 for n,d in graph.get(node_guro_l1,[])):
            graph[node_guro_l1].append((node_gasan_l1, DISTANCE_GURO_GASAN_KM))
            graph[node_gasan_l1].append((node_guro_l1, DISTANCE_GURO_GASAN_KM))
    
    node_geumcheon_l1 = ('1', '금천구청'); node_gwangmyeong_l1 = ('1', '광명')
    if node_geumcheon_l1 in all_nodes_in_scope and node_gwangmyeong_l1 in all_nodes_in_scope and not any(n==node_gwangmyeong_l1 for n,d in graph.get(node_geumcheon_l1,[])):
        graph[node_geumcheon_l1].append((node_gwangmyeong_l1, DISTANCE_GEUMCHEON_GWANGMYEONG_KM))
        graph[node_gwangmyeong_l1].append((node_geumcheon_l1, DISTANCE_GEUMCHEON_GWANGMYEONG_KM))

    node_byeongjeom_l1 = ('1', '병점'); node_seodongtan_l1 = ('1', '서동탄'); node_sema_l1 = ('1', '세마')
    if node_byeongjeom_l1 in all_nodes_in_scope:
        if node_seodongtan_l1 in all_nodes_in_scope and not any(n==node_seodongtan_l1 for n,d in graph.get(node_byeongjeom_l1,[])):
            graph[node_byeongjeom_l1].append((node_seodongtan_l1, DISTANCE_BYEONGJEOM_SEODONGTAN_KM))
            graph[node_seodongtan_l1].append((node_byeongjeom_l1, DISTANCE_BYEONGJEOM_SEODONGTAN_KM))
        if node_sema_l1 in all_nodes_in_scope and not any(n==node_sema_l1 for n,d in graph.get(node_byeongjeom_l1,[])):
            graph[node_byeongjeom_l1].append((node_sema_l1, DISTANCE_BYEONGJEOM_SEMA_KM))
            graph[node_sema_l1].append((node_byeongjeom_l1, DISTANCE_BYEONGJEOM_SEMA_KM))
    
    node_gangdong_l5 = ('5', '강동'); node_gildong_l5 = ('5', '길동'); node_dunchon_l5 = ('5', '둔촌동')
    if node_gangdong_l5 in all_nodes_in_scope:
        if node_gildong_l5 in all_nodes_in_scope and not any(n==node_gildong_l5 for n,d in graph.get(node_gangdong_l5,[])):
            graph[node_gangdong_l5].append((node_gildong_l5, DISTANCE_GANGDONG_GILDONG_KM))
            graph[node_gildong_l5].append((node_gangdong_l5, DISTANCE_GANGDONG_GILDONG_KM))
        if node_dunchon_l5 in all_nodes_in_scope and not any(n==node_dunchon_l5 for n,d in graph.get(node_gangdong_l5,[])):
            graph[node_gangdong_l5].append((node_dunchon_l5, DISTANCE_GANGDONG_DUNCHON_KM))
            graph[node_dunchon_l5].append((node_gangdong_l5, DISTANCE_GANGDONG_DUNCHON_KM))

    # --- 환승역 연결 (환승 '거리' 사용) ---
    for (from_line_t, station_name_t, to_line_t_str), transfer_dist_km in transfer_distances_data.items():
        if not (to_line_t_str.isdigit() and to_line_t_str in current_relevant_lines):
            if to_line_t_str in current_relevant_lines: pass
            else: continue 
        node1_t = (from_line_t, station_name_t)
        node2_t = (to_line_t_str, station_name_t)
        if node1_t in all_nodes_in_scope and node2_t in all_nodes_in_scope:
            if not any(neighbor == node2_t and abs(dist - transfer_dist_km) < 0.001 for neighbor, dist in graph.get(node1_t, [])):
                graph[node1_t].append((node2_t, transfer_dist_km))
            if not any(neighbor == node1_t and abs(dist - transfer_dist_km) < 0.001 for neighbor, dist in graph.get(node2_t, [])):
                graph[node2_t].append((node1_t, transfer_dist_km))
    return graph

# --- Dijkstra, reconstruct_path 함수 (이전과 동일) ---
def dijkstra(graph, start_node):
    all_graph_nodes = set(graph.keys())
    for destinations in graph.values():
        for dest_node, _ in destinations:
            all_graph_nodes.add(dest_node)
            
    distances_km = {node: float('inf') for node in all_graph_nodes}
    predecessors = {node: None for node in all_graph_nodes}
    
    if start_node not in all_graph_nodes :
             distances_km[start_node] = float('inf') 
             predecessors[start_node] = None
             return distances_km, predecessors

    distances_km[start_node] = 0
    priority_queue = [(0, start_node)] 

    while priority_queue:
        current_distance_km, current_node = heapq.heappop(priority_queue)
        if current_distance_km > distances_km.get(current_node, float('inf')): 
            continue
        for neighbor, weight_km in graph.get(current_node, []): 
            distance_km_val = current_distance_km + weight_km
            if distance_km_val < distances_km.get(neighbor, float('inf')): 
                distances_km[neighbor] = distance_km_val
                predecessors[neighbor] = current_node
                heapq.heappush(priority_queue, (distance_km_val, neighbor))
    return distances_km, predecessors

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
    while current != start_node:
        path.append(current)
        current = predecessors.get(current) 
        if current is None: return None 
        count += 1
        if max_path_len > 0 and count > max_path_len + 50 :
             print(f"경고: 경로 재구성 중 최대 길이를 초과했습니다 ({count} > {max_path_len}). {start_node} -> {end_node}")
             return None 
    path.append(start_node)
    return path[::-1]

if __name__ == "__main__":
    relevant_lines_for_project_list = ['1', '2', '3', '4', '5']
    # load_transfer_distances_data 에서 사용될 수 있도록 전역 변수에도 할당
    # 이 방식보다는 load_transfer_distances_data에 relevant_lines_for_project_list를 직접 전달하는 것이 좋음.
    # (이전 코드에서 build_subway_graph_by_distance를 통해 전달되도록 수정됨)
    # global relevant_lines_for_project_list_global
    # relevant_lines_for_project_list_global = relevant_lines_for_project_list

    print(f"최단 경로 기준: 거리(km)")
    print(f"참고: 2호선 분기점 연결 거리(km): 신도림-도림천 {DISTANCE_SINDORIM_DORIMCHEON_KM}, 성수-용답 {DISTANCE_SEONGSU_YONGDAP_KM}")
    print(f"참고: 1호선 분기점 연결 거리(km): 구로-가산 {DISTANCE_GURO_GASAN_KM}, 금천-광명 {DISTANCE_GEUMCHEON_GWANGMYEONG_KM} (가정치 또는 CSV 직접 참조)")
    print(f"환승 정보는 '{TRANSFER_INFO_CSV_PATH}' 파일에서 '환승거리'를 읽어옵니다. 누락 시 기본 {DEFAULT_TRANSFER_DISTANCE_KM}km 적용.")

    subway_graph_data = build_subway_graph_by_distance(
        STATION_TRAVEL_INFO_SEOUL_METRO_CSV_PATH,
        STATION_TRAVEL_INFO_KORAIL_CSV_PATH,
        TRANSFER_INFO_CSV_PATH,
        relevant_lines_for_project_list
    )

    if subway_graph_data is None or not subway_graph_data:
        print("지하철 그래프 데이터를 생성하지 못했습니다. 프로그램을 종료합니다.")
    else:
        # --- 기존 경로 탐색 (사용자 설정 출발지/목적지) ---
        start_node_tuple_main = (START_STATION_LINE, START_STATION_NAME)
        end_name_main = END_STATION_NAME

        is_start_node_valid_in_graph = False
        final_check_all_nodes_in_graph = set(subway_graph_data.keys())
        for _k_node_list_val in subway_graph_data.values():
            for _k_node_val, _ in _k_node_list_val: final_check_all_nodes_in_graph.add(_k_node_val)
        if start_node_tuple_main in final_check_all_nodes_in_graph: is_start_node_valid_in_graph = True
        
        if not is_start_node_valid_in_graph:
            print(f"오류: 출발역 {start_node_tuple_main}이(가) 생성된 그래프 데이터에 유효하게 포함되지 않았습니다.")
        else:
            distances_from_start_km, path_predecessors = dijkstra(subway_graph_data, start_node_tuple_main)
            possible_end_nodes = []
            for node_key in final_check_all_nodes_in_graph:
                if node_key[1] == end_name_main:
                    if node_key[0] in relevant_lines_for_project_list:
                         possible_end_nodes.append(node_key)
            
            if not possible_end_nodes:
                print(f"오류: 목적역 '{end_name_main}'({','.join(relevant_lines_for_project_list)}호선 내)에 해당하는 역이 그래프에 정의되어 있지 않습니다.")
            else:
                min_dist_to_destination = float('inf')
                actual_end_node_main = None # 변수명 변경
                for end_node_candidate in possible_end_nodes:
                    candidate_dist = distances_from_start_km.get(end_node_candidate, float('inf'))
                    if candidate_dist < min_dist_to_destination:
                        min_dist_to_destination = candidate_dist
                        actual_end_node_main = end_node_candidate
                
                if actual_end_node_main is None or min_dist_to_destination == float('inf'):
                    print(f"{start_node_tuple_main[1]}({start_node_tuple_main[0]}호선) 에서 {end_name_main}(으)로 가는 경로를 찾을 수 없습니다.")
                else:
                    final_path = reconstruct_path(path_predecessors, start_node_tuple_main, actual_end_node_main)
                    print(f"--- [{START_STATION_NAME}({START_STATION_LINE}호선) 에서 {END_STATION_NAME} (도착: {actual_end_node_main[0]}호선)] 최단 *거리* 경로 (알고리즘 결과) ---")
                    print(f"총 거리: {min_dist_to_destination:.2f} km")
                    if final_path:
                        print("경로:")
                        for i, step in enumerate(final_path):
                            print(f"  {i+1}. {step[1]} ({step[0]}호선)")
                    else:
                        print("  경로를 재구성할 수 없습니다.")
        
        print("\n" + "="*50)
        print("--- 경로 1 (서울역(1) → 신도림(1) → 신도림(2) → 까치산(2)) 거리 계산 ---")

        # 경로 1의 각 지점 정의
        route1_start = ('1', '서울역')
        route1_via_sindorim_l1 = ('1', '신도림')
        route1_via_sindorim_l2 = ('2', '신도림')
        route1_end_kkachisan_l2 = ('2', '까치산')

        dist_r1_part1 = float('inf')
        dist_r1_transfer_sindorim = float('inf')
        dist_r1_part3 = float('inf')

        # 1. 서울역(1) -> 신도림(1) 거리
        if route1_start in final_check_all_nodes_in_graph and route1_via_sindorim_l1 in final_check_all_nodes_in_graph:
            distances_r1_p1, _ = dijkstra(subway_graph_data, route1_start)
            dist_r1_part1 = distances_r1_p1.get(route1_via_sindorim_l1, float('inf'))
            if dist_r1_part1 == float('inf'):
                print(f"  경로 1 오류: {route1_start}에서 {route1_via_sindorim_l1}(으)로 가는 경로를 찾을 수 없습니다.")
            else:
                print(f"  1. {route1_start[1]}({route1_start[0]}) → {route1_via_sindorim_l1[1]}({route1_via_sindorim_l1[0]}) 거리: {dist_r1_part1:.2f} km")
        else:
            print(f"  경로 1 오류: {route1_start} 또는 {route1_via_sindorim_l1} 노드가 그래프에 없습니다.")

        # 2. 신도림역 1호선-2호선 환승 거리
        # 그래프에서 직접 해당 간선의 가중치를 찾습니다.
        found_transfer_dist_sindorim = False
        if route1_via_sindorim_l1 in subway_graph_data:
            for neighbor, dist_km in subway_graph_data.get(route1_via_sindorim_l1, []):
                if neighbor == route1_via_sindorim_l2:
                    dist_r1_transfer_sindorim = dist_km
                    found_transfer_dist_sindorim = True
                    break
        if found_transfer_dist_sindorim:
            print(f"  2. 신도림역 환승 ({route1_via_sindorim_l1[0]}호선 ↔ {route1_via_sindorim_l2[0]}호선) 거리: {dist_r1_transfer_sindorim:.3f} km")
        else:
            dist_r1_transfer_sindorim = DEFAULT_TRANSFER_DISTANCE_KM # CSV에 없거나 노드 문제시 기본값 사용 명시
            print(f"  2. 신도림역 환승 ({route1_via_sindorim_l1[0]}호선 ↔ {route1_via_sindorim_l2[0]}호선) 거리를 찾을 수 없어 기본값 사용: {dist_r1_transfer_sindorim:.3f} km")


        # 3. 신도림(2) -> 까치산(2) 거리
        if route1_via_sindorim_l2 in final_check_all_nodes_in_graph and route1_end_kkachisan_l2 in final_check_all_nodes_in_graph:
            distances_r1_p3, _ = dijkstra(subway_graph_data, route1_via_sindorim_l2)
            dist_r1_part3 = distances_r1_p3.get(route1_end_kkachisan_l2, float('inf'))
            if dist_r1_part3 == float('inf'):
                print(f"  경로 1 오류: {route1_via_sindorim_l2}에서 {route1_end_kkachisan_l2}(으)로 가는 경로를 찾을 수 없습니다.")
            else:
                print(f"  3. {route1_via_sindorim_l2[1]}({route1_via_sindorim_l2[0]}) → {route1_end_kkachisan_l2[1]}({route1_end_kkachisan_l2[0]}) 거리: {dist_r1_part3:.2f} km")
        else:
            print(f"  경로 1 오류: {route1_via_sindorim_l2} 또는 {route1_end_kkachisan_l2} 노드가 그래프에 없습니다.")

        if dist_r1_part1 != float('inf') and dist_r1_transfer_sindorim != float('inf') and dist_r1_part3 != float('inf'):
            total_dist_route1 = dist_r1_part1 + dist_r1_transfer_sindorim + dist_r1_part3
            print(f"  >> 경로 1 총 예상 거리: {total_dist_route1:.2f} km")
        else:
            print(f"  >> 경로 1 총 예상 거리: 계산 불가 (경로 중 일부를 찾을 수 없음)")

        print("="*50 + "\n")