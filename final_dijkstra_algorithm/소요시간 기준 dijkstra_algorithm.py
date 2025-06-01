# -*- coding: utf-8 -*-
import pandas as pd
import heapq
from collections import defaultdict
import re

# --- CSV 파일 경로 설정 ---
LINE_INFO_CSV_PATHS = {
    '1_soyo_guro': "1호선 소요산~구로.csv",
    '1_guro_incheon': "1호선 구로~인천.csv",
    '1_byeongjeom_sinchang': "1호선 병점~신창.csv",
    '1_guro_gwangmyeong': "1호선 구로~광명.csv",
    '1_guro_seodongtan': "1호선 구로~서동탄.csv",
    '2': "2호선.csv",
    '3': "3호선.csv",
    '4': "4호선.csv",
    '5': "5호선.csv"
}
TRANSFER_INFO_CSV_PATH = "서울교통공사_환승역거리 소요시간 정보_20250331.csv"

# --- 사용자 설정값 ---
DEFAULT_TRANSFER_TIME_MIN = 4.0
START_STATION_LINE = '1'
START_STATION_NAME = '서울역'
END_STATION_NAME ='신정네거리'

# --- 명시적 분기점 연결 소요 시간 (분) ---
TIME_SEONGSU_YONGDAP_MIN = 3.0
TIME_SINDORIM_DORIMCHEON_MIN = 1.5  # 1분 30초
TIME_GANGDONG_DUNCHON_MIN = 1 + 50/60  # 1분 50초
TIME_GURO_GASAN_MIN = 4.0
TIME_GEUMCHEON_GWANGMYEONG_MIN = 5.0
TIME_BYEONGJEOM_SEODONGTAN_MIN = 5.0
# ----------------------------------------------------------------------

def parse_time_to_minutes(time_str):
    if pd.isna(time_str):
        return 0.0
    time_str = str(time_str).strip()
    match = re.match(r'(\d+):(\d{1,2})', time_str)
    if match:
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        return minutes + seconds / 60.0
    else:
        try:
            val = float(time_str)
            if val < 0: return 0.0
            return val
        except ValueError:
            if time_str == "0:00" or time_str == "0": return 0.0
            return 0.0

def load_transfer_times_data(file_path, current_relevant_lines, default_transfer_time):
    transfer_data = {}
    try:
        # 이전 실행에서 'utf-8-sig'가 환승 정보 파일에 적합했으므로 유지
        df_transfer = pd.read_csv(file_path, encoding='utf-8-sig')
        for _, row in df_transfer.iterrows():
            from_line = str(row['호선']).strip()
            station_name = str(row['환승역명']).strip()
            to_line = str(row['환승노선']).replace('호선','').replace('선','').strip()
            
            is_to_line_relevant = False
            if to_line.isdigit() and to_line in current_relevant_lines:
                is_to_line_relevant = True
            elif not to_line.isdigit():
                if to_line in current_relevant_lines:
                     is_to_line_relevant = True

            if not is_to_line_relevant and not any(c.isdigit() for c in to_line) and to_line not in current_relevant_lines :
                 continue
            
            try:
                time_val_str = str(row['환승소요시간']).strip()
                transfer_time_min = parse_time_to_minutes(time_val_str)
            except KeyError:
                transfer_time_min = default_transfer_time
            except Exception:
                transfer_time_min = default_transfer_time

            if transfer_time_min <= 0:
                 transfer_time_min = default_transfer_time
            
            transfer_data[(from_line, station_name, to_line)] = transfer_time_min
    except FileNotFoundError:
        print(f"경고: 환승 정보 CSV 파일을 찾을 수 없습니다: {file_path}.")
    except Exception as e:
        print(f"경고: 환승 정보 CSV 파일을 읽는 중 오류 발생: {e}.")
    return transfer_data

def build_subway_graph_by_time(line_csv_paths, transfer_csv_path, current_relevant_lines, default_transfer_time_param):
    station_data_lookup = {} 

    # 1. 모든 노선 정보 CSV 파일 로드 및 파싱
    for line_key_suffix, file_path in line_csv_paths.items():
        try:
            df_line = pd.read_csv(file_path, encoding='utf-8') 
            current_line_num_from_file_key = line_key_suffix.split('_')[0]
            if not current_line_num_from_file_key.isdigit() and line_key_suffix in ['2','3','4','5']:
                current_line_num_from_file_key = line_key_suffix

            for _, row in df_line.iterrows():
                station_name = str(row['역명']).strip()
                # 중요: project_stations에 정의된 역 이름과 CSV의 역 이름이 정확히 일치해야 합니다.
                # 역명 클리닝(예: '성북'->'광운대')은 project_stations 정의 시 미리 반영되어 있거나,
                # 이 함수 초반에 project_stations 전체를 대상으로 일괄 적용하는 것이 좋습니다.
                # (현재 코드는 project_stations에 이미 최종 역명이 있다고 가정)

                time_from_prev_str = str(row['시간(분)']).strip()
                time_from_prev_min = parse_time_to_minutes(time_from_prev_str)
                
                key_tuple = (current_line_num_from_file_key, station_name)
                existing_time = station_data_lookup.get(key_tuple)

                if existing_time is None: 
                    station_data_lookup[key_tuple] = time_from_prev_min
                elif existing_time == 0.0 and time_from_prev_min > 0.0:
                    station_data_lookup[key_tuple] = time_from_prev_min
        except FileNotFoundError:
            print(f"경고: 노선 정보 CSV 파일을 찾을 수 없습니다: {file_path}")
        except Exception as e:
            print(f"경고: 노선 정보 CSV '{file_path}' 처리 중 오류: {e}.")

    transfer_times_data = load_transfer_times_data(transfer_csv_path, current_relevant_lines, default_transfer_time_param)
    graph = defaultdict(list)
    all_nodes_in_scope = set()

    project_stations = {
        '1_main_north_to_guro': ['소요산', '동두천', '보산', '동두천중앙', '지행', '덕정', '덕계', '양주', '녹양', '가능', '의정부', '회룡', '망월사', '도봉산', '도봉', '방학', '창동', '녹천', '월계', '광운대', '석계', '신이문', '외대앞', '회기', '청량리', '제기동', '신설동', '동묘앞', '동대문', '종로5가', '종로3가', '종각', '시청', '서울역', '남영', '용산', '노량진', '대방', '신길', '영등포', '신도림', '구로'],
        '1_gyeongin_branch': ['구로', '구일', '개봉', '오류동', '온수', '역곡', '소사', '부천', '중동', '송내', '부개', '부평', '백운', '동암', '간석', '주안', '도화', '제물포', '도원', '동인천', '인천'],
        '1_gyeongbu_main': ['구로', '가산디지털단지', '독산', '금천구청', '석수', '관악', '안양', '명학', '금정', '군포', '당정', '의왕', '성균관대', '화서', '수원', '세류', '병점', '세마', '오산대', '오산', '진위', '송탄', '서정리', '지제', '평택', '성환', '직산', '두정', '천안', '봉명', '쌍용', '아산', '배방', '온양온천', '신창'],
        '1_gwangmyeong_shuttle': ['금천구청', '광명'],
        '1_seodongtan_branch': ['병점', '서동탄'],
        '2_main': ['시청', '을지로입구', '을지로3가', '을지로4가', '동대문역사문화공원', '신당', '상왕십리', '왕십리', '한양대', '뚝섬', '성수', '건대입구', '구의', '강변', '잠실나루', '잠실', '잠실새내', '종합운동장', '삼성', '선릉', '역삼', '강남', '교대', '서초', '방배', '사당', '낙성대', '서울대입구', '봉천', '신림', '신대방', '구로디지털단지', '대림', '신도림', '문래', '영등포구청', '당산', '합정', '홍대입구', '신촌', '이대', '아현', '충정로'],
        '2_seongsu_branch': ['용답', '신답', '용두', '신설동'],
        '2_sinjeong_branch': ['도림천', '양천구청', '신정네거리', '까치산'],
        '3': ['대화', '주엽', '정발산', '마두', '백석', '대곡', '화정', '원당', '원흥','삼송', '지축', '구파발', '연신내', '불광', '녹번', '홍제', '무악재', '독립문', '경복궁', '안국', '종로3가', '을지로3가', '충무로', '동대입구', '약수', '금호', '옥수', '압구정', '신사', '잠원', '고속터미널', '교대', '남부터미널', '양재', '매봉', '도곡', '대치', '학여울', '대청', '일원', '수서', '가락시장', '경찰병원', '오금'],
        '4': ['진접', '오남', '별내별가람', '당고개', '상계', '노원', '창동', '쌍문', '수유', '미아', '미아사거리', '길음', '성신여대입구', '한성대입구', '혜화', '동대문', '동대문역사문화공원', '충무로', '명동', '회현', '서울역', '숙대입구', '삼각지', '신용산', '이촌', '동작', '총신대입구(이수)', '사당', '남태령', '선바위', '경마공원', '대공원', '과천', '정부과천청사', '인덕원', '평촌', '범계', '금정', '산본', '수리산', '대야미', '반월', '상록수', '한대앞', '중앙', '고잔', '초지', '안산', '신길온천', '정왕', '오이도'],
        '5_main_to_sangil': ['방화', '개화산', '김포공항', '송정', '마곡', '발산', '우장산', '화곡', '까치산', '신정', '목동', '오목교', '양평', '영등포구청', '영등포시장', '신길', '여의도', '여의나루', '마포', '공덕', '애오개', '충정로', '서대문', '광화문', '종로3가', '을지로4가', '동대문역사문화공원', '청구', '신금호', '행당', '왕십리', '마장', '답십리', '장한평', '군자', '아차산', '광나루', '천호', '강동', '길동', '굽은다리', '명일', '고덕', '상일동'],
        '5_gangdong_to_macheon': ['둔촌동', '올림픽공원', '방이', '오금', '개롱', '거여', '마천']
    }
    # 참고: 위 project_stations 내 역 이름들은 CSV와 정확히 일치하는 '최종 역명'이어야 합니다.
    # (예: '성북'->'광운대', '미아삼거리'->'미아사거리', '신천'->'잠실새내' 등의 변환이 이미 적용된 상태)
    # 만약 CSV를 읽은 후 역명 클리닝을 일괄적으로 하고 싶다면, 이 함수 초반 또는 station_data_lookup을 채운 후,
    # project_stations의 모든 역 이름과 station_data_lookup의 키에 있는 역 이름을 대상으로 클리닝 함수를 적용해야 합니다.

    for line_key, station_name_list_segment in project_stations.items():
        line_num_str = line_key.split('_')[0] 
        if line_num_str not in current_relevant_lines:
            continue

        previous_station_name_in_segment = None
        for station_name in station_name_list_segment:
            current_node = (line_num_str, station_name) 
            all_nodes_in_scope.add(current_node)
            
            if previous_station_name_in_segment:
                prev_node_tuple = (line_num_str, previous_station_name_in_segment)
                time_min = station_data_lookup.get(current_node, -1.0) 

                if time_min >= 0: 
                    graph[prev_node_tuple].append((current_node, time_min))
                    graph[current_node].append((prev_node_tuple, time_min))
            previous_station_name_in_segment = station_name
            
    # --- 특정 분기점 명시적 연결 ---
    node_seongsu_l2 = ('2', '성수'); node_yongdap_l2 = ('2', '용답')
    if node_seongsu_l2 in all_nodes_in_scope and node_yongdap_l2 in all_nodes_in_scope:
        graph[node_seongsu_l2].append((node_yongdap_l2, TIME_SEONGSU_YONGDAP_MIN))
        graph[node_yongdap_l2].append((node_seongsu_l2, TIME_SEONGSU_YONGDAP_MIN))

    node_sindorim_l2 = ('2', '신도림'); node_dorimcheon_l2 = ('2', '도림천')
    if node_sindorim_l2 in all_nodes_in_scope and node_dorimcheon_l2 in all_nodes_in_scope:
        graph[node_sindorim_l2].append((node_dorimcheon_l2, TIME_SINDORIM_DORIMCHEON_MIN))
        graph[node_dorimcheon_l2].append((node_sindorim_l2, TIME_SINDORIM_DORIMCHEON_MIN))

    node_gangdong_l5 = ('5', '강동'); node_dunchon_l5 = ('5', '둔촌동')
    if node_gangdong_l5 in all_nodes_in_scope and node_dunchon_l5 in all_nodes_in_scope:
        graph[node_gangdong_l5].append((node_dunchon_l5, TIME_GANGDONG_DUNCHON_MIN))
        graph[node_dunchon_l5].append((node_gangdong_l5, TIME_GANGDONG_DUNCHON_MIN))

    node_guro_l1 = ('1', '구로'); node_gasan_l1 = ('1', '가산디지털단지')
    if node_guro_l1 in all_nodes_in_scope and node_gasan_l1 in all_nodes_in_scope:
        graph[node_guro_l1].append((node_gasan_l1, TIME_GURO_GASAN_MIN))
        graph[node_gasan_l1].append((node_guro_l1, TIME_GURO_GASAN_MIN))
        
    node_geumcheon_l1 = ('1', '금천구청'); node_gwangmyeong_l1 = ('1', '광명')
    if node_geumcheon_l1 in all_nodes_in_scope and node_gwangmyeong_l1 in all_nodes_in_scope:
        graph[node_geumcheon_l1].append((node_gwangmyeong_l1, TIME_GEUMCHEON_GWANGMYEONG_MIN))
        graph[node_gwangmyeong_l1].append((node_geumcheon_l1, TIME_GEUMCHEON_GWANGMYEONG_MIN))

    node_byeongjeom_l1 = ('1', '병점'); node_seodongtan_l1 = ('1', '서동탄')
    if node_byeongjeom_l1 in all_nodes_in_scope and node_seodongtan_l1 in all_nodes_in_scope:
        graph[node_byeongjeom_l1].append((node_seodongtan_l1, TIME_BYEONGJEOM_SEODONGTAN_MIN))
        graph[node_seodongtan_l1].append((node_byeongjeom_l1, TIME_BYEONGJEOM_SEODONGTAN_MIN))

    # 4. 환승역 연결
    for (from_line_t, station_name_t, to_line_t_str), transfer_time_val_min in transfer_times_data.items():
        is_to_line_t_str_relevant = False
        if to_line_t_str.isdigit() and to_line_t_str in current_relevant_lines:
            is_to_line_t_str_relevant = True
        elif not to_line_t_str.isdigit():
            if to_line_t_str in current_relevant_lines:
                 is_to_line_t_str_relevant = True
        
        if not (is_to_line_t_str_relevant or (from_line_t in current_relevant_lines and to_line_t_str in current_relevant_lines)):
            continue

        node1_t = (from_line_t, station_name_t)
        node2_t = (to_line_t_str, station_name_t)

        if node1_t in all_nodes_in_scope and node2_t in all_nodes_in_scope:
            graph[node1_t].append((node2_t, transfer_time_val_min))
            graph[node2_t].append((node1_t, transfer_time_val_min))
            
    return graph

def dijkstra(graph, start_node):
    all_graph_nodes = set(graph.keys())
    for destinations in graph.values():
        for dest_node, _ in destinations:
            all_graph_nodes.add(dest_node)
            
    durations_min = {node: float('inf') for node in all_graph_nodes}
    predecessors = {node: None for node in all_graph_nodes}
    
    if start_node not in durations_min: # start_node가 all_graph_nodes에 없는 경우를 대비
        durations_min[start_node] = float('inf')
        predecessors[start_node] = None

    durations_min[start_node] = 0
    priority_queue = [(0, start_node)] 

    while priority_queue:
        current_duration_min, current_node = heapq.heappop(priority_queue)

        if current_node not in durations_min or current_duration_min > durations_min[current_node]: 
            continue

        for neighbor, weight_time_min in graph.get(current_node, []):
            if neighbor not in durations_min: # Initialize if neighbor somehow missed
                 durations_min[neighbor] = float('inf')
                 predecessors[neighbor] = None

            duration_val = current_duration_min + weight_time_min
            if duration_val < durations_min[neighbor]: 
                durations_min[neighbor] = duration_val
                predecessors[neighbor] = current_node
                heapq.heappush(priority_queue, (duration_val, neighbor))
    return durations_min, predecessors

def reconstruct_path(predecessors, start_node, end_node):
    path = []
    current = end_node
    
    if current not in predecessors and current != start_node :
        if current == start_node and predecessors.get(current) is None : # start_node이고 predecessors에 있지만 None으로 기록된 경우
             path.append(start_node)
             return path[::-1]
        return None # 도달 불가능 또는 end_node가 start_node가 아닌데 predecessors에 없는 경우
    
    # start_node이면서 predecessors에 자기 자신이 없는 경우 (단일 노드 경로)
    if current == start_node and predecessors.get(current) is None:
        path.append(start_node)
        return path[::-1]

    max_path_len = len(predecessors) if predecessors else 0 
    count = 0

    while current != start_node:
        path.append(current)
        current = predecessors.get(current) 
        if current is None: 
            return None 
        
        count += 1
        if max_path_len > 0 and count > max_path_len + 50:
             print(f"경고: 경로 재구성 중 최대 길이 초과 ({count} > {max_path_len}). {start_node} -> {end_node}")
             return None 
    path.append(start_node)
    return path[::-1]

if __name__ == "__main__":
    relevant_lines_for_project = ['1', '2', '3', '4', '5']
    
    print(f"최단 경로 기준: 소요 시간(분)")
    print(f"환승 정보는 '{TRANSFER_INFO_CSV_PATH}' 파일에서 '환승소요시간'을 읽어옵니다.")
    print(f"환승 시간이 없거나 유효하지 않은 경우 기본 {DEFAULT_TRANSFER_TIME_MIN}분 적용.")

    subway_graph_data = build_subway_graph_by_time(
        LINE_INFO_CSV_PATHS,
        TRANSFER_INFO_CSV_PATH,
        relevant_lines_for_project,
        DEFAULT_TRANSFER_TIME_MIN
    )

    if subway_graph_data is None or not subway_graph_data:
        print("지하철 그래프 데이터를 생성하지 못했습니다. 프로그램을 종료합니다.")
    else:
        start_node_tuple = (START_STATION_LINE, START_STATION_NAME)
        
        final_check_all_nodes_in_graph = set(subway_graph_data.keys())
        for _k_node_list_val in subway_graph_data.values():
            for _k_node_val, _ in _k_node_list_val:
                final_check_all_nodes_in_graph.add(_k_node_val)
        
        if start_node_tuple not in final_check_all_nodes_in_graph:
            print(f"오류: 출발역 {start_node_tuple}이(가) 생성된 그래프 데이터에 유효하게 포함되지 않았습니다.")
            print(f"     CSV데이터, relevant_lines 또는 project_stations 정의를 확인하세요.")
            print(f"     (팁: {START_STATION_NAME}역이 해당 호선({START_STATION_LINE})의 CSV 파일에 정확히 기재되어 있는지,")
            print(f"      또는 project_stations 리스트에 포함되어 있는지 확인하십시오.)")
        else:
            durations_from_start_min, path_predecessors = dijkstra(subway_graph_data, start_node_tuple)
            
            possible_end_nodes = []
            for node_key_line, node_key_name in final_check_all_nodes_in_graph: # Use a more robust set of all nodes
                if node_key_name == END_STATION_NAME:
                    if node_key_line in relevant_lines_for_project:
                         possible_end_nodes.append((node_key_line, node_key_name))
            
            if not possible_end_nodes:
                print(f"오류: 도착역 '{END_STATION_NAME}'({','.join(relevant_lines_for_project)}호선 내)에 해당하는 역이 그래프에 정의되어 있지 않습니다.")
            else:
                min_duration_to_destination = float('inf')
                actual_end_node = None
                for end_node_candidate in possible_end_nodes:
                    candidate_duration = durations_from_start_min.get(end_node_candidate, float('inf'))
                    if candidate_duration < min_duration_to_destination:
                        min_duration_to_destination = candidate_duration
                        actual_end_node = end_node_candidate
                
                if actual_end_node is None or min_duration_to_destination == float('inf'):
                    print(f"{start_node_tuple[1]}({start_node_tuple[0]}호선) 에서 {END_STATION_NAME}(으)로 가는 경로를 찾을 수 없습니다.")
                else:
                    final_path = reconstruct_path(path_predecessors, start_node_tuple, actual_end_node)
                    print(f"--- {START_STATION_NAME} ({START_STATION_LINE}호선) 에서 {END_STATION_NAME} (도착: {actual_end_node[0]}호선) 까지 최단 *소요 시간* 경로 ---")
                    print(f"총 소요 시간: {min_duration_to_destination:.2f} 분")
                    if final_path:
                        print("경로:")
                        for i, step in enumerate(final_path):
                            print(f"  {i+1}. {step[1]} ({step[0]}호선)")
                    else:
                        print("경로를 재구성할 수 없습니다. (도착역에 도달 불가능)")