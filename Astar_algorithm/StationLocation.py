import pandas as pd

# ──────────────────────────────────────────────────────────────────────────
# 1) 서울시 역사 마스터 정보 (Seoul_Station_Master.csv) 처리
# ──────────────────────────────────────────────────────────────────────────
# 1-1. CSV 불러오기 (인코딩: cp949)
master = pd.read_csv("Astar_algorithm/Seoul_Station_Master.csv", encoding="cp949")

# 1-2. 1~5호선만 필터
#    실제 '호선' 컬럼 값 예시: '1호선', '2호선', ... , 그 외에도 '수도권 광역급행철도' 등이 섞여 있음
mask_master = master["호선"].isin(["1호선", "2호선", "3호선", "4호선", "5호선"])
master_1to5 = master[mask_master].copy()

# 1-3. 필요한 컬럼만 추출하고, 컬럼명 한글→영문으로 변경
#    원본: ['역사_ID', '역사명', '호선', '위도', '경도']
#    → 추출: ['호선','역사명','위도','경도']
#    → rename: {'호선':'line', '역사명':'station_name', '위도':'latitude', '경도':'longitude'}
master_1to5 = master_1to5[["호선", "역사명", "위도", "경도"]].rename(columns={
    "호선": "line",
    "역사명": "station_name",
    "위도": "latitude",
    "경도": "longitude"
})

# 1-4. CSV로 저장 (인코딩은 필요에 따라 utf-8/utf-8-sig/CP949 등 조절 가능)
master_1to5.to_csv("Astar_algorithm/station_location/subway_1to5_master.csv", index=False, encoding="utf-8")
print("생성 완료 → subway_1to5_master.csv (1~5호선만 필터링, 컬럼명 변경)")

# ──────────────────────────────────────────────────────────────────────────
# 2) 수도권 1호선 역 위치 정보 (Metropolitan_1Line_Station_Location.csv) 처리
# ──────────────────────────────────────────────────────────────────────────
# 2-1. CSV 불러오기
line1 = pd.read_csv("Astar_algorithm/Metropolitan_1Line_Station_Location.csv", encoding="cp949")

# 2-2. 필요한 컬럼만 추출하고, 컬럼명 한글→영문으로 변경
#    원본: ['철도운영기관명', '선명', '역명', '경도', '위도']
#    → 추출: ['선명','역명','위도','경도']
#    → rename: {'선명':'line', '역명':'station_name', '위도':'latitude', '경도':'longitude'}
line1 = line1[["선명", "역명", "위도", "경도"]].rename(columns={
    "선명": "line",
    "역명": "station_name",
    "위도": "latitude",
    "경도": "longitude"
})

# 2-3. CSV로 저장
line1.to_csv("Astar_algorithm/station_location/subway_line1_coords.csv", index=False, encoding="utf-8")
print("생성 완료 → subway_line1_coords.csv (1호선 좌표, 컬럼명 변경)")

# ──────────────────────────────────────────────────────────────────────────
# 3) 수도권 5호선 역 위치 정보 (Metropolitan_5Line_Station_Location.csv) 처리
# ──────────────────────────────────────────────────────────────────────────
# 3-1. CSV 불러오기
line5 = pd.read_csv("Astar_algorithm/Metropolitan_5Line_Station_Location.csv", encoding="cp949")

# 3-2. 필요한 컬럼만 추출하고, 컬럼명 한글→영문으로 변경
#    원본: ['철도운영기관명', '선명', '역명', '경도', '위도']
#    → 추출: ['선명','역명','위도','경도']
#    → rename: {'선명':'line', '역명':'station_name', '위도':'latitude', '경도':'longitude'}
line5 = line5[["선명", "역명", "위도", "경도"]].rename(columns={
    "선명": "line",
    "역명": "station_name",
    "위도": "latitude",
    "경도": "longitude"
})

# 3-3. CSV로 저장
line5.to_csv("Astar_algorithm/station_location/subway_line5_coords.csv", index=False, encoding="utf-8")
print("생성 완료 → subway_line5_coords.csv (5호선 좌표, 컬럼명 변경)")
