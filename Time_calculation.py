# travel_time_extractor.py
import re, os, csv
from datetime import datetime, timedelta
import pandas as pd

def _parse_hhmmss(t: str) -> datetime:
    """'05:24:30' → datetime(1900-01-01 05:24:30)"""
    return datetime.strptime(t.strip(), "%H:%M:%S")

def collect_from_csv(path: str, encoding="cp949"):
    """도착(역명+시간) ↔ 출발(빈역명+시간) 짝으로 이동시간 산출"""
    df = pd.read_csv(path, header=None, names=["station", "time"], encoding=encoding)
    df["station"] = df["station"].fillna("")        # NaN → 빈문자
    records, prev_station, prev_dep_time = [], None, None

    # 두 줄씩 순회(도착 → 출발)
    rows = df.to_records(index=False)
    for i in range(0, len(rows), 2):
        arr_station, arr_time = rows[i]             # 역명, 도착시각
        dep_station_blank, dep_time = rows[i + 1]

        # 첫 역: 도착시각 없을 수 있음 → 출발정보만 저장
        if prev_station is None:
            prev_station = arr_station
            prev_dep_time = _parse_hhmmss(dep_time)
            continue

        # 이동시간 = 이번 역 도착 – 이전 역 출발
        travel_sec = int((_parse_hhmmss(arr_time) - prev_dep_time).total_seconds())
        records.append(
            {
                "구간": f"{prev_station}-{arr_station}",
                "이동시간": str(timedelta(seconds=travel_sec))[-8:]  # HH:MM:SS
            }
        )

        # 다음 반복 대비 업데이트
        prev_station = arr_station
        prev_dep_time = _parse_hhmmss(dep_time)

    return records

def collect_from_txt(path: str, encoding="utf-8"):
    """
    ex) '병점(대기1분)-(이동5분)->서동탄'
         → '병점-서동탄', '00:05:00'
    """
    pat = re.compile(r"(.+?)\(대기.*?\)-\(이동(\d+)분\)->(.+)")
    records = []
    with open(path, encoding=encoding) as f:
        for line in f:
            m = pat.search(line)
            if not m:
                continue
            s_from, minutes, s_to = m.groups()
            records.append(
                {
                    "구간": f"{s_from}-{s_to}",
                    "이동시간": str(timedelta(minutes=int(minutes)))[2:]  # MM:SS
                }
            )
    return records

if __name__ == "__main__":
    sources = [
        ("경인.csv",  collect_from_csv),
        ("경부_서동탄branch제외.csv", collect_from_csv),
        ("branch참고시간.txt", collect_from_txt),
    ]

    all_records = []
    for fname, loader in sources:
        path = os.path.join("역간시간표/1호선/각1콜럼씩", fname)  # Colab/로컬 경로에 맞게 조정
        all_records.extend(loader(path))

    # CSV 저장 (구간, 이동시간)
    out_path = "역간시간표/1호선/line1_segment_travel_times.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["구간", "이동시간"])
        writer.writeheader()
        writer.writerows(all_records)

    print(f"✅ 완료: {out_path}에 {len(all_records)}개 구간 저장")
