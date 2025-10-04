import requests
import xml.etree.ElementTree as ET
import pandas as pd
from datetime import datetime
import os
import subprocess
from .trival import start_timer, end_timer
from io import StringIO
from obspy import read_events
from glob import glob



"""
Download the ISC throught Crawler 
Now just support ISC_EHB bulletin


"""


def QuakeML2CSV(QuakeML_file,output_file_path):
    cat = read_events(QuakeML_file)
    records = []

    # 遍历每个事件
    for event in cat:
        event_id = event.resource_id.id.split('=')[-1]

        origin = event.preferred_origin() or event.origins[0]
        origin_time = origin.time.datetime
        lat = origin.latitude
        lon = origin.longitude
        depth_m = getattr(origin.depth, "value", None)
        depth_km = depth_m / 1000 if depth_m is not None else None

        # 遍历到时信息（arrivals）
        for arr in origin.arrivals:
            phase = arr.phase
            az = arr.azimuth
            dist = arr.distance
            tres = getattr(arr, "time_residual", None)   # 更安全的写法
            tused = getattr(arr, "time_used", None)      # 这里用 getattr
            # 根据 arrival 找对应 pick
            pick_id = arr.pick_id.id
            pick_time = None
            station = None
            network = None

            for pick in event.picks:
                if pick.resource_id.id == pick_id:
                    pick_time = pick.time.datetime
                    if pick.waveform_id is not None:
                        network = pick.waveform_id.network_code
                        station = pick.waveform_id.station_code
                    break

            records.append({
                "event_id": event_id,
                "origin_time": origin_time,
                "latitude": lat,
                "longitude": lon,
                "depth_km": depth_km,
                "phase": phase,
                "azimuth_deg": az,
                "distance_deg": dist,
                "time_residual": tres,
                "time_used": tused,
                "pick_time": pick_time,
                "network": network,
                "station": station
            })

    # 转换为 DataFrame 并保存 CSV
    df = pd.DataFrame(records)
    df.to_csv(output_file_path, index=False)
    print(f"✅ 转换完成，已保存为{output_file_path}")

from concurrent.futures import ThreadPoolExecutor, as_completed

def batch_convert_and_merge(input_dir, merged_csv_path):
    """
    批量将 QuakeML(xml) 转换为 CSV，并合并为一张总表
    """
    all_csv_paths = []

    # 遍历所有 .xml 文件
    for xml_file in glob(os.path.join(input_dir, "*.xml")):
        csv_path = xml_file + ".csv"
        print(f"Converting {xml_file} -> {csv_path}")
        try:
            QuakeML2CSV(QuakeML_file=xml_file, output_file_path=csv_path)
            all_csv_paths.append(csv_path)
        except Exception as e:
            print(f"❌ Failed to convert {xml_file}: {e}")

    # 合并所有 CSV
    dfs = []
    for csv_file in all_csv_paths:
        try:
            df = pd.read_csv(csv_file)
            # 可选：在合并前加一列来源文件名，方便溯源
            df["source_file"] = os.path.basename(csv_file)
            dfs.append(df)
        except Exception as e:
            print(f"❌ Failed to read {csv_file}: {e}")

    if dfs:
        merged_df = pd.concat(dfs, ignore_index=True)
        merged_df.drop_duplicates(inplace=True)
        merged_df.to_csv(merged_csv_path, index=False)
        print(f"✅ Merged CSV saved to: {merged_csv_path}")
    else:
        print("⚠️ No CSV files were merged.")

def split_time_ranges(start_time, end_time, months=6):
    """
    将时间按指定月数切分
    """
    start = pd.to_datetime(start_time)
    end   = pd.to_datetime(end_time)
    ranges = []
    cur = start

    while cur < end:
        next_ = cur + pd.DateOffset(months=months)
        if next_ > end:
            next_ = end
        ranges.append((cur, next_))
        cur = next_
    return ranges

def split_stations(station_list, bulk_size=5):
    """
    将台站列表按指定批次切分
    """
    if isinstance(station_list, str):
        stations = [s.strip() for s in station_list.split(',') if s.strip()]
    else:
        # 已经是列表就直接使用
        stations = station_list

    # 按批次切分
    return [stations[i:i + bulk_size] for i in range(0, len(stations), bulk_size)]

def fetch_isc_chunk(base_url, params, output_file):
    """
    下载单个块并保存
    """
    try:
        response = requests.get(base_url, params=params, timeout=300)
        response.raise_for_status()
        with open(output_file, 'wb') as f:
            f.write(response.content)
        print(f"[OK] Saved: {output_file}")
    except Exception as e:
        print(f"[FAIL] {output_file}: {e}")

def isc_crawler(
        Download_dir,
        start_time, end_time,
        lat_range, lon_range,
        magnitude_range, depth_range,
        phase_list="", station_list=None,
        out_format="QuakeML",
        months_per_chunk=6, stations_per_chunk=5,
        max_workers=4):
    
    time_start = start_timer()

    if station_list is None:
        station_list = []

    base_url = "https://www.isc.ac.uk/cgi-bin/web-db-run"
    os.makedirs(Download_dir, exist_ok=True)

    Save_dir = os.path.join(Download_dir,"QuakeML")
    os.makedirs(Save_dir, exist_ok=True)
    out_file_path = os.path.join(Download_dir,"ISC_arrivals.csv")
    # 时间切片
    time_chunks = split_time_ranges(start_time, end_time, months=months_per_chunk)
    # 台站切片
    station_chunks = split_stations(station_list, bulk_size=stations_per_chunk)

    tasks = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for (t_start, t_end) in time_chunks:
            for stns in station_chunks:
                params = {
                    "req_agcy": "ISC-EHB",
                    "out_format": out_format,
                    "tdef": "on",
                    "phaselist": phase_list,
                    "stnsearch": "STN",
                    "sta_list": ",".join(stns),
                    "searchshape": "RECT",
                    "bot_lat": lat_range[0],
                    "top_lat": lat_range[1],
                    "left_lon": lon_range[0],
                    "right_lon": lon_range[1],
                    "start_year": t_start.year,
                    "start_month": t_start.month,
                    "start_day": t_start.day,
                    "start_time": "00:00:00",
                    "end_year": t_end.year,
                    "end_month": t_end.month,
                    "end_day": t_end.day,
                    "end_time": "23:59:59",
                    "min_dep": depth_range[0],
                    "max_dep": depth_range[1],
                    "min_mag": magnitude_range[0],
                    "max_mag": magnitude_range[1],
                    "req_mag_type": "Any",
                    "include_links": "on",
                    "request": "STNARRIVALS",
                    "table_owner": "iscehb"
                }

                fname = f"ISC_{stns[0]}_{stns[-1]}_{t_start.strftime('%Y%m%d')}_{t_end.strftime('%Y%m%d')}.xml"
                output_file = os.path.join(Save_dir,fname)
  
                tasks.append(executor.submit(fetch_isc_chunk, base_url, params, output_file))

        for future in as_completed(tasks):
            pass

    
    batch_convert_and_merge(Save_dir,out_file_path)

    end_timer(time_start,task_name="ISC-EHB Crawler is Done")