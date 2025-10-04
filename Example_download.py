from pathlib import Path
import pandas as pd
import numpy as np
from Download_py import (station_download,event_USGS,merge_event,count_stations_fast,
                         filter_events,plot_raw_events,plot_selected_events,select_event,
                         WaveformDownloader,extract_earthquake_events,generate_gmt_rays,
                         contrast,isc_crawler,start_timer,end_timer)
import warnings
# 忽略所有 FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

"""

File Tree
Download_dir   
    |----metadata_channel.csv    metadata download info
    |----stations.csv            station info  
    |----networks_stats.csv   
    |----event_USGS.csv
    |----merged_event.csv        formatted event.csv
    |----filtered_event.csv
    |----download
        |---singular network
            |----waveforms
            |----log                  Just waveform log for now
            |----- Net_event.csv      from filtered event
            |----- Net_station.csv    basic info in station not channel    
            |            
            |----PickNet     if you use the PickNet too, there will be another new package soon
                |---QuakeML        save the temp bulk download  will be merged into arrival.csv    
                |----ISC_arrival.csv      from ISC-EHB  if not means don't contain this info
                |----Fomatted_Waveforms   save Fomatted Waveforms from waveforms
                |----catalog.txt          formatted event info
                |----station.txt          formatted station info

                
# channel may larger 3x station for different duration

"""



""" ["4B","4C","XG","CK","KC","KD","TJ","YT" ,"8H", "KR", "7B", "9H", "6C", "5C", "KN", "QZ", "AD", "XW", "XP"]
 """
""" Set up """
client_list = ["GFZ", "IRIS","ISC"] # depend on the station info
nets = ["8H", "KR", "7B", "9H", "6C", "5C", "KN", "QZ", "AD", "XW", "XP"]
Download_dir = Path("/media/baimanight/red/Tarim_Mag3+")  #Tarim_Mag3+
channel = "?H?"
lat_range = [35,45]
lon_range = [70,90]         # station and event reign
depth_range = [1,45]        # km
distance_range = [2,12]     # degree
magnitude_range = [1,10]
wave_duration = [-20,150]   # s  0=origin


"Get station metadata > channel.csv networks_stats.csv  station.csv "  
#  station_download(client_list,nets,Download_dir,lat_range,lon_range); breakpoint()

df_all_networks = pd.read_csv(Download_dir / "networks_stats.csv")
df_networks = df_all_networks[df_all_networks["Download"] == True].copy()
df_networks["min_start_date"] = pd.to_datetime(df_networks["min_start_date"])
df_networks["max_end_date"]   = pd.to_datetime(df_networks["max_end_date"])

start_time = df_networks["min_start_date"].min()
end_time = df_networks["max_end_date"].max()
net_client_map = dict(zip(df_networks["network"], df_networks["client_name"]))



"Get event form USGS Mag~4.5+"
#  event_USGS(start_time,end_time,magnitude_range,lat_range,lon_range,Download_dir); breakpoint()

"Merge event with USGS to Format"

# merge_event(cn_file="台网中心/正式目录09_25.xls",time_type="CST", # CST UTC suit xls and csv
#             USGS_file=Download_dir / "event_USGS.csv",output_dir=Download_dir, 
#             trans_name={
#                 "time": "发震日期（北京时间）",
#                 "latitude": "纬度(°)",
#                 "longitude": "经度(°)",
#                 "depth_km": "震源深度(Km)",
#                 "magnitude": "震级",
#                 "magnitude_type": "震级类型"
#             }  
# )

# merge_event(cn_file="台网中心/UTC_中国地震台网地震目录_1997_2017.xls", time_type="UTC",# CST UTC suit xls and csv
#             USGS_file=Download_dir / "merged_event.csv",output_dir=Download_dir, 
#             mag_format=True,   #  for different mag_type columns
#             trans_name={
#                 "time": "发震时刻(国际时)",
#                 "latitude": "纬度(°)",
#                 "longitude": "经度(°)",
#                 "depth_km": "震源深度(Km)",
#                 "magnitude": "mL",
#                 "magnitude_type": "震级类型"
#             }  
# )

# breakpoint()


""" event filter  choose suit event with mag,region,depth """

input_station = Download_dir / "stations.csv"
filter_event_path = Download_dir / "filtered_event.csv"

# filter_events(
#     input_catalog= Download_dir / "merged_event.csv",
#     input_station= input_station,
#     output_catalog=filter_event_path,

#     min_magnitude=magnitude_range[0], max_magnitude=magnitude_range[1],
#     min_distance_deg=distance_range[0], max_distance_deg=distance_range[1],
#     min_depth=depth_range[0], max_depth=depth_range[1]
# )

# breakpoint()

"""Plot filtered event map"""
# plot_raw_events(filter_event_path, input_station,resolution=0.1)
# breakpoint()

""" Download Waveform and ISC-EHB arrivals if True """
use_time = start_timer()

df_filtered_events = pd.read_csv(filter_event_path, parse_dates=["time"])
df_station = pd.read_csv(input_station)

for net,client_name in net_client_map.items():

    base_dir = Download_dir/"Download"/net
    base_dir.mkdir(parents=True, exist_ok=True)
    PickNet_dir = base_dir/"PickNet"
    PickNet_dir.mkdir(parents=True, exist_ok=True)
    # choose event duration
    try:
        row = df_networks[df_networks["network"] == net].iloc[0]

        row_start = pd.to_datetime(row["min_start_date"]) if pd.notna(row["min_start_date"]) else None
        row_end   = pd.to_datetime(row["max_end_date"])   if row["max_end_date"] != "None" else None

        start_time_net = row_start - pd.Timedelta(days=1) if row_start is not None else None

        if row_start is not None and row_end is None:
            end_time_net = pd.Timestamp.today().normalize()  # 仅取今天日期
        elif row_end is not None:
            end_time_net = row_end + pd.Timedelta(days=1)
        else:
            raise Exception("networks__stats.csv start/end_time is wrong Second")
    except:
        raise Exception("networks__stats.csv start/end_time is wrong First")

    df_duration_event = df_filtered_events[
            (df_filtered_events["time"] >= start_time_net) &
            (df_filtered_events["time"] <= end_time_net)
        ]
    
    net_event = base_dir/f"{net}_event.csv"
    if not net_event.exists():# if Get new Need to delete raw
        df_duration_event.to_csv(net_event, index=False)
    # Station 
    net_station = base_dir/f"{net}_station.csv"
    df_duration_station = df_station[df_station["network"] == net]
    df_duration_station.to_csv(net_station,index=False)

    station_list = ",".join(df_duration_station["station"])
    isc_crawler(PickNet_dir,start_time_net,end_time_net,lat_range,lon_range,magnitude_range,depth_range,
                phase_list="Pn,Sn",station_list=station_list,out_format="QuakeML",
                months_per_chunk=6, stations_per_chunk=5,max_workers=2)

    print(f"\nDownload {net} from {start_time_net} to {end_time_net} Event {len(df_duration_event)}\n")


    # waveform_download = WaveformDownloader(
    #              net_event,            # 挑选的地震事件目录（前置：数据源，必须先加载）
    #              net_station,           # 台站目录（前置：数据源，必须先加载）
    #              base_output_dir=base_dir,         # 输出文件夹（前置：路径配置）
    #              client_name=client_name,       # FDSN 客户端（默认GFZ）
    #              channel=channel,           # 通道选择（默认?H?）
    #              pre_event_time=abs(wave_duration[0]),       # 震前秒数
    #              post_time_rules=[(float('inf'), 150)],     # 震后秒数 (3, 100), (5, 150), 
    #              min_distance=distance_range[0],          # 最小震中距 (度)
    #              max_distance=distance_range[1],         # 最大震中距 (度)
    #              max_concurrent_events=8,  # 并行数
    #              request_timeout=5,       # 单次请求超时
    #              max_retries=2,           # 最大重试次数
    #              retry_delay=1,           # 重试延迟
    #              chunk_size=16,            # 分块大小
    #              resume=True         # something weird
    #              )
    # waveform_download.download()
    
    print(f"\nFormat {net}   Event {len(df_duration_event)}\n")

end_timer(use_time,task_name="Download Waveform is Done")
breakpoint()


