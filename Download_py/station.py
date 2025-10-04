from obspy.clients.fdsn import Client
import pandas as pd
from datetime import datetime
from pathlib import Path


def station_download(client_list,nets,output_path,lat_range,lon_range):

    # client IRIS GHZ  etc  depend on the station form FSDN
    # 定义要查询的台网列表
    # nets = ["8H", "KR", "7B", "9H", "6C", "5C", "KN", "QZ", "AD", "XW", "XP"]
    # 定义输出目录
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    # 定义地理边界
    minlat, maxlat = lat_range[0], lat_range[1]
    minlon, maxlon = lon_range[0], lon_range[1]


    # 存储所有结果的列表
    all_results = []
    # 存储台网时间边界
    network_time_bounds = {}
    # 存储成功台网列表
    successful_nets = []

    # 循环处理每个台网
    for net in nets:
        print(f"\n{'='*50}")
        print(f"开始处理台网: {net}")
        
        net_results = []
        net_min_start, net_max_end = None, None
        success = False    

        for client_name in client_list:
            try:
                # 查询台网的元数据
                client = Client(client_name)
                inventory = client.get_stations(
                    network=net,
                    minlatitude=minlat,
                    maxlatitude=maxlat,
                    minlongitude=minlon,
                    maxlongitude=maxlon,
                    level="channel"
                )
                
                # 遍历台网、台站、通道
                for network in inventory:
                    for station in network:
                        station_lat, station_lon = station.latitude, station.longitude
                        station_elevation = station.elevation if station.elevation is not None else "None"
                        for channel in station:
                            sensor_type = channel.sensor.description if channel.sensor else "unknown"
                            sample_rate = channel.sample_rate
                            start_date, end_date = channel.start_date, channel.end_date
                            
                            # 更新时间范围
                            if start_date:
                                if net_min_start is None or start_date < net_min_start:
                                    net_min_start = start_date
                            if end_date:
                                if net_max_end is None or end_date > net_max_end:
                                    net_max_end = end_date
                              
                            
                            # 保存一行数据
                            net_results.append({
                                "network": network.code,
                                "station": station.code,
                                "latitude": station_lat,
                                "longitude": station_lon,
                                "elevation":station_elevation,
                                "channel": channel.code,
                                "receiver_type": sensor_type,
                                "samp_rate_hz": sample_rate,
                                "start_date": start_date.strftime("%Y-%m-%d") if start_date else "None",
                                "end_date": end_date.strftime("%Y-%m-%d") if end_date else datetime.today().strftime("%Y-%m-%d")

                            })
                
                # 添加结果
                all_results.extend(net_results)
                
                if net_results:
                    successful_nets.append(net)
                    print(f"{net} 台网: 找到 {len(net_results)} 条通道数据")
                    network_time_bounds[net] = {
                        "min_start": net_min_start.strftime("%Y-%m-%d") if net_min_start else "None",
                        "max_end": net_max_end.strftime("%Y-%m-%d") if net_max_end else "None",
                        "client": client_name
                    }
                    success = True
                    
                    break  # 成功则跳出 client 循环

            except Exception as e:
                print(f"client {client_name} 处理 {net} 时出错: {e}")
                continue  # 尝试下一个 client

        if not success:
            print(f"{net} 台网: 所有 client 均失败")
            network_time_bounds[net] = {"min_start": "error", "max_end": "error","client":"error"}

    # 保存结果
    if all_results:
        df_all = pd.DataFrame(all_results)
        print("\n" + "="*50)
        print(f"所有台网汇总信息（共 {len(df_all)} 条记录）:")
        print(f"成功台网: {', '.join(successful_nets)} (共 {len(successful_nets)} 个)")
        print(f"唯一台站数量: {df_all[['network','station']].drop_duplicates().shape[0]}")
        print(f"唯一通道数量: {df_all[['network','station','channel']].drop_duplicates().shape[0]}")
              
        combined_filename = output_dir / f"metadata_channel.csv"
        df_all.to_csv(combined_filename, index=False)

        df_station = df_all.drop_duplicates(subset=["network","station"])[["network","station","latitude","longitude","elevation"]]
        comobined_station = output_dir / f"stations.csv"
        df_station.to_csv(comobined_station, index=False)

 
        # 保存台站数据

        print(f"\n所有台网数据已保存到: {combined_filename}")
        
        # 按台网分组统计
        group_stats = df_all.groupby("network").agg(
            stations=("station","nunique"),
            channels=("channel","count"),
            min_lat=("latitude","min"),
            max_lat=("latitude","max"),
            min_lon=("longitude","min"),
            max_lon=("longitude","max")
        ).reset_index()
        
        # 时间边界表
        time_bounds_data = []
        for net in nets:
            bounds = network_time_bounds.get(net, {"min_start": "None", "max_end": "None"})
            time_bounds_data.append({
                "network": net,
                "min_start_date": bounds["min_start"],
                "max_end_date": bounds["max_end"],
                "client_name": bounds["client"], 
                "Download": True if net in successful_nets else (False if bounds["min_start"] == "error" else "无数据")
            })
        time_bounds_df = pd.DataFrame(time_bounds_data)
        
        # 合并统计
        merged_stats = pd.merge(
            group_stats, time_bounds_df, on="network", how="outer"
        )
        merged_filename = output_dir / f"networks_stats.csv"
        merged_stats.to_csv(merged_filename, index=False)
        print(f"\n整合后的统计信息已保存为: {merged_filename}")
        
        # 保存成功台网列表
        with open("successful_networks.txt", "w") as f:
            f.write("\n".join(successful_nets))
    else:
        print("\n所有台网在指定范围内均未找到任何台站数据。")

    print("\n处理完成!")


if __name__ == "__main__":
    station_download()