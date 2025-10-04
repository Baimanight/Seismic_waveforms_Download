import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import contextily as ctx
import geopandas as gpd
from shapely.geometry import Point
from matplotlib.ticker import ScalarFormatter, FuncFormatter
from scipy.spatial import cKDTree
from obspy.geodetics import locations2degrees
from matplotlib.colors import BoundaryNorm, ListedColormap
import math
import requests
from pathlib import Path
import matplotlib.ticker as mticker
from datetime import timedelta




""" event form USGS ~M4.5+ """
def event_USGS(start_time,end_time,magnitude_range,lat_range,lon_range,output_path):

    # USGS API参数设置   min magnitude about 4.5
    url = (
        "https://earthquake.usgs.gov/fdsnws/event/1/query?"
        f"format=geojson&starttime={start_time}&endtime={end_time}"
        f"&minmagnitude={magnitude_range[0]}&maxmagnitude={magnitude_range[1]}"
        f"&minlatitude={lat_range[0]}&maxlatitude={lat_range[1]}"
        f"&minlongitude={lon_range[0]}&maxlongitude={lon_range[1]}"
    )

    # 发送请求获取数据
    response = requests.get(url)
    data = response.json()

    # 提取地震事件信息
    earthquakes = []
    for feature in data["features"]:
        props = feature["properties"]
        coords = feature["geometry"]["coordinates"]
        earthquakes.append({
            "time": props["time"],
            "latitude": coords[1],
            "longitude": coords[0],
            "depth_km": coords[2],
            "magnitude": props["mag"],
            "magnitude_type": props.get("magType", None),
            "place": props["place"]
        })

    # 转换为DataFrame并保存为CSV
    df = pd.DataFrame(earthquakes)
    df["time"] = pd.to_datetime(df["time"], unit="ms")  # 转换时间戳为日期

    # 保存CSV文件（原功能保留）
    df.to_csv( output_path / "event_USGS.csv", index=False)

    print(f"地震目录已保存为 {output_path}/event_USGS.csv")

""" merge different event """

def merge_event(cn_file, USGS_file, output_dir, time_type="UTC",
                mag_format=False,
                trans_name={
                    "time": "发震日期（北京时间）",
                    "latitude": "纬度(°)",
                    "longitude": "经度(°)",
                    "depth_km": "震源深度(Km)",
                    "magnitude": "magnitude",
                    "magnitude_type": "震级类型"
                }):
    """
    合并中国台网与国际地震目录，支持震级优先选择和时间窗口去重。
    
    参数:
        cn_file: 中国台网文件路径 (xls/xlsx/csv)
        USGS_file: 国际目录 CSV 文件路径
        output_dir: 输出目录 (Path 或字符串)
        time_type: "CST" 或 "UTC"
        mag_format: True 时优先选择 mL，False 使用原有 magnitude
        trans_name: 字段名称映射
    返回:
        合并后的 DataFrame
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 读取文件
    def read_file(file_path):
        file_path = Path(file_path)
        if file_path.suffix.lower() in [".xls", ".xlsx"]:
            return pd.read_excel(file_path)
        elif file_path.suffix.lower() == ".csv":
            return pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    cn_df = read_file(cn_file)

    # 时间处理
    if time_type.upper() == "CST":
        cn_df["time"] = pd.to_datetime(cn_df[trans_name["time"]]) - timedelta(hours=8)
    elif time_type.upper() == "UTC":
        cn_df["time"] = pd.to_datetime(cn_df[trans_name["time"]])
    else:
        raise ValueError("time_type must be either 'CST' or 'UTC'")



    if mag_format:  # 按优先级选择震级
        mag_cols = ["mL", "Ms", "Ms7", "mb", "mB"]

        def select_magnitude(row):
            # 按优先级查找
            for col in mag_cols:
                if col in row and not pd.isna(row[col]):
                    # 返回 [数值, 选择的列名]
                    return pd.Series([row[col], col])
            # 如果都没有命中，退回默认列
            return pd.Series([row.get(trans_name["magnitude"], None), trans_name["magnitude"]])

        # 应用到 DataFrame，生成两列
        cn_df[["magnitude", "magnitude_type"]] = cn_df.apply(select_magnitude, axis=1)

        # 坐标和深度列重命名
        rename_map = {
            trans_name["latitude"]: "latitude",
            trans_name["longitude"]: "longitude",
            trans_name["depth_km"]: "depth_km"
        }
    else:  # 直接使用原始列

        rename_map = {
            trans_name["latitude"]: "latitude",
            trans_name["longitude"]: "longitude",
            trans_name["depth_km"]: "depth_km",
            trans_name["magnitude"]: "magnitude",
            trans_name["magnitude_type"]:"magnitude_type"
        }


    cn_df.rename(columns=rename_map, inplace=True)
    cn_df = cn_df[["time", "latitude", "longitude", "depth_km", "magnitude", "magnitude_type"]]

    # 国际目录读取
    intl_df = pd.read_csv(USGS_file)
    intl_df["time"] = pd.to_datetime(intl_df["time"])
    intl_df = intl_df[["time", "latitude", "longitude", "depth_km", "magnitude", "magnitude_type"]]

    # 合并

    all_events = pd.concat([intl_df, cn_df], ignore_index=True)
    all_events.sort_values("time", inplace=True)

    # 时间 + 空间窗口去重 (1 分钟 + 0.1 度)
    keep = []
    last_row = None
    for i, row in all_events.iterrows():
        if last_row is None:
            keep.append(i)
            last_row = row
        else:
            dt = (row["time"] - last_row["time"]).total_seconds()
            if dt <= 60 and abs(row["latitude"] - last_row["latitude"]) <= 0.1 and abs(row["longitude"] - last_row["longitude"]) <= 0.1:
                continue
            else:
                keep.append(i)
                last_row = row

    result = all_events.loc[keep].reset_index(drop=True)

    # 保存
    out_file = output_dir / "merged_event.csv"
    result.to_csv(out_file, index=False)
    print(f"Merged event saved to: {out_file}")

    return result



# ===================== 筛选函数 =====================
def count_stations_fast(events_df, stations_df, min_deg, max_deg):
    """
    计算每个事件在给定台站表中的可用台站数量
    min_deg, max_deg: 事件与台站的震中距筛选范围 (单位: 度)
    """
    def latlon_to_xyz(lat, lon):
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        x = np.cos(lat_rad) * np.cos(lon_rad)
        y = np.cos(lat_rad) * np.sin(lon_rad)
        z = np.sin(lat_rad)
        return np.vstack((x, y, z)).T

    station_xyz = latlon_to_xyz(stations_df["latitude"].values,
                                stations_df["longitude"].values)
    tree = cKDTree(station_xyz)

    counts = []
    for lat, lon in zip(events_df["latitude"], events_df["longitude"]):
        event_xyz = latlon_to_xyz([lat], [lon])
        idxs = tree.query_ball_point(event_xyz, r=2.0)  # 先用弦长近似筛选
        count = 0
        for idx in idxs[0]:
            sta_lat, sta_lon = stations_df.iloc[idx][["latitude", "longitude"]]
            deg = locations2degrees(lat, lon, sta_lat, sta_lon)
            if min_deg <= deg <= max_deg:
                count += 1
        counts.append(count)

    events_df["station_count"] = counts
    return events_df

# ===================== 事件筛选 =====================
def filter_events(input_catalog, input_station, output_catalog,
                  min_magnitude=3, max_magnitude=10.0,
                  min_distance_deg=2.0, max_distance_deg=12.0,
                  min_depth=1, max_depth=45):
    """
    从事件表中过滤符合条件的事件:
      - 震级范围
      - 深度范围
      - 至少有一个满足距离条件的台站
    结果保存为 CSV
    """
    df = pd.read_csv(input_catalog)
    stations = pd.read_csv(input_station)

    # 计算台站数量
    df = count_stations_fast(df, stations, min_distance_deg, max_distance_deg)

    # 筛选
    filtered = df[
        (df["magnitude"] >= min_magnitude) &
        (df["magnitude"] < max_magnitude) &
        (df["depth_km"] >= min_depth) &
        (df["depth_km"] <= max_depth) &
        (df["station_count"] > 0)
    ].copy()

    if not filtered.empty:
        filtered.to_csv(output_catalog, index=False)
        print(f"✅ 事件数: {len(filtered)} 已保存到 {output_catalog}")
    else:
        print("⚠️ 没有符合条件的事件！")

    return filtered

# ===================== 绘图函数 =====================
def plot_raw_events(input_event, input_station,resolution=0.1):
    """
    绘制台站数量分布、网格事件数量曲线，以及事件分布热力图。
    """
    def assign_grid_label(lat_val, lon_val, lat_bins, lon_bins):
        lat_idx = np.digitize(lat_val, lat_bins) - 1
        lon_idx = np.digitize(lon_val, lon_bins) - 1
        lat_idx = max(0, min(lat_idx, len(lat_bins)-2))
        lon_idx = max(0, min(lon_idx, len(lon_bins)-2))
        return f"{lat_idx}_{lon_idx}"

    stations = pd.read_csv(input_station)
    df = pd.read_csv(input_event)
    try:
        station_count_stats = df["station_count"].value_counts().sort_index()
    except:
        station_count_stats = df["total_picks"].value_counts().sort_index()
    lat = df["latitude"].values
    lon = df["longitude"].values

    # 计算累加（≥当前索引的事件数量）
    cumulative_stats = station_count_stats[::-1].cumsum()[::-1]

    # 定义网格
    lat_bins = np.arange(lat.min(), lat.max() + resolution, resolution)
    lon_bins = np.arange(lon.min(), lon.max() + resolution, resolution)
    df["grid_label"] = df.apply(
        lambda row: assign_grid_label(row["latitude"], row["longitude"], lat_bins, lon_bins),
        axis=1
    )

    grid_counts = df["grid_label"].value_counts().sort_values(ascending=True)
    grid_ids_sorted = range(1, len(grid_counts) + 1)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df["longitude"], df["latitude"])],
        crs="EPSG:4326"
    ).to_crs(epsg=3857)


    # ===================== Figure 1: 台站数量分布 + 变化曲线 =====================
    fig1 = plt.figure(figsize=(7, 5))
    gs1 = plt.GridSpec(2, 1, figure=fig1, height_ratios=[1, 1])
    ax_top = fig1.add_subplot(gs1[0])
    ax_bottom = fig1.add_subplot(gs1[1])

    # 上：台站数量分布
    ax_top.bar(station_count_stats.index, station_count_stats.values,
            color="skyblue", alpha=0.7, edgecolor="black", width=0.8)
    ax_top.set_xlabel("Number of Station", fontsize=12)
    ax_top.set_ylabel("Events", fontsize=12)
    ax_top.set_title("Station Count Distribution", fontsize=14)
    ax_top.grid(True, alpha=0.3)

    # 下：网格事件数量曲线
    ax_bottom.plot(cumulative_stats.index,cumulative_stats.values,
                "b-", linewidth=2, label="All Events")
    ax_bottom.set_xlabel(" Associated Stations' Number ", fontsize=12)
    ax_bottom.set_ylabel(" Event Number ", fontsize=12)
    ax_bottom.set_title(" Event Count ", fontsize=14)
    ax_bottom.legend()
    ax_bottom.grid(True, alpha=0.3)

    total_plot = cumulative_stats.values.max()   # 总事件数（曲线最大值）
    ax_right = ax_bottom.twinx()            # 右侧坐标轴
    ax_right.set_ylim(ax_bottom.get_ylim())  # 保持与左侧相同的范围
    ax_right.set_ylabel("Percentage (%)", fontsize=12)
    # 刻度转换为百分比
    ax_right.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v / total_plot * 100:.0f}%")
    )

    plt.tight_layout()
    plt.show()

    # ===================== Figure 2: 单独热力图 =====================
    fig2, ax_heatmap = plt.subplots(figsize=(10, 8))

    lon_min, lon_max = df["longitude"].min(), df["longitude"].max()
    lat_min, lat_max = df["latitude"].min(), df["latitude"].max()

    lon_start = math.floor(lon_min)
    lon_end   = math.ceil(lon_max)
    lat_start = math.floor(lat_min)
    lat_end   = math.ceil(lat_max)

    lon_ticks = np.arange(lon_start, lon_end + 1, 2)
    lat_ticks = np.arange(lat_start, lat_end + 1, 2)

    # 期望 0.1° 分辨率
    nx = int((lon_max - lon_min) / resolution)
    ny = int((lat_max - lat_min) / resolution)

    hb = ax_heatmap.hexbin(
        df["longitude"], df["latitude"],
        gridsize=(nx, ny),
        extent=[lon_min, lon_max, lat_min, lat_max],
        cmap="viridis", alpha=0.6, mincnt=1
    )
    hb.set_clim(0, 50) 
    cbar = plt.colorbar(hb, ax=ax_heatmap, shrink=0.8)
    cbar.set_label("Number of Stations", fontsize=12)

    ax_heatmap.set_xlim(df["longitude"].min(), df["longitude"].max())
    ax_heatmap.set_ylim(df["latitude"].min(), df["latitude"].max())

    ax_heatmap.set_xticks(lon_ticks)
    ax_heatmap.set_xticklabels([f"{x:.0f}°" for x in lon_ticks])
    ax_heatmap.set_yticks(lat_ticks)
    ax_heatmap.set_yticklabels([f"{y:.0f}°" for y in lat_ticks])

    try:
        ctx.add_basemap(
            ax=ax_heatmap, crs='EPSG:4326',
            source=ctx.providers.Esri.WorldShadedRelief,
            alpha=0.8
        )
    except Exception as e:
        print(f"⚠️ 无法添加底图: {e}")

    ax_heatmap.set_xlabel("Longitude", fontsize=12)
    ax_heatmap.set_ylabel("Latitude", fontsize=12)
    ax_heatmap.set_title("Event Distribution", fontsize=14)

    ax_heatmap.scatter(
            stations["longitude"], stations["latitude"],
            marker="^", color="black", s=40, label="Stations", zorder=3
        )
    ax_heatmap.legend(loc="upper right")




    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    "event_filter";















