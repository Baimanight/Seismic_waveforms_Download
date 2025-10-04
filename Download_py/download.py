import os
import csv
import time
import random
import shutil
import traceback
import multiprocessing
from datetime import datetime
from functools import partial

import pandas as pd
import numpy as np
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from geopy.distance import geodesic



class WaveformDownloader:
    def __init__(
        self,
        input_event,              # 事件csv路径或DataFrame
        input_station,             # 台站csv路径
        base_output_dir,           # 输出目录
        client_name="GFZ",
        channel="?H?",
        pre_event_time=20,
        post_time_rules=[(float('inf'), 150)],
        min_distance=2,
        max_distance=12,
        max_concurrent_events=8,
        request_timeout=3,
        max_retries=2,
        retry_delay=1,
        chunk_size=14,
        resume=True
    ):
        self.input_event = input_event
        self.input_station = input_station
        self.base_output_dir = base_output_dir

        # 下载参数
        self.client_name = client_name
        self.channel = channel
        self.pre_event_time = pre_event_time
        self.post_time_rules = post_time_rules
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.max_concurrent_events = max_concurrent_events
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.chunk_size = chunk_size
        self.resume = resume

        # 目录
        self.WAVEFORM_DIR = os.path.join(self.base_output_dir, "waveforms")
        self.LOG_DIR = os.path.join(self.base_output_dir, "logs")
        os.makedirs(self.base_output_dir, exist_ok=True)
        os.makedirs(self.WAVEFORM_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR, exist_ok=True)

        self.log_file_path = os.path.join(self.LOG_DIR, "download_detailed.log")
        self.summary_csv_path = os.path.join(self.LOG_DIR, "download_summary.csv")
        self.config_path = os.path.join(self.LOG_DIR, "download_config.txt")

        # 读取输入
        self.df_stations = pd.read_csv(self.input_station).copy()

        self.full_df_earthquakes = pd.read_csv(self.input_event).copy()

        # 断点续传：初始化 download 列并过滤未下载事件
        if "download" not in self.full_df_earthquakes.columns:
            self.full_df_earthquakes["download"] = 0

        if self.resume: 
            self.df_earthquakes = self.full_df_earthquakes[self.full_df_earthquakes["download"] == 0].copy()
        else:
            self.df_earthquakes = self.full_df_earthquakes
        self.original_indices = list(self.df_earthquakes.index)  # 记录原始索引

        # 初始化日志文件
        with open(self.log_file_path, "w") as f:
            f.write(f"下载开始时间: {datetime.now().isoformat()}\n")
            f.write(f"输出目录: {self.base_output_dir}\n")
            f.write(f"地震事件数: {len(self.df_earthquakes)}\n")
            f.write(f"台站数: {len(self.df_stations)}\n")
            f.write("=" * 80 + "\n")

        with open(self.summary_csv_path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["EventID", "EventTime", "StationsFound", "StationsSuccess",
                             "ChannelsSuccess", "StationsSkipped", "StationsFailed",
                             "DownloadTime", "Status"])

    # ====================== 辅助工具 ======================
    def log_message(self, message, level="INFO"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        with open(self.log_file_path, "a") as f:
            f.write(log_entry + "\n")
        print(log_entry)

    def assign_post_event_time(distance, post_time_rules):
        for threshold, value in post_time_rules:
            if distance < threshold:
                return value


    def process_event(self, event_row, event_index, total_events):
        start_time = time.time()
        event_id = f"{event_index + 1}/{total_events}"
        event_dir_created = False
        files_saved = []

        try:
            eq = event_row
            eq_time = UTCDateTime(eq["time"])
            eq_lat = eq["latitude"]
            eq_lon = eq["longitude"]

            client = Client(self.client_name,timeout=150)

            # 计算震中距
            self.df_stations["distance_deg"] = self.df_stations.apply(
                lambda x: geodesic((eq_lat, eq_lon), (x["latitude"], x["longitude"])).km / 111.32,
                axis=1
            )
            mask = (self.df_stations["distance_deg"] >= self.min_distance) & \
                   (self.df_stations["distance_deg"] <= self.max_distance)
            far_stations = self.df_stations[mask].copy()

            self.log_message(f"事件 {event_id} 开始处理: {eq_time.isoformat()}, 找到 {len(far_stations)} 个台站")

            # for different window length 
            far_stations["post_event_time"] = far_stations["distance_deg"].apply(
                lambda d: self.assign_post_event_time(d, self.post_time_rules)
            )
            # 批量请求
            bulk = [
                (row["network"], row["station"], "*", self.channel,
                eq_time - self.pre_event_time, eq_time + row["post_event_time"])
                for _, row in far_stations.iterrows()
            ]

            successful_downloads = 0
            skipped_downloads = 0
            successful_stations = set()

            event_time_str = eq_time.strftime('%Y%m%d_%H%M%S')
            event_dir = os.path.join(self.WAVEFORM_DIR, event_time_str)
            for i in range(0, len(bulk), self.chunk_size):
                bulk_chunk = bulk[i:i + self.chunk_size]

                for attempt in range(self.max_retries + 1):
                    try:
                        time.sleep(random.uniform(0.5, 2.0))
                        stream = client.get_waveforms_bulk(
                            bulk_chunk, attach_response=True, request_timeout=self.request_timeout
                        )

                        if not event_dir_created and len(stream) > 0:
                            os.makedirs(event_dir, exist_ok=True)
                            event_dir_created = True
                            self.log_message(f"事件 {event_id} 创建目录: {event_dir}")

                        for tr in stream:
                            location_code = tr.stats.location if tr.stats.location else "00"
                            year = eq_time.year
                            julian_day = eq_time.julday
                            time_identifier = eq_time.strftime('%H%M%S')
                            filename = os.path.join(
                                event_dir,
                                f"{tr.stats.network}.{tr.stats.station}.{location_code}."
                                f"{tr.stats.channel}.{year}.{julian_day:03d}.{time_identifier}.SAC"
                            )

                            try:
                                tr.write(filename, format="SAC")
                                successful_downloads += 1
                                successful_stations.add(f"{tr.stats.network}.{tr.stats.station}")
                                files_saved.append(filename)
                            except Exception:
                                backup_filename = filename.replace(".SAC", ".mseed")
                                tr.write(backup_filename, format="MSEED")
                                files_saved.append(backup_filename)

                        break  # 成功下载后跳出重试
                    except Exception as e:
                        error_msg = str(e)
                        if "204" in error_msg or "No data available" in error_msg:
                            skipped_downloads += len(bulk_chunk)
                            self.log_message(f"事件 {event_id} 块 {i // self.chunk_size + 1}: 无数据(204)")
                            break
                        elif "timeout" in error_msg.lower() and attempt < self.max_retries:
                            wait_time = self.retry_delay * (attempt + 1)
                            self.log_message(f"事件 {event_id} 块 {i // self.chunk_size + 1}: 超时，{wait_time}秒后重试")
                            time.sleep(wait_time)
                        else:
                            self.log_message(f"事件 {event_id} 块 {i // self.chunk_size + 1}: 错误 - {error_msg}")
                            break

            successful_station_count = len(successful_stations)
            failed_count = len(far_stations) - successful_station_count - skipped_downloads

            # 清理空目录
            if event_dir_created and len(files_saved) == 0:
                shutil.rmtree(event_dir, ignore_errors=True)
                self.log_message(f"事件 {event_id} 删除空目录: {event_dir}")

            elapsed = time.time() - start_time
            summary = f"[{event_id}] {eq_time.isoformat()} | 台站: 找到 {len(far_stations)} | 成功 {successful_station_count} | 跳过 {skipped_downloads} | 失败 {failed_count} | 通道: {successful_downloads}"
            self.log_message(f"事件 {event_id} 完成: {summary}")

            return {
                "event_id": event_id,
                "event_time": eq_time.isoformat(),
                "stations_found": len(far_stations),
                "stations_success": successful_station_count,
                "channels_success": successful_downloads,
                "stations_skipped": skipped_downloads,
                "stations_failed": failed_count,
                "download_time": elapsed,
                "status": "success",
                "files_saved": len(files_saved)
            }

        except Exception as e:
            elapsed = time.time() - start_time
            if event_dir_created:
                shutil.rmtree(event_dir, ignore_errors=True)
            self.log_message(f"事件 {event_id} 错误: {traceback.format_exc()}", "ERROR")
            return {
                "event_id": event_id,
                "event_time": "N/A",
                "stations_found": 0,
                "stations_success": 0,
                "channels_success": 0,
                "stations_skipped": 0,
                "stations_failed": 0,
                "download_time": elapsed,
                "status": f"error: {str(e)}",
                "files_saved": 0
            }

    def make_update_callback(self):

        def update_and_save(result):
            if result["files_saved"] > 0:
                idx = int(result["event_id"].split("/")[0]) - 1
                original_idx = self.original_indices[idx]
                self.full_df_earthquakes.loc[original_idx, "download"] = 1
                self.full_df_earthquakes.to_csv(self.input_event, index=False)
        return update_and_save

    # ====================== 主入口 ======================
    def download(self):
        usetime = time.time()
        total_events = len(self.df_earthquakes)

        self.log_message(f"开始处理 {total_events} 个地震事件")
        self.log_message(f"并行处理: {self.max_concurrent_events} 个事件")
        self.log_message(f"数据目录: {self.WAVEFORM_DIR}")
        self.log_message(f"日志目录: {self.LOG_DIR}")

        pool = multiprocessing.Pool(processes=self.max_concurrent_events)
        update_callback = self.make_update_callback()

        results = []
        for i in range(total_events):
            res = pool.apply_async(
                self.process_event,
                args=(self.df_earthquakes.iloc[i].to_dict(), i, total_events),
                callback=update_callback
            )
            results.append(res)

        final_results = []
        for res in results:
            try:
                result = res.get()
                final_results.append(result)
            except Exception as e:
                self.log_message(f"进程错误: {traceback.format_exc()}", "ERROR")

        pool.close()
        pool.join()

        # 保存结果
        with open(self.summary_csv_path, "a", newline='') as f:
            writer = csv.writer(f)
            for r in final_results:
                writer.writerow([
                    r["event_id"], r["event_time"], r["stations_found"],
                    r["stations_success"], r["channels_success"],
                    r["stations_skipped"], r["stations_failed"],
                    r["download_time"], r["status"]
                ])

        # 统计与日志
        total_time = time.time()-usetime
        total_stations_found = sum(r["stations_found"] for r in final_results)
        total_stations_success = sum(r["stations_success"] for r in final_results)
        total_channels_success = sum(r["channels_success"] for r in final_results)
        total_stations_skipped = sum(r["stations_skipped"] for r in final_results)
        total_stations_failed = sum(r["stations_failed"] for r in final_results)
        total_files_saved = sum(r["files_saved"] for r in final_results)
        events_with_data = sum(1 for r in final_results if r["files_saved"] > 0)

        self.log_message("=" * 60)
        self.log_message(f"所有事件处理完成! 总耗时: {total_time:.1f} 秒 ({total_time/60:.1f} 分钟)")
        self.log_message(f"成功处理事件: {events_with_data}/{total_events} (包含数据的事件)")
        self.log_message(f"台站统计: 找到 {total_stations_found} | 成功 {total_stations_success} | 跳过 {total_stations_skipped} | 失败 {total_stations_failed}")
        self.log_message(f"通道统计: 成功下载 {total_channels_success} 个波形")
        self.log_message(f"文件统计: 保存 {total_files_saved} 个波形文件")
        self.log_message(f"平均下载速度: {total_files_saved/total_time:.2f} 文件/秒" if total_time > 0 else "N/A")
        self.log_message("=" * 60)

        # 保存最终统计
        final_summary_path = os.path.join(self.LOG_DIR, "final_summary.txt")
        with open(final_summary_path, "w") as f:
            f.write(f"处理完成时间: {datetime.now().isoformat()}\n")
            f.write(f"输出目录: {self.base_output_dir}\n")
            f.write(f"总事件数: {total_events}\n")
            f.write(f"包含数据的事件数: {events_with_data}\n")
            f.write(f"无数据的事件数: {total_events - events_with_data}\n")
            f.write(f"总耗时: {total_time/3600:.1f} h ({total_time/60:.1f} min)\n")
            f.write(f"总台站数: {total_stations_found}\n")
            f.write(f"成功台站: {total_stations_success}\n")
            f.write(f"跳过台站: {total_stations_skipped}\n")
            f.write(f"失败台站: {total_stations_failed}\n")
            f.write(f"总通道数: {total_channels_success}\n")
            f.write(f"总文件数: {total_files_saved}\n")
            f.write(f"平均下载速度: {total_files_saved/total_time:.2f} 文件/秒\n")

        self.log_message(f"最终统计已保存到: {final_summary_path}")
        self.log_message(f"下载总结已保存到: {self.summary_csv_path}")
        self.log_message(f"详细日志已保存到: {self.log_file_path}")

        # 保存配置信息
        with open(self.config_path, "w") as f:
            f.write("下载配置参数:\n")
            f.write(f"基础输出目录: {self.base_output_dir}\n")
            f.write(f"波形数据目录: {self.WAVEFORM_DIR}\n")
            f.write(f"日志目录: {self.LOG_DIR}\n")
            f.write(f"客户端: {self.client_name}\n")
            f.write(f"通道: {self.channel}\n")
            f.write(f"请求超时: {self.request_timeout}秒\n")
            f.write(f"最大重试次数: {self.max_retries}\n")
            f.write(f"重试延迟: {self.retry_delay}秒\n")
            f.write(f"分块大小: {self.chunk_size}\n")
            f.write(f"震前时间: {self.pre_event_time}秒\n")
            f.write(f"震后时间: {self.post_event_time}秒\n")
            f.write(f"最大并发事件数: {self.max_concurrent_events}\n")

        self.log_message(f"配置信息已保存到: {self.config_path}")

if __name__ == "__main__":
    "Wavefor Download"