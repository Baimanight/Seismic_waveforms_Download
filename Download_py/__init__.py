from .station import station_download
from .event import (count_stations_fast,filter_events,plot_raw_events,plot_selected_events,
                    select_event,merge_event,event_USGS)
from .download import WaveformDownloader
from .trival import extract_earthquake_events, generate_gmt_rays,contrast,start_timer,end_timer
from .ISC_crawler import isc_crawler


"""  
a simple package for download waveform
program for personal usage
Hopefully this can help your work 

2025-10-04 
details in https://github.com/Baimanight/Seismic_waveforms_Download
"""

__author__ = "Baimanight/Hua"


# all just for import   a whitelist
# __all__ = ["station_download", "event_USGS"]