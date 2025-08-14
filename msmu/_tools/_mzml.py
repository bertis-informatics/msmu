# import pandas as pd
# from pathlib import Path
# from typing import TypedDict
# import time

# from pymzml.run import Reader as MzmlReader
# from pymzml.file_classes import standardMzml as StandardMzml


# class FrameInfo(TypedDict):
#     ID: int
#     ms_level: int
#     scan_time: float
#     ms1_scan_id: int


# def read_mzml(file_path: str | Path) -> MzmlReader:
#     return MzmlReader(file_path)


# def get_frame_df(mzml: MzmlReader) -> pd.DataFrame:
#     start = time.time()

#     frames_info: list[FrameInfo] = []
#     ms1_scan_id: int = 0
#     for idx, frame in enumerate(mzml):
#         frame_info = _get_frame_info(frame)
#         if frame_info["ms_level"] == 1:
#             ms1_scan_id = frame.ID

#         frame_info["ms1_scan_id"] = ms1_scan_id
#         frames_info.append(frame_info)

#     end = time.time()
#     print(f"Read frame {idx} frames for {end-start}")

#     return pd.DataFrame(frames_info)


# def _get_frame_info(frame: StandardMzml) -> FrameInfo:
#     return {
#         "ID": frame.ID,
#         "ms_level": frame.ms_level,
#         "scan_time": frame.scan_time_in_minutes(),
#         "ms1_scan_id": -1,
#     }


# def get_ms2_isolation_window(ms2_frame):
#     isolation_window_target = float(ms2_frame["MS:1000827"])  # CV param for target window
#     isolation_window_lower_offset = float(ms2_frame["MS:1000828"])  # CV param for min target window offset
#     isolation_window_upper_offset = float(ms2_frame["MS:1000829"])  # CV param for max target window offset

#     if any([isolation_window_target, isolation_window_lower_offset, isolation_window_upper_offset]) is None:
#         raise ValueError("Isolation window values are not found in the frame. Possibly not an MS2 frame.")

#     isolation_window_low = isolation_window_target - isolation_window_lower_offset
#     isolation_window_high = isolation_window_target + isolation_window_upper_offset

#     return (isolation_window_low, isolation_window_high)
