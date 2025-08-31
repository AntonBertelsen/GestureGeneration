import time
import numpy as np

class PerformanceTracker:
    def __init__(self):
        self.frame_times = []
        self.timestamps = []
        self.cumulative_frames = []
        self.start_time = time.time()
    
    def record_frame(self, frame_time):
        self.frame_times.append(frame_time)
        current_time = time.time() - self.start_time
        self.timestamps.append(current_time)
        self.cumulative_frames.append(len(self.frame_times))
        
    def get_stats(self):
        return {
            "mean_frame_time": np.mean(self.frame_times),
            "median_frame_time": np.median(self.frame_times),
            "min_frame_time": min(self.frame_times),
            "max_frame_time": max(self.frame_times),
            "std_frame_time": np.std(self.frame_times),
            "total_frames": len(self.frame_times),
            "total_time": self.timestamps[-1] if self.timestamps else 0,
            "fps": len(self.frame_times) / self.timestamps[-1] if self.timestamps else 0
        }