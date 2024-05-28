import pstats
import os
from datetime import datetime

path_out_logs = os.path.join(os.path.dirname(__file__), "logs")
# timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

with open(os.path.join(path_out_logs, "performance.log"), "w") as stream:
    stats = pstats.Stats(os.path.join(path_out_logs, "output.prof"), stream=stream)
    stats.sort_stats("cumtime")
    stats.print_stats()
