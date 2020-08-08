from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from numpy.random import rand

from smallmatter.ds import MontagePager

plt.ioff()
mp = MontagePager(path=Path("output"))
for i in range(12):
    pd.DataFrame({"x": range(6), "y": rand(6)}).plot(ax=mp.pop(), x="x", y="y", title=f"chart\n{i:04d}")
mp.savefig()
