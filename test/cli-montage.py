from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from numpy.random import rand

from smallmatter.ds import MontagePager

plt.ioff()
mp = MontagePager(path=Path("output"), savefig_kwargs=dict(transparent=False))
for i in range(112):
    title = f"chart\n{i:04d}"
    pd.DataFrame({"x": range(6), "y": rand(6)}).plot(ax=mp.pop(subplot_id=title), x="x", y="y", title=title)
mp.savefig()
