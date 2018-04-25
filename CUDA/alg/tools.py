import numpy as np
import pandas as pd
import math
import sys
from shapely.geometry import Polygon, Point


def acc_check(result, EQ, r, grid=False):
    """вычисление точности алгоритма"""
    accEQ = 0
    if grid:
        r1, r2 = 15.5, 22
    else:
        r1, r2 = r, r*2

    for eq in EQ:
        if len(result) == 0:
            acc = 0
        else:
            evk = np.zeros((1, len(result)))
            for n, d in enumerate(eq):
                evk += (d - result[:, n]) ** 2
            evk = np.sqrt(evk[0])
            b_evk = evk[np.argmin(evk)]
            if b_evk <= r1:
                acc = 1
            elif r1 < b_evk <= r2:
                acc = (r2 - b_evk) / r1
            else:
                acc = 0
        accEQ += acc
    return round(accEQ / len(EQ), 6)