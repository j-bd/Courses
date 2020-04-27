#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 14:41:13 2020

@author: j-bd
"""

import pandas as pd

from explore.domain.data_explo import ExploData
from explore.settings import base


def main(d_explo=True):
    ''

    df = pd.read_csv(base.DATA_FILE)

    # TODO Create parser
    if d_explo:
        explo = ExploData(df)
        explo.get_pd_report()



if __name__ == '__main__':
    main()
