#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 12:51:53 2020

@author: j-bd
"""
from dataclasses import dataclass, field
import logging
import os

import pandas as pd
from pandas_profiling import ProfileReport

from explore.settings import base

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)



@dataclass
class ExploData:
    input_df: pd.DataFrame
    xs: pd.DataFrame = field(init=False)
    target: pd.DataFrame = field(init=False)

    def __post_init__(self):
        self.xs = self.input_df.drop(columns=[base.TARGET])
        self.target = self.input_df[base.TARGET]

    def get_pd_report(self):
        profile = ProfileReport(
            self.input_df, title='Pandas Profiling Report',
            html={'style':{'full_width':True}}
        )
        path_report = os.path.join(
            base.REPORTS_DIR, 'pandas_profiling_report.html'
        )
        profile.to_file(output_file=path_report)
        logging.INFO(f'Report export to {path_report}')

