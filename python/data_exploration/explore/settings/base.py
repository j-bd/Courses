#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 12:55:52 2020

@author: j-bd
"""
import os


# Path variables
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
DATA_DIR = os.path.join(REPO_DIR, 'data')
EXTERNAL_DATA_DIR = os.path.join(DATA_DIR, 'external')
INTERIM_DATA_DIR = os.path.join(DATA_DIR, 'interim')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
MODELS_DIR = os.path.join(REPO_DIR, 'models')
REPORTS_DIR = os.path.join(REPO_DIR, 'reports')

RAW_NAME = 'data.csv' # TODO adapt to your entry file

DATA_FILE = os.path.join(RAW_DATA_DIR, RAW_NAME)

# Global data information
COL_TO_DELETE = ['']
TARGET = 'SUBSCRIPTION'