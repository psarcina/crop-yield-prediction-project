#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 13:20:05 2020

@author: pasq
"""
from pull_Sentinel_cloud import upload
from multiprocessing import Process

path = "/Users/pasq/Documents/ML/crop-yield-prediction-project"
loc_folder = r"/2019_clean_data/2019_locations_final.csv"
dest_folder = "/landsat_data"
in_folder_opt = ""
years = list(range(2014, 2020))

def yearUpload(year):
    return upload(path, loc_folder, dest_folder, in_folder_opt, year)

def runInParallel(years):
    proc = []
    for year in years:
        p = Process(target=yearUpload, args=(year,))
        p.start()
        proc.append(p)
    for p in proc:
        p.join()

runInParallel(years)
