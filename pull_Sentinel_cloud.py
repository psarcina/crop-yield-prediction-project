#import os
#
#os.chdir("/Users/pasq/Documents/ML/crop-yield-prediction-project/clean_data")

import os
import ee
import time
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from IPython.display import Image, display

ee.Initialize()

def alert(func):
    def inner(*args, **kwargs):
        func(*args, **kwargs)
        os.system('say "Your program has finished"')
        
    return inner


def export_oneimage(img,folder,name,scale,crs):
  full_file_name = folder + '_' + name
  task = ee.batch.Export.image.toCloudStorage(img, \
          bucket='planet-kaggle',\
          fileNamePrefix=full_file_name,\
          scale=scale,\
          crs=crs,\
          maxPixels=1e12)
  task.start()
  i = 0
  while (task.status()['state'] == 'RUNNING') | (task.status()['state'] == 'READY'):
    i += 1
    print('Running...'+" "+str(i))
    print(task.status()["state"])
    # Perhaps task.cancel() at some point.
    time.sleep(10)
  print('Done.', task.status()) 

# Transforms an Image Collection with 1 band per Image into a single Image with items as bands
# Author: Jamie Vleeshouwer

def appendBand(current, previous):
    # Rename the band
    previous=ee.Image(previous)
    current = current.select(["B2", "B3", "B4", "NDVI"])
    # Append it to the result (Note: only return current item on first element/iteration)
    accum = ee.Algorithms.If(ee.Algorithms.IsEqual(previous,None), current, previous.addBands(ee.Image(current)))
    # Return the accumulation
    return accum

def weeklyNCRegion(new_region):
    filter_region = new_region
    def weeklyNoCloud(date_millis):
        date = ee.Date(date_millis)
        sentinel = ee.ImageCollection("LANDSAT/LC08/C01/T1_SR")\
                .filterBounds(filter_region)\
                .filterDate(date, date.advance(1, "month"))\
                .filterMetadata("CLOUD_COVER", "less_than", 50)\
                .median()
        
        return sentinel
    
    return weeklyNoCloud

#def weeklyNCRegion_first(new_region):
#    filter_region = new_region
#    def weeklyNoCloud(date_millis):
#        date = ee.Date(date_millis)
#        sentinel = ee.ImageCollection("COPERNICUS/S2_SR")\
#                .filterBounds(filter_region)\
#                .filterDate(date, date.advance(1, "month"))\
#                .sort("CLOUD_COVERAGE_ASSESSMENT")\
#                .first()
#        
#        return sentinel
#    
#    return weeklyNoCloud


def addNDVI(img):
    ndvi = img.normalizedDifference(["B5", "B4"]).select([0]).rename("NDVI")
    return img.addBands(ndvi)

@alert
def display_ee_img(image, params):
    url = image.getThumbUrl(params)
    img = Image(requests.get(url).content)
    display(img)

#params = {
#        "min": 0,
#        "max": 10000,
#        "dimensions": 512,
#        "bands": ["B4_52","B3_52","B2_52"],
#        "region": region.first().getInfo()["geometry"]
#        }

def fill_pad(img):
    small_dim = img.shape.index(min(img.shape[2:]))
    missing_pix = 512 - img.shape[small_dim]
    pad1 = missing_pix//2
    pad2 = missing_pix//2 + missing_pix%2
    padding = [(0,0),(0,0),(0,0),(0,0)]
    padding[small_dim] = (pad1,pad2)
    img = np.pad(img, padding, constant_values=0)
    return img



def upload(path, loc_folder, dest_folder, in_folder_opt, year):
    locations = pd.read_csv(path+loc_folder, header=None)
    county_region = ee.FeatureCollection("TIGER/2016/Counties")
    year = str(year)
    if in_folder_opt == "folder":
        np_list = pd.Series(os.listdir(path+dest_folder))
        sc_in_folder = np_list.str[:-4].str.split("_")
        in_folder_list = sc_in_folder.to_list()
    elif in_folder_opt == "log_ls":
        np_list = pd.read_fwf(path+"/log_ls", header=None)[0]
        sc_in_folder = np_list.str[:-4].str.split("_")
        in_folder_list = sc_in_folder.to_list()
    else:
        in_folder_list = []
    
#    if isinstance(dest_folder, pd.core.frame.DataFrame):
#        np_list = dest_folder.iloc[:,0]
#        sc_in_folder = np_list.str[:-4].str.split("_")
#        sc_in_folder = sc_in_folder.to_list()
#    else:
#        np_list = pd.Series(os.listdir(path+dest_folder))
#        if len(np_list) == 0:
#            sc_in_folder = []
#        else:
#            sc_in_folder = np_list.str[:-4].str.split("_")
#            sc_in_folder = sc_in_folder.to_list()

    for loc1, loc2, lon, lat in locations.values:
        try:
            s = "0"+str(int(loc1))
            c = "00"+str(int(loc2))
            s = s[-2:]
            c = c[-3:]
            if [year, s, c] in in_folder_list:
                print("Already in the folder")
#                continue
                
            fname = '{}_{}_{}'.format(str(year),s,c)
            print("Starting uploading file: %s" % fname)
            
            # filter for a county
            region = county_region.filterMetadata('STATEFP', 'equals', s)
            region = ee.FeatureCollection(region).filterMetadata('COUNTYFP', 'equals', c)
            
            start_date = str(year)+"-3-1"
            end_date = str(year)+"-9-1"
            month_diff = ee.Date(start_date).advance(1,"month").millis().subtract(ee.Date(start_date).millis())
            list_map_2019 = ee.List.sequence(ee.Date(start_date).millis(), ee.Date(end_date).millis(), month_diff)
#
#            start_date = "2020-3-1"
#            end_date = "2020-9-1"
#            month_diff = ee.Date(start_date).advance(1,"month").millis().subtract(ee.Date(start_date).millis())
#            list_map_2020 = ee.List.sequence(ee.Date(start_date).millis(), ee.Date(end_date).millis(), month_diff)
#            
            list_map = list_map_2019
#            list_map = list_map_2019.cat(list_map_2020)
#            list_map = ee.List.sequence(ee.Date(start_date).millis(), ee.Date(end_date).millis(), month_diff)
            
            sent_nc = ee.ImageCollection.fromImages(list_map.map(weeklyNCRegion(region)))
    #        sent_nc_first = ee.ImageCollection.fromImages(list_map.map(weeklyNCRegion_first(region)))
    #        j = 0
    #        info_job = ["median", "first"]
    #        for sent_nc in [sent_nc_median, sent_nc_first]:
    #            print("Selected: "+info_job[j])
    #            j += 1
            
    #        sent_ls = sent_nc.toList(sent_nc.size())
    #        for i in range(37):
    #            img = ee.Image(sent_ls.get(i))
    #            print(img.bandNames().getInfo())
                
            
            sent_nc_ndvi = sent_nc.map(addNDVI)
            sent_nc_ndvi.getInfo()
            
            img_0 = ee.Image(ee.Number(0))
            img_16000 = ee.Image(ee.Number(10000))
            def funzione(x):
                img = x.min(img_16000)
                img = img.max(img_0)
#                img = img.reproject(crs=crs, scale=scale)
                img = img.clipToBoundsAndScale(region, maxDimension=512)
#                img = img.clip(region)
#                img = img.reduceResolution(reducer=ee.Reducer.mean(), maxPixels=512)
                return img.select(["B2", "B3", "B4", "NDVI"])


            sent_nc_ndvi = sent_nc_ndvi.map(funzione)
            sent_list = sent_nc_ndvi.toList(sent_nc_ndvi.size())
            arrs = []
            for i in range(sent_nc_ndvi.size().getInfo()):
                print("Element: "+str(i))
#                try:
                img = ee.Image(sent_list.get(i)).sampleRectangle(defaultValue=0)
                r = np.array(img.get("B4").getInfo())
                g = np.array(img.get("B3").getInfo())
                b = np.array(img.get("B2").getInfo())
                ndvi = np.array(img.get("NDVI").getInfo())
                arrs.append(np.array([r,g,b,ndvi]))
                
            
            
            img = np.array(arrs)
            img = fill_pad(img)
            img = np.transpose(img, (1,0,2,3))            
            np.save(path+dest_folder+"/"+fname+".npy", img)
            
#            i = np.random.randint(6)
#            new_img = np.transpose(img[:3, i, :, :], (1,2,0))[:,:,:3]
#            plt.imshow(new_img/10000)
#            plt.show()
            print("Upload successfully completed")
            
            
#            idx = ""
#            ts = []
#            for i in range(18):
#                ch = ["B4"+idx, "B3"+idx, "B2"+idx]
#                img_np = img_scale.sampleRectangle(region).get(ch).getInfo()
#                ts.appen(img_np)
#                idx = "_"+str(i)
                
#            #random image selection
#            for i in range(1, 12):
#    #        i = 11
#                print("Month: %i" % i)
#            params = {
#                    "min": 0,
#                    "max": 10000,
#                    "dimensions": 512,
#                    "bands": ["B4","B3","B2"],
#                    "region": region.geometry()
#                    }
#                
#            display_ee_img(img_scale.clip(region), params)
#            export_oneimage(img_scale.clip(region), "data_image_full", fname, scale, crs)
        
        except Exception:
            print("Upload failed")
            continue
        
    return
        

#alabama = folium.Map(location=[32.01,-87.01], zoom_start=10)
#palette = ["red", "yellow", "green"]
#ndvi_params = {
#        "min":0,
#        "max":1,
#        "dimensions":512,
#        "palette": palette,
#        "bands": "NDVI_11",
#        "region":region.first().getInfo()["geometry"]
#        }
#
#ndvi = img.select("NDVI_"+str(i))
#import geehydro
#alabama.addLayer(ndvi, ndvi_params)
#alabama.save("/Users/pasq/Documents/ML/planet-understanding-the-amazon-from-space/alabama_map.html")


#Check if it is already updated to cloud
if __name__ == "__main__":
#    locations = pd.read_csv(path+'/2019_clean_data/2019_locations_final.csv',header=None)
#    county_region = ee.FeatureCollection("TIGER/2016/Counties")
#    np_list = pd.Series(os.listdir(path+"/numpy_data"))
#    sc_in_folder = np_list.str[:-4].str.split("_")
#    sc_in_folder = sc_in_folder.to_list()
#    sc_in_folder = r"/numpy_data" 

    path = "/Users/pasq/Documents/ML/crop-yield-prediction-project"
    loc_folder = r"/2019_clean_data/2019_locations_final.csv"
#    dest_folder = r"/landsat_data"
#    in_folder_opt = "lder" 
#    year = "2014"
    _, dest_folder, in_folder_opt, year = sys.argv
    print("Running the program with the following inputs...")
    print("dest_folder: "+dest_folder)
    print("in_folder_opt: "+in_folder_opt)
    print("year: "+str(year))
    
    
    upload(path, loc_folder, dest_folder, in_folder_opt, year)
