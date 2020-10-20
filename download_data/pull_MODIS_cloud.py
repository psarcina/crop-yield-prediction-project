#import os
#
#os.chdir("/Users/pasq/Documents/ML/crop-yield-prediction-project/clean_data")

import ee
import time
import pandas as pd
import requests
from IPython.display import Image

ee.Initialize()

def export_oneimage(img,folder,name,scale,crs):
  full_file_name = folder + '_' + name
  task = ee.batch.Export.image.toCloudStorage(img, \
          bucket='planet-kaggle',\
          fileNamePrefix=full_file_name,\
          scale=scale,\
          crs=crs,\
          maxPixels=1e12)
  task.start()
  while (task.status()['state'] == 'RUNNING') | (task.status()['state'] == 'READY'):
    print('Running...')
    # Perhaps task.cancel() at some point.
    time.sleep(10)
  print('Done.', task.status())

locations = pd.read_csv('locations_final.csv',header=None)

# Transforms an Image Collection with 1 band per Image into a single Image with items as bands
# Author: Jamie Vleeshouwer

def appendBand(current, previous):
    # Rename the band
    previous=ee.Image(previous)
    current = current.select([1,2,3])
    # Append it to the result (Note: only return current item on first element/iteration)
    accum = ee.Algorithms.If(ee.Algorithms.IsEqual(previous,None), current, previous.addBands(ee.Image(current)))
    # Return the accumulation
    return accum

county_region = ee.FeatureCollection("TIGER/2016/Counties")

loc1, loc2, lon, lat = locations.values[1]
s = "0"+str(int(loc1))
c = "00"+str(int(loc2))
s = s[-2:]
c = c[-3:]
fname = '{}_{}'.format(s,c)

# offset = 0.11
scale  = 10
crs='EPSG:4326'

# filter for a county
region = county_region.filterMetadata('STATEFP', 'equals', s)
region = ee.FeatureCollection(region).filterMetadata('COUNTYFP', 'equals', c)

start_date = "2019-7-31"
end_date = "2020-8-1"
week_diff = ee.Date(start_date).advance(1,"week").millis().subtract(ee.Date(start_date).millis())
list_map = ee.List.sequence(ee.Date(start_date).millis(), ee.Date(end_date).millis(), week_diff)

#geometry = region.first().getInfo()["geometry"]
#mask = ee.Image.constant(1).clip(geometry).mask()
#img_mask = img.updateMask(mask)

def weeklyNCRegion(new_region):
    filter_region = new_region
    def weeklyNoCloud(date_millis):
        date = ee.Date(date_millis)
        sentinel = ee.ImageCollection("COPERNICUS/S2_SR")\
                .filterBounds(filter_region)\
                .filterDate(date, date.advance(1, "week"))\
                .median()
        
        return sentinel
    
    return weeklyNoCloud
        

sent_nc = ee.ImageCollection.fromImages(list_map.map(weeklyNCRegion(region)))
#sent_list = sent_nc.toList(sent_nc.size())
#img = ee.Image(sent_list.get(0)).select(["B4","B3", "B2"])
#
#params = {
#        "min": 0,
#        "max": 5000,
#        "dimensions": 512,
#        "bands": ["B4","B3","B2"],
#        "region": region.first().getInfo()["geometry"]
#        }
#
#display_ee_img(img, params)
#
#x = img.getInfo()

#clouds = []
#for i in range(len(img_no_cloud)):
#    clouds.append(img_no_cloud[i]["properties"]["CLOUD_COVERAGE_ASSESSMENT"])
#
#
#
#x = sentinel.getInfo()
#clouds = []
#for i in range(len(x)):
#    k = x[i]["properties"]['CLOUD_COVERAGE_ASSESSMENT']
#    clouds.append(k)
#    print(k)


img = sent_nc.iterate(appendBand)
img = ee.Image(img)
img_0 = ee.Image(ee.Number(-100))
img_16000 = ee.Image(ee.Number(16000))
img = img.min(img_16000)
img = img.max(img_0)
img_scale = img.clipToBoundsAndScale(region, maxDimension=1024)
export_oneimage(img_scale, "data_image_full", fname, scale, crs)


#task_config = {
#        "description": "imageToDrive",
#        "scale": 10,
#        "region": region.first().getInfo()["geometry"]
#        }
#
#task = ee.batch.Export.image.toCloudStorage(sent_nc.select([1,2,3]), \
#      bucket='planet-kaggle',\
#      fileNamePrefix="exportExample",\
#      scale=scale,\
#      crs=crs)
#
#task.start()
#while (task.status()['state'] == 'RUNNING') | (task.status()['state'] == 'READY'):
#    print('Running...')
#    # Perhaps task.cancel() at some point.
#    time.sleep(10)
#
#print('Done.', task.status())




imgcoll = ee.ImageCollection('MODIS/MOD09A1') \
    .filterBounds(ee.Geometry.Rectangle(-106.5, 50,-64, 23))\
    .filterDate('2002-12-31','2016-8-4')

def display_ee_img(image, params):
    url = image.getThumbUrl(params)
    img = Image(requests.get(url).content)
    return img

##Visualize one image
#params = {
#        "min":0,
#        "max":1000,
#        "dimensions":512,
#        "bands":["sur_refl_b04", "sur_refl_b03", "sur_refl_b02"],
#        "region": ee.Geometry.Rectangle([-106.5, 50, -64, 23])
#        }
#url = imgcoll.first().getThumbUrl(params)
#import requests
#from IPython.display import Image
#Image(requests.get(url).content)

img=imgcoll.iterate(appendBand)
img=ee.Image(img)

img_0=ee.Image(ee.Number(-100))
img_16000=ee.Image(ee.Number(16000))

img=img.min(img_16000)
img=img.max(img_0)

for loc1, loc2, lon, lat in locations.values:
    s = "0"+str(int(loc1))
    c = "00"+str(int(loc2))
    s = s[-2:]
    c = c[-3:]
    fname = '{}_{}'.format(s,c)

    # offset = 0.11
    scale  = 500
    crs='EPSG:4326'

    # filter for a county
    region = county_region.filterMetadata('STATEFP', 'equals', s)
    region = ee.FeatureCollection(region).filterMetadata('COUNTYFP', 'equals', c)
#    region = ee.Feature(region.first())

    while True:
        try:
            export_oneimage(img.clip(region), 'data_image_full', fname, scale, crs)
        except Exception as e:
            print(e)
            print('retry')
            time.sleep(10)
            continue
        break
    

Export.image.toClo
##Analysis
#len(img.getInfo()["bands"])
#print(imgcoll.size().getInfo())
#params["bands"] = ["sur_refl_b07_622"]
#display_ee_img(img, params)
#
#
#counties = ee.FeatureCollection("TIGER/2016/Counties")
#info_counties = counties.getInfo()
#info_counties.keys()
#info_counties["features"][0].keys()
#len_counties = len(info_counties["features"])
#statefp = []
#for i in range(len_counties):
#    statefp.append(info_counties["features"][i]["properties"]["STATEFP"])
#
#county_statefp = []
#for i in range(len_counties):
#    county_i = info_counties["features"][i]["properties"]
#    county_statefp.append(county_i["STATEFP"]+"/"+county_i["COUNTYFP"])
#
#fp_sizes = []
#for cs in county_statefp:
#    s, c = cs.split("/")
#    region = counties.filterMetadata("STATEFP", "equals", s)
#    region = ee.FeatureCollection(region).filterMetadata("COUNTYFP","equals",c)
##    region = ee.Feature(region)
#    size = counties.filterMetadata("STATEFP", "equals", fp).size().getInfo()
#    
#    print(fp+": "+str(size))
#    fp_sizes.append(size)
#    
