'''wind.shelter.prep = function(radius,direction,tolerance,cellsize=90) {
    nc = nr = 2*ceiling(radius)+1
    mask = matrix(TRUE,ncol=nc,nrow=nr)
    for (j in 1:nc) {
        for (i in 1:nr) {
            if ((i==j) & (i==((nr+1)/2))) next
            xy = c( j-(nc+1)/2, (nr+1)/2-i )
            xy = xy / sqrt(xy[1]^2+xy[2]^2)
            if ( xy[2]>0)  a = asin(xy[1])  else a = pi - asin(xy[1])
            if (a < 0) a = a + 2*pi
            d = abs(direction-a)
            if (d>2*pi) d = d-2*pi
            d = min(d,2*pi-d)
            if (d<=tolerance) mask[i,j] = FALSE
        }
    }
    dist = matrix(NA,ncol=nc,nrow=nr)
    for (i in 1:nr) for (j in 1:nc) {
        xy = c( j-(nc+1)/2, (nr+1)/2-i )
        dist[i,j] = sqrt(xy[1]^2+xy[2]^2) * cellsize
    }
    list( mask = mask, dist = dist )
}'''

import numpy as np
import math
from elevation import getElevationMatrix, rasterToImage, getRasterRGB
from local_config import MAPBOX_TOKEN
from PIL import Image, ImageFile
import io
import mercantile
import basic_weather_calls

def wind_shelter_prep(radius,direction,tolerance,cellsize=90):
    nc = 2*int(radius)+1
    nr=nc
    mask=np.zeros((nc,nr),dtype=np.int8)
    for j in range(nc):
        for i in range(nr):
            if i==j and i==((nr+1)/2):
                continue
            xy = [j-(nc+1)/2, (nr+1)/2-i]
            xy = xy / np.sqrt(xy[0]**2+xy[1]**2)
            if ( xy[1]>0):
                a = np.arcsin(xy[0])  
            else:
                a = np.pi - np.arcsin(xy[0])
            if (a < 0):
                a = a + 2*np.pi
            d = abs(direction-a)
            if (d>2*np.pi):
                d = d-2*np.pi
            d = min(d,2*np.pi-d)
            if (d<=tolerance):
                mask[i,j] = 1
    
    

    return mask

'''wind.shelter = function(x,prob=NULL,control) {
    if (missing(x)) return("windshelter")
    if (missing(control)) stop("need 'control' argument - call 'wind.shelter.prep' first")
    ctr = centervalue(x)
    x[control$mask] = NA
    res = NA
    if (!all(is.na(x))) {
        x = atan((x-ctr)/control$dist)
        if (is.null(prob)) {
            res = max(x,na.rm=TRUE)
        } else res = stats::quantile(x,probs=prob,na.rm=TRUE)
    }
    return(res)
}

'''



#mask has a problem - fix so it's using a pi sized slice in the upwind direction
def centervalue(x): 
    i = math.ceil(x.shape[1] / 2)
    return(x[i,i],i)  

def wind_shelter(x,mask,radius,cellsize,coord_x=None,coord_y=None):
    
    ctr,coord_x = centervalue(x)
    ctr_c,coord_c = centervalue(mask)
    for i in range(int(radius)):
        for j in range(int(radius)):
            if mask[coord_c+i,coord_c+j]==1:
                x[coord_x+i,coord_x+j] = np.nan
            if mask[coord_c-i,coord_c-j]==1:
                x[coord_x-i,coord_x-j] = np.nan
            
    x = x[coord_x-radius:coord_x+1+radius,coord_x-radius:coord_x+1+radius]
    
            
    array_sum = np.sum(x)
    array_has_nan = np.isnan(array_sum)
    
    res = np.nan
    x_list =[]
    for i in range(1,radius):
        for j in range(1,radius):
            x_plus = (coord_c+i,coord_c+j)
            x_minus = (coord_c-i,coord_c-j)
            distance_plus = np.sqrt((coord_c-x_plus[0])**2+(coord_c-x_plus[1])**2) * cellsize
            distance_minus = np.sqrt((coord_c-x_minus[0])**2+(coord_c-x_minus[1])**2) * cellsize
            
            if not np.isnan(x[x_plus]): 
                print('x_plus')
                x_list.append(np.arctan((x[x_plus]-ctr)/distance_plus))
            if not np.isnan(x[x_minus]): 
                x_list.append(np.arctan((x[x_minus]-ctr)/distance_minus))
    res = max(x_list)
    
    
    return(res)


lng=-95.9326171875
lat=41.26129149391987
tile_coords = mercantile.tile(lng, lat, zoom=14)

elevation_mat = getElevationMatrix(MAPBOX_TOKEN, tile_coords.z, tile_coords.x, tile_coords.y)        

direction = np.pi/180*basic_weather_calls.wind_direction(lat,lng)


control = wind_shelter_prep(20,direction,0.52,20)

print(control)
shelter = wind_shelter(elevation_mat,control,20,20)
print(shelter)