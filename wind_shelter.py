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
def wind_shelter_prep(radius,direction,tolerance,cellsize=90):
    nc = 2*int(radius)+1
    nr=nc
    mask=np.ones((nc,nr))
    for j in range(nc):
        for i in range(nr):
            if i==j and i==((nr+1)/2):
                continue
            xy = [j-(nc+1)/2, (nr+1)/2-i]
            xy = xy / np.sqrt(xy[1]**2+xy[2]**2)
            if ( xy[2]>0):
                a = np.arcsin(xy[1])  
            else:
                a = np.pi - np.arcsin(xy[1])
            if (a < 0):
                a = a + 2*np.pi
            d = abs(direction-a)
            if (d>2*np.pi):
                d = d-2*np.pi
            d = min(d,2*np.pi-d)
            if (d<=tolerance):
                mask[i,j] = False
    
    dist = np.zeros((nc,nr))
    for i in range(1,nr):
        for j in range(1,nc):
            
            xy = ( j-(nc+1)/2, (nr+1)/2-i )
            dist[i,j] = np.sqrt(xy[1]^2+xy[2]^2) * cellsize

    return list( mask, dist )

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

def centervalue(x): 
    i = math.ceil(x.shape[1] / 2)
    return(x[i,i])  

def wind_shelter(x,control):
    
    ctr = centervalue(x)
    x[control[0]] = np.nan
    res = np.nan
    for i in x:
        if not np.isnan(i): 
            x = np.arctan((x-ctr)/control[1])
            res = max(x[~np.isnan(x)])
    
    
    return(res)



        
            
    