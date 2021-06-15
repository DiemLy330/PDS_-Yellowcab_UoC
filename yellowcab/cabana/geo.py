# import geojson
import numpy as np
from shapely.geometry import shape
from shapely.geometry import Point
from descartes import PolygonPatch
import geopandas as gpd
import matplotlib.pyplot as plt
from pyproj import Proj, transform
# from .trips_input import *

#-------------------------------------------------------------------------------
# Module using geo data to return information
    # get_centroid() returns a dictionary of location ID as keys and central points as values
    # get_area(location) illustrates area and its central point, default location = 1
    # get_map() prints NYC's map
#-------------------------------------------------------------------------------

class geo:
    # Get geo data of a specific location
    def __init__(self):
        # By geopandas
        self.f = gpd.read_file(r'C:\Users\kyral\Documents\GitHub\PDS_Yellowcab_UoC\data\input\taxi_zones.geojson')
        self.l = self.f['OBJECTID'].values
        self.id = self.f['OBJECTID'].unique()
        self.b = self.f['borough'].unique()

        # By geojson
        # with open(r'C:\Users\kyral\Documents\GitHub\PDS_Yellowcab_UoC\data\input\taxi_zones.geojson') as f:
        #     self.d = geojson.load(f)
        #
        # self.l = np.array([i['properties']['LocationID'] for i in self.d['features']])
        # self.b = np.unique(np.array([i['properties']['borough'] for i in self.d['features']]))
    def df(self):
        return self.f.crs

    def borough_ID(self):
        return self.b, self.id

    # Find central point of a specific location
    def get_centroid(self):
        # by geopandas
        def getXY(pt):
            return (pt.x, pt.y)
        centroidseries = self.f['geometry'].centroid
        centroidlist = list(map(getXY, centroidseries))
        result = []
        ny = Proj(init='ESRI:102718',preserve_units=True)
        for i in centroidlist:
            result.append(ny(i[0],i[1],inverse =True))
        return dict(zip(self.l,result))


        # By geojson
        # center = []
        # for i in range(len(self.l)):
        #     t = shape(self.d['features'][i]['geometry']).centroid
        #     point = Point(t)
        #     center.append([point.x,point.y])
        # return dict(zip(self.l,center))

    # Map a location ID with its central point
    def map_locationID(self,location = 1):
        # by geopandas
        df = self.f[self.f['OBJECTID']==location]
        def getXY(pt):
            return (pt.x, pt.y)
        centroidseries = self.f['geometry'].centroid
        centroidlist = list(map(getXY, centroidseries))
        centroid = dict(zip(self.l, centroidlist))
        point = centroid[location]
        df.plot()
        plt.plot(point[0], point[1], 'ro')
        plt.show()

        # by geojson
        # try:
        #     index = np.where(self.l == location)
        #     i = index[0].item()
        #     borough = self.d['features'][i]['properties']['borough']
        #     poly = self.d['features'][i]['geometry']
        #     point = self.get_centroid()[location]
        #
        #     BLUE = '#6699cc'
        #     fig = plt.figure()
        #     ax = fig.gca()
        #     ax.plot(point[0],point[1], 'ro')
        #     ax.add_patch(PolygonPatch(poly, fc=BLUE, ec=BLUE, alpha=0.5, zorder=2))
        #     ax.axis('scaled')
        #     plt.title(f'Map of location {location}, borough {borough}')
        #     plt.show()
        # except ValueError:
        #     print(f'No location ID {location} in borough {borough}')

        #Map NYC areas
    def get_map(self):
        # by geopandas
        self.f.plot()
        plt.show()

    # by geojson
    #     poly = [self.d['features'][i]['geometry'] for i in range(len(self.l))]
    #     BLUE = '#6699cc'
    #     fig = plt.figure()
    #     ax = fig.gca()
    #     for p in poly:
    #         ax.add_patch(PolygonPatch(p, fc=BLUE, ec=BLUE, alpha=0.5, zorder=2))
    #     ax.axis('scaled')
    #     plt.show()









