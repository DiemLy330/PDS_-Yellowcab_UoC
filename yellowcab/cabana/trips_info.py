import pandas as pd
from .trips_input import *
from .geo import *
import datetime
import os
import geopandas as gpd
from yellowcab.io import utils

# Modules returns detailed information of trips for a give trips dataframe
    # get_duration() returns trip duration
    # get_time(column) expands the given column into additional columns of Month, Weekday, Hour, Weekend (binary). Default column is pickup time
    # get_aggregate(column) calculate aggregate statistics description for trips duration, grouped by given column. Default column is month
    # get_position(column) adds additional columns of Longitude and Latitude. Default column is pick up location
    # outlier(series) returns lower boundary, upper boundary and number of outliers, respectively

# and visualize important findings such as:
    # map_best_month() visualize the number of started trips per region in a map for the month with the most trips

#----------------------------------------------------------------------------------------------------
class trips_info:
    def __init__(self, dataframe):
        self.df = dataframe
        self.g = gpd.read_file(os.path.join(utils.get_data_path(),'input','taxi_zones.geojson'))

        # Add new time columns of weekday, month and hour
    def get_time(self, column='tpep_pickup_datetime'):
        df = self.df
        df['PUmonth'] = df['tpep_pickup_datetime'].dt.month.astype('uint8')
        df['PUhour'] = df['tpep_pickup_datetime'].dt.hour.astype('uint8')
        df['DOmonth'] = df['tpep_dropoff_datetime'].dt.month.astype('uint8')
        df['DOhour'] = df['tpep_dropoff_datetime'].dt.hour.astype('uint8')

        # Return day of week
        df['PUweekday'] = df['tpep_pickup_datetime'].dt.weekday
        df["PUweekday"] = df["PUweekday"].replace(list(range(0, 7)), "Mon Tue Wed Thu Fri Sat Sun".split()).astype('category')
        df['DOweekday'] = df['tpep_dropoff_datetime'].dt.weekday
        df["DOweekday"] = df["DOweekday"].replace(list(range(0, 7)), "Mon Tue Wed Thu Fri Sat Sun".split()).astype(
            'category')

        # Return if a date is a weekend
        df['PUweekend'] = df['tpep_pickup_datetime'].dt.weekday
        df['PUweekend'] = df['PUweekend'].replace(list(range(0, 6)), 0).replace(6, 1).astype('uint8')
        df['DOweekend'] = df['tpep_dropoff_datetime'].dt.weekday
        df['DOweekend'] = df['DOweekend'].replace(list(range(0, 6)), 0).replace(6, 1).astype('uint8')

        columns = 'PUmonth PUhour DOmonth DOhour PUweekday DOweekday PUweekend DOweekend'.split( )
        return df[columns]

        # Calculate trip duration
    def get_duration(self):
        df = self.df
        df['duration'] = abs(df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds()
        df.duration = abs(df.duration).astype('uint16')
        return df

    # Return aggregate values by time
    def get_aggregate(self, column='month'):
        df = self.get_time(column='tpep_pickup_datetime')
        df['duration'] = abs(self.get_duration()['duration'])
        if column == 'day':
            d = df.groupby(['PUweekday'])
        if column == 'hour':
            d = df.groupby(['PUhour'])
        else:
            d = df.groupby(['PUmonth'])
        return d['duration'].describe()

        # Return longitude and latitude of a location ID
    def get_position(self):
        df = self.df
        f = geo().get_centroid()

        # Apply dictionary to column
        df['PUdummy'] = df['PULocationID'].map(f)
        df.fillna(method='ffill', inplace=True)
        df[['PUlon', 'PUlat']] = pd.DataFrame(df['PUdummy'].tolist(), index=df.index)
        df = df.drop(['PUdummy'], axis=1)
        df['PUlon'] = df['PUlon'].astype('float64')
        df['PUlat'] = df['PUlat'].astype('float64')

        df['DOdummy'] = df['DOLocationID'].map(f)
        df.fillna(method='ffill', inplace=True)
        df[['DOlon', 'DOlat']] = pd.DataFrame(df['DOdummy'].tolist(), index=df.index)
        df = df.drop(['DOdummy'], axis=1)
        df['DOlon'] = df['DOlon'].astype('float64')
        df['DOlat'] = df['DOlat'].astype('float64')
        return df

    def outlier(self, series):
        descriptive = series.describe().transpose()
        iqr = descriptive.loc['75%'] - descriptive.loc['25%']
        low_boundary = descriptive.loc['25%'] - 1.5 * iqr
        up_boundary = descriptive.loc['75%'] + 1.5 * iqr
        outliers = sum(0 if x >= low_boundary and x <= up_boundary else 1 for x in series)
        return low_boundary, up_boundary, outliers

    def boro(self):
        file = pd.read_csv(os.path.join(utils.get_data_path(),'input','taxi_zones.csv'), sep=',')
        boro = dict(zip(file.LocationID.values, file.Borough.values))
        return boro

    def map_best_month(self):
        df = self.df

        # Return borough from PULocationID of the chosen month
        file = pd.read_csv(os.path.join(utils.get_data_path(),'input','taxi_zones.csv'), sep=',')
        boro = dict(zip(file.LocationID.values, file.Borough.values))

        # get data of month with the most trip
        m = self.get_time().groupby(['PUmonth']).count()
        month = m[m['PUhour'] == m['PUhour'].max()].index[0].item()
        mon = df[df.PUmonth == month]
        mon['borough'] = mon.loc[:,'PULocationID'].map(boro)
        mon = mon[mon.borough != "Unknown"]

        gr = mon.groupby(['borough'])
        b = list(gr.groups.keys())
        gr = gr.count()
        counts = gr.iloc[:,0].values
        label = list(map(lambda x,y: f'{x}\n{y}',b,counts))

        # Map borough
        boros = self.g.dissolve(by='borough',aggfunc='count')
        boros['center'] = boros['geometry'].centroid
        boros_points=boros.copy()
        boros_points.set_geometry('center',inplace=True)
        boros.plot(cmap='Set2')

        t = []
        for x,y,l in zip(boros_points.geometry.x, boros_points.geometry.y, label):
            t.append(plt.text(x,y,l,fontsize=8))
        month_name ='January February March April May June July August September October November December'.split( )
        plt.title(f'{month_name[month-1]} has the most trips')
        plt.show()
        plt.clf()

    def structure(self):
        df = self.df
        df['PUhour_sin'] = np.sin(df.PUhour*(2.*np.pi/24))
        df['PUhour_cos'] = np.cos(df.PUhour * (2. * np.pi / 24))

        df['PUweekday_sin'] = np.sin(df.PUweekday*(2.*np.pi/7))
        df['PUweekday_cos'] = np.cos(df.PUweekday * (2. * np.pi / 7))

        df['PUmonth_sin'] = np.sin(df.PUmonth * (2. * np.pi / 12))
        df['PUmonth_cos'] = np.cos(df.PUmonth * (2. * np.pi / 12))