import pandas as pd
from .geo import *
from sys import getsizeof
from .trips_info import *
import os


# Module gets data trips and  additional information include:
    # get_borough_locationID(borough) returns a list of all location ID in that borough, default "borough = 'Queens'"
    # get_trips() returns a given fraction data of random trips data, default: fraction=0.05, optimize = True
    # get_queens() returns a dataframe of Queens borough only
#-------------------------------------------------------------------------------------------

class trips_input:
    def __init__(self):
        pass

    def get_data_path(self):
        if os.path.isdir(os.path.join(os.getcwd(), 'data')):
            return os.path.join(os.getcwd(), 'data')
        elif os.path.isdir(os.path.join(os.getcwd(), "..", 'data')):
            return os.path.join(os.getcwd(), "..", 'data')
        else:
            raise FileNotFoundError

    # Get list of borough
    def borough_list(self):
        file = pd.read_csv(os.path.join(self.get_data_path(),'input','taxi_zones.csv'), sep=',')
        return file.Borough.unique()

    # Location ID of a borough
    def get_borough_locationID(self,borough='Queens'):
        file = pd.read_csv(os.path.join(self.get_data_path(),'input','taxi_zones.csv'), sep=',')
        return file[file.Borough == borough]['LocationID'].unique()

    # Return a given fraction data of full trips data
    def get_trips(self, fraction=0.5, optimize=True):
        f = '01 02 03 04 05 06 07 08 09 10 11 12'.split( )
        result = []
        for i in f:
            raw = pd.read_parquet(os.path.join(self.get_data_path(),'input', f"{i}.parquet"), engine='pyarrow')
            df = raw.sample(frac=fraction,random_state=0)
            result.append(df)
        d = pd.concat(result)

        if optimize == True:
        # Optimize dtype to reduce file size
            d.passenger_count = d.passenger_count.astype('uint8')
            d.RatecodeID = d.RatecodeID.astype('uint8')
            d.payment_type = d.payment_type.astype('uint8')
            d.PULocationID = d.PULocationID.astype('uint16'  )
            d.DOLocationID = d.DOLocationID.astype('uint16')
            monetary = 'fare_amount tip_amount total_amount tolls_amount extra mta_tax'.split( )
            d[monetary]= abs(d[monetary].apply(lambda x: x.astype('float32')))
            d.improvement_surcharge = d.improvement_surcharge.astype('float32')
            d.congestion_surcharge = d.congestion_surcharge.astype('float32')
            d.trip_distance = d.trip_distance.astype('float32')
        # Filter unrealistic data
            d = d[(d.payment_type <=6)&(d.RatecodeID<=6)]
            d = d[(d.PULocationID<=263)&(d.DOLocationID<=263)]
            d = d[(d.passenger_count>0)&(d.trip_distance>0)]
            # FARE AMOUNT CONDITIONS
            # fc1 = d['fare_amount'] <= d['trip_distance'] * 3 + ((d['trip_distance'] * 0.5) / 0.2)
            # fc2 = d['fare_amount'] > d['trip_distance'] * 2.5
            # d= d[fc1 & fc2]
        elif optimize ==False:
            pass
        return d

    def get_queens(self):
        qb = self.get_borough_locationID(borough='Queens')
        f = '01 02 03 04 05 06 07 08 09 10 11 12'.split()
        result = []
        for i in f:
            raw = pd.read_parquet(os.path.join(self.get_data_path(), 'input', f"{i}.parquet"),engine='pyarrow')
            df = raw[raw.PULocationID.isin(qb)==True]
            result.append(df)
        full_queens = pd.concat(result)
        return full_queens