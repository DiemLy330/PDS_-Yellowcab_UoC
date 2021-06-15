from yellowcab.cabana import *
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import SGDRegressor, SGDClassifier
from math import radians

def transform_queens(data=None):
    np.random.seed(0)
    if data==None:
        x = trips_input()
        d = x.get_queens()
    else:
        d = data
    cols = 'tpep_dropoff_datetime tpep_pickup_datetime PULocationID DOLocationID'.split()
    t = d[cols]
    y = trips_info(t)
    df = y.get_position()
    df2 = y.get_time()
    df3 = y.get_duration()
    queens = pd.concat((df, df2), axis=1)
    queens['duration'] = df3.duration
    queens = queens.drop(cols, axis=1)
    full_queens = pd.concat((queens, d), axis=1)
    return full_queens

def pre_process_queens(full_queens):
    initial = full_queens.drop(['tpep_dropoff_datetime', 'tpep_pickup_datetime'], axis=1)
    # monetary = ['fare_amount', 'extra', 'mta_tax',
    #          'tip_amount', 'tolls_amount', 'total_amount']
    # initial[monetary] = initial[monetary] / 100

    #
    # Change day from categories to number
    dd = dict(zip(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], np.arange(0, 7)))
    initial['PUday'] = initial['PUweekday'].map(dd).astype('uint8')
    initial['DOday'] = initial['DOweekday'].map(dd).astype('uint8')
    initialdropped = initial.drop(['PUweekday', 'DOweekday'], axis=1)

    initialdropped = initialdropped[(initialdropped.payment_type <= 6) & (initialdropped.RatecodeID <= 6)]
    initialdropped = initialdropped[(initialdropped.PULocationID <= 263) & (initialdropped.DOLocationID <= 263)]
    initialdropped = initialdropped[(initialdropped.passenger_count > 0) & (initialdropped.trip_distance > 0)]


    # Get dummies for payment_type and RatecodeID
    pay = pd.get_dummies(initialdropped['payment_type'], drop_first=True)
    payname = pay.rename(columns=dict(zip(np.arange(2, 6), ['pay2', 'pay3', 'pay4', 'pay5'])))
    rate = pd.get_dummies(initialdropped['RatecodeID'], drop_first=True)
    ratename = rate.rename(columns=dict(zip(np.arange(2, 7), ['r2', 'r3', 'r4', 'r5', 'r6'])))
    raw = pd.concat((initialdropped, ratename, payname), axis=1).drop('RatecodeID', axis=1)
    #
    # DEALING WITH FEATURES THAT ARE ONLY MEANINGFUL WHEN IT IS >=0
    raw[['improvement_surcharge','congestion_surcharge']]= abs(raw[['improvement_surcharge','congestion_surcharge']])
    raw = raw[raw.duration > 0]

    # #
        # Speed conditions
    sc1 = (raw['trip_distance'] / raw['duration']) <= 0.009  # < 34MPH
    sc2 = (raw['trip_distance'] / raw['duration']) > 0.00097  # >3.5MPH
    raw = raw[sc1 & sc2]

        # FARE AMOUNT CONDITIONS
    fc1 = raw['fare_amount']<=raw['trip_distance']*3+ ((raw['trip_distance']*0.5)/0.2)
    fc2 = raw['fare_amount']>raw['trip_distance']*2.5
    raw = raw[fc1 & fc2]

    #
        # SIN COS TRANSFORM FOR MONTH DAY HOUR
    raw['PUhoursin'] = np.sin(raw['PUhour'] * (2. * np.pi / 24))  # coordinate values on Oy
    raw['PUhourcos'] = np.cos(raw['PUhour'] * (2. * np.pi / 24))  # coordinate values on Ox
    raw['DOhoursin'] = np.sin(raw['DOhour'] * (2. * np.pi / 24))  # coordinate values on Oy
    raw['DOhourcos'] = np.cos(raw['DOhour'] * (2. * np.pi / 24))  # coordinate values on Ox

    raw['PUdaysin'] = np.sin(raw['PUday'] * (2. * np.pi / 7))
    raw['PUdaycos'] = np.cos(raw['PUday'] * (2. * np.pi / 7))
    raw['DOdaysin'] = np.sin(raw['DOday'] * (2. * np.pi / 7))
    raw['DOdaycos'] = np.cos(raw['DOday'] * (2. * np.pi / 7))

    raw['PUmonthsin'] = np.sin(raw['PUmonth'] * (2. * np.pi / 12))
    raw['PUmonthcos'] = np.cos(raw['PUmonth'] * (2. * np.pi / 12))
    raw['DOmonthsin'] = np.sin(raw['DOmonth'] * (2. * np.pi / 12))
    raw['DOmonthcos'] = np.cos(raw['DOmonth'] * (2. * np.pi / 12))

        # TRIM IRRELAVANT COLUMNS

    raw.drop(['PUhour', 'DOhour', 'PUday', 'DOday', 'PUmonth', 'DOmonth'], axis=1, inplace=True)

    ###LON LAT OF SOME SPECIFIC PLACES
    JFKairport = (-73.780968, 40.641766)
    # JFKairport= (-73.78, 40.64)

    # Queens library = 40.757978766519294, -73.82900393688718
    Queenslib = -73.83, 40.76

    # LGDairport= 40.776902684722835, -73.87395517203288
    LGDairport = -73.87, 40.78

    # marinomar= 40.77123716721885, -73.80129679098286
    Marinomar = -73.80, 40.77

    L = [JFKairport, LGDairport, Queenslib, Marinomar]

    # Converting decimal lon lat to radians for each specific places mentioned above
    def detora(List):
        res = []
        for i in List:
            listc = list(map(radians, i))
            res.append(listc)
        return res
    res = detora(L)

    # convert decimal lon lat to radians for PUlon  PUlat , DOlon, DOlat
    raw['PUlonra'] = np.radians(raw['PUlon'])
    raw['PUlatra'] = np.radians(raw['PUlat'])
    raw['DOlonra'] = np.radians(raw['DOlon'])
    raw['DOlatra'] = np.radians(raw['DOlat'])

    # Compute the Harvesine distance between the datapoints and each specific place listed above
    def haversineindf(df):
        for i in np.arange(0, len(res)):
            dlonp = raw['PUlonra'] - res[i][0]
            dlatp = raw['PUlatra'] - res[i][1]
            ap = np.sin(dlatp / 2) ** 2 + np.cos(raw['PUlatra']) * np.cos(res[i][1]) * np.sin(dlonp / 2) ** 2
            cp = 2 * np.arcsin(np.sqrt(ap))
            sp = cp * 3956
            sp = sp.rename(i)
            df = pd.concat((df, sp), axis=1)
        return df

    # Compute the Harvesine distance between the PU DO
    def haversineindfpd(df):
        dlonpd = raw['PUlonra'] - raw['DOlonra']
        dlatpd = raw['PUlatra'] - raw['DOlatra']
        apd = np.sin(dlatpd / 2) ** 2 + np.cos(raw['PUlatra']) * np.cos(raw['DOlatra']) * np.sin(dlonpd / 2) ** 2
        cpd = 2 * np.arcsin(np.sqrt(apd))
        spd = cpd * 3956
        df = pd.concat((df, spd), axis=1)
        return df

    s1 = pd.DataFrame(np.zeros(len(raw.index)), index=raw.index)
    predf = haversineindf(s1)
    predfpd = haversineindfpd(s1)
    df_new = predf.iloc[:, 1::]
    dfpd = predfpd.iloc[:, 1::]
    dfname = df_new.rename(columns=dict(zip(np.arange(0,4),['tojfk','tolgd', 'toql','tomar'])))
    dfpdname = dfpd.rename(columns={0: 'pd'})
    Xfull = pd.concat((raw, dfname, dfpdname), axis=1)
    Xfull.drop(['PUlon', 'PUlat', 'DOlon', 'DOlat', 'PUlonra', 'PUlatra', 'DOlonra', 'DOlatra', 'PULocationID',
                'DOLocationID','pay5'], axis=1, inplace=True)

    # rs = RobustScaler()
    # rs.fit(Xfull)
    # Xscaled = pd.DataFrame(rs.transform(Xfull), index=Xfull.index, columns=Xfull.columns)
    return Xfull
