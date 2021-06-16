from yellowcab.cabana import *
import numpy as np
import matplotlib.pyplot as plt
import time
import geopandas as gpd
import pandas as pd
import seaborn as sns
from scipy.stats import norm
from yellowcab.model import *
import sklearn
from yellowcab.io import input

def main():
        # Test geo() module,
    # data = geo()
    # print(data.get_centroid()) # giving a dict of approx. 260 keys
    # print(data.map_locationID(location=10))
    # print(data.get_map())
    # print(data.df())
#-------------------------------------------------------------
        #Test trip_inputs() module
    # x = trips_input()
    # b = x.borough_list()
    # l = x.get_borough_locationID(borough="Bronx")
    # print(b,l)

    # t = x.get_trips(fraction=0.1,optimize=False)
    # print(t.describe())
    # print(t.info())
    # print(t.head())
#-------------------------------------------------------------
       #Test trips_info() module
    # y = trips_info(t)
    # df = y.get_time()
    # print(df.info())
    # print(df.head())

    # d = y.get_duration()
    # print(d.head())
    # print(d.info())
    #
    # df = y.get_position()
    # print(df.info())
    #
    # a,b,c = y.outlier(df['PUlon'])
    # d,e,f = y.outlier(df['PUlat'])
    # start_location = df[(df['PUlon']>a)&(df['PUlon']<b)&(df['PUlat']>d)&(df['PUlat']<e)]
    # sl = start_location.loc[:,['PUlon','PUlat']]
    # print(sl.describe())
    # print(f'Number of outlier for longitude is {c}')
    # print(f'Number of outlier for latitude is {f}')

    # m = y.get_aggregate('month')
    # h = y.get_aggregate('hour')
    # d = y.get_aggregate('weekday')
    # print(m,h,d)

    # d = y.map_best_month()
    # print(d)
#-----------------------------------------------------------------------------
    # # Create full NYC file
    # x = trips_input()
    # d = x.get_trips(fraction=0.1,optimize=True)
    # cols = 'tpep_dropoff_datetime tpep_pickup_datetime PULocationID DOLocationID'.split( )
    # t = d[cols]
    # y = trips_info(t)
    # df = y.get_position()
    # df2 = y.get_time()
    # df3 = y.get_duration()
    # nyc = pd.concat((df,df2),axis=1)
    # nyc['duration'] = df3.duration
    # nyc = nyc.drop(cols,axis=1)
    # full_nyc = pd.concat((nyc,d),axis=1)
    # full_nyc = full_nyc.drop(['tpep_dropoff_datetime', 'tpep_pickup_datetime'], axis=1)
    # #     #-------------------------------------
    #  #     # Plot duration distribution (2c)
    # cols = 'PUmonth PUhour PUweekday PUweekend duration'.split()
    # dur = full_nyc[cols]
    # gr = dur.groupby(['PUmonth'])
    # des = gr.duration.describe()
    #
    # month = dur[dur.PUmonth==12]
    # sns.distplot(month['duration'],color='r',hist=False)
    # mu = des.iloc[11,1]
    # sigma = des.iloc[11,2]
    # nd = np.linspace(mu-3*sigma,mu+3*sigma,100)
    # plt.plot(nd, norm.pdf(nd,mu,sigma))
    # # print(des)
    # plt.title('Duration distribution of December')
    # plt.show()
    # plt.clf()
    # #
    # fig,ax=plt.subplots(4,3)
    #
    # month = full_nyc[full_nyc.PUmonth==1]
    # sns.distplot(month['duration'],color='r',hist=False,ax=ax[0,0])
    # mu = des.iloc[0,1]
    # sigma = des.iloc[0,2]
    # nd = np.linspace(mu-3*sigma,mu+3*sigma,100)
    # ax[0,0].plot(nd, norm.pdf(nd,mu,sigma))
    # ax[0,0].set_title('January')
    #
    # month2 = full_nyc[full_nyc.PUmonth == 2]
    # sns.distplot(month2['duration'], color='r', hist=False,ax=ax[0,1])
    # mu2 = des.iloc[1, 1]
    # sigma2 = des.iloc[1, 2]
    # nd2 = np.linspace(mu2 - 3 * sigma2, mu2 + 3 * sigma2, 100)
    # ax[0, 1].plot(nd2, norm.pdf(nd2, mu2, sigma2))
    # ax[0, 1].set_title('February')
    #
    # month3 = full_nyc[full_nyc.PUmonth == 3]
    # sns.distplot(month3['duration'], color='r', hist=False, ax=ax[0, 2])
    # mu3 = des.iloc[2, 1]
    # sigma3 = des.iloc[2, 2]
    # nd3 = np.linspace(mu3 - 3 * sigma3, mu3 + 3 * sigma3, 100)
    # ax[0, 2].plot(nd3, norm.pdf(nd3, mu3, sigma3))
    # ax[0, 2].set_title('March')
    #
    # month4 = full_nyc[full_nyc.PUmonth == 4]
    # sns.distplot(month4['duration'], color='r', hist=False, ax=ax[1, 0])
    # mu4 = des.iloc[3, 1]
    # sigma4 = des.iloc[3, 2]
    # nd4 = np.linspace(mu4 - 3 * sigma4, mu4 + 3 * sigma4, 100)
    # ax[1, 0].plot(nd4, norm.pdf(nd4, mu4, sigma4))
    # ax[1, 0].set_title('April')
    #
    # month5 = full_nyc[full_nyc.PUmonth == 5]
    # sns.distplot(month5['duration'], color='r', hist=False, ax=ax[1, 1])
    # mu5 = des.iloc[4, 1]
    # sigma5 = des.iloc[4, 2]
    # nd5 = np.linspace(mu5 - 3 * sigma5, mu5 + 3 * sigma5, 100)
    # ax[1, 1].plot(nd5, norm.pdf(nd5, mu5, sigma5))
    # ax[1, 1].set_title('May')
    #
    # month6 = full_nyc[full_nyc.PUmonth == 6]
    # sns.distplot(month6['duration'], color='r', hist=False, ax=ax[1, 2])
    # mu6 = des.iloc[5, 1]
    # sigma6 = des.iloc[5, 2]
    # nd6 = np.linspace(mu6 - 3 * sigma6, mu6 + 3 * sigma6, 100)
    # ax[1, 2].plot(nd6, norm.pdf(nd6, mu6, sigma6))
    # ax[1, 2].set_title('June')
    #
    # month7 = full_nyc[full_nyc.PUmonth == 7]
    # sns.distplot(month7['duration'], color='r', hist=False, ax=ax[2, 0])
    # mu7 = des.iloc[6, 1]
    # sigma7 = des.iloc[6, 2]
    # nd7 = np.linspace(mu7 - 3 * sigma7, mu7 + 3 * sigma7, 100)
    # ax[2, 0].plot(nd7, norm.pdf(nd7, mu7, sigma7))
    # ax[2, 0].set_title('July')
    #
    # month8 = full_nyc[full_nyc.PUmonth == 8]
    # sns.distplot(month8['duration'], color='r', hist=False, ax=ax[2, 1])
    # mu8 = des.iloc[7, 1]
    # sigma8 = des.iloc[7, 2]
    # nd8 = np.linspace(mu8 - 3 * sigma8, mu8 + 3 * sigma8, 100)
    # ax[2, 1].plot(nd8, norm.pdf(nd8, mu8, sigma8))
    # ax[2, 1].set_title('August')
    #
    # month9 = full_nyc[full_nyc.PUmonth == 9]
    # sns.distplot(month9['duration'], color='r', hist=False, ax=ax[2, 2])
    # mu9 = des.iloc[8, 1]
    # sigma9 = des.iloc[8, 2]
    # nd9 = np.linspace(mu9 - 3 * sigma9, mu9 + 3 * sigma9, 100)
    # ax[2, 2].plot(nd9, norm.pdf(nd9, mu9, sigma9))
    # ax[2, 2].set_title('March')
    #
    # month10 = full_nyc[full_nyc.PUmonth == 10]
    # sns.distplot(month10['duration'], color='r', hist=False, ax=ax[3, 0])
    # mu10 = des.iloc[9, 1]
    # sigma10 = des.iloc[9, 2]
    # nd10 = np.linspace(mu10 - 3 * sigma10, mu10 + 3 * sigma10, 100)
    # ax[3, 0].plot(nd10, norm.pdf(nd10, mu10, sigma10))
    # ax[3, 0].set_title('October')
    #
    # month11 = full_nyc[full_nyc.PUmonth == 11]
    # sns.distplot(month11['duration'], color='r', hist=False, ax=ax[3, 1])
    # mu11 = des.iloc[10, 1]
    # sigma11 = des.iloc[10, 2]
    # nd11 = np.linspace(mu11 - 3 * sigma11, mu11 + 3 * sigma11, 100)
    # ax[3, 1].plot(nd11, norm.pdf(nd11, mu11, sigma11))
    # ax[3, 1].set_title('November')
    #
    # month12 = full_nyc[full_nyc.PUmonth == 12]
    # sns.distplot(month12['duration'], color='r', hist=False, ax=ax[3, 2])
    # mu12 = des.iloc[11, 1]
    # sigma12 = des.iloc[11, 2]
    # nd12 = np.linspace(mu12 - 3 * sigma12, mu12 + 3 * sigma12, 100)
    # ax[3, 2].plot(nd12, norm.pdf(nd12, mu12, sigma12))
    # ax[3, 2].set_title('December')
    # fig.tight_layout()
    # plt.show()
    # plt.clf()

    # 2a
    # print(dis.map_best_month())

    # 2b
    # boro_dict = dis.boro()
    # full_nyc['borough'] = full_nyc.PULocationID.map(boro_dict)
    # day_order = 'Mon Tue Wed Thu Fri Sat Sun'.split( )
    # heat = full_nyc.pivot_table(index='borough',columns='PUhour',values='trip_distance')
    # sns.heatmap(heat,cmap='YlGnBu')
    # plt.title('Trip distance by start borough and hour')
    # plt.show()
    # plt.clf()

    # 1d
    # cols = 'PUmonth PUhour PUweekday PUweekend duration'.split()
    # dur = full_nyc[cols]
    # m = dur.groupby(['PUweekday'])
    # mo = m['duration'].describe()
    # print(mo)
    # plt.plot(mo)
    # plt.locator_params(nbins=12)
    # plt.xticks(np.arange(1,13,1))
    # day_order = 'Mon Tue Wed Thu Fri Sat Sun'.split()
    # mo = m['duration'].mean().reindex(day_order).plot(kind='line')
    # plt.title('Number of trips by weekday and weekend for NYC')
    # plt.show()
    # plt.clf()
            # --------------------------------------------------------------
        # # Test Queens
    # q = x.get_trips(fraction=1,optimize=True)
    # qb = q.get_borough_locationID(borough='Queens')
    # full_queens = full_nyc[full_nyc.PULocationID.isin(qb) == True]
    # print(full_queens.info())
    # print(full_queens.head())

    # d = pd.read_csv(r"C:/Users/kyral/Documents/GitHub/PDS_Yellowcab_UoC/data/output/queens.csv",sep=',',index_col=0)
    # d = d[(d.payment_type <= 6) & (d.RatecodeID <= 6)]
    # d = d[(d.PULocationID <= 263) & (d.DOLocationID <= 263)]
    # d = d[(d.passenger_count > 0) & (d.trip_distance > 0)]
    # #     # FARE AMOUNT CONDITIONS
    # d.fare_amount = d.fare_amount/100
    # fc1 = d['fare_amount'] <= d['trip_distance'] * 3 + ((d['trip_distance'] * 0.5) / 0.2)
    # fc2 = d['fare_amount'] > d['trip_distance'] * 2.5
    # d = d[fc1 & fc2]
    # d.duration = d.duration*60
    # sc1 = (d['trip_distance'] / d['duration']) <= 0.009  # 17mil JFK to timesquare 45 min # < 34MPH
    # sc2 = (d['trip_distance'] /d['duration']) > 0.00097  # >3.5MPH
    # d = d[sc1 & sc2]
    # cols = 'PUmonth PUhour PUweekday PUweekend duration'.split( )
    # dur = d[cols]
    # #
    # gr = dur.groupby(['PUmonth'])
    # des = gr.duration.describe()
    # # month = dur[dur.PUmonth==12]
    # # sns.distplot(month['duration'],color='r',hist=False)
    # # mu = des.iloc[11,1]
    # # sigma = des.iloc[11,2]
    # # nd = np.linspace(mu-3*sigma,mu+3*sigma,100)
    # # plt.plot(nd, norm.pdf(nd,mu,sigma))
    # # plt.title('Duration distribution of December for Queens')
    #
    # fig,ax=plt.subplots(4,3,sharey=True)
    #
    # month = dur[dur.PUmonth==1]
    # sns.distplot(month['duration'],color='r',hist=False,ax=ax[0,0])
    # mu = des.iloc[0,1]
    # sigma = des.iloc[0,2]
    # nd = np.linspace(mu-3*sigma,mu+3*sigma,100)
    # ax[0,0].plot(nd, norm.pdf(nd,mu,sigma))
    # ax[0,0].set_title('January')
    #
    # month2 = dur[dur.PUmonth == 2]
    # sns.distplot(month2['duration'], color='r', hist=False,ax=ax[0,1])
    # mu2 = des.iloc[1, 1]
    # sigma2 = des.iloc[1, 2]
    # nd2 = np.linspace(mu2 - 3 * sigma2, mu2 + 3 * sigma2, 100)
    # ax[0, 1].plot(nd2, norm.pdf(nd2, mu2, sigma2))
    # ax[0, 1].set_title('February')
    #     #
    # month3 = dur[dur.PUmonth == 3]
    # sns.distplot(month3['duration'], color='r', hist=False, ax=ax[0, 2])
    # mu3 = des.iloc[2, 1]
    # sigma3 = des.iloc[2, 2]
    # nd3 = np.linspace(mu3 - 3 * sigma3, mu3 + 3 * sigma3, 100)
    # ax[0, 2].plot(nd3, norm.pdf(nd3, mu3, sigma3))
    # ax[0, 2].set_title('March')
    #     #
    # month4 = dur[dur.PUmonth == 4]
    # sns.distplot(month4['duration'], color='r', hist=False, ax=ax[1, 0])
    # mu4 = des.iloc[3, 1]
    # sigma4 = des.iloc[3, 2]
    # nd4 = np.linspace(mu4 - 3 * sigma4, mu4 + 3 * sigma4, 100)
    # ax[1, 0].plot(nd4, norm.pdf(nd4, mu4, sigma4))
    # ax[1, 0].set_title('April')
    #     #
    # month5 = dur[dur.PUmonth == 5]
    # sns.distplot(month5['duration'], color='r', hist=False, ax=ax[1, 1])
    # mu5 = des.iloc[4, 1]
    # sigma5 = des.iloc[4, 2]
    # nd5 = np.linspace(mu5 - 3 * sigma5, mu5 + 3 * sigma5, 100)
    # ax[1, 1].plot(nd5, norm.pdf(nd5, mu5, sigma5))
    # ax[1, 1].set_title('May')
    #     #
    # month6 = dur[dur.PUmonth == 6]
    # sns.distplot(month6['duration'], color='r', hist=False, ax=ax[1, 2])
    # mu6 = des.iloc[5, 1]
    # sigma6 = des.iloc[5, 2]
    # nd6 = np.linspace(mu6 - 3 * sigma6, mu6 + 3 * sigma6, 100)
    # ax[1, 2].plot(nd6, norm.pdf(nd6, mu6, sigma6))
    # ax[1, 2].set_title('June')
    #     #
    # month7 = dur[dur.PUmonth == 7]
    # sns.distplot(month7['duration'], color='r', hist=False, ax=ax[2, 0])
    # mu7 = des.iloc[6, 1]
    # sigma7 = des.iloc[6, 2]
    # nd7 = np.linspace(mu7 - 3 * sigma7, mu7 + 3 * sigma7, 100)
    # ax[2, 0].plot(nd7, norm.pdf(nd7, mu7, sigma7))
    # ax[2, 0].set_title('July')
    #     #
    # month8 = dur[dur.PUmonth == 8]
    # sns.distplot(month8['duration'], color='r', hist=False, ax=ax[2, 1])
    # mu8 = des.iloc[7, 1]
    # sigma8 = des.iloc[7, 2]
    # nd8 = np.linspace(mu8 - 3 * sigma8, mu8 + 3 * sigma8, 100)
    # ax[2, 1].plot(nd8, norm.pdf(nd8, mu8, sigma8))
    # ax[2, 1].set_title('August')
    #     #
    # month9 = dur[dur.PUmonth == 9]
    # sns.distplot(month9['duration'], color='r', hist=False, ax=ax[2, 2])
    # mu9 = des.iloc[8, 1]
    # sigma9 = des.iloc[8, 2]
    # nd9 = np.linspace(mu9 - 3 * sigma9, mu9 + 3 * sigma9, 100)
    # ax[2, 2].plot(nd9, norm.pdf(nd9, mu9, sigma9))
    # ax[2, 2].set_title('March')
    #     #
    # month10 = dur[dur.PUmonth == 10]
    # sns.distplot(month10['duration'], color='r', hist=False, ax=ax[3, 0])
    # mu10 = des.iloc[9, 1]
    # sigma10 = des.iloc[9, 2]
    # nd10 = np.linspace(mu10 - 3 * sigma10, mu10 + 3 * sigma10, 100)
    # ax[3, 0].plot(nd10, norm.pdf(nd10, mu10, sigma10))
    # ax[3, 0].set_title('October')
    #     #
    # month11 = dur[dur.PUmonth == 11]
    # sns.distplot(month11['duration'], color='r', hist=False, ax=ax[3, 1])
    # mu11 = des.iloc[10, 1]
    # sigma11 = des.iloc[10, 2]
    # nd11 = np.linspace(mu11 - 3 * sigma11, mu11 + 3 * sigma11, 100)
    # ax[3, 1].plot(nd11, norm.pdf(nd11, mu11, sigma11))
    # ax[3, 1].set_title('November')
    #     #
    # month12 = dur[dur.PUmonth == 12]
    # sns.distplot(month12['duration'], color='r', hist=False, ax=ax[3, 2])
    # mu12 = des.iloc[11, 1]
    # sigma12 = des.iloc[11, 2]
    # nd12 = np.linspace(mu12 - 3 * sigma12, mu12 + 3 * sigma12, 100)
    # ax[3, 2].plot(nd12, norm.pdf(nd12, mu12, sigma12))
    # ax[3, 2].set_title('December')
    # for a in ax.flat:
    #         a.label_outer()
    # fig.tight_layout()
    # plt.show()
    # plt.clf()
    # m = dur.groupby(['PUweekend'])
    # mo = m['duration'].count().plot(kind='bar')
    # mo = m['duration'].count()
    # plt.plot(mo.index.tolist(),mo, tick_label = ['Weekday','Weekend'] )
    # print(mo)
    # m =dur.groupby(['PUweekday'],sort=False)
    # day_order = 'Mon Tue Wed Thu Fri Sat Sun'.split()
    # mo = m['duration'].count().reindex(day_order).plot(kind='bar')
    # plt.title('Average duration in weekdays and weekends for Queens')
    # plt.show()
    # plt.clf()
#-----------------------------------------------------------
    # Test model
    # y = model_nyc()
    # df = y.transform()
    # df = y.predict_distance_nyc()
    # df = y.predict_fare_nyc()
    # df = y.predict_payment_type_nyc()
    # df = y.predict()
    x = model_queens()
    # df = x.predict_distance_queens()
    # df = x.predict_fare_queens()
    # df = x.predict_payment_type_queens()
    df = x.predict()
    print(df)
    # print(df.info())
    # print(df.head())

    # a = pd.read_parquet(r'C:\Users\kyral\Documents\GitHub\PDS_Yellowcab_UoC\data\input\04.parquet',engine='pyarrow')
    # x = input.read_model(name='predict_payment_type_queens.pkl')
    # print(x)

if __name__ == '__main__':
    main()
