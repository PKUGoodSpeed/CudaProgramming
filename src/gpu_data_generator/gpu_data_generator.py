import market_data2 as md2
from light.util import selectDates
import gsampler as gs
import numpy as np
import pandas as pd
from IPython.display import display
import os


def main():
    symbol = 'KRW_G0.KRX'
    colo = 'KRX'
    dates = ['20171120']
    features = []
    for i in range(5):
        features.append(gs.FeatureAskPrice(symbol = symbol, level = i))
        features.append(gs.FeatureAskQty(symbol = symbol, level = i))
    for i in range(5):
        features.append(gs.FeatureBidPrice(symbol = symbol, level = i))
        features.append(gs.FeatureBidQty(symbol = symbol, level = i))
    features.append(gs.FeatureTradePrice(symbol = symbol))
    features.append(gs.FeatureTradeSide(symbol = symbol))
    features.append(gs.FeatureTradeSize(symbol = symbol))
    features.append(gs.FeatureTickerIdCast())
    sampler = gs.SamplerRunner(features = features, dates = dates, colo = colo, cb_version = 'v2')
    sampler.run()
    df = sampler.load_data_df()
    df = df.between_time(start_time='10:00:00', end_time='13:00:00')
    df = df.fillna(0.)
    df.columns = ['ap1', 'az1', 'ap2', 'az2', 'ap3' ,'az3' ,'ap4' ,'az4' ,'ap5' , 'az5', 
                  'bp1', 'bz1', 'bp2', 'bz2', 'bp3', 'bz3' ,'bp4' ,'bz4' ,'bp5' , 'bz5',
                  'tp', 'ts', 'tv', 'a', 'b', 'time']
    df.drop('a', axis=1, inplace=True)
    df.drop('b', axis=1, inplace=True)
    filename = '../data/{0}.20171120.csv'.format(symbol)
    df.to_csv(filename)
    
if __name__ == '__main__':
    main()