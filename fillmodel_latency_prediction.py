import gsampler as gs
import time
import pandas as pd
import numpy as np
from sklearn import svm, preprocessing
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt
import os
import tables
import gdb.gdb as gdb
from gutil import Repo
# import time
from parser.parser import parseStrategyLog
from market_data.common import timeAdjustment
# from market_data import selectDatesAndFiles
from market_data2 import get_file_selection
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor


class Features:

    candidate_features = [
        #         'ask_book_up_time_trade',
        #         'bid_book_down_time_trade',
        'order_num_trade_sell_side',
        'order_size_trade_sell_side',
        'order_num_trade_buy_side',
        'order_size_trade_buy_side',
        'order_num_trigger_sell_side',
        'order_size_trigger_sell_side',
        'order_num_trigger_buy_side',
        'order_size_trigger_buy_side',
        'ask_num_accum_trade',
        'ask_num_accum_trigger',
        'bid_num_accum_trade',
        'bid_num_accum_trigger',
        'ask_size_accum_trade',
        'ask_size_accum_trigger',
        'bid_size_accum_trade',
        'bid_size_accum_trigger',
        'ask_num_cancel_trade',
        'ask_num_cancel_trigger',
        'bid_num_cancel_trade',
        'bid_num_cancel_trigger',
        'ask_size_cancel_trade',
        'ask_size_cancel_trigger',
        'bid_size_cancel_trade',
        'bid_size_cancel_trigger',
        'trd_through_num_buy_side_trigger',
        'trd_through_num_sell_side_trigger',
        'trd_through_num_buy_side_trade',
        'trd_through_num_sell_side_trade',
        'trd_through_levels_buy_side_trigger',
        'trd_through_levels_sell_side_trigger',
        'trd_through_levels_buy_side_trade',
        'trd_through_levels_sell_side_trade'
    ]


class Fillmodel_Latency_Prediction:

    def __init__(self, trade_symbol=None, trigger_symbols=[], filepath=None):
        self.trade_symbol = trade_symbol
        self.trigger_symbols = trigger_symbols
        if not filepath:
            self.filepath = '/share/tmp/tian/latency_learn'
        else:
            self.filepath = filepath
        self.model = None
        self.scaler = None
        self.training_stat = None
        exch = trade_symbol.split('.')[1]
        if exch in ['CME', 'ICE']:
            self.company = 'ginkgo'
            Repo.set('us_fut')
        elif exch in ['KRX']:
            self.company = 'ginkgo'
            Repo.set('kr_fut')

    def test_latency_table(self, df_latency_log=None, df_latency_table=None, date=None):
        if df_latency_log is None:
            raise Exception('Dataframe for Latency Log does not exist')
        if df_latency_table is None:
            print Exception('Dataframe for Latency Table does not exist')
        df_latency_log = df_latency_log[
            df_latency_log['date'] == int(date)][['time_new', 'time_response', 'side', 'corrected_flag']].sort(['time_new'])
        df_latency_log['side'] = df_latency_log[
            'side'].replace([1, 0], ['BUY', 'SELL'])
        predict_res = []
        for i in range(len(df_latency_log)):
            order_time = df_latency_log['time_new'].iloc[i]
            order_side = df_latency_log['side'].iloc[i]
            latency = self._model_predict_interpolation(
                df_latency_table, order_time, order_side)
            predict_res.append(latency)
        if self.training_stat is not None:
            training_res = self.training_stat[
                self.training_stat.date == int(date)].drop(['date', 'actual_latency', 'side'], axis=1)
        df_res = pd.DataFrame(
            {'time_new': df_latency_log['time_new'], 'side': df_latency_log['side'],
             'corrected_flag': df_latency_log['corrected_flag'],
             'actual_latency': (df_latency_log['time_response'] - df_latency_log['time_new']),
             'predict_latency(table intp)': predict_res})
        result = pd.merge(df_res, training_res, how='left', on=['time_new']).rename(
            columns={'predict_latency': 'predict_latency(direct)'})
        print 'Interpolated Latency Prediction:'
        print result.to_string()
        return result

    def build_latency_table(self, df_group, selected_features=None, dropped_features=None, compress=True, threshold=50):
        df_predict_table = pd.DataFrame({'time': df_group.time})
        for order_side in [1, 0]:
            res = self.predict_latency(
                df_group, order_side, selected_features, dropped_features)
            side = 'BUY' if order_side == 1 else 'SELL'
            df_predict_table.loc[:, side] = res
            print 'The %s side is finished' % side
        df_predict_table.index = range(len(df_predict_table))
        print 'original table is finished'
        print 'table compression session ...'
        if not compress:
            print 'no compression request'
            return df_predict_table

        if threshold is None:
            threshold = min(
                min(df_predict_table.BUY), min(df_predict_table.SELL)) * 0.2
        begin = time.time()
        idx = self._table_compressor(df_predict_table, threshold)
        end = time.time()
        df_predict_table_comp = df_predict_table.ix[idx]
        df_predict_table_comp.index = range(len(df_predict_table_comp))
        print 'finish table compression in %.3f seconds' % (end - begin)
        return df_predict_table_comp

    def load_training_statistics(self):
        df = self._return_training_statistics()
        return df

    def predict_latency(self, df, order_side, selected_features=None, dropped_features=None):

        df_learn = self._feature_processor(
            df, selected_features, dropped_features, is_train=False)
        df_learn.loc[:, 'side'] = [order_side] * len(df_learn)

        X = np.array(df_learn.values)
        if not self.model:
            raise Exception(
                "The Machine Learning Model is not yet established")
        if not self.scaler:
            raise Exception(
                "The Machine Learning Scale is not yet established")
        X = self.scaler.transform(X)
        res = self.model.predict(X)
        return res

    def select_features(self, df, corr_threshold=None):
        m = self._get_features_correlation(df, corr_threshold)
        features = m['corrected_latency'].dropna().index.tolist()
        features.remove('corrected_latency')
        if 'date' in features:
            features.remove('date')
        return features

    def train_SVM_model(self, df, latency_outlier=6000, kernel='rbf', C=5000, epsilon=100, gamma=None, param_grid={},
                        training_portion=0.8, testing_portion=0.2, selected_features=None, dropped_features=None, shuffle=False, N=1):

        df_learn = self._feature_processor(
            df, selected_features, dropped_features, is_train=True)
        if C is None:
            C = 5000
        if epsilon is None:
            epsilon = 100
        if gamma is None:
            gamma = 0.0

        if N > 1 and not shuffle:
            print 'Iteration statistics, turn on the random shuffle'
            shuffle = True

        time_new_data = []
        side_data = []
        date_data = []
        latency_data = []
        predict_data = []

        for i in range(N):

            if shuffle:
                reindex = self._random_shuffle(df_learn.index)
                df_learn = df_learn.reindex(reindex)
            total_size = len(df_learn)
            train_size = min(int(total_size * training_portion), total_size)
            test_start = max(int(total_size * (1 - testing_portion)), 0)
            df_learn_train = df_learn.iloc[:train_size]
            df_learn_test = df_learn.iloc[test_start:]
            df_learn_train = df_learn_train[df_learn_train.corrected_flag == 1]
            if latency_outlier is not None:
                df_learn_train = df_learn_train[
                    df_learn_train.corrected_latency < latency_outlier]
            X_train_pre = np.array(
                df_learn_train.drop(['date', 'time_new', 'corrected_latency', 'corrected_flag'], axis=1).values)
            y_train = df_learn_train['corrected_latency'].tolist()
            train_size_pre = len(X_train_pre)
            print 'Drop %d outlier data points in the training' % (train_size - train_size_pre)
            X_test_pre = np.array(
                df_learn_test.drop(['date', 'time_new', 'corrected_latency', 'corrected_flag'], axis=1).values)

            X = np.concatenate((X_train_pre, X_test_pre), axis=0)
            X_train = X[:train_size_pre]
            X_test = X[train_size_pre:]
            scaler = preprocessing.StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

            if param_grid:
                print ('The Grid Search Mode')
                clf = GridSearchCV(
                    svm.SVR(kernel='rbf'), param_grid)
                clf = clf.fit(X_train, y_train)
                print("Best estimator found by grid search:")
                print(clf.best_estimator_)
                return (None, None)

            model = svm.SVR(kernel=kernel, C=C, epsilon=epsilon)
            model.fit(X_train, y_train)
            predict = model.predict(X_test)

            time_new = df_learn_test['time_new'].tolist()
            time_new_data += time_new
            side = [
                'BUY' if i == 1 else 'SELL' for i in df_learn_test['side'].tolist()]
            side_data += side
            latency = df_learn_test['corrected_latency'].tolist()
            latency_data += latency
            date = df_learn_test['date'].tolist()
            date_data += date
            predict_data += predict.tolist()

        df_res = pd.DataFrame(
            {'date': date_data, 'time_new': time_new_data, 'side': side_data, 'actual_latency': latency_data, 'predict_latency': predict_data})
        self._gather_training_statistics(df_res)
        self._register_model(model, scaler)

        print df_res.head(200).to_string()

    def train_ADA_model(self, df, latency_outlier=6000, corrected_flag_only=True, n_estimators=500, learning_rate=1, training_portion=0.8, testing_portion=0.2,
                        selected_features=None, dropped_features=None, shuffle=False, N=1):

        df_learn = self._feature_processor(
            df, selected_features, dropped_features, is_train=True)

        if N > 1 and not shuffle:
            print 'Iteration statistics, turn on the random shuffle'
            shuffle = True

        time_new_data = []
        side_data = []
        date_data = []
        latency_data = []
        predict_data = []

        for i in range(N):

            if shuffle:
                reindex = self._random_shuffle(df_learn.index)
                df_learn = df_learn.reindex(reindex)
            total_size = len(df_learn)
            train_size = min(int(total_size * training_portion), total_size)
            test_start = max(int(total_size * (1 - testing_portion)), 0)
            df_learn_train = df_learn.iloc[:train_size]
            df_learn_test = df_learn.iloc[test_start:]
            if corrected_flag_only:
                df_learn_train = df_learn_train[
                    df_learn_train.corrected_flag == 1]
            if latency_outlier is not None:
                df_learn_train = df_learn_train[
                    df_learn_train.corrected_latency < latency_outlier]
            X_train_pre = np.array(
                df_learn_train.drop(['date', 'time_new', 'corrected_latency', 'corrected_flag'], axis=1).values)
            y_train = df_learn_train['corrected_latency'].tolist()
            train_size_pre = len(X_train_pre)
            print 'Drop %d outlier data points in the training' % (train_size - train_size_pre)
            X_test_pre = np.array(
                df_learn_test.drop(['date', 'time_new', 'corrected_latency', 'corrected_flag'], axis=1).values)
#             y_test = df_learn_test['corrected_latency'].tolist()

            X = np.concatenate((X_train_pre, X_test_pre), axis=0)
            X_train = X[:train_size_pre]
            X_test = X[train_size_pre:]
            scaler = preprocessing.StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

            model = DecisionTreeRegressor()
            adb = AdaBoostRegressor(
                model, n_estimators=n_estimators, learning_rate=learning_rate)
            adb.fit(X_train, y_train)

            predict = adb.predict(X_test)

            time_new = df_learn_test['time_new'].tolist()
            time_new_data += time_new
            side = [
                'BUY' if i == 1 else 'SELL' for i in df_learn_test['side'].tolist()]
            side_data += side
            latency = df_learn_test['corrected_latency'].tolist()
            latency_data += latency
            date = df_learn_test['date'].tolist()
            date_data += date
            predict_data += predict.tolist()

        df_res = pd.DataFrame(
            {'date': date_data, 'time_new': time_new_data, 'side': side_data, 'actual_latency': latency_data, 'predict_latency': predict_data})
        self._gather_training_statistics(df_res)
        self._register_model(adb, scaler)

        print df_res.tail(200).to_string()
        return df_res

    def gather_latency_table_features(self, colo, date, grid_unit, trade_grids=5, trigger_grids=4, interpolation_roundoff=True, rerun=False):
        print "-" * 66
        print "Gather Latency Table Features for date: %s" % date
        df_group = self._latency_table_data_collector(
            date, colo, grid_unit, trade_grids, trigger_grids, interpolation_roundoff, rerun)
        print "-" * 66
        return df_group

    def gather_training_features(self, colo, colo_servers=[], dates=[], time_unit=1000, trade_upper_limit=5, trigger_lower_limit=4,
                                 interpolation_roundoff=True,
                                 minimum_latency=300, search_end=300, IOC_response_to_ack_limit=800,
                                 symbol_mode=None, latency_correction=True, rerun=False, force_log_file=None):
        df_concat = []
        for date in dates:
            for colo_server in colo_servers:
                print "-" * 66
                print "Gather Training Features for date: %s, colo_server: %s" % (date, colo_server)
                df_res = self._training_data_collector(
                    date, colo, colo_server, time_unit, trade_upper_limit, trigger_lower_limit,
                    interpolation_roundoff,
                    minimum_latency, search_end, IOC_response_to_ack_limit,
                    symbol_mode, latency_correction, rerun, force_log_file)
                print "-" * 66
                if df_res is None:
                    continue
                df_concat.append(df_res)
        df_features = pd.concat(df_concat, ignore_index=True)
        return df_features

    def concatenate_training_features(self, df_old, df_new):
        if df_new.date.isin(df_old.date):
            x = raw_input(
                'the new training features have overlaping dates with old training features, continue? [Y/N] ')
        if x.upper() != 'Y':
            print 'Abort concatenation'
            return None
        df_features = pd.concat([df_old, df_new], ignore_index=True)
        return df_features.astype(float)

    def drop_training_features(self, df, dates=[]):
        dates = [int(date) for date in dates]
        to_drop = df[df['date'].isin(dates)].index.tolist()
        df = df.drop(to_drop)
        df.index = range(len(df))
        return df

    def has_date_in_training_features(self, df, date):
        date = int(date)
        if df['date'].isin([date]).any():
            return True
        return False

    def save_data_df(self, df, filename):
        fname = '%s/%s' % (self.filepath, filename)
        print 'writing file: %s' % fname
        if df is None or df.empty:
            raise Exception('Dataframe not exists/has no data for saving')
        h5file = tables.open_file(fname, 'w', driver='H5FD_CORE')
        q = df.astype(float)
        r = q.to_records(index=False)
        f = tables.Filters(complevel=5)
        h5file.create_table(h5file.root, 'DATA', filters=f, obj=r)
        h5file.close()

    def load_data_df(self, filename, table='/DATA'):
        fname = '%s/%s' % (self.filepath, filename)
        if not os.path.exists(fname):
            raise Exception('.h5 file not exists')
        print 'loading file: %s' % fname
        with tables.open_file(fname) as f:
            df = pd.DataFrame(f.get_node(table).read())
        return df

    def _get_features_correlation(self, df, corr_threshold=None):
        df_learn = self._feature_processor(
            df, selected_features=None, dropped_features=None, is_train=True)
        X = df_learn.drop(
            ['date', 'time_new'], axis=1)
        m = X.corr()
        if corr_threshold is not None:
            m = m[m.abs() > corr_threshold]
        self._colorMatrix(m, corr_threshold)
        return m

    def _model_predict_interpolation(self, df_predict_table, order_time, order_side):
        time_table = df_predict_table.time
        time_index_l, time_index_r = self._time_window(time_table, order_time)
        if time_index_l == -1:
            return df_predict_table[order_side].ix[0]
        elif time_index_r == len(time_table):
            return df_predict_table[order_side].ix[-1]
        else:
            time_l = df_predict_table['time'].ix[time_index_l]
            time_r = df_predict_table['time'].ix[time_index_r]
            val_l = df_predict_table[order_side].ix[time_index_l]
            val_r = df_predict_table[order_side].ix[time_index_r]
            latency = ((order_time - time_l) * val_r +
                       (time_r - order_time) * val_l) / (time_r - time_l)
            return latency

    def _time_window(self, time_table, target):
        l = 0
        r = len(time_table) - 1
        while(l <= r):
            mid = l + (r - l) / 2
            if(target == time_table[mid]):
                return (mid - 1, mid)
            elif(target < time_table[mid]):
                r = mid - 1
            else:
                l = mid + 1
        return (r, l)

    def _table_compressor(self, df, threshold=50):
        start = 0
        prev = 0
        cur = 0
        idx = [0]
        prev_record = True
        size = len(df)
        for i in range(size)[1:]:
            cur = i
            if ((abs(df['BUY'].ix[cur] - df['BUY'].ix[prev]) > threshold) or (abs(df['BUY'].ix[cur] - df['BUY'].ix[start]) > threshold) or
                    (abs(df['SELL'].ix[cur] - df['SELL'].ix[prev]) > threshold) or (abs(df['SELL'].ix[cur] - df['SELL'].ix[start]) > threshold)):
                if not prev_record:
                    idx.append(prev)
                idx.append(cur)
                prev_record = True
                start = cur
            else:
                prev_record = False
            prev = cur
        return idx

    def _register_model(self, model=None, scaler=None):
        if not model or not scaler:
            raise Exception('No model or scaler is available')
        self.model = model
        self.scaler = scaler

    def _gather_training_statistics(self, df):
        self.training_stat = df

    def _return_training_statistics(self):
        if self.training_stat is None:
            print 'No training statistics dataframe is recorded'
            return None
        return self.training_stat

    def _random_shuffle(self, index):
        reindex = np.random.permutation(index)
        return reindex

    def _grouper(self, row, grid_unit):
        return int(row.time) / int(grid_unit)

    def _latency_table_data_collector(self, date, colo, grid_unit=1000, trade_grids=5, trigger_grids=4,
                                      interpolation_roundoff=True,
                                      symbol_mode=None, rerun=False):
        print '''
                Reminder: ***************************************************************
                To make the calculation correct, the following is the MUST:
                  The arguments here should be consistent with those in self._training_data_collector()
                  grid_unit == time_unit, trade_grids == trade_upper_limit, trigger_grids == trigger_lower_limit
                  interpolation_roundoff setting is the same
                *************************************************************************
                '''

        def _future_roll_sum(index, grids):
            dd = df_group[(df_group['group_flag'] >= index) & (df_group['group_flag'] < index + grids)].sum()
            return dd

        def _history_roll_sum(index, grids, interpolation_roundoff):
            if interpolation_roundoff:
                dd = df_group[(df_group['group_flag'] > index - grids) & (df_group['group_flag'] <= index)].sum()
            else:
                dd = df_group[(df_group['group_flag'] > index - grids) & (df_group['group_flag'] < index)].sum()
            return dd

        print 'Run Sampler ...'
        dmap = self._runSampler(date, colo, symbol_mode, rerun)
        if not dmap:
            print 'No data to run the sampler, skip'
            return None

        print 'Get Latency Table Features ...'
        d_concat = [dmap[key] for key in dmap]
        d = pd.concat(d_concat, join='outer').sort('time')
        d = d.replace(0, np.nan)

        print 'rows of d is %d' % len(d)

        d['group_flag'] = d.apply(self._grouper, args=(grid_unit,), axis=1)
        obj_group = d.groupby('group_flag')
        del d
        df_group = obj_group['time'].first().reset_index()

        print 'rows of df_group is %d' % len(df_group)

        print df_group.head().to_string()

        for symbol in set([self.trade_symbol] + self.trigger_symbols):
            print 'features are being discretized to the sub-window for %s' % symbol
            symbol_name = symbol.split('.')[0]
            df_group[['ask_num_accum_%s' % symbol_name, 'bid_num_accum_%s' % symbol_name,
                      'ask_num_cancel_%s' % symbol_name, 'bid_num_cancel_%s' % symbol_name,
                      'order_num_sell_side_%s' % symbol_name, 'order_num_buy_side_%s' % symbol_name,
                      'trd_through_num_sell_side_%s' % symbol_name,
                      'trd_through_num_buy_side_%s' % symbol_name]] = obj_group[[
                          'F_EventAskQtyDeltaPlus_%s' % symbol_name,
                          'F_EventBidQtyDeltaPlus_%s' % symbol_name,
                          'F_EventAskQtyDeltaMinus_%s' % symbol_name,
                          'F_EventBidQtyDeltaMinus_%s' % symbol_name,
                          'F_TradeSize_Sell_%s' % symbol_name,
                          'F_TradeSize_Buy_%s' % symbol_name,
                          'F_TradeThru_Sell_%s' % symbol_name,
                          'F_TradeThru_Buy_%s' % symbol_name]].count().reset_index(drop=True)
            print "finish count features for %s" % symbol_name

            df_group[['ask_size_accum_%s' % symbol_name, 'bid_size_accum_%s' % symbol_name,
                      'ask_size_cancel_%s' % symbol_name, 'bid_size_cancel_%s' % symbol_name,
                      'order_size_sell_side_%s' % symbol_name, 'order_size_buy_side_%s' % symbol_name,
                      'trd_through_levels_sell_side_%s' % symbol_name,
                      'trd_through_levels_buy_side_%s' % symbol_name]] = obj_group[[
                          'F_EventAskQtyDeltaPlus_%s' % symbol_name,
                          'F_EventBidQtyDeltaPlus_%s' % symbol_name,
                          'F_EventAskQtyDeltaMinus_%s' % symbol_name,
                          'F_EventBidQtyDeltaMinus_%s' % symbol_name,
                          'F_TradeSize_Sell_%s' % symbol_name,
                          'F_TradeSize_Buy_%s' % symbol_name,
                          'F_TradeThru_Sell_%s' % symbol_name,
                          'F_TradeThru_Buy_%s' % symbol_name]].sum().abs().fillna(0).reset_index(drop=True)
            print "finish size features for %s" % symbol_name

#         print df_group.columns
        df_roll_flags = df_group[['time', 'group_flag']]

        print 'start to work on rolling sum for future'
        start = time.time()
#         df_roll_future = df_group.group_flag.apply(_future_roll_sum, args=(trade_grids,))
#         df_roll_future = df_roll_future.drop(['group_flag', 'time'], axis=1)
        df_roll_future = []
        for index in df_group.group_flag:
            df_roll_future.append(df_group[(df_group['group_flag'] >= index) & (df_group['group_flag'] < index + trade_grids)].sum().tolist())

        df_roll_future = pd.DataFrame(df_roll_future, columns=df_group.columns).drop(['group_flag', 'time'], axis=1)

        for symbol in set([self.trade_symbol] + self.trigger_symbols):
            print 'rolling sum for future is being implemented for %s' % symbol
            symbol_name = symbol.split('.')[0]
            df_roll_future.rename(columns={'ask_num_accum_%s' % symbol_name: 'ask_num_accum_trade_%s' % symbol_name,
                                           'ask_size_accum_%s' % symbol_name: 'ask_size_accum_trade_%s' % symbol_name,
                                           'bid_num_accum_%s' % symbol_name: 'bid_num_accum_trade_%s' % symbol_name,
                                           'bid_size_accum_%s' % symbol_name: 'bid_size_accum_trade_%s' % symbol_name,
                                           'ask_num_cancel_%s' % symbol_name: 'ask_num_cancel_trade_%s' % symbol_name,
                                           'ask_size_cancel_%s' % symbol_name: 'ask_size_cancel_trade_%s' % symbol_name,
                                           'bid_num_cancel_%s' % symbol_name: 'bid_num_cancel_trade_%s' % symbol_name,
                                           'bid_size_cancel_%s' % symbol_name: 'bid_size_cancel_trade_%s' % symbol_name,
                                           'order_num_sell_side_%s' % symbol_name: 'order_num_trade_sell_side_%s' % symbol_name,
                                           'order_size_sell_side_%s' % symbol_name: 'order_size_trade_sell_side_%s' % symbol_name,
                                           'order_num_buy_side_%s' % symbol_name: 'order_num_trade_buy_side_%s' % symbol_name,
                                           'order_size_buy_side_%s' % symbol_name: 'order_size_trade_buy_side_%s' % symbol_name,
                                           'trd_through_num_sell_side_%s' % symbol_name: 'trd_through_num_sell_side_trade_%s' % symbol_name,
                                           'trd_through_levels_sell_side_%s' % symbol_name: 'trd_through_levels_sell_side_trade_%s' % symbol_name,
                                           'trd_through_num_buy_side_%s' % symbol_name: 'trd_through_num_buy_side_trade_%s' % symbol_name,
                                           'trd_through_levels_buy_side_%s' % symbol_name: 'trd_through_levels_buy_side_trade_%s' % symbol_name},
                                  inplace=True)
        end = time.time()
        print 'Finish rolling sum for future in %.3f seconds' % (end - start)

        print 'start to work on rolling sum for history'
        start = time.time()
        trigger_grids += 1
        df_roll_history = df_group.group_flag.apply(_history_roll_sum, args=(trigger_grids, interpolation_roundoff))
        df_roll_history = df_roll_history.drop(['group_flag', 'time'], axis=1)

        for symbol in set([self.trade_symbol] + self.trigger_symbols):
            print 'rolling sum for history is being implemented for %s' % symbol
            symbol_name = symbol.split('.')[0]

            df_roll_history.rename(columns={'ask_num_accum_%s' % symbol_name: 'ask_num_accum_trigger_%s' % symbol_name,
                                            'ask_size_accum_%s' % symbol_name: 'ask_size_accum_trigger_%s' % symbol_name,
                                            'bid_num_accum_%s' % symbol_name: 'bid_num_accum_trigger_%s' % symbol_name,
                                            'bid_size_accum_%s' % symbol_name: 'bid_size_accum_trigger_%s' % symbol_name,
                                            'ask_num_cancel_%s' % symbol_name: 'ask_num_cancel_trigger_%s' % symbol_name,
                                            'ask_size_cancel_%s' % symbol_name: 'ask_size_cancel_trigger_%s' % symbol_name,
                                            'bid_num_cancel_%s' % symbol_name: 'bid_num_cancel_trigger_%s' % symbol_name,
                                            'bid_size_cancel_%s' % symbol_name: 'bid_size_cancel_trigger_%s' % symbol_name,
                                            'order_num_sell_side_%s' % symbol_name: 'order_num_trigger_sell_side_%s' % symbol_name,
                                            'order_size_sell_side_%s' % symbol_name: 'order_size_trigger_sell_side_%s' % symbol_name,
                                            'order_num_buy_side_%s' % symbol_name: 'order_num_trigger_buy_side_%s' % symbol_name,
                                            'order_size_buy_side_%s' % symbol_name: 'order_size_trigger_buy_side_%s' % symbol_name,
                                            'trd_through_num_sell_side_%s' % symbol_name: 'trd_through_num_sell_side_trigger_%s' % symbol_name,
                                            'trd_through_levels_sell_side_%s' % symbol_name: 'trd_through_levels_sell_side_trigger_%s' % symbol_name,
                                            'trd_through_num_buy_side_%s' % symbol_name: 'trd_through_num_buy_side_trigger_%s' % symbol_name,
                                            'trd_through_levels_buy_side_%s' % symbol_name: 'trd_through_levels_buy_side_trigger_%s' % symbol_name},
                                   inplace=True)
        end = time.time()
        print 'Finish rolling sum for history in %.3f seconds' % (end - start)

        df_roll = pd.concat([df_roll_flags, df_roll_future, df_roll_history], axis=1)

        print 'history window size for trigger symbols is %d, future window size for trade symbol is %d' % (
            grid_unit * trigger_grids, grid_unit * trade_grids)
        return df_roll

    def _latency_table_data_collector_deprecate(self, date, colo, grid_unit=1000, trade_grids=5, trigger_grids=4,
                                                interpolation_roundoff=True,
                                                symbol_mode=None, rerun=False):
        print '''
                Reminder: ***************************************************************
                To make the calculation correct, the following is the MUST:
                  The arguments here should be consistent with those in self._training_data_collector()
                  grid_unit == time_unit, trade_grids == trade_upper_limit, trigger_grids == trigger_lower_limit
                  interpolation_roundoff setting is the same
                *************************************************************************
                '''
        print 'Run Sampler ...'
        dmap = self._runSampler(date, colo, symbol_mode, rerun)
        if not dmap:
            print 'No data to run the sampler, skip'
            return None

        print 'Get Latency Table Features ...'
        d_concat = [dmap[key] for key in dmap]
        d = pd.concat(d_concat, join='outer').sort('time')
        d = d.replace(0, np.nan)

        print 'rows of d is %d' % len(d)

        d['group_flag'] = d.apply(self._grouper, args=(grid_unit,), axis=1)
        obj_group = d.groupby('group_flag')
        del d
        df_group = obj_group['time'].first().reset_index()

        print 'rows of df_group is %d' % len(df_group)

        print df_group.head().to_string()

        for symbol in set([self.trade_symbol] + self.trigger_symbols):
            print 'features are being discretized to the sub-window for %s' % symbol
            symbol_name = symbol.split('.')[0]
            df_group[['ask_num_accum_%s' % symbol_name, 'bid_num_accum_%s' % symbol_name,
                      'ask_num_cancel_%s' % symbol_name, 'bid_num_cancel_%s' % symbol_name,
                      'order_num_sell_side_%s' % symbol_name, 'order_num_buy_side_%s' % symbol_name,
                      'trd_through_num_sell_side_%s' % symbol_name,
                      'trd_through_num_buy_side_%s' % symbol_name]] = obj_group[[
                          'F_EventAskQtyDeltaPlus_%s' % symbol_name,
                          'F_EventBidQtyDeltaPlus_%s' % symbol_name,
                          'F_EventAskQtyDeltaMinus_%s' % symbol_name,
                          'F_EventBidQtyDeltaMinus_%s' % symbol_name,
                          'F_TradeSize_Sell_%s' % symbol_name,
                          'F_TradeSize_Buy_%s' % symbol_name,
                          'F_TradeThru_Sell_%s' % symbol_name,
                          'F_TradeThru_Buy_%s' % symbol_name]].count().reset_index(drop=True)
            print "finish count features for %s" % symbol_name

            df_group[['ask_size_accum_%s' % symbol_name, 'bid_size_accum_%s' % symbol_name,
                      'ask_size_cancel_%s' % symbol_name, 'bid_size_cancel_%s' % symbol_name,
                      'order_size_sell_side_%s' % symbol_name, 'order_size_buy_side_%s' % symbol_name,
                      'trd_through_levels_sell_side_%s' % symbol_name,
                      'trd_through_levels_buy_side_%s' % symbol_name]] = obj_group[[
                          'F_EventAskQtyDeltaPlus_%s' % symbol_name,
                          'F_EventBidQtyDeltaPlus_%s' % symbol_name,
                          'F_EventAskQtyDeltaMinus_%s' % symbol_name,
                          'F_EventBidQtyDeltaMinus_%s' % symbol_name,
                          'F_TradeSize_Sell_%s' % symbol_name,
                          'F_TradeSize_Buy_%s' % symbol_name,
                          'F_TradeThru_Sell_%s' % symbol_name,
                          'F_TradeThru_Buy_%s' % symbol_name]].sum().abs().fillna(0).reset_index(drop=True)
            print "finish size features for %s" % symbol_name

        group_flag_list = []
        gap = max(trade_grids, trigger_grids + 1)
        ll = len(df_group)
        begin = time.time()

        print "before group flag expansion, the rows of df_group is %d" % ll

        for i in range(ll - 1):
            if(df_group['group_flag'].iloc[i + 1] - df_group['group_flag'].iloc[i] > 1):
                val = df_group['group_flag'].iloc[i]
                count = 0
                while(val < df_group['group_flag'].iloc[i + 1] and count < gap):
                    group_flag_list.append(val)
                    val += 1
                    count += 1
            else:
                group_flag_list.append(df_group['group_flag'].iloc[i])
        group_flag_list.append(df_group['group_flag'].iloc[-1])
        end = time.time()
        print 'finish group_flag_list in %.3f seconds, the rows is %d' % ((end - begin), len(group_flag_list))

        df_temp = pd.DataFrame({'group_flag_cont': group_flag_list})
        df_group = pd.merge(df_group, df_temp, left_on='group_flag',
                            right_on='group_flag_cont', how='outer',
                            sort=True).drop(['group_flag_cont'], 1).fillna(0)

        print 'rows of df_group after flag expansion is %d' % len(df_group)

        df_rollsum = df_group[['time', 'group_flag']]

        for symbol in set([self.trade_symbol] + self.trigger_symbols):
            print 'rolling sum for future is being implemented for %s' % symbol
            symbol_name = symbol.split('.')[0]
            df_rollsum['ask_num_accum_trade_%s' % symbol_name] = pd.rolling_sum(
                df_group['ask_num_accum_%s' % symbol_name][::-1], window=trade_grids)[::-1]
            df_rollsum['ask_size_accum_trade_%s' % symbol_name] = pd.rolling_sum(
                df_group['ask_size_accum_%s' % symbol_name][::-1], window=trade_grids)[::-1]
            df_rollsum['bid_num_accum_trade_%s' % symbol_name] = pd.rolling_sum(
                df_group['bid_num_accum_%s' % symbol_name][::-1], window=trade_grids)[::-1]
            df_rollsum['bid_size_accum_trade_%s' % symbol_name] = pd.rolling_sum(
                df_group['bid_size_accum_%s' % symbol_name][::-1], window=trade_grids)[::-1]
            df_rollsum['ask_num_cancel_trade_%s' % symbol_name] = pd.rolling_sum(
                df_group['ask_num_cancel_%s' % symbol_name][::-1], window=trade_grids)[::-1]
            df_rollsum['ask_size_cancel_trade_%s' % symbol_name] = pd.rolling_sum(
                df_group['ask_size_cancel_%s' % symbol_name][::-1], window=trade_grids)[::-1]
            df_rollsum['bid_num_cancel_trade_%s' % symbol_name] = pd.rolling_sum(
                df_group['bid_num_cancel_%s' % symbol_name][::-1], window=trade_grids)[::-1]
            df_rollsum['bid_size_cancel_trade_%s' % symbol_name] = pd.rolling_sum(
                df_group['bid_size_cancel_%s' % symbol_name][::-1], window=trade_grids)[::-1]
            df_rollsum['order_num_trade_sell_side_%s' % symbol_name] = pd.rolling_sum(
                df_group['order_num_sell_side_%s' % symbol_name][::-1], window=trade_grids)[::-1]
            df_rollsum['order_size_trade_sell_side_%s' % symbol_name] = pd.rolling_sum(
                df_group['order_size_sell_side_%s' % symbol_name][::-1], window=trade_grids)[::-1]
            df_rollsum['order_num_trade_buy_side_%s' % symbol_name] = pd.rolling_sum(
                df_group['order_num_buy_side_%s' % symbol_name][::-1], window=trade_grids)[::-1]
            df_rollsum['order_size_trade_buy_side_%s' % symbol_name] = pd.rolling_sum(
                df_group['order_size_buy_side_%s' % symbol_name][::-1], window=trade_grids)[::-1]
            df_rollsum['trd_through_num_sell_side_trade_%s' % symbol_name] = pd.rolling_sum(
                df_group['trd_through_num_sell_side_%s' % symbol_name][::-1], window=trade_grids)[::-1]
            df_rollsum['trd_through_levels_sell_side_trade_%s' % symbol_name] = pd.rolling_sum(
                df_group['trd_through_levels_sell_side_%s' % symbol_name][::-1], window=trade_grids)[::-1]
            df_rollsum['trd_through_num_buy_side_trade_%s' % symbol_name] = pd.rolling_sum(
                df_group['trd_through_num_buy_side_%s' % symbol_name][::-1], window=trade_grids)[::-1]
            df_rollsum['trd_through_levels_buy_side_trade_%s' % symbol_name] = pd.rolling_sum(
                df_group['trd_through_levels_buy_side_%s' % symbol_name][::-1], window=trade_grids)[::-1]

        trigger_grids += 1
        for symbol in set([self.trade_symbol] + self.trigger_symbols):
            print 'rolling sum for history is being implemented for %s' % symbol
            symbol_name = symbol.split('.')[0]
            if not interpolation_roundoff:
                # if not consider the interpolation roundoff, the history view
                # does not include one future point
                df_rollsum['ask_num_accum_trigger_%s' % symbol_name] = pd.rolling_sum(
                    df_group['ask_num_accum_%s' % symbol_name], window=trigger_grids) - df_group['ask_num_accum_%s' % symbol_name]
                df_rollsum['ask_size_accum_trigger_%s' % symbol_name] = pd.rolling_sum(
                    df_group['ask_size_accum_%s' % symbol_name], window=trigger_grids) - df_group['ask_size_accum_%s' % symbol_name]
                df_rollsum['bid_num_accum_trigger_%s' % symbol_name] = pd.rolling_sum(
                    df_group['bid_num_accum_%s' % symbol_name], window=trigger_grids) - df_group['bid_num_accum_%s' % symbol_name]
                df_rollsum['bid_size_accum_trigger_%s' % symbol_name] = pd.rolling_sum(
                    df_group['bid_size_accum_%s' % symbol_name], window=trigger_grids) - df_group['bid_size_accum_%s' % symbol_name]
                df_rollsum['ask_num_cancel_trigger_%s' % symbol_name] = pd.rolling_sum(
                    df_group['ask_num_cancel_%s' % symbol_name], window=trigger_grids) - df_group['ask_num_cancel_%s' % symbol_name]
                df_rollsum['ask_size_cancel_trigger_%s' % symbol_name] = pd.rolling_sum(
                    df_group['ask_size_cancel_%s' % symbol_name], window=trigger_grids) - df_group['ask_size_cancel_%s' % symbol_name]
                df_rollsum['bid_num_cancel_trigger_%s' % symbol_name] = pd.rolling_sum(
                    df_group['bid_num_cancel_%s' % symbol_name], window=trigger_grids) - df_group['bid_num_cancel_%s' % symbol_name]
                df_rollsum['bid_size_cancel_trigger_%s' % symbol_name] = pd.rolling_sum(
                    df_group['bid_size_cancel_%s' % symbol_name], window=trigger_grids) - df_group['bid_size_cancel_%s' % symbol_name]
                df_rollsum['order_num_trigger_sell_side_%s' % symbol_name] = pd.rolling_sum(
                    df_group['order_num_sell_side_%s' % symbol_name], window=trigger_grids) - df_group['order_num_sell_side_%s' % symbol_name]
                df_rollsum['order_size_trigger_sell_side_%s' % symbol_name] = pd.rolling_sum(
                    df_group['order_size_sell_side_%s' % symbol_name], window=trigger_grids) - df_group['order_size_sell_side_%s' % symbol_name]
                df_rollsum['order_num_trigger_buy_side_%s' % symbol_name] = pd.rolling_sum(
                    df_group['order_num_buy_side_%s' % symbol_name], window=trigger_grids) - df_group['order_num_buy_side_%s' % symbol_name]
                df_rollsum['order_size_trigger_buy_side_%s' % symbol_name] = pd.rolling_sum(
                    df_group['order_size_buy_side_%s' % symbol_name], window=trigger_grids) - df_group['order_size_buy_side_%s' % symbol_name]
                df_rollsum['trd_through_num_sell_side_trigger_%s' % symbol_name] = pd.rolling_sum(
                    df_group['trd_through_num_sell_side_%s' % symbol_name], window=trigger_grids) - df_group['trd_through_num_sell_side_%s' % symbol_name]
                df_rollsum['trd_through_levels_sell_side_trigger_%s' % symbol_name] = pd.rolling_sum(
                    df_group['trd_through_levels_sell_side_%s' % symbol_name], window=trigger_grids) - df_group['trd_through_levels_sell_side_%s' % symbol_name]
                df_rollsum['trd_through_num_buy_side_trigger_%s' % symbol_name] = pd.rolling_sum(
                    df_group['trd_through_num_buy_side_%s' % symbol_name], window=trigger_grids) - df_group['trd_through_num_buy_side_%s' % symbol_name]
                df_rollsum['trd_through_levels_buy_side_trigger_%s' % symbol_name] = pd.rolling_sum(
                    df_group['trd_through_levels_buy_side_%s' % symbol_name], window=trigger_grids) - df_group['trd_through_levels_buy_side_%s' % symbol_name]
            else:
                #                 print 'I am okay here 1'
                df_rollsum['ask_num_accum_trigger_%s' % symbol_name] = pd.rolling_sum(
                    df_group['ask_num_accum_%s' % symbol_name], window=trigger_grids)
                df_rollsum['ask_size_accum_trigger_%s' % symbol_name] = pd.rolling_sum(
                    df_group['ask_size_accum_%s' % symbol_name], window=trigger_grids)
                df_rollsum['bid_num_accum_trigger_%s' % symbol_name] = pd.rolling_sum(
                    df_group['bid_num_accum_%s' % symbol_name], window=trigger_grids)
                df_rollsum['bid_size_accum_trigger_%s' % symbol_name] = pd.rolling_sum(
                    df_group['bid_size_accum_%s' % symbol_name], window=trigger_grids)
                df_rollsum['ask_num_cancel_trigger_%s' % symbol_name] = pd.rolling_sum(
                    df_group['ask_num_cancel_%s' % symbol_name], window=trigger_grids)
                df_rollsum['ask_size_cancel_trigger_%s' % symbol_name] = pd.rolling_sum(
                    df_group['ask_size_cancel_%s' % symbol_name], window=trigger_grids)
                df_rollsum['bid_num_cancel_trigger_%s' % symbol_name] = pd.rolling_sum(
                    df_group['bid_num_cancel_%s' % symbol_name], window=trigger_grids)
#                 print 'I am okay here 2'
                df_rollsum['bid_size_cancel_trigger_%s' % symbol_name] = pd.rolling_sum(
                    df_group['bid_size_cancel_%s' % symbol_name], window=trigger_grids)
                df_rollsum['order_num_trigger_sell_side_%s' % symbol_name] = pd.rolling_sum(
                    df_group['order_num_sell_side_%s' % symbol_name], window=trigger_grids)
                df_rollsum['order_size_trigger_sell_side_%s' % symbol_name] = pd.rolling_sum(
                    df_group['order_size_sell_side_%s' % symbol_name], window=trigger_grids)
                df_rollsum['order_num_trigger_buy_side_%s' % symbol_name] = pd.rolling_sum(
                    df_group['order_num_buy_side_%s' % symbol_name], window=trigger_grids)
                df_rollsum['order_size_trigger_buy_side_%s' % symbol_name] = pd.rolling_sum(
                    df_group['order_size_buy_side_%s' % symbol_name], window=trigger_grids)
#                 print 'I am okay here 3'
                df_rollsum['trd_through_num_sell_side_trigger_%s' % symbol_name] = pd.rolling_sum(
                    df_group['trd_through_num_sell_side_%s' % symbol_name], window=trigger_grids)
                df_rollsum['trd_through_levels_sell_side_trigger_%s' % symbol_name] = pd.rolling_sum(
                    df_group['trd_through_levels_sell_side_%s' % symbol_name], window=trigger_grids)
                df_rollsum['trd_through_num_buy_side_trigger_%s' % symbol_name] = pd.rolling_sum(
                    df_group['trd_through_num_buy_side_%s' % symbol_name], window=trigger_grids)
                df_rollsum['trd_through_levels_buy_side_trigger_%s' % symbol_name] = pd.rolling_sum(
                    df_group['trd_through_levels_buy_side_%s' % symbol_name], window=trigger_grids)
#                 print 'I am okay here 4'

        print 'rows of df_group is %d' % len(df_group)
        print 'rows of df_rollsum is', len(df_rollsum)
#         print 'start to convert data type'
#         df_group = df_group.astype(np.int32)
#         print 'data type converts to int32'
        del df_group
        df_rollsum = df_rollsum[df_rollsum.group_flag > 0].fillna(0)
        print 'I finish fillna'
        print 'history window size for trigger symbols is %d, future window size for trade symbol is %d' % (
            grid_unit * trigger_grids, grid_unit * trade_grids)
        return df_rollsum

    def _training_data_collector(self, date, colo, colo_server,
                                 time_unit=1000,
                                 trade_upper_limit=5, trigger_lower_limit=4,
                                 interpolation_roundoff=True,
                                 minimum_latency=300, search_end=300, IOC_response_to_ack_limit=800,
                                 symbol_mode=None, latency_correction=True, rerun=False, force_log_file=None):
        print 'Get Latency Log ...'
        df_latency = self._latency_in_log(
            date, colo, colo_server, symbol_mode, latency_correction, minimum_latency, search_end,
            IOC_response_to_ack_limit, rerun, force_log_file)
        if df_latency is None:
            return None
        trade_lower_limit = 0
        trade_upper_limit = trade_upper_limit * time_unit
        trigger_lower_limit = -1 * trigger_lower_limit * time_unit
        if interpolation_roundoff:
            trigger_upper_limit = time_unit
        else:
            trigger_upper_limit = 0

        df_concat = [df_latency]
        order_new_time = df_latency['time_new']
        print 'Run Sampler ...'
        dmap = self._runSampler(date, colo, symbol_mode, rerun)
        if not dmap:
            print 'No data to run the sampler, skip'
            return None

        print 'Get Training Features ...'
        for s in dmap:
            # replace 0 with NAN to avoid multiple 0-value callbacks in the
            # count()
            dmap[s] = dmap[s].replace(0, np.nan)

        for symbol in set([self.trade_symbol] + self.trigger_symbols):
            symbol_name = symbol.split('.')[0]
            data = []
            col = [
                #                'ask_book_up_time_trade',
                #                'bid_book_down_time_trade',
                'ask_num_accum_trade',
                'bid_num_accum_trade',
                'ask_num_cancel_trade',
                'bid_num_cancel_trade',
                'order_num_trade_sell_side',
                'order_num_trade_buy_side',
                'trd_through_num_sell_side_trade',
                'trd_through_num_buy_side_trade',
                'ask_size_accum_trade',
                'bid_size_accum_trade',
                'ask_size_cancel_trade',
                'bid_size_cancel_trade',
                'order_size_trade_sell_side',
                'order_size_trade_buy_side',
                'trd_through_levels_sell_side_trade',
                'trd_through_levels_buy_side_trade']
            col = ['%s_%s' % (c, symbol_name) for c in col]
            d = dmap[symbol_name]
            for order_time in order_new_time:
                #             ask_up_time = -1
                #             bid_down_time = -1
                #             ask_up_time_df = d[
                #                 (d['time'] - order_time > 0) & (d['F_AskPrice_Up_%s' % symbol_name] > 0)]['time']
                #             if len(ask_up_time_df) > 0:
                #                 ask_up_time = ask_up_time_df.iloc[0] - order_time
                #             bid_down_time_df = d[
                #                 (d['time'] - order_time > 0) & (d['F_BidPrice_Down_%s' % symbol_name] < 0)]['time']
                #             if len(bid_down_time_df) > 0:
                #                 bid_down_time = bid_down_time_df.iloc[0] - order_time
                num_data = d[(d['time'] - order_time >= trade_lower_limit) &
                             (d['time'] - order_time <= trade_upper_limit)][[
                                 'F_EventAskQtyDeltaPlus_%s' % symbol_name,
                                 'F_EventBidQtyDeltaPlus_%s' % symbol_name,
                                 'F_EventAskQtyDeltaMinus_%s' % symbol_name,
                                 'F_EventBidQtyDeltaMinus_%s' % symbol_name,
                                 'F_TradeSize_Sell_%s' % symbol_name,
                                 'F_TradeSize_Buy_%s' % symbol_name,
                                 'F_TradeThru_Sell_%s' % symbol_name,
                                 'F_TradeThru_Buy_%s' % symbol_name]].count().tolist()
                size_data = d[(d['time'] - order_time >= trade_lower_limit) &
                              (d['time'] - order_time <= trade_upper_limit)][[
                                  'F_EventAskQtyDeltaPlus_%s' % symbol_name,
                                  'F_EventBidQtyDeltaPlus_%s' % symbol_name,
                                  'F_EventAskQtyDeltaMinus_%s' % symbol_name,
                                  'F_EventBidQtyDeltaMinus_%s' % symbol_name,
                                  'F_TradeSize_Sell_%s' % symbol_name,
                                  'F_TradeSize_Buy_%s' % symbol_name,
                                  'F_TradeThru_Sell_%s' % symbol_name,
                                  'F_TradeThru_Buy_%s' % symbol_name]].sum().abs().fillna(0).tolist()
                data.append((num_data + size_data))
            df_trd = pd.DataFrame(data, columns=col)
            df_concat.append(df_trd)

        for symbol in set([self.trade_symbol] + self.trigger_symbols):
            symbol_name = symbol.split('.')[0]
            data = []
            col = ['ask_num_accum_trigger',
                   'bid_num_accum_trigger',
                   'ask_num_cancel_trigger',
                   'bid_num_cancel_trigger',
                   'order_num_trigger_sell_side',
                   'order_num_trigger_buy_side',
                   'trd_through_num_sell_side_trigger',
                   'trd_through_num_buy_side_trigger',
                   'ask_size_accum_trigger',
                   'bid_size_accum_trigger',
                   'ask_size_cancel_trigger',
                   'bid_size_cancel_trigger',
                   'order_size_trigger_sell_side',
                   'order_size_trigger_buy_side',
                   'trd_through_levels_sell_side_trigger',
                   'trd_through_levels_buy_side_trigger']
            col = ['%s_%s' % (c, symbol_name) for c in col]
            d = dmap[symbol_name]
            for order_time in order_new_time:
                num_data = d[(d['time'] - order_time >= trigger_lower_limit) &
                             (d['time'] - order_time <= trigger_upper_limit)][[
                                 'F_EventAskQtyDeltaPlus_%s' % symbol_name,
                                 'F_EventBidQtyDeltaPlus_%s' % symbol_name,
                                 'F_EventAskQtyDeltaMinus_%s' % symbol_name,
                                 'F_EventBidQtyDeltaMinus_%s' % symbol_name,
                                 'F_TradeSize_Sell_%s' % symbol_name,
                                 'F_TradeSize_Buy_%s' % symbol_name,
                                 'F_TradeThru_Sell_%s' % symbol_name,
                                 'F_TradeThru_Buy_%s' % symbol_name]].count().tolist()
                size_data = d[(d['time'] - order_time >= trigger_lower_limit) &
                              (d['time'] - order_time <= trigger_upper_limit)][[
                                  'F_EventAskQtyDeltaPlus_%s' % symbol_name,
                                  'F_EventBidQtyDeltaPlus_%s' % symbol_name,
                                  'F_EventAskQtyDeltaMinus_%s' % symbol_name,
                                  'F_EventBidQtyDeltaMinus_%s' % symbol_name,
                                  'F_TradeSize_Sell_%s' % symbol_name,
                                  'F_TradeSize_Buy_%s' % symbol_name,
                                  'F_TradeThru_Sell_%s' % symbol_name,
                                  'F_TradeThru_Buy_%s' % symbol_name]].sum().abs().fillna(0).tolist()
                data.append((num_data + size_data))
            df_concat.append(pd.DataFrame(data, columns=col))
        df_res = pd.concat(df_concat, axis=1)
        return df_res

    def _feature_processor(self, df, selected_features, dropped_features, is_train):

        col = []
        if selected_features is None:
            features = Features.candidate_features
            for f in features:
                if 'trade' in f:
                    col.append('%s_%s' % (f, self.trade_symbol.split('.')[0]))
                elif 'trigger' in f:
                    for symbol in set([self.trade_symbol] + self.trigger_symbols):
                        col.append('%s_%s' % (f, symbol.split('.')[0]))
            if dropped_features:
                for f in dropped_features:
                    col.remove(f)

        else:
            col = selected_features

        for c in col:
            if c not in df.columns:
                col.remove(c)
                print 'Not account for %s, no data for this features' % c
        df_learn = df[col]

        if is_train:
            df_learn.loc[:, 'date'] = df['date']
            df_learn.loc[:, 'side'] = df['side']
            df_learn.loc[:, 'time_new'] = df['time_new']
            df_learn.loc[:, 'corrected_flag'] = df['corrected_flag']
            if 'corrected_latency' in df.columns:
                df_learn.loc[:, 'corrected_latency'] = df['corrected_latency']
            else:
                df_learn.loc[:, 'corrected_latency'] = df[
                    'time_response'] - df['time_new']
            df_learn = df_learn[df_learn['corrected_latency'] > 0]

        df_learn = df_learn.astype(float)
        return df_learn

    def _runSampler(self, date, colo, symbol_mode=None, rerun=False):
        #         res = selectDatesAndFiles(
        #             [self.trade_symbol] + self.trigger_symbols, [date], '%s_colo' % colo.lower())
        res = get_file_selection(
            [self.trade_symbol] + self.trigger_symbols, date, colo.lower())
        if not res:
            print 'No Data for the symbol %s , colo %s, date %s' % (self.trade_symbol, colo, str(date))
            return None
        df = {}
        for symbol in set([self.trade_symbol] + self.trigger_symbols):
            symbol_name = symbol.split('.')[0]
            ask = gs.FeatureAskPrice(symbol)
            bid = gs.FeatureBidPrice(symbol)
#             ask_price_delta = gs.FeatureEventDelta(ask, gs.FeatureEventEOP())
#             bid_price_delta = gs.FeatureEventDelta(bid, gs.FeatureEventEOP())
            ask_delta = gs.FeatureEventAskQtyDelta(
                gs.FeatureAskQty(symbol), ask, gs.FeatureEventEOP())
            bid_delta = gs.FeatureEventBidQtyDelta(
                gs.FeatureBidQty(symbol), bid, gs.FeatureEventEOP())
            cons = gs.FeatureConstant(0)
#             ask_price_delta_up = gs.FeatureMax(cons, ask_price_delta)
#             bid_price_delta_down = gs.FeatureMin(cons, bid_price_delta)
            ask_delta_accum = gs.FeatureMax(cons, ask_delta)
            bid_delta_accum = gs.FeatureMax(cons, bid_delta)
#             ask_delta_cancel = gs.FeatureMin(cons, ask_delta)
            ask_delta_decre = gs.FeatureMin(cons, ask_delta)
            buy_delta = gs.FeatureBuyAggrTradeDelta(symbol)
            ask_delta_cancel = gs.FeatureMin(
                cons, gs.FeaturePlus(ask_delta_decre, buy_delta))
#             bid_delta_cancel = gs.FeatureMin(cons, bid_delta)
            bid_delta_decre = gs.FeatureMin(cons, bid_delta)
            sell_delta = gs.FeatureSellAggrTradeDelta(symbol)
            bid_delta_cancel = gs.FeatureMin(
                cons, gs.FeaturePlus(bid_delta_decre, sell_delta))
            trd_sz = gs.FeatureTradeSize(symbol)
            trd_sd = gs.FeatureTradeSide(symbol)
            trd_sz_sell = gs.FeatureMin(
                cons, gs.FeatureMultiply(trd_sz, trd_sd))
            trd_sz_buy = gs.FeatureMax(
                cons, gs.FeatureMultiply(trd_sz, trd_sd))
            trd_through = gs.FeatureTradeThruLevels(symbol)
            trd_through_sell = gs.FeatureMin(cons, trd_through)
            trd_through_buy = gs.FeatureMax(cons, trd_through)

#             ask_price_delta_up.name = 'F_AskPrice_Up_%s' % symbol_name
#             bid_price_delta_down.name = 'F_BidPrice_Down_%s' % symbol_name
            ask_delta_accum.name = 'F_EventAskQtyDeltaPlus_%s' % symbol_name
            bid_delta_accum.name = 'F_EventBidQtyDeltaPlus_%s' % symbol_name
            ask_delta_cancel.name = 'F_EventAskQtyDeltaMinus_%s' % symbol_name
            bid_delta_cancel.name = 'F_EventBidQtyDeltaMinus_%s' % symbol_name
            trd_sz_sell.name = 'F_TradeSize_Sell_%s' % symbol_name
            trd_sz_buy.name = 'F_TradeSize_Buy_%s' % symbol_name
            trd_through_sell.name = 'F_TradeThru_Sell_%s' % symbol_name
            trd_through_buy.name = 'F_TradeThru_Buy_%s' % symbol_name
            features = [
                #                         ask_price_delta_up, bid_price_delta_down,
                ask_delta_accum, bid_delta_accum,
                ask_delta_cancel, bid_delta_cancel,
                trd_sz_sell, trd_sz_buy,
                trd_through_sell, trd_through_buy]
            s = gs.SamplerRunner(
                features=features, colo=colo, dates=[date], symbol_mode=symbol_mode)
            if rerun:
                s.rerun()
            else:
                s.run()
            if date in s.dates_with_caches:
                df[symbol_name] = s.load_data_df()
                print 'sampler dataframe of %s has been loaded' % symbol_name
            else:
                print 'sampler dataframe of %s is empty' % symbol_name
                return None
        return df

    def _latency_in_log(self, date, colo, colo_server, symbol_mode=None, latency_correction=True, minimum_latency=300, search_end=300,
                        IOC_response_to_ack_limit=800,
                        #                         Limit_response_to_ack_limit=5000,
                        minimum_samples_for_analysis=6,
                        rerun=False, force_log_file=None):
        time_adjustment = timeAdjustment(colo, date)
        eod_path = Repo.getEodPath()
        if force_log_file:
            stgy_log_path = '%s/%s/strategy'
        stgy_log_path = (
            '%s/%s/strategy/stgy_log.ACTIVE_%s_%s.%s.txt' % (eod_path, date, self.trade_symbol.split('.')[0], colo_server.upper(), date))
#         stgy_log_path = '/data/prod/eod/%s/strategy/stgy_log.ACTIVE_%s_%s.%s.txt' % (
#             date, self.trade_symbol.split('.')[0], colo_server.upper(), date)
        if not os.path.exists(stgy_log_path):
            print("Stgy Log file not exist for %s" % stgy_log_path)
            return None
        try:
            df = parseStrategyLog(stgy_log_path, date)
        except:
            print("Stgy Log file for %s is not parsed correctly" %
                  stgy_log_path)
            return None

        if df is None:
            return None
        order_new_time = df[df['MSG_TYPE'] == 'ORDER_NEW']['TIME'].tolist()
        print "The length of Order_New is %d" % len(order_new_time)
        if len(order_new_time) < minimum_samples_for_analysis:
            print "Two few samples (%d samples), skip the statistics" % len(order_new_time)
            return None

        order_new_time_adjust = [i - time_adjustment for i in order_new_time]
        side_string = df[df['MSG_TYPE'] == 'ORDER_NEW']['SIDE'].tolist()
        order_side = [
            1 if i == 'BUY' else -1 if i == 'SELL' else 0 for i in side_string]
        order_ids = df[df['MSG_TYPE'] == 'ORDER_NEW']['ORDER_ID'].tolist()
        order_qty = df[df['MSG_TYPE'] == 'ORDER_NEW']['QTY'].tolist()
        order_price = df[df['MSG_TYPE'] == 'ORDER_NEW']['PRICE'].tolist()

        if 'ORDER_TYPE' in df.columns:
            order_type = df[df['MSG_TYPE'] == 'ORDER_NEW'][
                'ORDER_TYPE'].tolist()
        else:
            order_type = ['IOC'] * len(order_ids)

        order_ack_time = []
        order_response_time = []
        filled_qty = []
        filled_price = []
        for i in range(len(order_ids)):
            oid = order_ids[i]
#             otype = order_type[i]
            tmp_df = df[df['ORDER_ID'] == oid]

            tmp_df_acked = tmp_df[tmp_df['MSG_TYPE'] == 'ORDER_ACKED']
            if len(tmp_df_acked) > 0:
                ack_time = tmp_df_acked['TIME'][0]
            else:
                ack_time = 0

            tmp_df_response = tmp_df[(tmp_df['MSG_TYPE'] == 'ORDER_FILLED') | (
                tmp_df['MSG_TYPE'] == 'ORDER_CANCELED')]
            if len(tmp_df_response) > 0:
                response_time = tmp_df_response['TIME'][0]
                f_qty = tmp_df_response[
                    tmp_df_response['MSG_TYPE'] == 'ORDER_FILLED']['QTY'].sum()
                if len(tmp_df_response[tmp_df_response['MSG_TYPE'] == 'ORDER_FILLED']) > 0:
                    f_price = tmp_df_response[
                        tmp_df_response['MSG_TYPE'] == 'ORDER_FILLED']['PRICE'][0]
                else:
                    f_price = 0
            else:
                response_time = 0
                f_qty = 0
                f_price = 0

            order_response_time.append(response_time)
            order_ack_time.append(ack_time)
            filled_qty.append(f_qty)
            filled_price.append(f_price)

        order_ack_time_adjust = [i - time_adjustment for i in order_ack_time]
        order_response_time_adjust = [
            i - time_adjustment for i in order_response_time]

        if latency_correction:
            if 'LIMIT_ORDER' not in order_type:
                order_response_time_corrected, corrected_flag = self._latency_correction_IOC(colo, date,
                                                                                             order_new_time_adjust,
                                                                                             order_response_time_adjust,
                                                                                             order_type, order_side, filled_qty, filled_price,
                                                                                             minimum_latency, search_end,
                                                                                             symbol_mode, rerun)
            else:
                order_response_time_corrected, corrected_flag = self._latency_correction_LIMIT(colo, date,
                                                                                               order_new_time_adjust,
                                                                                               order_ack_time_adjust,
                                                                                               order_response_time_adjust,
                                                                                               order_type, order_price, order_qty, order_side,
                                                                                               filled_qty, filled_price,
                                                                                               minimum_latency, search_end,
                                                                                               IOC_response_to_ack_limit,
                                                                                               symbol_mode, rerun)
        df = pd.DataFrame({'date': [date] * len(order_ids),
                           'time_new': order_new_time_adjust, 'time_response': order_response_time_corrected,
                           'corrected_flag': corrected_flag, 'side': side_string,
                           'corrected_latency': [i - j for (i, j) in zip(order_response_time_corrected, order_new_time_adjust)]
                           })
        df = df.replace(['BUY', 'SELL'], [1, 0]).astype(float)
        return df

    def _latency_correction_IOC(self, colo, date, order_new_time_adjust, order_response_time_adjust,
                                order_type, order_side, filled_qty, filled_price, minimum_latency, search_end, symbol_mode, rerun):
        p = gs.FeatureTradePrice(self.trade_symbol)
        q = gs.FeatureTradeSize(self.trade_symbol)
        sd = gs.FeatureTradeSide(self.trade_symbol)
        s = gs.SamplerRunner(features=[p, q, sd],
                             colo=colo, dates=[date], symbol_mode=symbol_mode)
        trade_symbol_name = self.trade_symbol.split('.')[0]
        p.name = 'F_Trade_Price_%s' % trade_symbol_name
        q.name = 'F_Trade_Size_%s' % trade_symbol_name
        sd.name = 'F_Trade_Side_%s' % trade_symbol_name
        if rerun:
            s.rerun()
        else:
            s.run()
        df = s.load_data_df()
        order_response_time_corrected = [0] * len(order_response_time_adjust)
        corrected_flag = [0] * len(order_response_time_adjust)

        tick_size = gdb.getTickSize(self.trade_symbol.split('_')[0], date)
        epsilon = tick_size / 2.0

        for i in range(len(order_new_time_adjust)):
            o_new_time = order_new_time_adjust[i]
            o_response_time = order_response_time_adjust[i]
            o_side = order_side[i]
            f_qty = filled_qty[i]
            f_price = filled_price[i]
            if o_response_time <= 0:
                continue

            tmp_df = df[(df['time'] > o_new_time + minimum_latency) &
                        (df['time'] < o_response_time + search_end)]
            order_fill_time = tmp_df[((tmp_df['F_Trade_Price_%s' % trade_symbol_name] - f_price).abs() < epsilon) &
                                     (tmp_df['F_Trade_Side_%s' % trade_symbol_name] == o_side) &
                                     (tmp_df['F_Trade_Size_%s' % trade_symbol_name] == f_qty)]['time'].tolist()
            tm = self._locate_corrected_time(
                order_fill_time, None, o_response_time)
            flag = 1
            if tm == -1:
                tm = o_response_time
                flag = 0
            order_response_time_corrected[i] = tm
            corrected_flag[i] = flag

        return (order_response_time_corrected, corrected_flag)

    def _latency_correction_LIMIT(self, colo, date, order_new_time_adjust, order_ack_time_adjust, order_response_time_adjust,
                                  order_type, order_price, order_qty, order_side, filled_qty, filled_price,
                                  minimum_latency, search_end, IOC_response_to_ack_limit,
                                  symbol_mode, rerun):
        ask = gs.FeatureAskPrice(self.trade_symbol)
        ask_delta = gs.FeatureEventAskQtyDelta(
            gs.FeatureAskQty(self.trade_symbol), ask, gs.FeatureEventEOP())
        bid = gs.FeatureBidPrice(self.trade_symbol)
        bid_delta = gs.FeatureEventBidQtyDelta(
            gs.FeatureBidQty(self.trade_symbol), bid, gs.FeatureEventEOP())
        p = gs.FeatureTradePrice(self.trade_symbol)
        q = gs.FeatureTradeSize(self.trade_symbol)
        sd = gs.FeatureTradeSide(self.trade_symbol)
        s = gs.SamplerRunner(features=[ask, ask_delta, bid, bid_delta, p, q, sd], colo=colo, dates=[
                             date], symbol_mode=symbol_mode)
        trade_symbol_name = self.trade_symbol.split('.')[0]
        ask.name = 'F_Ask_Price_%s' % trade_symbol_name
        ask_delta.name = 'F_Ask_Delta_%s' % trade_symbol_name
        bid.name = 'F_Bid_Price_%s' % trade_symbol_name
        bid_delta.name = 'F_Bid_Delta_%s' % trade_symbol_name
        p.name = 'F_Trade_Price_%s' % trade_symbol_name
        q.name = 'F_Trade_Size_%s' % trade_symbol_name
        sd.name = 'F_Trade_Side_%s' % trade_symbol_name
        if rerun:
            s.rerun()
        else:
            s.run()
        df = s.load_data_df()
        order_response_time_corrected = [0] * len(order_response_time_adjust)
        corrected_flag = [0] * len(order_response_time_adjust)

        tick_size = gdb.getTickSize(self.trade_symbol.split('_')[0], date)
        epsilon = tick_size / 2.0

        for i in range(len(order_new_time_adjust)):
            o_price = order_price[i]
            o_qty = order_qty[i]
            o_side = order_side[i]
            o_type = order_type[i]
            f_qty = filled_qty[i]
            f_price = filled_price[i]
            o_new_time = order_new_time_adjust[i]
            o_ack_time = order_ack_time_adjust[i]
            o_response_time = order_response_time_adjust[i]
            if o_response_time <= 0:
                continue

            if o_type == 'IOC':
                tmp_df = df[(df['time'] > o_new_time + minimum_latency) &
                            (df['time'] < o_response_time + search_end)]
                order_fill_time = tmp_df[((tmp_df['F_Trade_Price_%s' % trade_symbol_name] - f_price).abs() < epsilon) &
                                         (tmp_df['F_Trade_Side_%s' % trade_symbol_name] == o_side) &
                                         (tmp_df['F_Trade_Size_%s' % trade_symbol_name] == f_qty)]['time'].tolist()
                tm = self._locate_corrected_time(
                    order_fill_time, None, o_response_time)
                flag = 1
                if tm == -1:
                    tm = o_response_time
                    flag = 0

            elif o_type == 'LIMIT_ORDER':
                tmp_df = df[(df['time'] > o_new_time + minimum_latency) &
                            (df['time'] < o_ack_time + search_end)]
                if o_side == 1:
                    book_add_time = tmp_df[((tmp_df['F_Bid_Price_%s' % trade_symbol_name] - o_price).abs() < epsilon) & (
                        tmp_df['F_Bid_Delta_%s' % trade_symbol_name] == o_qty)]['time'].tolist()
                elif o_side == -1:
                    book_add_time = tmp_df[((tmp_df['F_Ask_Price_%s' % trade_symbol_name] - o_price).abs() < epsilon) & (
                        tmp_df['F_Ask_Delta_%s' % trade_symbol_name] == o_qty)]['time'].tolist()

                order_fill_time = None
                if o_response_time - o_ack_time < IOC_response_to_ack_limit:
                    order_fill_time = tmp_df[((tmp_df['F_Trade_Price_%s' % trade_symbol_name] - f_price).abs() < epsilon) &
                                             (tmp_df['F_Trade_Side_%s' % trade_symbol_name] == o_side) &
                                             (tmp_df['F_Trade_Size_%s' % trade_symbol_name] == f_qty)]['time'].tolist()
                tm = self._locate_corrected_time(
                    order_fill_time, book_add_time, o_ack_time)
                flag = 1
                if tm == -1:
                    tm = (
                        o_new_time + minimum_latency + o_ack_time + search_end) / 2.0
                    flag = 0
            order_response_time_corrected[i] = tm
            corrected_flag[i] = flag
        return (order_response_time_corrected, corrected_flag)

    def _locate_corrected_time(self, order_fill_time, book_add_time, ref_time):
        candidate_time = []
        if order_fill_time:
            candidate_time += order_fill_time
        if book_add_time:
            candidate_time += book_add_time
        if not candidate_time:
            return -1
        time_diff = [abs(i - ref_time) for i in candidate_time]
        ind = time_diff.index(min(time_diff))
        tm = candidate_time[ind]
        return tm

    def _colorMatrix(self, m, corr_threshold):
        m_mask = np.ma.masked_where(np.isnan(m), m)
        plt.ion()
        plt.pcolor(m_mask)
        plt.colorbar()
        labels = [str(x) for x in m.columns]
        plt.xticks(range(len(m)), labels, rotation=90)
        plt.yticks(range(len(m)), labels)
        plt.ylim(0, len(m.index))
        plt.xlim(0, len(m.columns))
        if corr_threshold is None:
            plt.title('Feature Correlations')
        else:
            plt.title(
                'Feature Correlations (abs(correlation) < %.3f are masked with white)' % corr_threshold)
        plt.draw()

    def _latency_correction_Deprecated(self, colo, date, order_new_time_adjust, order_response_time_adjust,
                                       order_type, order_side, filled_qty, filled_price, symbol_mode, rerun):
        p = gs.FeatureTradePrice(self.trade_symbol)
        q = gs.FeatureTradeSize(self.trade_symbol)
        sd = gs.FeatureTradeSide(self.trade_symbol)
        s = gs.SamplerRunner(features=[p, q, sd],
                             colo=colo, dates=[date], symbol_mode=symbol_mode)
        trade_symbol_name = self.trade_symbol.split('.')[0]
        p.name = 'F_Trade_Price_%s' % trade_symbol_name
        q.name = 'F_Trade_Size_%s' % trade_symbol_name
        sd.name = 'F_Trade_Side_%s' % trade_symbol_name
        if rerun:
            s.rerun()
        else:
            s.run()
        trd_df = s.load_data_df()
        order_response_time_corrected = [0] * len(order_response_time_adjust)
        corrected_flag = [0] * len(order_response_time_adjust)
        for i in range(len(order_new_time_adjust)):
            o_new_time = order_new_time_adjust[i]
            o_response_time = order_response_time_adjust[i]
            if order_type[i] is 'LIMIT_ORDER':
                order_response_time_corrected[i] = o_response_time
                continue

            o_side = order_side[i]
            o_fill = filled_qty[i]
            fill_price = filled_price[i]
            time_left = 0
            time_right = 0
            tmp_df = trd_df[
                (trd_df['time'] > o_new_time + 300) & (trd_df['time'] <= o_response_time)][['F_Trade_Price_%s' % trade_symbol_name,
                                                                                            'F_Trade_Size_%s' % trade_symbol_name,
                                                                                            'F_Trade_Side_%s' % trade_symbol_name,
                                                                                            'time']].dropna()
            if len(tmp_df > 0):
                actual_fill_time = tmp_df[(tmp_df['F_Trade_Price_%s' % trade_symbol_name] == fill_price) &
                                          (tmp_df['F_Trade_Side_%s' % trade_symbol_name] == o_side) &
                                          (tmp_df['F_Trade_Size_%s' % trade_symbol_name] == o_fill)]['time']
                if len(actual_fill_time) > 0:
                    if len(actual_fill_time) <= 3:
                        time_left = actual_fill_time.iloc[-1]

            tmp_df = trd_df[
                (trd_df['time'] > o_response_time) & (trd_df['time'] < o_response_time + 300)][['F_Trade_Price_%s' % trade_symbol_name,
                                                                                                'F_Trade_Size_%s' % trade_symbol_name,
                                                                                                'F_Trade_Side_%s' % trade_symbol_name,
                                                                                                'time']].dropna()
            if len(tmp_df > 0):
                actual_fill_time = tmp_df[(tmp_df['F_Trade_Price_%s' % trade_symbol_name] == fill_price) &
                                          (tmp_df['F_Trade_Side_%s' % trade_symbol_name] == o_side) &
                                          (tmp_df['F_Trade_Size_%s' % trade_symbol_name] == o_fill)]['time']
                if len(actual_fill_time) > 0:
                    time_right = actual_fill_time.iloc[0]

            if time_left == 0:
                order_response_time_corrected[i] = o_response_time
            elif time_right == 0:
                order_response_time_corrected[i] = time_left
                corrected_flag[i] = 1
            elif o_response_time - time_left <= abs(time_right - o_response_time):
                order_response_time_corrected[i] = time_left
                corrected_flag[i] = 1
            else:
                order_response_time_corrected[i] = o_response_time

        return (order_response_time_corrected, corrected_flag)