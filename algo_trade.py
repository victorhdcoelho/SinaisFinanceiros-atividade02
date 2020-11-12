import pandas as pd
from tqdm import tqdm
# import matplotlib.pyplot as plt


# class CrossOverInvest:
#    def __init__(self, df, windown_s, windown_l):
#        self.df = df
#        self.windown_s = windown_s
#        self.windown_l = windown_l
#        self.short = None
#        self.long = None
#        self.long_short = None
#        self.long_short_diff = None
#        self.all_long_short = {}
#
#    def create_long_short(self):
#        self.short = self.df.rolling(
#            window=self.windown_s).mean()[self.windown_l:]
#
#        self.long = self.df.rolling(
#            window=self.windown_l).mean()[self.windown_l:]
#        self.short = self.short.dropna()
#        self.long = self.long.dropna()
#
#    def plot_long_short(self):
#        for each in self.df.columns:
#            ativo = each
#            _ = plt.subplots(figsize=(20, 12))
#            plt.style.use('seaborn-whitegrid')
#            plt.plot(self.short.index, self.short[ativo],
#                     color='green', label='Short')
#            plt.plot(self.long.index, self.long[ativo],
#                     color='b', label='Long')
#            plt.title("Médias {}".format(each))
#            plt.legend(loc='upper left', fontsize=20)
#            plt.show()
#
#    def create_empty_long_short_list(self):
#        long_short = {}
#        for each in self.short.columns:
#            long_short[each] = [0 for i in range(self.short.shape[0])]
#        long_short = pd.DataFrame(data=long_short)
#        self.long_short = long_short
#
#    def compare_long_short(self):
#        for each in tqdm(self.short.columns):
#            for i in range(self.short.shape[0]):
#                if self.short[each].iloc[i] > self.long[each].iloc[i]:
#                    self.long_short[each].loc[i] = 1
#                else:
#                    self.long_short[each].loc[i] = 0
#        self.long_short.index = self.short.index
#        self.long_short_diff = self.long_short.diff()
#
#    def plot_buy_sell_long_short(self):
#        for each in self.df.columns:
#            ativo = each
#            _ = plt.subplots(figsize=(20, 15))
#            plt.style.use('seaborn-whitegrid')
#            plt.plot(self.short[ativo], label='Short')
#            plt.plot(self.long[ativo], label='Long')
#            plt.ylabel("Preço [R$]", fontsize=25)
#            plt.xlabel("Data", fontsize=25)
#            plt.title("Médias {}".format(each), fontsize=25)
#            plt.plot(self.short[ativo][
#                self.long_short_diff[ativo] == 1.0].index,
#                    self.short[ativo][self.long_short_diff[ativo] == 1.0],
#                    '^', markersize=10, color='m', label='Buy')
#            plt.plot(self.short[ativo][
#                self.long_short_diff[ativo] == -1.0].index,
#                    self.short[ativo][self.long_short_diff[ativo] == -1.0],
#                    'v', markersize=10, color='k', label='Sell')
#            plt.legend(loc='upper left', fontsize=25)
#            plt.show()
#
#    def set_action_long_short(self):
#        for ativo in tqdm(self.df.columns):
#            self.long_short_diff[ativo].loc[
#                self.long_short_diff[ativo] == 1.0] = "buy"
#            self.long_short_diff[ativo].loc[
#                self.long_short_diff[ativo] == -1.0] = "sell"
#        self.long_short_diff = self.long_short_diff.dropna()
#
#    def get_all_long_short_info(self):
#        for each in self.df.columns:
#            ativo_df = self.long_short_diff[
#                [each]][self.long_short_diff[each] != 0]
#            ativo_df = ativo_df.rename(columns={each: "action"})
#            ativo_df["close"] = self.df[each][self.df[each].index.isin(
#                ativo_df.index)]
#            ativo_df["return"] = ativo_df["close"].diff()/100
#            self.all_long_short[each] = ativo_df
#
#    def get_amount(self, capital, pesos=[0.08, 0.39, 0.05, 0.35, 0.12]):
#        print(capital)
#        for each, p in tqdm(zip(self.df.columns, pesos)):
#            capital = capital * p
#            print(capital)
#            montante = []
#            montante.append(capital)
#            for re in self.all_long_short[each]["return"][1:]:
#                capital = capital * (1 + re)
#                montante.append(capital)
#            self.all_long_short[each]["montante"] = montante
#
#    def report_active_long_short_result(self):
#        total = 0
#        for each in self.all_long_short.keys():
#            initial = self.all_long_short[each].head(1)["montante"][0]
#            final = initial = self.all_long_short[each].tail(1)["montante"][0]
#            taxa = (10 * self.all_long_short[each].shape[0])
#            result = (final-initial)-taxa
#            total += result
#            print("{}: {}".format(each, round(result, 2)))
#        print("Total de lucro da carteira: {}".format(round(total, 2)))
#
#    def run(self, capital, pesos=[0.08, 0.39, 0.05, 0.35, 0.12]):
#        self.create_long_short()
#        # self.plot_long_short()
#        self.create_empty_long_short_list()
#        self.compare_long_short()
#        # self.plot_buy_sell_long_short()
#        self.set_action_long_short()
#        self.get_all_long_short_info()
#        self.get_amount(capital, pesos)
#        self.report_active_long_short_result()


def long_short_invest(df, window_s, window_l, capital,
                      peso=[0.08, 0.39, 0.05, 0.35, 0.12]):
    short = df.rolling(window=window_s).mean()[window_l:]
    long = df.rolling(window=window_l).mean()[window_l:]
    short = short.dropna()
    long = long.dropna()

    long_short = {}
    for each in short.columns:
        long_short[each] = [0 for i in range(short.shape[0])]
    long_short = pd.DataFrame(data=long_short)

    for each in tqdm(short.columns):
        for i in range(short.shape[0]):
            if short[each].iloc[i] > long[each].iloc[i]:
                long_short[each].loc[i] = 1
            else:
                long_short[each].loc[i] = 0

    long_short.index = short.index
    long_short_diff = long_short.diff()

    for ativo in tqdm(df.columns):
        long_short_diff[ativo].loc[long_short_diff[ativo] == 1.0] = "buy"
        long_short_diff[ativo].loc[long_short_diff[ativo] == -1.0] = "sell"
    long_short_diff = long_short_diff.dropna()
    all_actives = {}
    for each in df.columns:
        ativo_df = long_short_diff[[each]][long_short_diff[each] != 0]
        ativo_df = ativo_df.rename(columns={each: "action"})
        ativo_df["close"] = df[each][df[each].index.isin(ativo_df.index)]
        ativo_df["return"] = ativo_df["close"].diff() / 100
        all_actives[each] = ativo_df

    for each, p in tqdm(zip(df.columns, peso)):
        capital = 100000 * p
        montante = []
        montante.append(capital)
        for re in all_actives[each]["return"][1:]:
            capital = capital * (1 + re)
            montante.append(capital)
        all_actives[each]["montante"] = montante

    total = 0
    for each in all_actives.keys():
        result = (all_actives[each][
            "montante"].iloc[all_actives[each].shape[0]-1] -
            all_actives[each]["montante"].iloc[0]) -\
            10 * all_actives[each].shape[0]
        total += result
        print("{}: {}".format(each, round(result, 2)))
    print("Total de lucro da carteira: {}".format(round(total, 2)))
    return short, long, long_short, long_short_diff, all_actives
