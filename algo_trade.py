import pandas as pd
from tqdm import tqdm


def get_ema_cols(df, short, long):
    short = df.ewm(ignore_na=False, min_periods=short, com=short, adjust=True).mean()
    long = df.ewm(ignore_na=False, min_periods=long, com=long, adjust=True).mean()
    return short, long

def get_macd(df, signal):
    macd = df["short"] - df["long"]
    macd_signal= macd.ewm(ignore_na=False, min_periods=0, com=signal, adjust=True).mean()
    return macd, macd_signal

def get_montante(df, capital):
    montante = []
    montante.append(capital)
    for each in df["return"][1:]:
        capital = capital * (1 + each)
        montante.append(capital)
    return montante

def get_return_info(df, ativo, capital):
    temp_df = df[df["position"] != 0]
    temp_df["return"] = temp_df[ativo].diff() /100
    temp_df["montante"] = get_montante(temp_df, capital)
    acertos_compra = temp_df[(temp_df["position"] == 1.0) & (temp_df["return"] <= 0)].shape[0]
    acertos_venda = temp_df[(temp_df["position"] == -1.0) & (temp_df["return"] > 0)].shape[0]
    result = (temp_df["montante"].tail(1)[0] - temp_df["montante"].head(1)[0]) - (10 * temp_df.shape[0])
    print("ATIVO: {} RETORNO: {} ACERTO BUY: {} ACERTO SELL: {} TOTAL TRADES: {} PERC_ACERTO: {}".format(
        ativo, round(result, 2), acertos_compra, acertos_venda, temp_df.shape[0],
        ((acertos_compra + acertos_venda)/temp_df.shape[0])*100))
    accs = acertos_compra + acertos_venda
    trades = temp_df.shape[0]
    return result, accs, trades

def get_new_invest(df, posi, capital):
    total = 0
    total_acc = 0
    total_trades = 0
    for each, pos in zip(df.keys(), posi):
        parte = capital * pos
        result, accs, trades = get_return_info(df[each], each, parte)
        total += result
        total_acc += accs
        total_trades += trades
    print("Total de lucro da carteira: {}\nMédia de acerto: {}%".format(round(total, 2), round((total_acc/total_trades)*100, 2)))

def get_returns_info_new(temp_df, ativo):
    acertos_compra = temp_df[(temp_df["position"] == 1.0) & (temp_df["return"] <= 0)].shape[0]
    acertos_venda = temp_df[(temp_df["position"] == -1.0) & (temp_df["return"] > 0)].shape[0]
    result = (temp_df["montante"].tail(1)[0] - temp_df["montante"].head(1)[0]) - (10 * temp_df.shape[0])
    print("ATIVO: {} RETORNO: {} ACERTO BUY: {} ACERTO SELL: {} TOTAL TRADES: {} PERC_ACERTO: {}".format(
        ativo, round(result, 2), acertos_compra, acertos_venda, temp_df.shape[0],
        ((acertos_compra + acertos_venda)/temp_df.shape[0])*100))
    accs = acertos_compra + acertos_venda
    trades = temp_df.shape[0]
    return result, accs, trades

def get_info_invest(df, capital, posi):
    total = 0
    total_acc = 0
    total_trades = 0
    for each, pos in zip(df.keys(), posi):
        parte = capital * pos
        result, accs, trades = get_return_info(df[each], each, parte)
        total += result
        total_acc += accs
        total_trades += trades
    print("Total de lucro da carteira: {}\nMédia de acerto: {}%".format(round(total, 2), round((total_acc/total_trades)*100, 2)))

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
