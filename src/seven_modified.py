import pandas as pd
import numpy as np
from jqdata import *
import datetime as dt
import statsmodels.api as sm
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

#########################################
#工具函数 
#########################################
def get_month_end_trading_days(start_date, end_date):
    """
    获取区间内每个月最后一个交易日
    """
    trade_days = list(get_trade_days(start_date=start_date, end_date=end_date))
    month_ends = []
    prev_month = None

    for i, d in enumerate(trade_days):
        cur_month = (d.year, d.month)
        next_month = None if i == len(trade_days) - 1 else (trade_days[i + 1].year, trade_days[i + 1].month)
        if cur_month != next_month:
            month_ends.append(d)

    return month_ends
    
def get_hs300_universe(date):
    """
    获取某日沪深300成分股
    """
    return list(get_index_stocks('000300.XSHG', date=date))

def safe_log(x):
    x = pd.Series(x)
    x = x.replace([np.inf, -np.inf], np.nan)
    x[x <= 0] = np.nan
    return np.log(x)

def exp_weights(n, half_life):
    """
    生成长度为 n 的指数衰减权重，越近权重越大
    """
    idx = np.arange(n)
    w = 0.5 ** ((n - 1 - idx) / half_life)
    return w / w.sum()

# 抓取最原始的数据
def get_price_panel(stocks, end_date, count=260, fields=None):
    """
    拉取过去 count 个交易日的行情面板
    返回 dict: field -> DataFrame(index=date, columns=code)
    形如
    {
    "close":  DataFrame(日期×股票),
    "high":   DataFrame(日期×股票),
    "low":    DataFrame(日期×股票),
    "volume": DataFrame(日期×股票)
    }
    如果需要close，就写panel['close']
    """
    if fields is None:
        fields = ['close', 'high', 'low', 'volume']

    panel = get_price(
        stocks,
        end_date=end_date,
        count=count,
        frequency='daily',
        fields=fields,
        skip_paused=False,
        fq='pre'
    )
    return panel
    
def get_return_panel(close_df):
    return close_df.pct_change()

def get_index_return_series(index_code, end_date, count=260):
    """
    获取沪深三百过去一段时间的每日涨跌幅序列，count=260表示默认取一年
    """
    idx_close = get_price(
        index_code,
        end_date=end_date,
        count=count,
        frequency='daily',
        fields=['close'],
        fq='pre'
    )['close']
    return idx_close.pct_change()
    
def get_valuation_snapshot(stocks, date):
    """
    获取某日估值截面，index=code，column=pb_ratio,market_cap，circulating_market_cap和turnover_ratio
    """
    q = query(
        valuation.code,
        valuation.pb_ratio,
        valuation.pe_ratio,
        valuation.ps_ratio,
        valuation.pcf_ratio,
        valuation.market_cap,
        valuation.circulating_market_cap,
        valuation.turnover_ratio
    ).filter(
        valuation.code.in_(stocks)
    )
    df = get_fundamentals(q, date=date)
    df = df.set_index('code')
    return df
    
def get_valuation_panel(stocks, end_date, count, fields):
    """
    获取过去 count 个交易日的历史估值面板
    用于 turnover_ratio 等日频估值字段
    返回 dict[field] = DataFrame(index=day, columns=code)
    即{
    'pb_ratio': DataFrame(行=天, 列=股票),
    'market_cap': DataFrame(行=天, 列=股票),
    'turnover_ratio': DataFrame(行=天, 列=股票)
    }
    """
    trade_days = get_trade_days(end_date=end_date, count=count)
    start_date = trade_days[0]
    end_date = trade_days[-1]
    val = get_valuation(
        stocks,
        start_date=start_date,
        end_date=end_date,
        fields=['code', 'day'] + fields
    )
    if val is None or len(val) == 0:
        raise ValueError("get_valuation 返回为空")
    val['day'] = pd.to_datetime(val['day'])

    out = {}
    for f in fields:
        out[f] = val.pivot(index='day', columns='code', values=f).sort_index()
    return out
    
def get_industry_factor(stocks, date):
    """
    行业因子：先取行业编码/名称
    中信行业在聚宽有时不稳定，先写成通用接口。
    如果你环境里能直接拿到中信一级，就优先用中信。
    否则先用申万一级替代，后面再切。
    """
    ind = get_industry(stocks, date=date)

    records = []
    for s in stocks:
        info = ind.get(s, {})
        # 优先尝试申万一级
        if 'sw_l1' in info and info['sw_l1'] is not None:
            industry_name = info['sw_l1']['industry_name']
        elif 'zjw' in info and info['zjw'] is not None:
            industry_name = info['zjw']['industry_name']
        else:
            industry_name = 'UNKNOWN'
        records.append((s, industry_name))

    df = pd.DataFrame(records, columns=['code', 'industry']).set_index('code')
    return df

def get_financial_snapshot_v1(stocks, date):
    """
    直接使用聚宽当前date可获得的最新财务字段
    """
    q = query(
        valuation.code,
        indicator.roe,
        # 营业收入同比
        indicator.inc_total_revenue_year_on_year,
        # 归母净利润同比
        indicator.inc_net_profit_to_shareholders_year_on_year,
        # 净利润同比
        indicator.inc_net_profit_year_on_year
    ).filter(
        valuation.code.in_(stocks)
    )
    df = get_fundamentals(q, date=date)
    df = df.set_index('code')
    return df


# =========================
# 构造因子
# =========================

# =========================
# 价值因子
# =========================
def calc_value_factors(val_snapshot,industry_df=None):
    """
    构造价值类因子。
    当前可严格构造：
    - ETOP: 1 / PE，近似历史EP
    - BTOP: 1 / PB，等价于 BP
    - stop1: 1 / PS，市销率 TTM 倒数
    - ocfp: 1 / PCF，经营现金流/市值的近似版本
    - ep_rel: PE / 行业PE
    - bp_rel: PB / 行业 PB

    当前版本暂不能严格构造：
    - epcut: 扣非净利润 / 总市值
    - ncfp: 净现金流 / 总市值
    - dividendyield: 过去一年分红 / 总市值
    - stop2: LYR 营收 / 总市值
    - PEG: PE / 盈利增长率，依赖净利润增速字段
    - EPIBS: 预期 EP，依赖分析师一致预期数据
    """
    out = {}
    pe = val_snapshot['pe_ratio'].replace([np.inf, -np.inf], np.nan)
    pb = val_snapshot['pb_ratio'].replace([np.inf, -np.inf], np.nan)
    ps = val_snapshot['ps_ratio'].replace([np.inf, -np.inf], np.nan)
    pcf = val_snapshot['pcf_ratio'].replace([np.inf, -np.inf], np.nan)
    
    ETOP = 1 / pe
    ETOP.name = 'ETOP'
    out['ETOP'] = ETOP
    
    BTOP = 1 / pb
    BTOP.name = 'BTOP'
    out['BTOP'] = BTOP
    
    stop1 = 1 / ps
    stop1.name = 'stop1'
    out['stop1'] = stop1
    
    ocfp = 1 / pcf
    ocfp.name = 'ocfp'
    out['ocfp'] = ocfp
    
    if industry_df is not None:
        tmp = pd.DataFrame({
            'pe_ratio': pe,
            'pb_ratio': pb
        }).join(industry_df, how='left')

        industry_pe = tmp.groupby('industry')['pe_ratio'].transform('median')
        industry_pb = tmp.groupby('industry')['pb_ratio'].transform('median')

        ep_rel = tmp['pe_ratio'] / industry_pe
        ep_rel = ep_rel.replace([np.inf, -np.inf], np.nan)
        ep_rel.name = 'ep_rel'
        out['ep_rel'] = ep_rel

        bp_rel = tmp['pb_ratio'] / industry_pb
        bp_rel = bp_rel.replace([np.inf, -np.inf], np.nan)
        bp_rel.name = 'bp_rel'
        out['bp_rel'] = bp_rel

    return out
    

  
# =========================
# 市值因子
# =========================  
def calc_size_factor(val_snapshot):
    """
    流通市值对数
    """
    size = safe_log(val_snapshot['circulating_market_cap'])
    size.name = 'ln_float_mktcap'
    return size
    

def calc_nonlinear_size_factor(val_snapshot):
    """
    非线性市值因子（Barra NLSIZE 的简化横截面版本）
    常见做法：先取 ln(size)，标准化后立方，再对 ln(size) 回归，取残差
    """
    x = safe_log(val_snapshot['circulating_market_cap'])
    # 标准化
    x = (x - x.mean()) / x.std()

    y = x ** 3
    # 把有有效值的股票挑出来，去掉空值
    valid = x.notna() & y.notna()
    xv = x[valid].values
    yv = y[valid].values

    X = np.column_stack([np.ones(len(xv)), xv])
    beta = np.linalg.lstsq(X, yv, rcond=None)[0]
    y_hat = X @ beta
    resid = pd.Series(index=x[valid].index, data=yv - y_hat)

    out = pd.Series(index=x.index, dtype=float)
    out.loc[resid.index] = resid
    out.name = 'nonlinear_size'
    return out

# =========================
# 波动率因子
# =========================
def calc_volume_volatility_factors(volume_df):
    """
    1/3/6/12月成交量波动率
    这里按 20/60/120/240 个交易日滚动标准差
    输出四个因子
    """
    out = {}
    windows = {
        'vol_1m': 20,
        'vol_3m': 60,
        'vol_6m': 120,
        'vol_12m': 240
    }

    for name, w in windows.items():
        fac = volume_df.iloc[-w:].std()
        fac.name = name
        out[name] = fac

    return out

def calc_yield_volatility_factors(ret_df):
    """
    波动率因子：1/3/6月日收益率波动率
    yieldvol_1: 近1个月日收益率标准差
    yieldvol_3: 近3个月日收益率标准差
    yieldvol_6: 近6个月日收益率标准差
    
    这里按 20/60/120 个交易日近似1/3/6个月。
    输入的是ret_df: index=date, columns=code
    返回 dict[name -> Series]
    """
    windows = {
        '1': 21,
        '3': 63,
        '6': 126,
    }
    out = {}
    for name, w in windows.items():
        fac = ret_df.iloc[-w:].std()
        fac.name = name
        out[name] = fac

    return out

def calc_high_low_factors(high_df, low_df):
    """
    波动率因子：1/3/6/12 月股价振幅
    high_low_1:  近1个月最高价/最低价
    high_low_3:  近3个月最高价/最低价
    high_low_6:  近6个月最高价/最低价
    high_low_12: 近12个月最高价/最低价

    这里按 20/60/120/240 个交易日近似 1/3/6/12 个月。
    输入 high_df, low_df: DataFrame(index=date, columns=code)
    返回 dict[name -> Series]
    """
    out = {}
    for name, w in windows.items():
        high_max = high_df.iloc[-w:].max()
        low_min = low_df.iloc[-w:].min()

        fac = high_max / low_min
        fac = fac.replace([np.inf, -np.inf], np.nan)
        fac.name = name
        out[name] = fac
    return out


def calc_barra_liquidity_factors(turnover_ratio_df):
    """
    turnover_ratio_df: 日频换手率面板, index=date, columns=code
    聚宽 valuation.turnover_ratio 的单位通常是百分比，需要先转成小数
    Barra 常见写法：
        STOM = log(sum 最近21日日换手率)
        STOQ = log(1/3 * sum 最近3个月STOM对应月换手率和)
        STOA = log(1/12 * sum 最近12个月STOM对应月换手率和)
    这里先做一个可运行版本：
        STOM: 最近20日日换手率之和取log
        STOQ: 最近60日日换手率之和 / 3 取log
        STOA: 最近240日日换手率之和 / 12 取log
    """

    tr = turnover_ratio_df / 100.0  # 百分比转小数
    eps = 1e-12

    stom = np.log(tr.iloc[-20:].sum() + eps)
    stoq = np.log(tr.iloc[-60:].sum() / 3.0 + eps)
    stoa = np.log(tr.iloc[-240:].sum() / 12.0 + eps)

    stom.name = 'STOM_barra'
    stoq.name = 'STOQ_barra'
    stoa.name = 'STOA_barra'
    return {
        'STOM_barra': stom,
        'STOQ_barra': stoq,
        'STOA_barra': stoa
    }

# =========================
# 反转因子：rstr_20，rstr_60，rstr_120，rstr_180，rstr_240，rstr_barra
# =========================
def calc_weighted_rstr(ret_series, lookback, half_life, lag=1):
    """
    ret_series: 单只股票日收益率Series
    特色反转因子：指数加权的历史对数收益和
    lag=1 表示跳过最近1天，避免短期微观结构噪音
    """
    s = ret_series.dropna()
    if len(s) < lookback + lag:
        return np.nan

    window = s.iloc[-(lookback + lag):-lag]
    log_ret = np.log1p(window)

    w = exp_weights(len(log_ret), half_life)
    return np.sum(log_ret.values * w)

def calc_reversal_factors(ret_df):
    """
    ret_df是所有股票的收益率数据
    返回 20 / 60 / 120 / 180 /240 日特色反转因子
    """
    specs = {
        'rstr_20': (20, 5,1),
        'rstr_60': (60, 15,1),
        'rstr_120':(120,30,1),
        'rstr_180': (180, 45,1),
        'rstr_240':(240,60,1),
        'rstr_barra':(500,120,21)
    }

    out = {}
    for fac_name, (lookback, half_life,lag) in specs.items():
        vals = {}
        for code in ret_df.columns:
            vals[code] = calc_weighted_rstr(
                ret_df[code],
                lookback=lookback,
                half_life=half_life,
                lag=lag
            )
        out[fac_name] = pd.Series(vals, name=fac_name)

    return out

# =========================
# 6. Beta因子
# =========================

def weighted_beta(stock_ret, bench_ret, half_life=60):
    """
    用指数衰减权重估计 beta
    """
    df = pd.concat([stock_ret, bench_ret], axis=1).dropna()
    if len(df) < 60:
        return np.nan

    y = df.iloc[:, 0].values
    x = df.iloc[:, 1].values

    w = exp_weights(len(df), half_life)
    x_bar = np.sum(w * x)
    y_bar = np.sum(w * y)

    cov_xy = np.sum(w * (x - x_bar) * (y - y_bar))
    var_x = np.sum(w * (x - x_bar) ** 2)

    if var_x == 0:
        return np.nan
    return cov_xy / var_x


def calc_beta_factor(ret_df, bench_ret, half_life=60):
    vals = {}
    for code in ret_df.columns:
        vals[code] = weighted_beta(ret_df[code], bench_ret, half_life=half_life)
    return pd.Series(vals, name='beta')


# =========================
# 7. 财务因子
# =========================
def calc_financial_factors_v1(fin_df):
    """
    第一版财务因子：
    - 单季度ROE：先用 roe 近似
    - 单季度营业收入增速：先用营收同比字段近似
    - 单季度归母净利润增速：先用归母净利同比字段近似
    """
    out = {}

    roe_q = fin_df['roe'].copy()
    roe_q.name = 'roe_q'
    out['roe_q'] = roe_q

    sales_yoy_q = fin_df['inc_total_revenue_year_on_year'].copy()
    sales_yoy_q.name = 'sales_yoy_q'
    out['sales_yoy_q'] = sales_yoy_q

    np_yoy_q = fin_df['inc_net_profit_to_shareholders_year_on_year'].copy()
    np_yoy_q.name = 'np_yoy_q'
    out['np_yoy_q'] = np_yoy_q

    return out
    

# =========================
# 8.总构造函数
# =========================
def build_factor_snapshot(date, index_code='000300.XSHG'):
    """
    在某一天,把沪深300所有股票的所有因子（估值、市值、流动性、动量、Beta、财务、行业）一次性全部算出来，
    index=code，column=因子值
    """
    stocks = get_hs300_universe(date)
    # 行情数据
    price_panel = get_price_panel(
        stocks,
        end_date=date,
        count=780,
        fields=['close', 'high', 'low', 'volume']
    )
    close_df = price_panel['close']
    volume_df = price_panel['volume']
    high_df = price_panel['high']
    low_df = price_panel['low']
    ret_df = get_return_panel(close_df)
    # 基准收益
    bench_ret = get_index_return_series(index_code, end_date=date, count=260)
    # 估值快照 + 历史turnover_ratio
    val_snapshot = get_valuation_snapshot(stocks, date)
    val_hist = get_valuation_panel(stocks, end_date=date, count=260, fields=['turnover_ratio'])
    turnover_ratio_df = val_hist['turnover_ratio']
      # 行业
    industry_df = get_industry_factor(stocks, date)
    # 财务
    fin_df = get_financial_snapshot_v1(stocks, date)
    # 各类因子
    factor_dict = {}
    # 估值 / 市值
    value_factors = calc_value_factors(
        val_snapshot=val_snapshot,
        industry_df=industry_df
    )
    factor_dict.update(value_factors)
    raw_float_mktcap = val_snapshot['circulating_market_cap'].copy()
    raw_float_mktcap.name = 'raw_float_mktcap'
    factor_dict['raw_float_mktcap'] = raw_float_mktcap
    factor_dict['ln_float_mktcap'] = calc_size_factor(val_snapshot)
    factor_dict['nonlinear_size'] = calc_nonlinear_size_factor(val_snapshot)
    # 波动率
    vol_factors = calc_volume_volatility_factors(volume_df)
    factor_dict.update(vol_factors)
    yield_factors=calc_yield_volatility_factors(res_df)
    factor_dict.update(yield_factors)
    high_low_factors=calc_high_low_factors(high_df,low_df)
    factor_dict.update(high_low_factors)
    # Barra流动性
    liq_factors = calc_barra_liquidity_factors(turnover_ratio_df)
    factor_dict.update(liq_factors)
    # 动量 / 特色反转
    rev_factors = calc_reversal_factors(ret_df)
    factor_dict.update(rev_factors)
     # Beta
    factor_dict['beta'] = calc_beta_factor(ret_df, bench_ret, half_life=60)
    # 财务因子
    fin_factors = calc_financial_factors_v1(fin_df)
    factor_dict.update(fin_factors)
    # 合并
    fac_df = pd.concat(factor_dict.values(), axis=1,sort=False)
    # 拼行业
    fac_df = fac_df.join(industry_df, how='left')
    fac_df.index.name = 'code'
    fac_df['date'] = pd.to_datetime(date)

    return fac_df

# test_part
# fac_df=build_factor_snapshot('2024-12-31',index_code='000300.XSHG')
# print(fac_df)
# # 统计每一列缺失值数量
# missing_count = fac_df.isnull().sum()
# # 打印出来
# print(missing_count)

# =========================
# 基础预处理函数
# =========================
def winsorize_mad(s, n=3):
    """
    中位数去极值法（MAD）
    """
    s = s.copy()
    med = s.median()
    mad = (s - med).abs().median()

    if pd.isna(mad) or mad == 0:
        return s

    lower = med - n * 1.4826 * mad
    upper = med + n * 1.4826 * mad
    s = s.clip(lower=lower, upper=upper)
    return s

def zscore_standardize(s):
    s = s.copy()
    std = s.std()
    if pd.isna(std) or std == 0:
        return s * np.nan
    return (s - s.mean()) / std

def fillna_by_industry_median(s, industry):
    """
    按行业中位数填补缺失值
    """
    df = pd.DataFrame({'factor': s, 'industry': industry})
    filled = df.groupby('industry')['factor'].transform(lambda x: x.fillna(x.median()))

    # 如果某个行业全是nan，再退化到全市场中位数
    filled = filled.fillna(df['factor'].median())
    return filled

def neutralize_factor(factor, size, industry):
    """
    对流通市值 + 行业哑变量做线性回归，取残差
    """
    df = pd.DataFrame({
        'factor': factor,
        'size': size,
        'industry': industry
    }).dropna()

    if len(df) < 10:
        return pd.Series(index=factor.index, dtype=float)

    industry_dummies = pd.get_dummies(df['industry'], prefix='ind', drop_first=True)

    X = pd.concat(
        [pd.Series(1.0, index=df.index, name='const'),
         df[['size']],
         industry_dummies],
        axis=1
    )

    y = df['factor']
    beta = np.linalg.lstsq(X.values, y.values, rcond=None)[0]
    y_hat = X.values @ beta
    resid = y.values - y_hat

    out = pd.Series(index=factor.index, dtype=float)
    out.loc[df.index] = resid
    return out


RISK_FACTOR_COLS = [
    'vol_1m',
    'vol_3m',
    'vol_6m',
    'vol_12m',
    'ln_float_mktcap',
    'nonlinear_size',
    'beta',
    'industry'
]

# 候选收益因子：每年筛选
CANDIDATE_ALPHA_FACTORS = [
    'BP',
    'roe_q',
    'sales_yoy_q',
    'np_yoy_q',
    'rstr_20',
    'rstr_60',
    'rstr_180',
    'rstr_240',
    'rstr_barra',
    'STOM_barra',
    'STOQ_barra',
    'STOA_barra',
]

FACTOR_CATEGORY_MAP = {
    'BP': 'value',
    'roe_q': 'profitability',
    'sales_yoy_q': 'growth',
    'np_yoy_q': 'growth',
    'rstr_20': 'reversal',
    'rstr_60': 'reversal',
    'rstr_180': 'reversal',
    'STOM_barra': 'liquidity',
    'STOQ_barra': 'liquidity',
    'STOA_barra': 'liquidity'
}

#########################################
# 1. 当前截面预处理：收益因子和风险因子分开处理
#########################################

def preprocess_factor_snapshot(fac_df):
    """
    对一天的因子值做处理
    收益因子：
    - 行业中位数填补
    - 去极值
    - 标准化
    - 行业市值中性化
    - 再标准化

    风险因子：
    - 基础清洗（填补/去极值/标准化）
    - 不做行业市值中性化
    """
    df = fac_df.copy()
    processed = pd.DataFrame(index=df.index)

    # ===== 收益因子 =====
    alpha_cols = [c for c in CANDIDATE_ALPHA_FACTORS if c in df.columns]

    for col in alpha_cols:
        s = df[col].copy()
        s = fillna_by_industry_median(s, df['industry'])
        s = winsorize_mad(s, n=3)
        s = zscore_standardize(s)
        s = neutralize_factor(
            factor=s,
            size=df['ln_float_mktcap'],
            industry=df['industry']
        )
        s = zscore_standardize(s)
        processed[col] = s

    # ===== 风险因子 =====
    numeric_risk_cols = [c for c in RISK_FACTOR_COLS if c in df.columns and c != 'industry']
    for col in numeric_risk_cols:
        s = df[col].copy()
        s = fillna_by_industry_median(s, df['industry'])
        s = winsorize_mad(s, n=3)
        s = zscore_standardize(s)
        processed[col] = s
    # ===== 保留原始流通市值（WLS权重用）=====
    if 'raw_float_mktcap' in df.columns:
        processed['raw_float_mktcap'] = df['raw_float_mktcap']
    # ===== 行业 & 日期 =====
    if 'industry' in df.columns:
        processed['industry'] = df['industry']
    if 'date' in df.columns:
        processed['date'] = pd.to_datetime(df['date'])

    return processed

#########################################
# 2. 年度收益因子筛选
#########################################

#########################################
# 单因子检测
#########################################
def run_single_factor_wls_one_date(df_date, factor_col):
    """
    df_date是某一个月末那一期的整张股票截面数据表，factor_col是某一因子列
    形如
    [
    code        date   BP roe_q rstr_20 next_ret
    0001.XSHE   12-31   ……
    0002.XSHE   12-31   ……
    ]
    单个月末，对一个因子做横截面WLS回归：
    next_ret ~ factor + ln_float_mktcap + industry_dummies
    权重 = sqrt(float market cap)
    """
    cols = ['next_ret', factor_col, 'ln_float_mktcap', 'industry']
    df = df_date[cols].copy().dropna()

    if len(df) < 30:
        return {
            'factor_return': np.nan,
            't_value': np.nan,
            'n_obs': len(df)
        }

    ind_dummies = pd.get_dummies(df['industry'], prefix='ind', drop_first=True)
    X = pd.concat([
        df[[factor_col, 'ln_float_mktcap']],
        ind_dummies
    ], axis=1)

    X = sm.add_constant(X)
    y = df['next_ret']
    w = np.sqrt(df_date.loc[df.index, 'raw_float_mktcap'])
    model = sm.WLS(y, X, weights=w)
    res = model.fit()

    return {
        'factor_return': res.params.get(factor_col, np.nan),
        't_value': res.tvalues.get(factor_col, np.nan),
        'n_obs': len(df)
    }
    

def run_single_factor_wls_panel(data, factor_col):
    """
    data是total_training data，每一年测量因子的时候用三年左右的数据跑wls
    对某个因子，逐月做WLS横截面回归
    返回每个月的 factor_return / t_value
    返回的表形如
    date	  factor  factor_return	 t_value	n_obs
    2022-01-31	BP	    0.023	       2.1	     300
    2022-02-28	BP	    -0.005	      -0.4	     300
    2022-03-31	BP	    0.015	      1.8	     300
    """
    out = []

    for dt, df_date in data.groupby('date'):
        res = run_single_factor_wls_one_date(df_date, factor_col)
        out.append({
            'date': dt,
            'factor': factor_col,
            'factor_return': res['factor_return'],
            't_value': res['t_value'],
            'n_obs': res['n_obs']
        })

    return pd.DataFrame(out).sort_values('date')
    

def summarize_single_factor_stats(res_df):
    """
    输入就是run_single_factor_wls_panel(data, factor_col)的输出
    形式请看上面。
    输出是一些某个因子具体指标，如t值绝对值平均值、t值绝对值>2概率、因子收益平均值、因子收益标准差
    因子收益t值和因子收益>0概率
    """
    x = res_df['factor_return'].dropna()
    t = res_df['t_value'].dropna()

    if len(x) == 0:
        return pd.Series(dtype=float)

    factor_ret_mean = x.mean()
    factor_ret_std = x.std()

    if factor_ret_std == 0 or pd.isna(factor_ret_std):
        factor_ret_t = np.nan
    else:
        factor_ret_t = factor_ret_mean / (factor_ret_std / np.sqrt(len(x)))

    summary = pd.Series({
        'abs_t_mean': t.abs().mean(),
        'abs_t_gt_2_ratio': (t.abs() > 2).mean(),
        'factor_ret_mean': factor_ret_mean,
        'factor_ret_std': factor_ret_std,
        'factor_ret_tvalue': factor_ret_t,
        'factor_ret_gt_0_ratio': (x > 0).mean(),
        'wls_n_months': len(x)
    })

    return summary
    

def calc_ic_one_date(df_date, factor_col):
    """
    单个月末计算一个因子的IC：
    1) 因子值对市值+行业做截面回归，取残差
    2) 残差因子与 next_ret 做 Spearman 相关
    """
    cols = [factor_col, 'ln_float_mktcap', 'industry', 'next_ret']
    df = df_date[cols].copy().dropna()

    if len(df) < 30:
        return {
            'ic': np.nan,
            'n_obs': len(df)
        }

    # 因子先做截面中性化
    fac_neu = neutralize_factor(
        factor=df[factor_col],
        size=df['ln_float_mktcap'],
        industry=df['industry']
    )
    tmp = pd.DataFrame({
        'fac_neu': fac_neu,
        'next_ret': df['next_ret']
    }).dropna()

    if len(tmp) < 30:
        return {
            'ic': np.nan,
            'n_obs': len(tmp)
        }

    ic, _ = spearmanr(tmp['fac_neu'], tmp['next_ret'])

    return {
        'ic': ic,
        'n_obs': len(tmp)
    }

def run_single_factor_ic_panel(data, factor_col):
    """
    输入是data，也就是hist_data表
    对某个因子逐月计算IC
    返回每个月的 ic 序列
    """
    out = []

    for dt, df_date in data.groupby('date'):
        res = calc_ic_one_date(df_date, factor_col)
        out.append({
            'date': dt,
            'factor': factor_col,
            'ic': res['ic'],
            'n_obs': res['n_obs']
        })

    return pd.DataFrame(out).sort_values('date')

def summarize_ic_stats(ic_df):
    """
    输出某个因子的IC指标：IC平均值、IC标准差、IRIC和IC>0的概率
    """
    x = ic_df['ic'].dropna()

    if len(x) == 0:
        return pd.Series(dtype=float)

    ic_mean = x.mean()
    ic_std = x.std()

    if ic_std == 0 or pd.isna(ic_std):
        iric = np.nan
    else:
        iric = ic_mean / ic_std

    summary = pd.Series({
        'ic_mean': ic_mean,
        'ic_std': ic_std,
        'iric': iric,
        'ic_gt_0_ratio': (x > 0).mean(),
        'ic_n_months': len(x)
    })

    return summary

#########################################
# 分层回测
#########################################
def build_history_training_data(end_date, n_months=24, index_code='000300.XSHG'):
    """
    构建截至 end_date 的历史训练样本

    返回：
    DataFrame，每行表示：
    [某个月末, 某只股票, 当期因子值..., 下一期收益 next_ret]
    """
    end_date = pd.Timestamp(end_date)

    # 取历史月末交易日
    month_ends = get_month_end_trading_days('2010-01-01', end_date)
    month_ends = [pd.Timestamp(d) for d in month_ends]
    month_ends = [d for d in month_ends if d <= end_date]

    # 多取一个月末，用来生成最后一期的 next_ret
    month_ends = month_ends[-(n_months + 1):]

    if len(month_ends) < n_months + 1:
        return pd.DataFrame()

    all_df = []

    for i in range(len(month_ends) - 1):
        d0 = month_ends[i]     # 因子截面日
        d1 = month_ends[i + 1] # 下一期结束日

        # 1) d0 当天的因子截面
        fac_df = build_factor_snapshot(d0, index_code=index_code)
        fac_df = preprocess_factor_snapshot(fac_df)

        # 2) 下一期收益标签
        stocks = fac_df.index.tolist()

        close_df = get_price(
            stocks,
            start_date=d0,
            end_date=d1,
            frequency='daily',
            fields=['close'],
            skip_paused=False,
            fq='pre'
        )['close']

        next_ret = close_df.iloc[-1] / close_df.iloc[0] - 1
        fac_df['next_ret'] = next_ret.reindex(fac_df.index)

        # 3) 存成长表
        fac_df = fac_df.reset_index()
        all_df.append(fac_df)

    if len(all_df) == 0:
        return pd.DataFrame()

    hist_data = pd.concat(all_df, axis=0, ignore_index=True, sort=False)
    hist_data['date'] = pd.to_datetime(hist_data['date'])

    return hist_data

# test_code
hist_data=build_history_training_data('2019-12-31',n_months=24,index_code='000300.XSHG')
print(hist_data.head())
print(hist_data.tail())


def assign_factor_group(df_date, factor_col, n_groups=5):
    """
    df_date是total training data里这个date的截面数据
    对单个月末截面，按某一个因子的值分组
    返回带 group 列的 DataFrame
    group = 1,2,...,n_groups
    """
    df = df_date[[factor_col, 'next_ret']].copy().dropna()

    if len(df) < n_groups:
        return pd.DataFrame()

    # 排序后等分
    df = df.sort_values(factor_col).copy()
    df['rank'] = np.arange(1, len(df) + 1)

    # 等频分组
    df['group'] = pd.qcut(df['rank'], q=n_groups, labels=False) + 1

    return df

def calc_group_return_one_date(df_date, factor_col, n_groups=5):
    """
    单个月末，计算某个因子的分组下一期收益
    """
    grouped_df = assign_factor_group(df_date, factor_col, n_groups=n_groups)
    if grouped_df.empty:
        return pd.Series(dtype=float)

    group_ret = grouped_df.groupby('group')['next_ret'].mean()
    group_ret.name = pd.to_datetime(df_date['date'].iloc[0])

    return group_ret
    
def run_factor_group_backtest(data, factor_col, n_groups=5):
    """
    输入：data就是hist_data
    对某个因子逐月做分层回测
    返回 DataFrame:
    index = date
    columns = group1, group2, ..., groupN
    values = 下一期组收益
    输出index=date，columns=group1……group5的下一期平均收益
    形如
    date	    group_1	group_2	group_3	group_4	group_5
    2023-08-31	0.021	0.015	0.009	-0.005	-0.018
    2023-09-30	0.012	0.008	0.002	-0.010	-0.022
    """
    out = []

    for dt, df_date in data.groupby('date'):
        group_ret = calc_group_return_one_date(df_date, factor_col, n_groups=n_groups)

        if len(group_ret) == 0:
            continue

        row = {'date': dt}
        for g in range(1, n_groups + 1):
            row[f'group_{g}'] = group_ret.get(g, np.nan)

        out.append(row)

    res = pd.DataFrame(out).sort_values('date').set_index('date')
    return res




def calc_group_nav(group_ret_df):
    """
    根据组收益序列计算各组累计净值，就是每一列做累乘
    输入是run_factor_group_backtest的res
    """
    nav = (1 + group_ret_df).cumprod()
    return nav




def calc_monotonic_score_from_group(group_ret_df):
    """
    用最终净值和组号的 Spearman 相关来衡量分层单调性
    每个因子会返回一个score，接近+1代表正向强相关，0代表因子无效，接近-1代表负向强相关
    """
    if group_ret_df is None or len(group_ret_df) == 0:
        return np.nan

    nav_df = calc_group_nav(group_ret_df)
    if len(nav_df) == 0:
        return np.nan

    final_nav = nav_df.iloc[-1]
    group_rank = np.arange(1, len(final_nav) + 1)
    score, _ = spearmanr(group_rank, final_nav.values)
    return score
    
#test
# test_columns=['BP','roe_q','rstr_60','rstr_180','rstr_240','rstr_barra','STOM_barra','STOQ_barra','STOA_barra']
# for factor in test_columns:
#     wls=run_single_factor_wls_panel(hist_data,factor)
#     ic=run_single_factor_ic_panel(hist_data, factor)
#     wls_sum = summarize_single_factor_stats(wls)
#     ic_sum = summarize_ic_stats(ic)
#     print('wls_sum')
#     print(wls_sum)
#     print('ic_sum')
#     print(ic_sum)
#     res=run_factor_group_backtest(hist_data,factor,n_groups=5)
#     score=calc_monotonic_score_from_group(res)
#     print(factor)
#     print(score)
    

def build_alpha_factor_score_table(train_data, candidate_alpha_factors):
    """
    对候选收益因子做单因子打分,
    输出形如
    factor    direction   mono_score   score   t_value  ic_mean  ...
    BP        1           0.6         0.82     4.2      0.08
    rstr_60  -1          -0.6        0.78     3.9      0.07
    ...
    """
    rows = []

    for fac in candidate_alpha_factors:
        # 对每个因子做以下操作
        try:
            # 单因子检测
            wls_df = run_single_factor_wls_panel(train_data, fac)
            ic_df = run_single_factor_ic_panel(train_data, fac)

            wls_sum = summarize_single_factor_stats(wls_df)
            ic_sum = summarize_ic_stats(ic_df)
            
            # 分层回测
            group_ret = run_factor_group_backtest(train_data, fac, n_groups=5)
            mono_score = calc_monotonic_score_from_group(group_ret)

            ic_mean = ic_sum.get('ic_mean', np.nan)
            fac_ret_mean = wls_sum.get('factor_ret_mean', np.nan)

            # 方向判断：优先IC，其次分层，其次因子收益
            # 只要ic_mean或者mono_score或者fac_ret_mean其中之一是负的就给它的direction标记为-1
            if pd.notna(ic_mean) and ic_mean != 0:
                direction = 1 if ic_mean > 0 else -1
            elif pd.notna(mono_score) and mono_score != 0:
                direction = 1 if mono_score > 0 else -1
            elif pd.notna(fac_ret_mean) and fac_ret_mean != 0:
                direction = 1 if fac_ret_mean > 0 else -1
            else:
                direction = 1

            score = (
                0.35 * abs(wls_sum.get('factor_ret_tvalue', 0))
                + 0.25 * abs(ic_sum.get('iric', 0))
                + 0.20 * abs(wls_sum.get('abs_t_mean', 0))
                + 0.20 * abs(mono_score if pd.notna(mono_score) else 0)
            )

            row = {
                'factor': fac,
                'direction': direction,
                'mono_score': mono_score,
                'score': score
            }
            row.update(wls_sum.to_dict())
            row.update(ic_sum.to_dict())
            rows.append(row)

        except Exception as e:
            rows.append({
                'factor': fac,
                'direction': 1,
                'mono_score': np.nan,
                'score': np.nan,
                'error': str(e)
            })

    res = pd.DataFrame(rows)
    return res.sort_values('score', ascending=False)

# test
res=build_alpha_factor_score_table(hist_data,CANDIDATE_ALPHA_FACTORS)
print(res.columns.tolist())
print(res[['factor', 'score']])
print('res is here')
print(res)



def preselect_alpha_factors(score_table, top_k_per_cat=4):
    """
    每个类别先取前 top_k_per_cat 个
    """
    tab = score_table.copy()
    tab = tab[pd.notna(tab['score'])].copy()
    tab['category'] = tab['factor'].map(FACTOR_CATEGORY_MAP)

    picked = []
    for cat, sub in tab.groupby('category'):
        sub = sub.sort_values('score', ascending=False).head(top_k_per_cat)
        picked.extend(sub['factor'].tolist())

    return picked

picked=preselect_alpha_factors(res)
print(picked)
    
def apply_factor_directions(data, direction_map):
    """
    输入是整张hist_data大表，给它新增若干列
    把反向因子统一翻成正方向
    输出 *_dir 列
    """
    df = data.copy()

    for fac, d in direction_map.items():
        if fac in df.columns:
            df[fac + '_dir'] = df[fac] * d

    return df

#########################################
# 二次筛选
#########################################
def run_multifactor_wls_one_date_with_aic(df_date, factor_cols):
    """
    单个月末，多因子横截面WLS回归，并返回当期截面的拟合值

    回归形式：
    next_ret ~ factor_cols + ln_float_mktcap + industry_dummies

    权重：
    sqrt(raw_float_mktcap)

    参数
    ----
    df_date : DataFrame
        某一个 date 的横截面数据
    factor_cols : list[str]
        收益因子列名列表

    返回
    ----
    (pred, actual) 或 None
        pred   : Series, index=code, name='pred'
        actual : Series, index=code, name='actual'
    """
    needed_cols = ['code', 'next_ret', 'ln_float_mktcap', 'industry', 'raw_float_mktcap'] + factor_cols
    df = df_date[needed_cols].copy().dropna()

    # 样本太少就跳过
    if len(df) < max(30, len(factor_cols) + 10):
        return None

    # 行业哑变量
    ind_dummies = pd.get_dummies(df['industry'], prefix='ind', drop_first=True)

    # 回归自变量：收益因子 + 市值控制 + 行业控制
    X = pd.concat([
        df[factor_cols],
        df[['ln_float_mktcap']],
        ind_dummies
    ], axis=1)

    X = sm.add_constant(X)
    y = df['next_ret']

    # WLS 权重：流通市值平方根
    w = np.sqrt(df['raw_float_mktcap'].clip(lower=1e-8))

    model = sm.WLS(y, X, weights=w)
    res = model.fit()

    return {
        'aic': res.aic,
        'bic': res.bic,
        'n_obs': len(df)
    }

def evaluate_alpha_factor_subset(train_data, factor_cols,criterion='aic'):
    """
    看这组因子在历史训练期里，按月做多因子回归后
    用平均月度AIC/BIC评价一组收益因子
    AIC/BIC越小越好
    """
    if len(factor_cols) == 0:
        return -np.inf

    values = []

    for dt, df_date in train_data.groupby('date'):
        out = run_multifactor_wls_one_date_with_aic(df_date, factor_cols)
        if out is None:
            continue

        val = out.get(criterion, np.nan)
        if pd.notna(val):
            values.append(val)


    if len(values) == 0:
        return np.inf

    return np.mean(values)
    

def forward_select_alpha_factors(train_data, candidate_directed_cols, max_factors=6, min_improve=0.002):
    """
    前向逐步选择收益因子
    """
    selected = []
    remaining = candidate_directed_cols.copy()
    best_score = np.inf

    while len(remaining) > 0 and len(selected) < max_factors:
        best_candidate = None
        best_candidate_score = best_score

        for fac in remaining:
            trial = selected + [fac]
            score = evaluate_alpha_factor_subset(train_data, trial)
            # 如果提升的score>0.002就保留这个因子
            if score < best_candidate_score:
                best_candidate_score = score
                best_candidate = fac

        if best_candidate is None:
            break

        if best_score == np.inf or (best_score - best_candidate_score) >= min_improve:
            selected.append(best_candidate)
            remaining.remove(best_candidate)
            best_score = best_candidate_score
        else:
            break

    return selected, best_score

def select_alpha_factors_annually(train_data):
    """
    年度收益因子筛选总入口
    """
    score_table = build_alpha_factor_score_table(train_data, CANDIDATE_ALPHA_FACTORS)
    direction_map = dict(zip(score_table['factor'], score_table['direction']))
    train_data_dir = apply_factor_directions(train_data, direction_map)

    preselected_raw = preselect_alpha_factors(score_table, top_k_per_cat=2)
    preselected_dir = [fac + '_dir' for fac in preselected_raw]

    selected_dir, model_score = forward_select_alpha_factors(
        train_data=train_data_dir,
        candidate_directed_cols=preselected_dir,
        max_factors=8,
        min_improve=0.002
    )

    selected_raw = [c.replace('_dir', '') for c in selected_dir]

    return {
        'score_table': score_table,
        'direction_map': direction_map,
        'train_data_dir': train_data_dir,
        'selected_alpha_factors_raw': selected_raw,
        'selected_alpha_factors_dir': selected_dir,
        'model_score': model_score
    }
    
print('最终选的因子数据')
alpha_res=select_alpha_factors_annually(hist_data)
direction_map=alpha_res['direction_map']
hist_data_dir=alpha_res['train_data_dir']
alpha_factor_cols=alpha_res['selected_alpha_factors_dir']
print(hist_data_dir.columns.tolist())
print(alpha_factor_cols)

#########################################
# 5. 年度缓存：只对收益因子做
#########################################
# def should_refresh_alpha_factor_selection(decision_date):
#     year_key = pd.Timestamp(decision_date).year
#     return ('alpha_factor_selection_cache' not in g.__dict__) or (year_key not in g.alpha_factor_selection_cache)
def is_last_trading_day_of_year(date):
    """
    找到一年最后一个调仓日
    """
    date = pd.Timestamp(date)
    year = date.year
    trade_days = get_trade_days(
        start_date='{}-01-01'.format(year),
        end_date='{}-12-31'.format(year)
    )
    trade_days = [pd.Timestamp(d) for d in trade_days]

    if len(trade_days) == 0:
        return False
    return date == trade_days[-1]

def update_alpha_factors_if_year_end(context):
    today = pd.Timestamp(context.current_dt.date())
    if not is_last_trading_day_of_year(today):
        return
    decision_date = get_prev_trade_date(today)
    log.info('year-end alpha factor selection update: {}'.format(today))
    log.info('today = {}'.format(today))
    log.info('decision_date = {}'.format(decision_date))
    train_data = build_history_training_data(
        end_date=decision_date,
        n_months=36,
        index_code=g.index_code
    )
    if train_data is None or len(train_data) == 0:
        log.info('train_data empty, skip annual alpha selection')
        return
    result = select_alpha_factors_annually(train_data)
    g.current_alpha_direction_map = result['direction_map']
    g.current_alpha_factors_raw = result['selected_alpha_factors_raw']
    g.current_alpha_factors_dir = result['selected_alpha_factors_dir']
    g.current_alpha_selection_year = today.year
    
    log.info('alpha selection updated for year {}'.format(today.year))
    log.info('selected alpha raw: {}'.format(g.current_alpha_factors_raw))
    log.info('selected alpha dir: {}'.format(g.current_alpha_factors_dir))



#########################################
# 3. 收益模型：只吃筛选后的收益因子
#########################################

def build_alpha_factor_return_panel(train_data, alpha_factor_cols):
    """
    alpha_factor_cols是挑选出来的alpha factors，注意是加了dir进行过方向处理的
    逐月回归，生成历史收益因子收益序列
    返回形如
    date	    BP_dir	rstr_60_dir	STOQ_dir
    2023-08-31	0.007	-0.004	    0.006
    2023-09-30	0.009	-0.002	    0.005
    """
    ret_rows = []
    for dt, df_date in train_data.groupby('date'):
        needed_cols = ['next_ret', 'ln_float_mktcap', 'industry', 'raw_float_mktcap'] + alpha_factor_cols
        df = df_date[needed_cols].copy().dropna()

        if len(df) < max(30, len(alpha_factor_cols) + 10):
            continue

        ind_dummies = pd.get_dummies(df['industry'], prefix='ind', drop_first=True)
        X = pd.concat([df[alpha_factor_cols], df[['ln_float_mktcap']], ind_dummies], axis=1)
        X = sm.add_constant(X)
        y = df['next_ret']
        w = np.sqrt(df['raw_float_mktcap'].clip(lower=1e-8))

        model = sm.WLS(y, X, weights=w)
        res = model.fit()

        row = {'date': dt}
        for fac in alpha_factor_cols:
            row[fac] = res.params.get(fac, np.nan)
        ret_rows.append(row)

    if len(ret_rows) == 0:
        return pd.DataFrame(columns=alpha_factor_cols)

    return pd.DataFrame(ret_rows).sort_values('date').set_index('date')

def forecast_alpha_factor_returns(alpha_factor_ret_df, window=12):
    """
    12个月移动平均预测下一期收益因子收益
    """
    return alpha_factor_ret_df.rolling(window=window, min_periods=window).mean().shift(1)

def build_expected_return_for_current_date(train_data, current_fac_df, alpha_factor_cols, ma_window=12):
    """
    计算当前截面的股票预期收益
    train_data是历史数据
    current_fac_df是现在这个月末的股票的因子暴露
    alpha_factor_cols是这一年已经选好的最优因子
    返回形如：
    date	    code	expected_ret	BP_dir	rstr_60_dir
    2025-01-23	000001	0.0124	        1.2	    0.8
    2025-01-23	000002	0.0087	        0.9	    0.6
    """
    alpha_ret_df = build_alpha_factor_return_panel(train_data, alpha_factor_cols)
    # 计算因子预期收益
    alpha_ret_forecast = forecast_alpha_factor_returns(alpha_ret_df, window=ma_window)

    current_date = pd.to_datetime(current_fac_df['date'].iloc[0])

    if current_date in alpha_ret_forecast.index:
        fac_forecast = alpha_ret_forecast.loc[current_date]
    else:
        valid = alpha_ret_forecast.dropna()
        if len(valid) == 0:
            return pd.DataFrame()
        fac_forecast = valid.iloc[-1]

    if fac_forecast[alpha_factor_cols].isna().any():
        return pd.DataFrame()

    tmp = current_fac_df.reset_index()[['code', 'date'] + alpha_factor_cols].copy()
    # 计算股票预期收益= Σ(因子暴露 × 预测因子收益)
    contri = tmp[alpha_factor_cols].mul(fac_forecast[alpha_factor_cols], axis=1)
    tmp['expected_ret'] = contri.sum(axis=1, min_count=1)

    return tmp[['date', 'code', 'expected_ret'] + alpha_factor_cols]
    
# test
raw_fac_df = build_factor_snapshot('2019-12-31')
current_fac_df = preprocess_factor_snapshot(raw_fac_df)
current_fac_df_dir = apply_factor_directions(current_fac_df, direction_map)
print(current_fac_df_dir[alpha_factor_cols].isna().sum())
expected_Return=build_expected_return_for_current_date(hist_data_dir,current_fac_df_dir,alpha_factor_cols,ma_window=12)
print(expected_Return.columns.tolist())
print('expected_return')
print(expected_Return['expected_ret'])
count_zero = (expected_Return['expected_ret'] == 0).sum()
print(f"等于0的数量：{count_zero}")

#########################################
# 4. 风险模型：只吃固定风险因子
#########################################

def build_risk_exposure_for_current_date(current_fac_df):
    """
    当前截面的风险因子暴露矩阵 X_risk
    形如，
             ln_float_mktcap  beta  volatility  ind_银行  ind_医药  ind_消费
    code
    000001        1.2          0.8      0.5          1        0        0
    000002        0.7          1.1      0.9          0        1        0
    """
    df = current_fac_df.reset_index()[['code'] + [c for c in RISK_FACTOR_COLS if c in current_fac_df.columns]].copy()
    X = df.set_index('code')

    # 行业哑变量
    if 'industry' in X.columns:
        ind = pd.get_dummies(X['industry'], prefix='ind', drop_first=False)
        X = pd.concat([X.drop(columns=['industry']), ind], axis=1)

    return X

def build_risk_factor_return_panel(train_data):
    """
    逐月回归生成风险因子收益
    返回两张表
    一张是risk_ret_df，形如
                ln_float_mktcap  beta  volatility  ind_银行  ind_医药  ...
    2023-08-31    0.0012        0.0020  -0.0015      0.0032   -0.0010
    2023-09-30    0.0020        0.0011  -0.0022      0.0015   -0.0025
    另一张是残差表，形如
        date       code    residual
    0   2023-08-31  000001   0.021
    1   2023-08-31  000002  -0.011
    2   2023-09-30  000001   0.005
    """
    rows = []
    residual_rows = []

    numeric_risk_cols = [c for c in RISK_FACTOR_COLS if c != 'industry']

    for dt, df_date in train_data.groupby('date'):
        needed_cols = ['code', 'next_ret', 'raw_float_mktcap', 'industry'] + numeric_risk_cols
        df = df_date[needed_cols].copy().dropna()

        if len(df) < max(30, len(numeric_risk_cols) + 10):
            continue

        ind_dummies = pd.get_dummies(df['industry'], prefix='ind', drop_first=False)
        X = pd.concat([df[numeric_risk_cols], ind_dummies], axis=1)
        X = sm.add_constant(X)
        y = df['next_ret']
        w = np.sqrt(df['raw_float_mktcap'].clip(lower=1e-8))

        model = sm.WLS(y, X, weights=w)
        res = model.fit()

        row = {'date': dt}
        for col in X.columns:
            if col != 'const':
                row[col] = res.params.get(col, np.nan)
        rows.append(row)

        resid_df = pd.DataFrame({
            'date': dt,
            'code': df['code'].values,
            'residual': res.resid.values
        })
        residual_rows.append(resid_df)

    risk_ret_df = pd.DataFrame(rows).sort_values('date').set_index('date') if len(rows) > 0 else pd.DataFrame()
    residual_panel_df = pd.concat(residual_rows, axis=0, ignore_index=True) if len(residual_rows) > 0 else pd.DataFrame(columns=['date', 'code', 'residual'])

    return risk_ret_df, residual_panel_df

def build_sigma_for_current_date(train_data, current_fac_df, cov_window=12, spec_window=12):
    """
    Sigma = X F X' + D
    """
    X = build_risk_exposure_for_current_date(current_fac_df)

    risk_ret_df, residual_panel_df = build_risk_factor_return_panel(train_data)

    current_date = pd.to_datetime(current_fac_df['date'].iloc[0])

    hist_risk_ret = risk_ret_df.loc[risk_ret_df.index < current_date].tail(cov_window)
    if len(hist_risk_ret) < cov_window:
        return None

    F = hist_risk_ret.cov()

    common_cols = [c for c in X.columns if c in F.columns]
    X = X[common_cols]
    F = F.loc[common_cols, common_cols]

    sys_cov = X.values @ F.values @ X.values.T
    sys_cov = pd.DataFrame(sys_cov, index=X.index, columns=X.index)

    resid_hist = residual_panel_df[residual_panel_df['date'] < current_date].copy()
    recent_dates = sorted(resid_hist['date'].unique())[-spec_window:] if len(resid_hist) > 0 else []
    resid_hist = resid_hist[resid_hist['date'].isin(recent_dates)] if len(recent_dates) > 0 else resid_hist

    spec_var = resid_hist.groupby('code')['residual'].std() ** 2 if len(resid_hist) > 0 else pd.Series(dtype=float)
    spec_var = spec_var.reindex(X.index)

    if spec_var.notna().sum() == 0:
        spec_var = pd.Series(1e-4, index=X.index)
    else:
        spec_var = spec_var.fillna(spec_var.median())

    D = pd.DataFrame(np.diag(spec_var.values), index=X.index, columns=X.index)

    Sigma = sys_cov + D
    return Sigma

    
## 组合优化
def optimize_portfolio_current_date(
    expected_ret_df,
    Sigma,
    risk_aversion=10.0,
    max_weight=0.10,
    top_n=100,
    min_expected_ret_quantile=None
):
    """
    对当前截面做组合优化

    参数
    ----
    expected_ret_df : DataFrame
        至少包含 ['date', 'code', 'expected_ret']
    Sigma : DataFrame
        股票协方差矩阵，index/columns 都是股票代码
    risk_aversion : float
        风险厌恶系数 lambda
    max_weight : float
        单票最大权重
    top_n : int
        只在 expected_ret 最高的前 top_n 只股票里做优化
    min_expected_ret_quantile : float or None
        如果不为 None，则只保留 expected_ret 高于某个分位数的股票
        比如 0.7 表示只保留前 30% 的股票

    返回
    ----
    weight_df : DataFrame
        ['date', 'code', 'weight']
    """
    if expected_ret_df is None or len(expected_ret_df) == 0:
        return pd.DataFrame(columns=['date', 'code', 'weight'])

    if Sigma is None or len(Sigma) == 0:
        return pd.DataFrame(columns=['date', 'code', 'weight'])

    df = expected_ret_df.copy()

    # 当前日期
    current_date = pd.to_datetime(df['date'].iloc[0])

    # 去缺失
    df = df[['date', 'code', 'expected_ret']].dropna().copy()
    if len(df) == 0:
        return pd.DataFrame(columns=['date', 'code', 'weight'])

    # 只保留 Sigma 里有的股票
    common_codes = [c for c in df['code'] if c in Sigma.index]
    df = df[df['code'].isin(common_codes)].copy()

    if len(df) == 0:
        return pd.DataFrame(columns=['date', 'code', 'weight'])

    # 可选：按 expected_ret 分位数先过滤
    if min_expected_ret_quantile is not None:
        threshold = df['expected_ret'].quantile(min_expected_ret_quantile)
        df = df[df['expected_ret'] >= threshold].copy()

    # 再按 expected_ret 排序，只保留 top_n
    df = df.sort_values('expected_ret', ascending=False).head(top_n).copy()

    if len(df) == 0:
        return pd.DataFrame(columns=['date', 'code', 'weight'])

    codes = df['code'].tolist()

    # 对齐 mu 和 Sigma
    mu = df.set_index('code')['expected_ret'].reindex(codes).values
    cov = Sigma.reindex(index=codes, columns=codes).values

    n = len(codes)
    if n == 0:
        return pd.DataFrame(columns=['date', 'code', 'weight'])

    # 如果 max_weight 太小，导致根本不可行，自动放宽到可行下界
    # 例如 n=30, max_weight=0.02，则总上限只有0.6，不可能满仓
    effective_max_weight = max_weight
    if n * effective_max_weight < 1.0:
        effective_max_weight = 1.0 / n + 1e-6

    # 数值稳健性：给协方差矩阵加一个很小的对角扰动
    cov = np.array(cov, dtype=float)
    cov = 0.5 * (cov + cov.T)   # 强制对称
    cov = cov + np.eye(n) * 1e-8

    # 初始值：等权
    w0 = np.ones(n) / n

    # 目标函数：min( -mu'w + lambda * w' Sigma w )
    def objective(w):
        ret_term = -np.dot(mu, w)
        risk_term = risk_aversion * np.dot(w, cov @ w)
        return ret_term + risk_term
    # 权重和为1
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    ]
    # 边界：不做空 + 单票上限
    bounds = [(0.0, effective_max_weight) for _ in range(n)]
    try:
        res = minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-9, 'disp': False}
        )
        
        log.info('opt success = {}'.format(res.success))
        log.info('opt message = {}'.format(res.message))
        log.info('opt fun = {}'.format(res.fun))
        log.info('weight std = {}'.format(np.std(res.x)))

        if (not res.success) or (res.x is None):
            # 优化失败就退化成等权
            weights = np.ones(n) / n
        else:
            weights = res.x

    except Exception as e:
        log.info('optimize_portfolio_current_date failed: {}'.format(str(e)))
        weights = np.ones(n) / n

    # 数值清理
    weights = np.clip(weights, 0, None)
    if weights.sum() <= 0:
        weights = np.ones(n) / n
    else:
        weights = weights / weights.sum()

    weight_df = pd.DataFrame({
        'date': current_date,
        'code': codes,
        'weight': weights
    })

    return weight_df

   
#########################################
# 6. rebalance 主流程
#########################################
def get_prev_trade_date(date):
    tds = get_trade_days(end_date=date, count=2)
    return pd.Timestamp(tds[-2])
    
def initialize(context):
    set_benchmark('000300.XSHG')
    set_option('use_real_price', True)
    set_option('avoid_future_data', True)
    
    g.index_code = '000300.XSHG'
    g.ma_window = 12
    g.top_n = 100
    g.risk_aversion = 10.0
    g.max_weight = 0.10
    
    g.current_alpha_direction_map = {}
    g.current_alpha_factors_raw = []
    g.current_alpha_factors_dir = []
    g.current_alpha_selection_year = None
    # 每个月最后一个交易日的下午14:30，自动执行rebalance函数
    run_monthly(update_alpha_factors_if_year_end, monthday=-1, time='14:00')
    run_monthly(rebalance, monthday=-1, time='14:30')
    
    
def can_buy_today(stock):
    data = get_current_data()
    return (stock in data) and (not data[stock].paused)

def can_sell_today(stock):
    data = get_current_data()
    return (stock in data) and (not data[stock].paused)
    

def execute_trades(context, weight_df, min_weight=1e-4, min_value=1000):
    if weight_df is None or len(weight_df) == 0:
        return

    df = weight_df[['code', 'weight']].copy()
    df = df[df['weight'] > min_weight].copy()

    if len(df) == 0:
        return

    df['weight'] = df['weight'] / df['weight'].sum()

    total_value = context.portfolio.total_value
    df['target_value'] = df['weight'] * total_value
    df = df[df['target_value'] >= min_value].copy()

    target_dict = dict(zip(df['code'], df['target_value']))
    current_positions = context.portfolio.positions

    # 先卖
    for stock in list(current_positions.keys()):
        if stock not in target_dict:
            o = order_target_value(stock, 0)
            log.info('sell order: stock={}, order={}'.format(stock, o))

    # 再买/调
    for stock, target_value in target_dict.items():
        if stock in current_positions:
            current_value = current_positions[stock].value
        else:
            current_value = 0

        if abs(target_value - current_value) < min_value:
            continue

        o = order_target_value(stock, target_value)
        log.info(
            'adjust order: stock={}, current_value={}, target_value={}, order={}'.format(
                stock, current_value, target_value, o
            )
        )
    
    
def rebalance(context):
    today = pd.Timestamp(context.current_dt.date())
    decision_date = get_prev_trade_date(today)
    if len(g.current_alpha_factors_dir) == 0:
        log.info('alpha factors not initialized yet, skip rebalance')
        return
    # 当日截面数据
    raw_fac_df = build_factor_snapshot(decision_date, index_code=g.index_code)
    current_fac_df = preprocess_factor_snapshot(raw_fac_df)
    current_fac_df_dir = apply_factor_directions(current_fac_df, g.current_alpha_direction_map)
    train_data = build_history_training_data(
        end_date=decision_date,
        n_months=36,
        index_code=g.index_code
    )
    if train_data is None or len(train_data) == 0:
        log.info('train_data empty, skip')
        return
    hist_data_dir = apply_factor_directions(train_data, g.current_alpha_direction_map)
    expected_return = build_expected_return_for_current_date(
        hist_data_dir,
        current_fac_df_dir,
        g.current_alpha_factors_dir,
        ma_window=g.ma_window
    )
    if expected_return is None or len(expected_return) == 0:
        log.info('expected_return empty, skip')
        return

    Sigma = build_sigma_for_current_date(train_data, current_fac_df)

    if Sigma is None:
        log.info('Sigma is None, skip')
        return
    weight_df = optimize_portfolio_current_date(
        expected_ret_df=expected_return,
        Sigma=Sigma,
        risk_aversion=g.risk_aversion,
        max_weight=g.max_weight,
        top_n=g.top_n
    )

    if weight_df is None or len(weight_df) == 0:
        log.info('weight_df empty, skip')
    else:
        log.info('weight_df shape: {}'.format(weight_df.shape))
        log.info('weight sum: {}'.format(weight_df['weight'].sum()))
        log.info('positive weight count: {}'.format((weight_df['weight'] > 0).sum()))
        log.info('max weight: {}'.format(weight_df['weight'].max()))
        log.info('top 10 weights:\n{}'.format(
            weight_df.sort_values('weight', ascending=False).head(10)
        ))

    execute_trades(context, weight_df)

