import pandas as pd
import numpy as np
from jqdata import *
import datetime as dt
import statsmodels.api as sm
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.optimize import minimize

# 工具函数
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
    获取某日估值截面
    """
    q = query(
        valuation.code,
        valuation.pb_ratio,
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
        # 优先尝试中信行业
        if 'zjw' in info and info['zjw'] is not None:
            industry_name = info['zjw']['industry_name']
        elif 'sw_l1' in info and info['sw_l1'] is not None:
            industry_name = info['sw_l1']['industry_name']
        else:
            industry_name = 'UNKNOWN'
        records.append((s, industry_name))

    df = pd.DataFrame(records, columns=['code', 'industry']).set_index('code')
    return df

# =========================
# 3. 截面因子：估值 / 市值 / 波动率
# =========================
def calc_bp_factor(val_snapshot):
    """
    BP = 1 / PB
    """
    bp = 1.0 / val_snapshot['pb_ratio']
    bp.name = 'BP'
    return bp


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
    x = (x - x.mean()) / x.std()

    y = x ** 3

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


def calc_volume_volatility_factors(volume_df):
    """
    1/3/6/12月成交量波动率
    这里按 20/60/120/240 个交易日滚动标准差
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

# =========================
# 4. Barra流动性因子
# =========================
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
# 5. 动量 / 特色反转因子
# =========================
def calc_weighted_rstr(ret_series, lookback, half_life, lag=1):
    """
    ret_series: 单只股票日收益率 Series
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
    返回 20 / 60 / 180 日特色反转因子
    """
    specs = {
        'rstr_20': (20, 5),
        'rstr_60': (60, 15),
        'rstr_180': (180, 45),
    }

    out = {}
    for fac_name, (lookback, half_life) in specs.items():
        vals = {}
        for code in ret_df.columns:
            vals[code] = calc_weighted_rstr(
                ret_df[code],
                lookback=lookback,
                half_life=half_life,
                lag=1
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
# 7. 财务因子（第一版：先跑通）
# =========================

def get_financial_snapshot_v1(stocks, date):
    """
    第一版：直接使用聚宽当前 date 可获得的最新财务字段
    先保证流程能跑通
    """
    q = query(
        valuation.code,
        indicator.roe,
        indicator.inc_total_revenue_year_on_year,
        indicator.inc_net_profit_to_shareholders_year_on_year
    ).filter(
        valuation.code.in_(stocks)
    )
    df = get_fundamentals(q, date=date)
    df = df.set_index('code')
    return df


def calc_financial_factors_v1(fin_df):
    """
    第一版财务因子：
    - 单季度ROE：先用 roe 近似
    - 单季度营业收入增速：先用营收同比字段近似
    - 单季度归母净利润增速：先用归母净利同比字段近似
    - 单季度ROE增速：先临时留空，后面做严格版
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

    roe_yoy_q = pd.Series(index=fin_df.index, dtype=float, name='roe_yoy_q')
    out['roe_yoy_q'] = roe_yoy_q   # 先占位，后面做严格版本

    return out

# =========================
# 8. 单个月末：生成因子截面
# =========================
def build_factor_snapshot(date, index_code='000300.XSHG'):
    stocks = get_hs300_universe(date)

    # 行情数据
    price_panel = get_price_panel(
        stocks,
        end_date=date,
        count=260,
        fields=['close', 'high', 'low', 'volume']
    )
    close_df = price_panel['close']
    volume_df = price_panel['volume']
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

    # 估值 / 市值 / 波动率
    factor_dict['BP'] = calc_bp_factor(val_snapshot)
    raw_float_mktcap = val_snapshot['circulating_market_cap'].copy()
    raw_float_mktcap.name = 'raw_float_mktcap'
    factor_dict['raw_float_mktcap'] = raw_float_mktcap
    factor_dict['ln_float_mktcap'] = calc_size_factor(val_snapshot)
    factor_dict['nonlinear_size'] = calc_nonlinear_size_factor(val_snapshot)

    vol_factors = calc_volume_volatility_factors(volume_df)
    factor_dict.update(vol_factors)

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


# =========================
# 1. 基础预处理函数
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

# =========================
# 2. 单月截面预处理
# =========================
def preprocess_factor_snapshot(fac_df):
    """
    输入: build_factor_snapshot(date) 生成的原始因子表
    输出: 预处理后的因子表
    """
    df = fac_df.copy()

    # 你当前这版的收益因子列
    return_factor_cols = [
        'BP',
        'roe_q',
        'sales_yoy_q',
        'np_yoy_q',
        'roe_yoy_q',
        'rstr_20',
        'rstr_60',
        'rstr_180',
        'STOM_barra',
        'STOQ_barra',
        'STOA_barra'
    ]
    processed = pd.DataFrame(index=df.index)
    for col in return_factor_cols:
        s = df[col].copy()
        # 1) 行业中位数填补
        s = fillna_by_industry_median(s, df['industry'])
        # 2) 去极值
        s = winsorize_mad(s, n=3)
        # 3) 标准化
        s = zscore_standardize(s)
        # 4) 行业市值中性化
        s = neutralize_factor(
            factor=s,
            size=df['ln_float_mktcap'],
            industry=df['industry']
        )
        # 5) 再标准化一次（常见做法）
        s = zscore_standardize(s)

        processed[col] = s
        # 保留风险因子和辅助列
    risk_factor_cols = [
        'ln_float_mktcap',
        'nonlinear_size',
        'beta',
        'vol_1m',
        'vol_3m',
        'vol_6m',
        'vol_12m'
    ]
    for col in risk_factor_cols:
        if col in df.columns:
            s = df[col].copy()
            s = fillna_by_industry_median(s, df['industry'])
            s = winsorize_mad(s, n=3)
            s = zscore_standardize(s)
            processed[col] = s
    if 'industry' in df.columns:
        processed['industry'] = df['industry']

    if 'date' in df.columns:
        processed['date'] = df['date']
        
    if 'raw_float_mktcap' in df.columns:
        processed['raw_float_mktcap']=df['raw_float_mktcap']
    return processed

# test
test_date = '2019-01-31'
raw_fac = build_factor_snapshot(test_date)
proc_fac = preprocess_factor_snapshot(raw_fac)
print(raw_fac[['BP', 'roe_q', 'rstr_20','ln_float_mktcap','raw_float_mktcap']].head())
print(proc_fac[['BP', 'roe_q', 'rstr_20','ln_float_mktcap','raw_float_mktcap']].head())

# 把所有月份的数据concat在一起
def build_factor_dataset(start_date, end_date,preprocess=True):
    month_ends = get_month_end_trading_days(start_date, end_date)
    all_df = []

    for d in month_ends:
        print("building factors for", d)
        fac_df = build_factor_snapshot(d)
        if preprocess:
            fac_df = preprocess_factor_snapshot(fac_df)

        fac_df = fac_df.reset_index()
        all_df.append(fac_df)

    res = pd.concat(all_df, axis=0, ignore_index=True,sort=False)
    return res

def build_monthly_return_labels(start_date, end_date):
    """
    生成每个月末截面的下一期收益标签
    返回 DataFrame: [date, code, next_ret]
    """
    month_ends = get_month_end_trading_days(start_date, end_date)
    records = []

    for i in range(len(month_ends) - 1):
        d0 = month_ends[i]
        d1 = month_ends[i + 1]

        stocks = get_hs300_universe(d0)

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

        tmp = pd.DataFrame({
            'code': next_ret.index,
            'date': pd.to_datetime(d0),
            'next_ret': next_ret.values
        })
        records.append(tmp)

    return pd.concat(records, axis=0, ignore_index=True, sort=False)

def run_single_factor_wls_one_date(df_date, factor_col):
    """
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
    对某个因子，逐月做WLS横截面回归
    返回每个月的 factor_return / t_value
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
    
def summarize_single_factor_all(wls_df, ic_df):
    wls_sum = summarize_single_factor_stats(wls_df)
    ic_sum = summarize_ic_stats(ic_df)
    out = {}
    out.update(wls_sum.to_dict())
    out.update(ic_sum.to_dict())
    return out


def run_factor_stat_test(data, factor_cols):
    all_summary = []

    for fac in factor_cols:
        print("testing factor:", fac)

        wls_df = run_single_factor_wls_panel(data, fac)
        ic_df = run_single_factor_ic_panel(data, fac)

        summary = summarize_single_factor_all(wls_df, ic_df)
        summary['factor'] = fac
        all_summary.append(summary)

    res = pd.DataFrame(all_summary)
    cols = ['factor'] + [c for c in res.columns if c != 'factor']
    return res[cols]

factor_data = build_factor_dataset('2011-02-01', '2019-01-31', preprocess=True)
label_data = build_monthly_return_labels('2011-02-01', '2019-01-31')
factor_data['date'] = pd.to_datetime(factor_data['date'])
label_data['date'] = pd.to_datetime(label_data['date'])
data_for_test = factor_data.merge(label_data, on=['date', 'code'], how='left')
factor_cols = [
    'BP',
    'roe_q',
    'sales_yoy_q',
    'np_yoy_q',
    'rstr_20',
    'rstr_60',
    'rstr_180',
    'STOM_barra',
    'STOQ_barra',
    'STOA_barra'
]
stat_table = run_factor_stat_test(data_for_test, factor_cols)
print(stat_table)

# 分层回测
def assign_factor_group(df_date, factor_col, n_groups=5):
    """
    对单个月末截面，按因子值分组
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
    对某个因子逐月做分层回测
    返回 DataFrame:
    index = date
    columns = group1, group2, ..., groupN
    values = 下一期组收益
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
    根据组收益序列计算各组累计净值
    """
    nav = (1 + group_ret_df).cumprod()
    return nav

def plot_group_nav(nav_df, factor_col):
    plt.figure(figsize=(10, 6))
    for col in nav_df.columns:
        plt.plot(nav_df.index, nav_df[col], label=col)
    plt.legend()
    plt.title(f'{factor_col} grouped backtest NAV')
    plt.grid(True)
    plt.show()
    
    
def summarize_group_backtest(group_ret_df):
    """
    输入:
        group_ret_df: index=date, columns=group_1,...,group_5
                     值为每组每月收益率
    输出:
        每组的业绩指标
    """
    summary = {}

    for col in group_ret_df.columns:
        x = group_ret_df[col].dropna()

        if len(x) == 0:
            summary[col] = {
                'mean_monthly_ret': np.nan,
                'ann_ret': np.nan,
                'ann_vol': np.nan,
                'sharpe': np.nan,
                'win_rate': np.nan,
                'n_months': 0
            }
            continue
        # 方法1：用月均收益近似年化
        mean_monthly_ret = x.mean()
        ann_ret = (1 + mean_monthly_ret) ** 12 - 1
        # 月度波动率年化
        ann_vol = x.std() * np.sqrt(12)
        # 夏普比率（先假设无风险利率=0）
        if ann_vol == 0 or pd.isna(ann_vol):
            sharpe = np.nan
        else:
            sharpe = ann_ret / ann_vol
        # 月胜率
        win_rate = (x > 0).mean()
        summary[col] = {
            'mean_monthly_ret': mean_monthly_ret,
            'ann_ret': ann_ret,
            'ann_vol': ann_vol,
            'sharpe': sharpe,
            'win_rate': win_rate,
            'n_months': len(x)
        }

    return pd.DataFrame(summary).T
   
# test part 
# for fac in ['BP', 'roe_q','np_yoy_q','rstr_20','rstr_60','rstr_180','STOM_barra','STOQ_barra','STOA_barra']:
#     print("factor =", fac)
#     group_ret = run_factor_group_backtest(data_for_test, fac, n_groups=5)
#     summary = summarize_group_backtest(group_ret)
#     print(summary)

# 相关性检测
def calc_factor_corr_matrix(data, factor_cols, method='spearman'):
    """
    在全样本面板上计算因子相关性矩阵
    默认用 spearman，更稳一点
    """
    df = data[factor_cols].copy()
    corr = df.corr(method=method)
    return corr

def find_high_corr_pairs(corr_df, threshold=0.7):
    """
    从相关性矩阵中找出高相关因子对
    """
    pairs = []
    cols = corr_df.columns.tolist()

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c1, c2 = cols[i], cols[j]
            val = corr_df.loc[c1, c2]
            if pd.notna(val) and abs(val) >= threshold:
                pairs.append({
                    'factor_1': c1,
                    'factor_2': c2,
                    'corr': val,
                    'abs_corr': abs(val)
                })

    res = pd.DataFrame(pairs)
    if len(res) > 0:
        res = res.sort_values('abs_corr', ascending=False)
    return res

def calc_vif_one_date(df_date, factor_cols):
    """
    单个月末截面计算 VIF
    这里只对 factor_cols 做，不把行业哑变量和市值控制项混进去
    因为我们主要想看候选收益因子之间的共线性
    """
    X = df_date[factor_cols].copy().dropna()

    if len(X) < len(factor_cols) + 5:
        return pd.Series(index=factor_cols, dtype=float)

    # 防止常数列
    valid_cols = []
    for col in factor_cols:
        s = X[col]
        if s.std() > 0 and s.notna().sum() > 0:
            valid_cols.append(col)

    X = X[valid_cols]

    if X.shape[1] <= 1:
        return pd.Series(index=factor_cols, dtype=float)

    vif_vals = []
    arr = X.values

    for i in range(X.shape[1]):
        try:
            vif_vals.append(variance_inflation_factor(arr, i))
        except:
            vif_vals.append(np.nan)

    out = pd.Series(vif_vals, index=X.columns)

    # 补回被剔除列
    full_out = pd.Series(index=factor_cols, dtype=float)
    full_out.loc[out.index] = out
    return full_out

def calc_vif_panel(data, factor_cols):
    """
    逐月计算 VIF，返回:
    1) 每月 VIF 面板
    2) VIF 汇总统计
    """
    rows = []

    for dt, df_date in data.groupby('date'):
        vif_s = calc_vif_one_date(df_date, factor_cols)
        row = {'date': dt}
        for fac in factor_cols:
            row[fac] = vif_s.get(fac, np.nan)
        rows.append(row)

    vif_df = pd.DataFrame(rows).sort_values('date').set_index('date')

    summary = pd.DataFrame({
        'vif_mean': vif_df.mean(),
        'vif_median': vif_df.median(),
        'vif_max': vif_df.max(),
        'vif_gt_5_ratio': (vif_df > 5).mean(),
        'vif_gt_10_ratio': (vif_df > 10).mean(),
        'n_months': vif_df.notna().sum()
    }).sort_values('vif_mean', ascending=False)

    return vif_df, summary

def run_multicollinearity_check(data, factor_cols, corr_method='spearman', corr_threshold=0.7):
    """
    输出:
    - corr_df: 相关性矩阵
    - high_corr_pairs: 高相关因子对
    - vif_df: 每月VIF
    - vif_summary: VIF汇总
    """
    corr_df = calc_factor_corr_matrix(data, factor_cols, method=corr_method)
    high_corr_pairs = find_high_corr_pairs(corr_df, threshold=corr_threshold)
    vif_df, vif_summary = calc_vif_panel(data, factor_cols)

    return corr_df, high_corr_pairs, vif_df, vif_summary

factor_cols_model = [
    'BP',
    'roe_q',
    'np_yoy_q',
    'rstr_20',
    'rstr_60',
    'rstr_180'
]

corr_df, high_corr_pairs, vif_df, vif_summary = run_multicollinearity_check(
    data=data_for_test,
    factor_cols=factor_cols_model,
    corr_method='spearman',
    corr_threshold=0.7
)

print("相关性矩阵：")
print(corr_df)

print("高相关因子对：")
print(high_corr_pairs)

print("VIF汇总：")
print(vif_summary)

# 收益预测模型
def add_flipped_factors(data):
    """
    把反向有效因子翻转为正向因子
    """
    df = data.copy()

    if 'rstr_20' in df.columns:
        df['rev_20'] = -df['rstr_20']
    if 'rstr_60' in df.columns:
        df['rev_60'] = -df['rstr_60']
    if 'rstr_180' in df.columns:
        df['rev_180'] = -df['rstr_180']

    return df

def run_multifactor_wls_one_date(df_date, factor_cols):
    """
    单个月末多因子横截面回归:
    next_ret ~ factor_cols + ln_float_mktcap + industry_dummies
    WLS权重 = sqrt(raw_float_mktcap)

    返回:
        params: 各因子收益
        tvalues: 各因子t值
        n_obs: 样本数
    """
    needed_cols = ['next_ret', 'ln_float_mktcap', 'industry', 'raw_float_mktcap'] + factor_cols
    df = df_date[needed_cols].copy().dropna()

    if len(df) < max(30, len(factor_cols) + 10):
        return {
            'params': {fac: np.nan for fac in factor_cols},
            'tvalues': {fac: np.nan for fac in factor_cols},
            'n_obs': len(df)
        }

    # 行业哑变量
    ind_dummies = pd.get_dummies(df['industry'], prefix='ind', drop_first=True)

    X = pd.concat([
        df[factor_cols],
        df[['ln_float_mktcap']],
        ind_dummies
    ], axis=1)

    X = sm.add_constant(X)
    y = df['next_ret']

    # WLS 权重
    w = np.sqrt(df['raw_float_mktcap'].clip(lower=1e-8))

    model = sm.WLS(y, X, weights=w)
    res = model.fit()

    params = {}
    tvalues = {}
    for fac in factor_cols:
        params[fac] = res.params.get(fac, np.nan)
        tvalues[fac] = res.tvalues.get(fac, np.nan)

    return {
        'params': params,
        'tvalues': tvalues,
        'n_obs': len(df)
    }
    
def build_factor_return_panel(data, factor_cols):
    """
    逐月回归，生成历史因子收益时间序列表
    返回:
        factor_ret_df: index=date, columns=factor_cols
        factor_t_df:   index=date, columns=factor_cols
        meta_df:       每月样本数
    """
    ret_rows = []
    t_rows = []
    meta_rows = []

    for dt, df_date in data.groupby('date'):
        out = run_multifactor_wls_one_date(df_date, factor_cols)

        ret_row = {'date': dt}
        t_row = {'date': dt}
        for fac in factor_cols:
            ret_row[fac] = out['params'][fac]
            t_row[fac] = out['tvalues'][fac]

        meta_row = {
            'date': dt,
            'n_obs': out['n_obs']
        }

        ret_rows.append(ret_row)
        t_rows.append(t_row)
        meta_rows.append(meta_row)

    factor_ret_df = pd.DataFrame(ret_rows).sort_values('date').set_index('date')
    factor_t_df = pd.DataFrame(t_rows).sort_values('date').set_index('date')
    meta_df = pd.DataFrame(meta_rows).sort_values('date').set_index('date')

    return factor_ret_df, factor_t_df, meta_df


def forecast_factor_returns_ma(factor_ret_df, window=12):
    """
    用过去 window 个月的移动平均预测下一期因子收益
    注意 shift(1)，避免用到当期信息
    """
    factor_ret_forecast = factor_ret_df.rolling(window=window, min_periods=window).mean().shift(1)
    return factor_ret_forecast

def build_stock_expected_return(data, factor_ret_forecast_df, factor_cols):
    """
    对每个月的股票:
    expected_ret = sum(当期因子暴露 * 预测因子收益)

    返回:
        DataFrame with columns:
        ['date', 'code', 'expected_ret'] + factor_cols
    """
    out = []

    for dt, df_date in data.groupby('date'):
        if dt not in factor_ret_forecast_df.index:
            continue

        fac_forecast = factor_ret_forecast_df.loc[dt, factor_cols]
        # 如果这一期预测因子收益还不完整，跳过
        if fac_forecast.isna().any():
            continue

        tmp = df_date[['code', 'date'] + factor_cols].copy()

        # 点积
        X = tmp[factor_cols]
        tmp['expected_ret'] = X.mul(fac_forecast, axis=1).sum(axis=1)

        out.append(tmp[['date', 'code', 'expected_ret'] + factor_cols])

    if len(out) == 0:
        return pd.DataFrame(columns=['date', 'code', 'expected_ret'] + factor_cols)

    return pd.concat(out, axis=0, ignore_index=True, sort=False)

def build_return_model_pipeline(data_for_test, factor_cols, ma_window=12):
    """
    收益预测模型总流程:
    1. 多因子横截面回归 -> 历史因子收益
    2. 12个月移动平均 -> 预测下一期因子收益
    3. 股票层面 expected_ret

    返回:
        factor_ret_df
        factor_t_df
        factor_ret_forecast_df
        stock_expected_ret_df
        meta_df
    """
    # 1) 历史因子收益
    factor_ret_df, factor_t_df, meta_df = build_factor_return_panel(data_for_test, factor_cols)

    # 2) 因子收益预测
    factor_ret_forecast_df = forecast_factor_returns_ma(factor_ret_df, window=ma_window)

    # 3) 股票预期收益
    stock_expected_ret_df = build_stock_expected_return(
        data=data_for_test,
        factor_ret_forecast_df=factor_ret_forecast_df,
        factor_cols=factor_cols
    )

    return factor_ret_df, factor_t_df, factor_ret_forecast_df, stock_expected_ret_df, meta_df


# test part
# data_for_test = add_flipped_factors(data_for_test)
# factor_cols_model = [
#     'BP',
#     'roe_q',
#     'np_yoy_q',
#     'rev_20',
#     'rev_60',
#     'rev_180'
# ]
# factor_ret_df, factor_t_df, factor_ret_forecast_df, stock_expected_ret_df, meta_df = \
#     build_return_model_pipeline(
#         data_for_test=data_for_test,
#         factor_cols=factor_cols_model,
#         ma_window=12
#     )
# print("历史因子收益：")
# print(factor_ret_df.head())

# print("历史因子t值：")
# print(factor_t_df.head())

# print("预测的下一期因子收益：")
# print(factor_ret_forecast_df.head(15))

# print("股票预期收益：")
# print(stock_expected_ret_df[['date', 'code', 'expected_ret']].head())

# print("每期回归样本数：")
# print(meta_df.head())

# 风险收益模型
RISK_FACTOR_COLS = [
    'vol_1m',
    'vol_3m',
    'vol_6m',
    'vol_12m',
    'ln_float_mktcap',
    'nonlinear_size',
    'beta'
]
def get_risk_exposure_one_date(df_date, risk_factor_cols=RISK_FACTOR_COLS, use_industry=True):
    """
    单个月末风险因子暴露矩阵
    返回:
        X: DataFrame, index=code, columns=风险因子(+行业哑变量)
    """
    needed_cols = ['code', 'industry'] + risk_factor_cols
    df = df_date[needed_cols].copy()

    X = df.set_index('code')[risk_factor_cols].copy()

    if use_industry:
        ind_dummies = pd.get_dummies(df.set_index('code')['industry'], prefix='ind', drop_first=False)
        X = pd.concat([X, ind_dummies], axis=1)

    return X

def build_risk_exposure_panel(data, risk_factor_cols=RISK_FACTOR_COLS, use_industry=True):
    """
    生成全样本的风险因子暴露字典
    返回:
        exposures[date] = X
    """
    exposures = {}

    for dt, df_date in data.groupby('date'):
        X = get_risk_exposure_one_date(
            df_date,
            risk_factor_cols=risk_factor_cols,
            use_industry=use_industry
        )
        exposures[dt] = X

    return exposures
    

def run_risk_factor_regression_one_date(df_date, risk_factor_cols=RISK_FACTOR_COLS, use_industry=True):
    """
    单个月末风险因子横截面回归:
    next_ret ~ 风险因子 + 行业哑变量
    WLS 权重 = sqrt(raw_float_mktcap)

    返回:
        factor_returns: Series
        residuals: Series(index=code)
        n_obs: int
    """
    base_cols = ['code', 'next_ret', 'raw_float_mktcap', 'industry'] + risk_factor_cols
    df = df_date[base_cols].copy().dropna()

    if len(df) < max(30, len(risk_factor_cols) + 10):
        factor_names = list(risk_factor_cols)
        if use_industry:
            ind_names = sorted(pd.get_dummies(df['industry'], prefix='ind', drop_first=False).columns.tolist())
            factor_names += ind_names

        return {
            'factor_returns': pd.Series(index=factor_names, dtype=float),
            'residuals': pd.Series(index=df['code'], dtype=float),
            'n_obs': len(df)
        }

    X = df[risk_factor_cols].copy()

    if use_industry:
        ind_dummies = pd.get_dummies(df['industry'], prefix='ind', drop_first=False)
        X = pd.concat([X, ind_dummies], axis=1)

    X = sm.add_constant(X)
    y = df['next_ret']
    w = np.sqrt(df['raw_float_mktcap'].clip(lower=1e-8))

    model = sm.WLS(y, X, weights=w)
    res = model.fit()

    factor_returns = res.params.drop('const', errors='ignore')
    residuals = pd.Series(res.resid.values, index=df['code'].values, name='residual')

    return {
        'factor_returns': factor_returns,
        'residuals': residuals,
        'n_obs': len(df)
    }

def build_risk_factor_return_panel(data, risk_factor_cols=RISK_FACTOR_COLS, use_industry=True):
    """
    逐月估计风险因子收益和特异收益
    返回:
        risk_factor_ret_df: index=date, columns=风险因子(+行业因子)
        residual_panel_df:  长表 [date, code, residual]
        meta_df:            每月样本数
    """
    factor_ret_rows = []
    residual_rows = []
    meta_rows = []

    all_factor_names = None

    for dt, df_date in data.groupby('date'):
        out = run_risk_factor_regression_one_date(
            df_date,
            risk_factor_cols=risk_factor_cols,
            use_industry=use_industry
        )

        fac_ret = out['factor_returns']

        if all_factor_names is None:
            all_factor_names = fac_ret.index.tolist()

        row = {'date': dt}
        for fac in fac_ret.index:
            row[fac] = fac_ret[fac]
        factor_ret_rows.append(row)

        resid = out['residuals']
        if len(resid) > 0:
            tmp = pd.DataFrame({
                'date': dt,
                'code': resid.index,
                'residual': resid.values
            })
            residual_rows.append(tmp)

        meta_rows.append({
            'date': dt,
            'n_obs': out['n_obs']
        })

    risk_factor_ret_df = pd.DataFrame(factor_ret_rows).sort_values('date').set_index('date')
    residual_panel_df = pd.concat(residual_rows, axis=0, ignore_index=True, sort=False) if len(residual_rows) > 0 else \
        pd.DataFrame(columns=['date', 'code', 'residual'])
    meta_df = pd.DataFrame(meta_rows).sort_values('date').set_index('date')

    return risk_factor_ret_df, residual_panel_df, meta_df

def calc_factor_cov_matrix(risk_factor_ret_df, as_of_date, window=12):
    """
    计算某个时点的风险因子协方差矩阵 F
    用过去 window 个月的风险因子收益做样本协方差
    """
    df = risk_factor_ret_df.copy()
    df = df.loc[df.index < as_of_date].tail(window)

    if len(df) < window:
        return None

    F = df.cov()
    return F

def calc_specific_risk(residual_panel_df, as_of_date, window=12):
    """
    计算某个时点的个股特异风险
    返回:
        spec_var: Series(index=code), 值为特异方差
    """
    df = residual_panel_df.copy()
    df = df[df['date'] < as_of_date].copy()

    if df.empty:
        return pd.Series(dtype=float)

    # 仅取过去 window 个月
    recent_dates = sorted(df['date'].unique())[-window:]
    df = df[df['date'].isin(recent_dates)]

    spec_std = df.groupby('code')['residual'].std()
    spec_var = spec_std ** 2
    spec_var.name = 'specific_var'

    return spec_var

def calc_stock_cov_matrix_one_date(
    data,
    risk_factor_ret_df,
    residual_panel_df,
    as_of_date,
    risk_factor_cols=RISK_FACTOR_COLS,
    use_industry=True,
    cov_window=12,
    spec_window=12
):
    """
    某个月末计算股票协方差矩阵:
    Sigma = X F X' + D
    """
    df_date = data[data['date'] == as_of_date].copy()
    if df_date.empty:
        return None

    X = get_risk_exposure_one_date(
        df_date,
        risk_factor_cols=risk_factor_cols,
        use_industry=use_industry
    )

    F = calc_factor_cov_matrix(risk_factor_ret_df, as_of_date, window=cov_window)
    if F is None:
        return None

    # 对齐 X 和 F 的列
    common_cols = [c for c in X.columns if c in F.columns]
    X = X[common_cols]
    F = F.loc[common_cols, common_cols]

    # 系统性风险
    sys_cov = X.values @ F.values @ X.values.T
    sys_cov = pd.DataFrame(sys_cov, index=X.index, columns=X.index)

    # 特异风险
    spec_var = calc_specific_risk(residual_panel_df, as_of_date, window=spec_window)
    spec_var = spec_var.reindex(X.index).fillna(spec_var.median() if len(spec_var) > 0 else 0.0)

    D = np.diag(spec_var.values)
    D = pd.DataFrame(D, index=X.index, columns=X.index)

    Sigma = sys_cov + D
    return Sigma

def build_risk_model_pipeline(
    data_for_test,
    risk_factor_cols=RISK_FACTOR_COLS,
    use_industry=True
):
    """
    风险模型第一阶段:
    1. 风险因子暴露
    2. 风险因子收益
    3. 特异收益

    返回:
        exposures
        risk_factor_ret_df
        residual_panel_df
        meta_df
    """
    exposures = build_risk_exposure_panel(
        data=data_for_test,
        risk_factor_cols=risk_factor_cols,
        use_industry=use_industry
    )

    risk_factor_ret_df, residual_panel_df, meta_df = build_risk_factor_return_panel(
        data=data_for_test,
        risk_factor_cols=risk_factor_cols,
        use_industry=use_industry
    )

    return exposures, risk_factor_ret_df, residual_panel_df, meta_df

risk_factor_cols_model = [
    'vol_1m',
    'vol_3m',
    'vol_6m',
    'vol_12m',
    'ln_float_mktcap',
    'nonlinear_size',
    'beta'
]

exposures, risk_factor_ret_df, residual_panel_df, risk_meta_df = build_risk_model_pipeline(
    data_for_test=data_for_test,
    risk_factor_cols=risk_factor_cols_model,
    use_industry=True
)

print("某个月的风险暴露矩阵：")
sample_dt = sorted(exposures.keys())[0]
print(sample_dt)
print(exposures[sample_dt].head())

print("风险因子收益：")
print(risk_factor_ret_df.head())

print("特异收益：")
print(residual_panel_df.head())

print("每期风险回归样本数：")
print(risk_meta_df.head())

# available_dates = sorted(data_for_test['date'].unique())
# test_dt = available_dates[20]   # 往后取一点，确保前面有足够窗口

# Sigma = calc_stock_cov_matrix_one_date(
#     data=data_for_test,
#     risk_factor_ret_df=risk_factor_ret_df,
#     residual_panel_df=residual_panel_df,
#     as_of_date=test_dt,
#     risk_factor_cols=risk_factor_cols_model,
#     use_industry=True,
#     cov_window=12,
#     spec_window=12
# )

# print("协方差矩阵形状：")
# print(Sigma.shape if Sigma is not None else None)

# print("协方差矩阵前5x5：")
# print(Sigma.iloc[:5, :5] if Sigma is not None else None)


def optimize_portfolio_one_date(
    expected_ret_df,
    Sigma,
    as_of_date,
    risk_aversion=10.0,
    max_weight=0.02,
    top_n=None
):
    """
    单个月末组合优化

    参数:
    expected_ret_df: DataFrame, 至少包含 ['date', 'code', 'expected_ret']
    Sigma: DataFrame, 股票协方差矩阵, index/columns = code
    as_of_date: 当前调仓日
    risk_aversion: 风险厌恶系数 lambda
    max_weight: 单只股票最大权重
    top_n: 若不为 None，则只在 expected_ret 最高的 top_n 只股票里优化

    返回:
    weight_df: DataFrame ['date', 'code', 'weight']
    """
    df = expected_ret_df[expected_ret_df['date'] == as_of_date].copy()
    if df.empty or Sigma is None:
        return pd.DataFrame(columns=['date', 'code', 'weight'])

    # 只保留协方差矩阵里有的股票
    common_codes = [c for c in df['code'] if c in Sigma.index]
    df = df[df['code'].isin(common_codes)].copy()

    if df.empty:
        return pd.DataFrame(columns=['date', 'code', 'weight'])

    # 只取 top_n
    if top_n is not None:
        df = df.sort_values('expected_ret', ascending=False).head(top_n).copy()

    codes = df['code'].tolist()

    # 对齐 mu 和 Sigma
    mu = df.set_index('code')['expected_ret'].reindex(codes).values
    cov = Sigma.reindex(index=codes, columns=codes).values

    n = len(codes)
    if n == 0:
        return pd.DataFrame(columns=['date', 'code', 'weight'])

    # 初始权重：等权
    w0 = np.ones(n) / n

    # 目标函数: min( -mu'w + lambda * w' Sigma w )
    def objective(w):
        ret_term = -np.dot(mu, w)
        risk_term = risk_aversion * np.dot(w, cov @ w)
        return ret_term + risk_term

    # 约束: sum(w)=1
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    ]

    # 边界: 0 <= w <= max_weight
    bounds = [(0.0, max_weight) for _ in range(n)]

    # 先检查是否可行：n * max_weight 必须 >= 1
    if n * max_weight < 1.0:
        # 不可行时退化成等权，但等权只放到前 min feasible 个
        feasible_n = int(np.floor(1.0 / max_weight))
        feasible_n = min(feasible_n, n)

        if feasible_n == 0:
            return pd.DataFrame(columns=['date', 'code', 'weight'])

        weights = np.zeros(n)
        weights[:feasible_n] = 1.0 / feasible_n
    else:
        res = minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if not res.success:
            # 失败时退化到等权
            weights = np.ones(n) / n
        else:
            weights = res.x

    weight_df = pd.DataFrame({
        'date': as_of_date,
        'code': codes,
        'weight': weights
    })

    return weight_df

def build_optimized_portfolios(
    stock_expected_ret_df,
    data_for_test,
    risk_factor_ret_df,
    residual_panel_df,
    risk_factor_cols,
    use_industry=True,
    cov_window=12,
    spec_window=12,
    risk_aversion=10.0,
    max_weight=0.02,
    top_n=100
):
    """
    对所有调仓日做组合优化

    返回:
    portfolio_df: ['date', 'code', 'weight']
    """
    all_weights = []

    dates = sorted(stock_expected_ret_df['date'].unique())

    for dt in dates:
        print("optimizing portfolio for", dt)

        Sigma = calc_stock_cov_matrix_one_date(
            data=data_for_test,
            risk_factor_ret_df=risk_factor_ret_df,
            residual_panel_df=residual_panel_df,
            as_of_date=dt,
            risk_factor_cols=risk_factor_cols,
            use_industry=use_industry,
            cov_window=cov_window,
            spec_window=spec_window
        )

        if Sigma is None:
            continue

        weight_df = optimize_portfolio_one_date(
            expected_ret_df=stock_expected_ret_df,
            Sigma=Sigma,
            as_of_date=dt,
            risk_aversion=risk_aversion,
            max_weight=max_weight,
            top_n=top_n
        )

        if len(weight_df) > 0:
            all_weights.append(weight_df)

    if len(all_weights) == 0:
        return pd.DataFrame(columns=['date', 'code', 'weight'])

    portfolio_df = pd.concat(all_weights, axis=0, ignore_index=True, sort=False)
    return portfolio_df




#### 回测核心段
CANDIDATE_FACTOR_COLS = [
    'BP',
    'roe_q',
    'sales_yoy_q',
    'np_yoy_q',
    'rstr_20',
    'rstr_60',
    'rstr_180',
    'STOM_barra',
    'STOQ_barra',
    'STOA_barra'
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

def calc_monotonic_score_from_group(group_ret_df):
    """
    用最终净值与组号的 Spearman 相关，衡量分层单调性
    +1 接近正向单调，-1 接近反向单调
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
    
def build_factor_selection_table(train_data, candidate_factor_cols):
    """
    对候选因子做年度筛选前的综合评分
    返回表包含：
    - 单因子统计检验
    - IC
    - 分层单调性
    - 建议方向 direction (+1/-1)
    - 综合分数 score
    """
    rows = []

    for fac in candidate_factor_cols:
        try:
            wls_df = run_single_factor_wls_panel(train_data, fac)
            ic_df = run_single_factor_ic_panel(train_data, fac)

            wls_sum = summarize_single_factor_stats(wls_df)
            ic_sum = summarize_ic_stats(ic_df)

            group_ret = run_factor_group_backtest(train_data, fac, n_groups=5)
            mono_score = calc_monotonic_score_from_group(group_ret)

            ic_mean = ic_sum.get('ic_mean', np.nan)
            factor_ret_mean = wls_sum.get('factor_ret_mean', np.nan)

            # 方向判断：优先看IC，其次看分层单调性
            if pd.notna(ic_mean) and ic_mean != 0:
                direction = 1 if ic_mean > 0 else -1
            elif pd.notna(mono_score) and mono_score != 0:
                direction = 1 if mono_score > 0 else -1
            elif pd.notna(factor_ret_mean) and factor_ret_mean != 0:
                direction = 1 if factor_ret_mean > 0 else -1
            else:
                direction = 1

            # 一个简单综合分数，可后续微调
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

def apply_factor_directions(data, direction_map):
    """
    根据 direction_map 对因子做方向翻转
    direction_map: {factor: +1/-1}
    返回新的 DataFrame，并生成 directed 因子列：
    如 BP -> BP_dir, rstr_20 -> rstr_20_dir
    """
    df = data.copy()

    for fac, d in direction_map.items():
        new_col = fac + '_dir'
        df[new_col] = df[fac] * d

    return df
    
def run_multifactor_wls_one_date_with_pred(df_date, factor_cols):
    """
    单个月末多因子回归，并返回当期截面的预测值
    """
    needed_cols = ['code', 'next_ret', 'ln_float_mktcap', 'industry', 'raw_float_mktcap'] + factor_cols
    df = df_date[needed_cols].copy().dropna()

    if len(df) < max(30, len(factor_cols) + 10):
        return None

    ind_dummies = pd.get_dummies(df['industry'], prefix='ind', drop_first=True)
    X = pd.concat([df[factor_cols], df[['ln_float_mktcap']], ind_dummies], axis=1)
    X = sm.add_constant(X)
    y = df['next_ret']
    w = np.sqrt(df['raw_float_mktcap'].clip(lower=1e-8))

    model = sm.WLS(y, X, weights=w)
    res = model.fit()

    pred = pd.Series(res.predict(X), index=df['code'], name='pred')
    actual = pd.Series(y.values, index=df['code'], name='actual')

    return pred, actual


def evaluate_factor_subset(train_data, factor_cols):
    """
    评估一个因子子集的训练期表现
    用月度预测值 vs next_ret 的 Spearman 平均值作为评分
    """
    if len(factor_cols) == 0:
        return -np.inf

    monthly_ic = []

    for dt, df_date in train_data.groupby('date'):
        out = run_multifactor_wls_one_date_with_pred(df_date, factor_cols)
        if out is None:
            continue

        pred, actual = out
        tmp = pd.concat([pred, actual], axis=1).dropna()
        if len(tmp) < 30:
            continue

        ic, _ = spearmanr(tmp['pred'], tmp['actual'])
        if pd.notna(ic):
            monthly_ic.append(ic)

    if len(monthly_ic) == 0:
        return -np.inf

    return np.mean(monthly_ic)

def forward_select_factors(train_data, candidate_directed_cols, max_factors=6, min_improve=0.002):
    """
    前向逐步选择：
    每次加入一个因子，只要能提升模型平均月度IC，就保留
    """
    selected = []
    remaining = candidate_directed_cols.copy()
    best_score = -np.inf

    while len(remaining) > 0 and len(selected) < max_factors:
        best_candidate = None
        best_candidate_score = best_score

        for fac in remaining:
            trial = selected + [fac]
            score = evaluate_factor_subset(train_data, trial)

            if score > best_candidate_score:
                best_candidate_score = score
                best_candidate = fac

        if best_candidate is None:
            break

        if best_score == -np.inf or (best_candidate_score - best_score) >= min_improve:
            selected.append(best_candidate)
            remaining.remove(best_candidate)
            best_score = best_candidate_score
        else:
            break

    return selected, best_score

def preselect_factors(selection_table, factor_category_map=None, top_k_per_cat=2, min_score=0.0):
    """
    初筛：
    - 如果给了类别映射，则每个类别取前 top_k_per_cat 个
    - 否则按 score 全局排序
    """
    tab = selection_table.copy()
    tab = tab[pd.notna(tab['score'])]
    tab = tab[tab['score'] >= min_score]

    if factor_category_map is None:
        return tab['factor'].tolist()

    tab['category'] = tab['factor'].map(factor_category_map)

    picked = []
    for cat, sub in tab.groupby('category'):
        sub = sub.sort_values('score', ascending=False).head(top_k_per_cat)
        picked.extend(sub['factor'].tolist())

    return picked

def select_factors_annually(
    train_data,
    candidate_factor_cols,
    factor_category_map=None,
    top_k_per_cat=2,
    max_factors=6,
    min_improve=0.002
):
    """
    年度因子筛选总流程
    返回:
    {
        'selection_table': ...,
        'direction_map': ...,
        'preselected_factors': ...,
        'selected_directed_cols': ...,
        'selected_raw_factors': ...
    }
    """
    # 1. 单因子评分表
    selection_table = build_factor_selection_table(train_data, candidate_factor_cols)

    # 2. 方向映射
    direction_map = dict(zip(selection_table['factor'], selection_table['direction']))

    # 3. 对训练数据应用方向
    train_data_dir = apply_factor_directions(train_data, direction_map)

    # 4. 初筛
    preselected_raw_factors = preselect_factors(
        selection_table,
        factor_category_map=factor_category_map,
        top_k_per_cat=top_k_per_cat,
        min_score=0.0
    )

    # 5. directed 列名
    preselected_directed_cols = [fac + '_dir' for fac in preselected_raw_factors]

    # 6. 共线性检查可以先做，但当前先记录，不在这里硬剔除
    #    如果你愿意，后面可以在这里接自动剔除

    # 7. 前向逐步选择
    selected_directed_cols, model_score = forward_select_factors(
        train_data=train_data_dir,
        candidate_directed_cols=preselected_directed_cols,
        max_factors=max_factors,
        min_improve=min_improve
    )

    # 8. 还原原始因子名
    selected_raw_factors = [c.replace('_dir', '') for c in selected_directed_cols]

    return {
        'selection_table': selection_table,
        'direction_map': direction_map,
        'train_data_dir': train_data_dir,
        'preselected_factors': preselected_raw_factors,
        'selected_directed_cols': selected_directed_cols,
        'selected_raw_factors': selected_raw_factors,
        'model_score': model_score
    }



def initialize(context):
    set_benchmark('000300.XSHG')   # 基准
    set_option('use_real_price', True)    # 用真实价格
    set_option('avoid_future_data', True) # 关闭未来函数
    
    set_order_cost(
        OrderCost(
            close_tax=0.001,
            open_commission=0.0003,
            close_commission=0.0003,
            min_commission=5
        ),
        type='stock'
    )
    g.index_code = '000300.XSHG'
    g.top_n = 100
    g.max_weight = 0.02
    g.risk_aversion = 10.0
    g.ma_window = 12
    # g.return_factor_cols = [
    #     'BP',
    #     'roe_q',
    #     'np_yoy_q',
    #     'rstr_20',
    #     'rstr_60',
    #     'rstr_180'
    # ]
    g.risk_factor_cols = [
        'vol_1m',
        'vol_3m',
        'vol_6m',
        'vol_12m',
        'ln_float_mktcap',
        'nonlinear_size',
        'beta'
    ]
    g.factor_selection_cache = {}
    g.current_selected_factors = None
    g.current_direction_map = None
    g.current_selected_directed_cols = None
    g.current_selection_year = None
    run_monthly(rebalance, monthday=-1, time='14:30')

def get_prev_trade_date(date):
    trade_days = get_trade_days(end_date=date, count=2)
    return trade_days[-2]

def get_or_update_annual_factor_selection(context, hist_data, decision_date):
    year_key = pd.Timestamp(decision_date).year

    if year_key in g.factor_selection_cache:
        result = g.factor_selection_cache[year_key]
    else:
        result = select_factors_annually(
            train_data=hist_data,
            candidate_factor_cols=CANDIDATE_FACTOR_COLS,
            factor_category_map=FACTOR_CATEGORY_MAP,
            top_k_per_cat=2,
            max_factors=6,
            min_improve=0.002
        )
        g.factor_selection_cache[year_key] = result

    g.current_selected_factors = result['selected_raw_factors']
    g.current_direction_map = result['direction_map']
    g.current_selected_directed_cols = result['selected_directed_cols']
    g.current_selection_year = year_key

    log.info('selection year: {}'.format(year_key))
    log.info('selected raw factors: {}'.format(g.current_selected_factors))
    log.info('selected directed cols: {}'.format(g.current_selected_directed_cols))

    return result

def apply_current_factor_selection(current_fac_df, direction_map):
    return apply_factor_directions(current_fac_df, direction_map)

def rebalance(context):
    today = context.current_dt.date()
    decision_date = get_prev_trade_date(today)
    log.info('rebalance date: {}'.format(today))
    log.info('decision date: {}'.format(decision_date))
    # 1. 取当期截面因子
    fac_df = build_factor_snapshot(decision_date, index_code=g.index_code)
    # 2. 预处理
    fac_df = preprocess_factor_snapshot(fac_df)
    # 3. 加翻转因子
    fac_df = add_flipped_factors(fac_df)
    hist_data = build_history_training_data(
        end_date=decision_date,
        n_months=max(36, g.ma_window + 24)
    )
    if hist_data is None or len(hist_data) == 0:
        log.info('hist_data empty, skip')
        return
    selection_result = get_or_update_annual_factor_selection(
        context=context,
        hist_data=hist_data,
        decision_date=decision_date
    )
    
    direction_map = selection_result['direction_map']
    selected_directed_cols = selection_result['selected_directed_cols']

    if selected_directed_cols is None or len(selected_directed_cols) == 0:
        log.info('no selected factors, skip')
        return

    # 历史样本和当前截面都按方向生成 *_dir 列
    hist_data_dir = apply_factor_directions(hist_data, direction_map)
    fac_df_dir = apply_current_factor_selection(fac_df, direction_map)

    # 收益模型：只用年度选出来的 directed 因子
    expected_ret_df = build_expected_return_for_current_date(
        hist_data=hist_data_dir,
        current_fac_df=fac_df_dir,
        factor_cols=selected_directed_cols,
        ma_window=g.ma_window
    )


    if expected_ret_df is None or len(expected_ret_df) == 0:
        log.info('expected_ret_df empty, skip')
        return

    # 6. 风险模型：得到当期股票协方差矩阵
    Sigma = build_sigma_for_current_date(
        hist_data=hist_data,
        current_data=fac_df,
        risk_factor_cols=g.risk_factor_cols
    )

    if Sigma is None:
        log.info('Sigma is None, skip')
        return
    weight_df = optimize_portfolio_current_date(
        expected_ret_df=expected_ret_df,
        Sigma=Sigma,
        risk_aversion=g.risk_aversion,
        max_weight=g.max_weight,
        top_n=g.top_n
    )

    if weight_df is None or len(weight_df) == 0:
        log.info('weight_df empty, skip')
        return

    # 8. 下单调仓
    execute_trades(context, weight_df)
    
def build_history_training_data(end_date, n_months=24):
    end_date = pd.Timestamp(end_date)
    month_ends = get_month_end_trading_days('2010-01-01', end_date)
    month_ends = [pd.Timestamp(d) for d in month_ends]
    month_ends = [d for d in month_ends if d <= end_date]
    month_ends = month_ends[-(n_months + 1):]

    if len(month_ends) < n_months + 1:
        return pd.DataFrame()

    all_df = []

    for i in range(len(month_ends) - 1):
        d0 = pd.Timestamp(month_ends[i])
        d1 = pd.Timestamp(month_ends[i + 1])

        fac_df = build_factor_snapshot(d0)
        fac_df = preprocess_factor_snapshot(fac_df)
        fac_df = add_flipped_factors(fac_df)

        # 下一期收益标签
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

        fac_df = fac_df.reset_index()
        all_df.append(fac_df)

    if len(all_df) == 0:
        return pd.DataFrame()

    hist_data = pd.concat(all_df, axis=0, ignore_index=True, sort=False)
    hist_data['date'] = pd.to_datetime(hist_data['date'])
    return hist_data

def build_expected_return_for_current_date(hist_data, current_fac_df, factor_cols, ma_window=12):
    factor_ret_df, factor_t_df, meta_df = build_factor_return_panel(hist_data, factor_cols)
    factor_ret_forecast_df = forecast_factor_returns_ma(factor_ret_df, window=ma_window)

    current_date = pd.to_datetime(current_fac_df['date'].iloc[0])
    if current_date not in factor_ret_forecast_df.index:
        # 当前日期不在历史收益表里，所以取最后一个可用预测
        fac_forecast = factor_ret_forecast_df.dropna().iloc[-1]
    else:
        fac_forecast = factor_ret_forecast_df.loc[current_date]

    if fac_forecast[factor_cols].isna().any():
        return pd.DataFrame()

    tmp = current_fac_df.reset_index()[['code', 'date'] + factor_cols].copy()
    X = tmp[factor_cols]
    tmp['expected_ret'] = X.mul(fac_forecast[factor_cols], axis=1).sum(axis=1)

    return tmp[['date', 'code', 'expected_ret'] + factor_cols]

def build_sigma_for_current_date(hist_data, current_data, risk_factor_cols):
    risk_factor_ret_df, residual_panel_df, risk_meta_df = build_risk_factor_return_panel(
        hist_data,
        risk_factor_cols=risk_factor_cols,
        use_industry=True
    )

    current_date = pd.to_datetime(current_data['date'].iloc[0])

    # 为了复用你前面的函数，把 current_data 拼回去
    all_data = pd.concat([hist_data, current_data.reset_index()], axis=0, ignore_index=True, sort=False)

    Sigma = calc_stock_cov_matrix_one_date(
        data=all_data,
        risk_factor_ret_df=risk_factor_ret_df,
        residual_panel_df=residual_panel_df,
        as_of_date=current_date,
        risk_factor_cols=risk_factor_cols,
        use_industry=True,
        cov_window=12,
        spec_window=12
    )

    return Sigma

def optimize_portfolio_current_date(expected_ret_df, Sigma, risk_aversion=10.0, max_weight=0.02, top_n=100):
    current_date = pd.to_datetime(expected_ret_df['date'].iloc[0])

    return optimize_portfolio_one_date(
        expected_ret_df=expected_ret_df,
        Sigma=Sigma,
        as_of_date=current_date,
        risk_aversion=risk_aversion,
        max_weight=max_weight,
        top_n=top_n
    )

def execute_trades(context, weight_df, min_weight=1e-4, min_value=1000):
    """
    按目标权重调仓
    min_weight: 小于这个权重就视为 0
    min_value: 小于这个目标市值就不下单
    """
    if weight_df is None or len(weight_df) == 0:
        return

    weight_df = weight_df.copy()

    total_value = context.portfolio.total_value

    # 1. 先把极小权重视为 0
    weight_df.loc[weight_df['weight'].abs() < min_weight, 'weight'] = 0.0

    # 2. 过滤掉极小目标市值
    weight_df['target_value'] = weight_df['weight'] * total_value
    weight_df.loc[weight_df['target_value'].abs() < min_value, 'target_value'] = 0.0

    target_weights = dict(zip(weight_df['code'], weight_df['target_value']))

    # 先卖出不在目标池或目标市值为0的股票
    current_positions = list(context.portfolio.positions.keys())
    for stock in current_positions:
        if stock not in target_weights or target_weights[stock] == 0:
            order_target_value(stock, 0)

    # 再买入 / 调整目标市值
    for stock, target_value in target_weights.items():
        if target_value > 0:
            order_target_value(stock, target_value)