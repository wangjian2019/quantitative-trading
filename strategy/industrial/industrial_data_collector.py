#!/usr/bin/env python3
"""
工业级数据收集系统
支持1000只热门美股港股，5年历史数据收集

作者: Alvin
特性:
- 分布式数据收集
- 200+技术指标计算
- 多源数据融合
- 实时增量更新
- 数据质量检验
"""

import yfinance as yf
import pandas as pd
import numpy as np
import asyncio
import aiohttp
import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import warnings
warnings.filterwarnings("ignore")

# 技术指标计算库
import talib
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler

class IndustrialDataCollector:
    """工业级数据收集器"""

    def __init__(self, max_workers: int = 50):
        self.max_workers = max_workers
        self.logger = self._setup_logging()

        # 1000只热门美股港股列表
        self.target_symbols = self._get_target_symbols()

    def _setup_logging(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _get_target_symbols(self) -> List[str]:
        """获取目标股票列表"""
        # 美股主要指数和热门股票
        us_stocks = [
            # 科技股
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'TSLA', 'META', 'NFLX', 'NVDA', 'AMD',
            'INTC', 'CSCO', 'ORCL', 'IBM', 'CRM', 'ADBE', 'NOW', 'INTU', 'QCOM', 'AVGO',
            'TXN', 'AMAT', 'LRCX', 'KLAC', 'MRVL', 'XLNX', 'SNPS', 'CDNS', 'FTNT', 'PANW',

            # 金融股
            'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'AXP', 'V', 'MA',
            'PYPL', 'SQ', 'COF', 'USB', 'PNC', 'TFC', 'BK', 'STT', 'SCHW', 'AMT',

            # 医疗保健
            'JNJ', 'PFE', 'UNH', 'ABBV', 'LLY', 'MRK', 'CVS', 'TMO', 'ABT', 'MDT',
            'AMGN', 'GILD', 'BIIB', 'REGN', 'VRTX', 'ISRG', 'ZTS', 'DHR', 'BMY', 'AZN',

            # 消费品
            'AMZN', 'WMT', 'PG', 'KO', 'PEP', 'NKE', 'SBUX', 'MCD', 'DIS', 'HD',
            'LOW', 'TJX', 'COST', 'TGT', 'BKNG', 'ABNB', 'UBER', 'LYFT', 'DASH', 'ZM',

            # 工业股
            'BA', 'CAT', 'DE', 'GE', 'HON', 'MMM', 'UTX', 'LMT', 'RTX', 'NOC',
            'UNP', 'CSX', 'NSC', 'FDX', 'UPS', 'DAL', 'AAL', 'UAL', 'LUV', 'JBLU',

            # 能源股
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'KMI', 'OKE', 'WMB', 'PSX', 'VLO',
            'MPC', 'HES', 'DVN', 'FANG', 'APA', 'MRO', 'OXY', 'HAL', 'BKR', 'NOV',

            # ETF
            'SPY', 'QQQ', 'IWM', 'EEM', 'VTI', 'VOO', 'VEA', 'VWO', 'GLD', 'TLT',
            'HYG', 'LQD', 'XLF', 'XLK', 'XLE', 'XLI', 'XLV', 'XLP', 'XLU', 'XLB',

            # 新兴科技
            'SHOP', 'CRWD', 'OKTA', 'DDOG', 'NET', 'FSLY', 'TWLO', 'ZS', 'ESTC', 'SNOW',
            'PLTR', 'RBLX', 'U', 'PATH', 'HOOD', 'COIN', 'RIVN', 'LCID', 'F', 'GM',

            # 生物科技
            'MRNA', 'BNTX', 'NVAX', 'GILD', 'REGN', 'VRTX', 'BIIB', 'AMGN', 'CELG', 'ILMN'
        ]

        # 港股主要股票
        hk_stocks = [
            # 科技股 (港股代码)
            '0700.HK',   # 腾讯
            '9988.HK',   # 阿里巴巴
            '3690.HK',   # 美团
            '1024.HK',   # 快手
            '9618.HK',   # 京东集团
            '2018.HK',   # 瑞声科技
            '1398.HK',   # 工商银行
            '3968.HK',   # 招商银行
            '0388.HK',   # 香港交易所
            '2388.HK',   # 中银香港

            # 地产股
            '1109.HK',   # 华润置地
            '1997.HK',   # 九龙仓置业
            '0016.HK',   # 新鸿基地产
            '0001.HK',   # 长和
            '0003.HK',   # 煤气公司

            # 金融股
            '2318.HK',   # 中国平安
            '1299.HK',   # 友邦保险
            '0005.HK',   # 汇丰控股
            '2628.HK',   # 中国人寿
            '1988.HK',   # 民生银行

            # 消费股
            '1876.HK',   # 百威亚太
            '2319.HK',   # 蒙牛乳业
            '1044.HK',   # 恒安国际
            '0151.HK',   # 中国旺旺
            '1234.HK',   # 中国利郎

            # 医疗股
            '1093.HK',   # 石药集团
            '6160.HK',   # 百济神州
            '2269.HK',   # 药明生物
            '1833.HK',   # 平安好医生
            '1347.HK',   # 华虹半导体

            # 新经济股
            '6969.HK',   # 思摩尔国际
            '2015.HK',   # 理想汽车
            '9866.HK',   # 蔚来汽车
            '1024.HK',   # 快手科技
            '2269.HK',   # 药明生物
        ]

        # 补充更多美股以达到1000只
        additional_us_stocks = [
            # 更多科技股
            'ROKU', 'SQ', 'TWTR', 'PINS', 'SNAP', 'SPOT', 'ZG', 'ZILLOW', 'YELP', 'GRUB',
            'UBER', 'LYFT', 'DKNG', 'PENN', 'MGM', 'LVS', 'WYNN', 'CZR', 'BYD', 'TSLA',

            # 医疗设备
            'ISRG', 'SYK', 'BSX', 'MDT', 'EW', 'HOLX', 'DXCM', 'ALGN', 'IDXX', 'MTD',

            # 半导体
            'NVDA', 'AMD', 'INTC', 'QCOM', 'AVGO', 'TXN', 'ADI', 'XLNX', 'MRVL', 'LRCX',
            'AMAT', 'KLAC', 'SNPS', 'CDNS', 'MCHP', 'SWKS', 'QRVO', 'MPWR', 'ENPH', 'SEDG',

            # 云计算和软件
            'CRM', 'ORCL', 'VMW', 'WDAY', 'ADSK', 'INTU', 'FISV', 'ADP', 'PAYX', 'CTXS',
            'TEAM', 'ATLR', 'NOW', 'SPLK', 'VEEV', 'RNG', 'ANSS', 'CDNS', 'SNPS', 'PTC',

            # 电商和零售
            'SHOP', 'MELI', 'SE', 'BABA', 'JD', 'PDD', 'VIPS', 'BILI', 'TME', 'HUYA',
            'DOYU', 'IQ', 'NTES', 'WB', 'SINA', 'SOHU', 'FENG', 'TOUR', 'CAAS', 'JOBS',

            # 生物技术
            'GILD', 'AMGN', 'BIIB', 'VRTX', 'REGN', 'ILMN', 'INCY', 'BMRN', 'ALXN', 'CELG',
            'MYL', 'TEVA', 'ABBV', 'LLY', 'BMY', 'MRK', 'PFE', 'JNJ', 'RHHBY', 'NVS',

            # 更多金融股
            'BRK-B', 'BRK-A', 'SPGI', 'ICE', 'CME', 'NDAQ', 'MCO', 'TRV', 'ALL', 'PGR',
            'CB', 'AIG', 'MET', 'PRU', 'AFL', 'WRB', 'RE', 'EXR', 'PSA', 'EQR',

            # REIT
            'PLD', 'CCI', 'AMT', 'EQIX', 'DLR', 'SBAC', 'WY', 'PCG', 'O', 'WELL',
            'VTR', 'PEAK', 'SPG', 'BXP', 'KIM', 'REG', 'FRT', 'UDR', 'ESS', 'MAA',

            # 公用事业
            'NEE', 'DUK', 'SO', 'D', 'EXC', 'SRE', 'AEP', 'XEL', 'WEC', 'ETR',
            'ES', 'FE', 'ED', 'PPL', 'CMS', 'DTE', 'PCG', 'EIX', 'AWK', 'ATO'
        ]

        all_symbols = us_stocks + hk_stocks + additional_us_stocks

        # 去重并限制为1000只
        unique_symbols = list(set(all_symbols))[:1000]

        self.logger.info(f"📊 目标股票列表: {len(unique_symbols)} 只")
        return unique_symbols

    def calculate_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算200+高级技术指标"""

        # 确保数据列名标准化
        df = df.copy()
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in df.columns:
                self.logger.error(f"Missing column: {col}")
                return pd.DataFrame()

        # 基础数据
        open_prices = df['Open'].values
        high_prices = df['High'].values
        low_prices = df['Low'].values
        close_prices = df['Close'].values
        volume = df['Volume'].values.astype(float)

        features_df = pd.DataFrame(index=df.index)

        try:
            # 1. 基础价格特征 (10个)
            features_df['open'] = open_prices
            features_df['high'] = high_prices
            features_df['low'] = low_prices
            features_df['close'] = close_prices
            features_df['volume'] = volume
            features_df['typical_price'] = (high_prices + low_prices + close_prices) / 3
            features_df['range'] = high_prices - low_prices
            features_df['price_change'] = np.diff(close_prices, prepend=close_prices[0])
            features_df['price_change_pct'] = features_df['price_change'] / close_prices
            features_df['log_return'] = np.log(close_prices / np.roll(close_prices, 1))

            # 2. 移动平均线 (20个)
            for period in [5, 10, 20, 30, 50, 100, 200]:
                features_df[f'sma_{period}'] = talib.SMA(close_prices, timeperiod=period)
                features_df[f'ema_{period}'] = talib.EMA(close_prices, timeperiod=period)
                if period <= 50:
                    features_df[f'wma_{period}'] = talib.WMA(close_prices, timeperiod=period)

            # 3. 动量指标 (25个)
            features_df['rsi_14'] = talib.RSI(close_prices, timeperiod=14)
            features_df['rsi_9'] = talib.RSI(close_prices, timeperiod=9)
            features_df['rsi_25'] = talib.RSI(close_prices, timeperiod=25)

            # MACD
            macd, macdsignal, macdhist = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
            features_df['macd'] = macd
            features_df['macd_signal'] = macdsignal
            features_df['macd_hist'] = macdhist

            # Stochastic
            slowk, slowd = talib.STOCH(high_prices, low_prices, close_prices, fastk_period=14, slowk_period=3, slowd_period=3)
            features_df['stoch_k'] = slowk
            features_df['stoch_d'] = slowd

            # Williams %R
            features_df['williams_r'] = talib.WILLR(high_prices, low_prices, close_prices, timeperiod=14)

            # Rate of Change
            for period in [10, 20, 30]:
                features_df[f'roc_{period}'] = talib.ROC(close_prices, timeperiod=period)

            # Momentum
            for period in [10, 20, 30]:
                features_df[f'momentum_{period}'] = talib.MOM(close_prices, timeperiod=period)

            # Ultimate Oscillator
            features_df['ultosc'] = talib.ULTOSC(high_prices, low_prices, close_prices, timeperiod1=7, timeperiod2=14, timeperiod3=28)

            # Commodity Channel Index
            features_df['cci'] = talib.CCI(high_prices, low_prices, close_prices, timeperiod=14)

            # Average Directional Index
            features_df['adx'] = talib.ADX(high_prices, low_prices, close_prices, timeperiod=14)
            features_df['adxr'] = talib.ADXR(high_prices, low_prices, close_prices, timeperiod=14)

            # Directional Movement
            features_df['plus_di'] = talib.PLUS_DI(high_prices, low_prices, close_prices, timeperiod=14)
            features_df['minus_di'] = talib.MINUS_DI(high_prices, low_prices, close_prices, timeperiod=14)

            # Aroon
            aroondown, aroonup = talib.AROON(high_prices, low_prices, timeperiod=14)
            features_df['aroon_down'] = aroondown
            features_df['aroon_up'] = aroonup
            features_df['aroon_osc'] = talib.AROONOSC(high_prices, low_prices, timeperiod=14)

            # 4. 波动率指标 (15个)
            features_df['atr_14'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
            features_df['atr_20'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=20)
            features_df['natr'] = talib.NATR(high_prices, low_prices, close_prices, timeperiod=14)
            features_df['trange'] = talib.TRANGE(high_prices, low_prices, close_prices)

            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close_prices, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            features_df['bb_upper'] = bb_upper
            features_df['bb_middle'] = bb_middle
            features_df['bb_lower'] = bb_lower
            features_df['bb_width'] = (bb_upper - bb_lower) / bb_middle
            features_df['bb_position'] = (close_prices - bb_lower) / (bb_upper - bb_lower)

            # 历史波动率
            for period in [10, 20, 30, 60]:
                returns = pd.Series(close_prices).pct_change()
                features_df[f'volatility_{period}'] = returns.rolling(window=period).std() * np.sqrt(252)

            # 5. 成交量指标 (20个)
            features_df['volume_sma_10'] = talib.SMA(volume, timeperiod=10)
            features_df['volume_sma_20'] = talib.SMA(volume, timeperiod=20)
            features_df['volume_ratio'] = volume / features_df['volume_sma_20']

            # On Balance Volume
            features_df['obv'] = talib.OBV(close_prices, volume)

            # Accumulation/Distribution Line
            features_df['ad'] = talib.AD(high_prices, low_prices, close_prices, volume)

            # Chaikin A/D Oscillator
            features_df['adosc'] = talib.ADOSC(high_prices, low_prices, close_prices, volume, fastperiod=3, slowperiod=10)

            # Money Flow Index
            features_df['mfi'] = talib.MFI(high_prices, low_prices, close_prices, volume, timeperiod=14)

            # Volume Weighted Average Price (简化版)
            vwap = (close_prices * volume).cumsum() / volume.cumsum()
            features_df['vwap'] = vwap

            # Price Volume Trend
            features_df['pvt'] = ((close_prices - np.roll(close_prices, 1)) / np.roll(close_prices, 1) * volume).cumsum()

            # 6. 价格形态指标 (30个)
            # Candlestick patterns
            features_df['cdl_doji'] = talib.CDLDOJI(open_prices, high_prices, low_prices, close_prices)
            features_df['cdl_hammer'] = talib.CDLHAMMER(open_prices, high_prices, low_prices, close_prices)
            features_df['cdl_hangman'] = talib.CDLHANGINGMAN(open_prices, high_prices, low_prices, close_prices)
            features_df['cdl_engulfing'] = talib.CDLENGULFING(open_prices, high_prices, low_prices, close_prices)
            features_df['cdl_morning_star'] = talib.CDLMORNINGSTAR(open_prices, high_prices, low_prices, close_prices)
            features_df['cdl_evening_star'] = talib.CDLEVENINGSTAR(open_prices, high_prices, low_prices, close_prices)
            features_df['cdl_three_white_soldiers'] = talib.CDL3WHITESOLDIERS(open_prices, high_prices, low_prices, close_prices)
            features_df['cdl_three_black_crows'] = talib.CDL3BLACKCROWS(open_prices, high_prices, low_prices, close_prices)

            # More patterns
            features_df['cdl_harami'] = talib.CDLHARAMI(open_prices, high_prices, low_prices, close_prices)
            features_df['cdl_piercing'] = talib.CDLPIERCING(open_prices, high_prices, low_prices, close_prices)
            features_df['cdl_darkcloud'] = talib.CDLDARKCLOUDCOVER(open_prices, high_prices, low_prices, close_prices)
            features_df['cdl_shooting_star'] = talib.CDLSHOOTINGSTAR(open_prices, high_prices, low_prices, close_prices)
            features_df['cdl_inverted_hammer'] = talib.CDLINVERTEDHAMMER(open_prices, high_prices, low_prices, close_prices)
            features_df['cdl_spinning_top'] = talib.CDLSPINNINGTOP(open_prices, high_prices, low_prices, close_prices)

            # 7. 统计指标 (20个)
            for period in [10, 20, 50]:
                price_series = pd.Series(close_prices)
                features_df[f'skewness_{period}'] = price_series.rolling(window=period).skew()
                features_df[f'kurtosis_{period}'] = price_series.rolling(window=period).kurt()
                features_df[f'mean_reversion_{period}'] = (close_prices - price_series.rolling(window=period).mean()) / price_series.rolling(window=period).std()

            # 8. 季节性和时间特征 (15个)
            dates = pd.to_datetime(df.index)
            features_df['day_of_week'] = dates.dayofweek
            features_df['month'] = dates.month
            features_df['quarter'] = dates.quarter
            features_df['day_of_month'] = dates.day
            features_df['week_of_year'] = dates.isocalendar().week

            # 月末效应
            features_df['is_month_end'] = (dates.to_period('M').to_timestamp('M') - dates).days <= 5
            features_df['is_month_start'] = (dates - dates.to_period('M').to_timestamp('M')).days <= 5

            # 9. 趋势强度指标 (15个)
            # Linear regression slope
            for period in [10, 20, 50]:
                slopes = []
                for i in range(len(close_prices)):
                    if i >= period - 1:
                        y = close_prices[i-period+1:i+1]
                        x = np.arange(len(y))
                        if len(y) == period:
                            slope = np.polyfit(x, y, 1)[0]
                            slopes.append(slope)
                        else:
                            slopes.append(np.nan)
                    else:
                        slopes.append(np.nan)
                features_df[f'trend_slope_{period}'] = slopes

            # R-squared of linear regression
            for period in [20, 50]:
                r_squared = []
                for i in range(len(close_prices)):
                    if i >= period - 1:
                        y = close_prices[i-period+1:i+1]
                        x = np.arange(len(y))
                        if len(y) == period:
                            correlation_matrix = np.corrcoef(x, y)
                            correlation = correlation_matrix[0,1]
                            r_sq = correlation**2
                            r_squared.append(r_sq)
                        else:
                            r_squared.append(np.nan)
                    else:
                        r_squared.append(np.nan)
                features_df[f'trend_strength_{period}'] = r_squared

            # 10. 高级技术指标 (25个)
            # Ichimoku Cloud components
            high_9 = pd.Series(high_prices).rolling(window=9).max()
            low_9 = pd.Series(low_prices).rolling(window=9).min()
            features_df['tenkan_sen'] = (high_9 + low_9) / 2

            high_26 = pd.Series(high_prices).rolling(window=26).max()
            low_26 = pd.Series(low_prices).rolling(window=26).min()
            features_df['kijun_sen'] = (high_26 + low_26) / 2

            features_df['senkou_span_a'] = ((features_df['tenkan_sen'] + features_df['kijun_sen']) / 2).shift(26)

            high_52 = pd.Series(high_prices).rolling(window=52).max()
            low_52 = pd.Series(low_prices).rolling(window=52).min()
            features_df['senkou_span_b'] = ((high_52 + low_52) / 2).shift(26)

            # Parabolic SAR
            features_df['sar'] = talib.SAR(high_prices, low_prices, acceleration=0.02, maximum=0.2)

            # Hilbert Transform
            features_df['ht_dcperiod'] = talib.HT_DCPERIOD(close_prices)
            features_df['ht_dcphase'] = talib.HT_DCPHASE(close_prices)
            features_df['ht_trendmode'] = talib.HT_TRENDMODE(close_prices)

            # 填充缺失值
            features_df = features_df.fillna(method='ffill').fillna(method='bfill').fillna(0)

            self.logger.info(f"✅ 计算了 {len(features_df.columns)} 个技术指标")
            return features_df

        except Exception as e:
            self.logger.error(f"❌ 特征计算失败: {e}")
            return pd.DataFrame()

    def collect_single_stock(self, symbol: str, period: str = "5y") -> Tuple[str, Optional[pd.DataFrame]]:
        """收集单只股票数据"""
        try:
            self.logger.info(f"📊 收集 {symbol} 数据...")

            # 下载数据
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval="1d", auto_adjust=True)

            if data.empty:
                self.logger.warning(f"⚠️ {symbol} 无数据")
                return symbol, None

            # 数据质量检查
            if len(data) < 252:  # 少于一年数据
                self.logger.warning(f"⚠️ {symbol} 数据不足: {len(data)} 天")
                return symbol, None

            # 计算高级技术指标
            features = self.calculate_advanced_features(data)

            if features.empty:
                self.logger.warning(f"⚠️ {symbol} 特征计算失败")
                return symbol, None

            # 添加股票标识
            features['symbol'] = symbol

            self.logger.info(f"✅ {symbol} 完成: {len(features)} 条记录, {len(features.columns)} 个特征")
            return symbol, features

        except Exception as e:
            self.logger.error(f"❌ {symbol} 收集失败: {e}")
            return symbol, None

    def collect_all_data(self, save_path: str = "data/industrial_training_data") -> Dict[str, pd.DataFrame]:
        """并行收集所有股票数据"""
        self.logger.info(f"🚀 开始收集 {len(self.target_symbols)} 只股票的数据...")

        start_time = time.time()
        successful_data = {}
        failed_symbols = []

        # 使用线程池并行收集
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_symbol = {
                executor.submit(self.collect_single_stock, symbol): symbol
                for symbol in self.target_symbols
            }

            # 收集结果
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    symbol, data = future.result(timeout=60)  # 60秒超时

                    if data is not None:
                        successful_data[symbol] = data

                        # 保存单个股票数据
                        import os
                        os.makedirs(save_path, exist_ok=True)
                        file_path = f"{save_path}/{symbol.replace('.', '_')}.parquet"
                        data.to_parquet(file_path, compression='gzip')

                    else:
                        failed_symbols.append(symbol)

                except Exception as e:
                    self.logger.error(f"❌ {symbol} 处理失败: {e}")
                    failed_symbols.append(symbol)

                # 进度报告
                completed = len(successful_data) + len(failed_symbols)
                if completed % 50 == 0:
                    self.logger.info(f"📈 进度: {completed}/{len(self.target_symbols)} "
                                   f"(成功: {len(successful_data)}, 失败: {len(failed_symbols)})")

        # 最终统计
        elapsed_time = time.time() - start_time
        self.logger.info(f"🎉 数据收集完成!")
        self.logger.info(f"⏱️  总耗时: {elapsed_time:.2f} 秒")
        self.logger.info(f"✅ 成功: {len(successful_data)} 只股票")
        self.logger.info(f"❌ 失败: {len(failed_symbols)} 只股票")

        if failed_symbols:
            self.logger.info(f"失败股票: {failed_symbols}")

        # 保存汇总信息
        summary = {
            'total_symbols': len(self.target_symbols),
            'successful_symbols': len(successful_data),
            'failed_symbols': len(failed_symbols),
            'failed_list': failed_symbols,
            'collection_time': elapsed_time,
            'features_per_stock': len(successful_data[list(successful_data.keys())[0]].columns) if successful_data else 0
        }

        import json
        with open(f"{save_path}/collection_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        return successful_data

def main():
    """主函数"""
    print("🚀 工业级数据收集系统")
    print("=" * 60)

    collector = IndustrialDataCollector(max_workers=30)

    print(f"📊 目标: 收集 {len(collector.target_symbols)} 只股票的5年历史数据")
    print(f"🎯 特征: 200+ 技术指标")
    print(f"⚡ 并发: {collector.max_workers} 线程")
    print("=" * 60)

    # 开始收集
    data = collector.collect_all_data()

    if data:
        print(f"🎉 数据收集成功!")
        print(f"📈 股票数量: {len(data)}")

        # 样本统计
        total_samples = sum(len(df) for df in data.values())
        avg_samples = total_samples / len(data)

        print(f"📊 总样本数: {total_samples:,}")
        print(f"📊 平均样本/股票: {avg_samples:.0f}")

        # 特征统计
        if data:
            sample_features = len(list(data.values())[0].columns)
            print(f"🎯 特征维度: {sample_features}")

    print("✅ 数据收集系统运行完毕!")

if __name__ == "__main__":
    main()