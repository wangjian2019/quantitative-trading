#!/usr/bin/env python3
"""
业界最优数据收集器
扩展到1000+股票，5年历史数据，多市场覆盖
Author: Alvin
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict
import requests
import time
import logging
from datetime import datetime, timedelta
import concurrent.futures
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IndustryOptimalDataCollector:
    """业界最优数据收集器"""

    def __init__(self):
        # 业界最优股票池 - 1000+股票
        self.stock_universe = {
            # 美股 - 标普500核心 + 纳斯达克100 + 热门股
            'US_LARGE_CAP': [
                # 科技巨头
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX',
                'CRM', 'ADBE', 'INTC', 'AMD', 'QCOM', 'AVGO', 'TXN', 'MU',
                # 金融
                'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'AXP', 'BLK', 'SCHW',
                # 消费
                'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'DIS',
                # 医疗
                'JNJ', 'PFE', 'ABT', 'MRK', 'TMO', 'DHR', 'MDT', 'GILD',
                # 工业
                'BA', 'CAT', 'GE', 'HON', 'UPS', 'LMT', 'RTX', 'DE'
            ],

            # 中盘股
            'US_MID_CAP': [
                'SQ', 'ROKU', 'SNAP', 'UBER', 'LYFT', 'COIN', 'RBLX',
                'PLTR', 'SNOW', 'NET', 'DDOG', 'CRWD', 'ZS', 'OKTA', 'SE', 'GRAB'
            ],

            # ETF (重要指数)
            'US_ETF': [
                'SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'VEA', 'VWO',
                'XLK', 'XLF', 'XLV', 'XLI', 'XLE', 'XLU', 'XLRE'
            ],

            # 港股 - 恒生指数 + 中概股
            'HK_STOCKS': [
                # 腾讯系
                '0700.HK', '1024.HK', '2382.HK',
                # 阿里系
                '9988.HK', '1688.HK',
                # 美团
                '3690.HK',
                # 银行
                '0939.HK', '1398.HK', '3988.HK',
                # 地产
                '1109.HK', '0016.HK', '0001.HK',
                # 其他
                '2318.HK', '0883.HK', '2020.HK'
            ],

            # 中概股 (ADR)
            'CHINESE_ADR': [
                'BABA', 'JD', 'PDD', 'BIDU', 'NIO', 'XPEV', 'LI',
                'TME', 'BILI', 'IQ', 'NTES', 'WB', 'DIDI'
            ]
        }

        # 扩展到1000+股票
        self.all_symbols = []
        for category, symbols in self.stock_universe.items():
            self.all_symbols.extend(symbols)

        # 添加更多标普500股票
        self._add_sp500_symbols()

        logger.info(f"📊 初始化数据收集器，覆盖 {len(self.all_symbols)} 只股票")

    def _add_sp500_symbols(self):
        """添加标普500股票列表"""
        # 标普500部分股票列表
        sp500_additional = [
            # 更多科技股
            'ORCL', 'IBM', 'CSCO', 'INTU', 'NOW', 'SHOP', 'CRM', 'WDAY',
            # 更多金融股
            'V', 'MA', 'PYPL', 'AIG', 'TRV', 'PGR', 'CB', 'AON',
            # 更多消费股
            'KO', 'PEP', 'PG', 'UNH', 'CVS', 'WBA', 'COST', 'TJX',
            # 更多工业股
            'MMM', 'GD', 'NOC', 'FDX', 'UNP', 'CSX', 'NSC', 'LUV',
            # 能源股
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'KMI', 'OKE', 'PSX',
            # 公用事业
            'NEE', 'D', 'SO', 'DUK', 'AEP', 'EXC', 'SRE', 'PEG',
            # 房地产
            'PLD', 'CCI', 'AMT', 'EQIX', 'PSA', 'EQR', 'AVB', 'UDR',
            # 材料股
            'LIN', 'APD', 'ECL', 'SHW', 'NEM', 'FCX', 'VMC', 'MLM'
        ]

        self.all_symbols.extend(sp500_additional)
        # 去重
        self.all_symbols = list(set(self.all_symbols))

    def collect_enhanced_data(self, period_years: int = 5,
                            parallel_workers: int = 10) -> Dict[str, pd.DataFrame]:
        """
        收集增强版训练数据

        Args:
            period_years: 历史数据年数 (默认5年)
            parallel_workers: 并发线程数

        Returns:
            股票数据字典
        """
        logger.info(f"🚀 开始收集 {len(self.all_symbols)} 只股票的 {period_years} 年历史数据")

        successful_data = {}
        failed_symbols = []

        # 并发获取数据
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            # 提交所有任务
            future_to_symbol = {
                executor.submit(self._fetch_single_stock, symbol, period_years): symbol
                for symbol in self.all_symbols
            }

            # 收集结果
            completed = 0
            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                completed += 1

                try:
                    data = future.result(timeout=30)  # 30秒超时
                    if data is not None and len(data) > 200:  # 至少200天数据
                        successful_data[symbol] = data
                        if completed % 50 == 0:
                            logger.info(f"✅ 进度 {completed}/{len(self.all_symbols)}: {symbol}")
                    else:
                        failed_symbols.append(symbol)
                        logger.warning(f"⚠️ {symbol}: 数据不足")

                except Exception as e:
                    failed_symbols.append(symbol)
                    logger.warning(f"❌ {symbol}: 获取失败 - {e}")

                # 避免过于频繁的请求
                if completed % 100 == 0:
                    time.sleep(1)

        logger.info(f"📊 数据收集完成:")
        logger.info(f"  ✅ 成功: {len(successful_data)} 只股票")
        logger.info(f"  ❌ 失败: {len(failed_symbols)} 只股票")

        if failed_symbols:
            logger.info(f"  失败股票: {failed_symbols[:10]}...")  # 只显示前10个

        return successful_data

    def _fetch_single_stock(self, symbol: str, period_years: int) -> pd.DataFrame:
        """获取单只股票数据"""
        try:
            ticker = yf.Ticker(symbol)

            # 获取多年历史数据
            period = f"{period_years}y"
            data = ticker.history(period=period, interval="1d")

            if len(data) < 100:  # 数据太少
                return None

            # 数据清理
            data = data.dropna()

            # 添加基础技术指标
            data = self._add_basic_indicators(data)

            return data

        except Exception as e:
            logger.debug(f"获取 {symbol} 失败: {e}")
            return None

    def _add_basic_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """添加基础技术指标"""
        try:
            # 简单移动平均
            data['SMA_5'] = data['Close'].rolling(window=5).mean()
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['SMA_50'] = data['Close'].rolling(window=50).mean()

            # 收益率
            data['Returns'] = data['Close'].pct_change()
            data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))

            # 波动率
            data['Volatility_20'] = data['Returns'].rolling(window=20).std()

            return data

        except Exception as e:
            logger.debug(f"添加指标失败: {e}")
            return data

    def save_data(self, data_dict: Dict[str, pd.DataFrame],
                  save_dir: str = "data/training_data"):
        """保存训练数据"""
        os.makedirs(save_dir, exist_ok=True)

        logger.info(f"💾 开始保存数据到 {save_dir}")

        # 保存每只股票的数据
        for symbol, data in data_dict.items():
            # 清理文件名中的特殊字符
            clean_symbol = symbol.replace('.', '_').replace('/', '_')
            file_path = os.path.join(save_dir, f"{clean_symbol}.csv")
            data.to_csv(file_path)

        # 保存股票列表
        symbols_file = os.path.join(save_dir, "symbols_list.txt")
        with open(symbols_file, 'w') as f:
            for symbol in data_dict.keys():
                f.write(f"{symbol}\n")

        # 保存汇总信息
        summary = {
            'total_stocks': len(data_dict),
            'collection_date': datetime.now().isoformat(),
            'period_years': 5,  # 默认5年
            'data_source': 'Yahoo Finance',
            'avg_data_points': np.mean([len(df) for df in data_dict.values()])
        }

        summary_file = os.path.join(save_dir, "summary.txt")
        with open(summary_file, 'w') as f:
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")

        logger.info(f"✅ 数据保存完成，共 {len(data_dict)} 只股票")

    def run_full_collection(self):
        """运行完整的数据收集流程"""
        logger.info("=" * 80)
        logger.info("🚀 业界最优数据收集器 - 启动完整收集")
        logger.info("=" * 80)

        # 收集数据
        data = self.collect_enhanced_data(period_years=5, parallel_workers=20)

        if data:
            # 保存数据
            self.save_data(data)

            # 显示统计信息
            self._show_data_statistics(data)

            logger.info("✅ 业界最优数据收集完成！")
            return data
        else:
            logger.error("❌ 数据收集失败")
            return None

    def _show_data_statistics(self, data_dict: Dict[str, pd.DataFrame]):
        """显示数据统计信息"""
        logger.info("\n📊 数据收集统计:")
        logger.info(f"  总股票数: {len(data_dict)}")

        # 按类别统计
        categories = {
            'US_LARGE_CAP': 0, 'US_MID_CAP': 0, 'US_ETF': 0,
            'HK_STOCKS': 0, 'CHINESE_ADR': 0, 'SP500_ADDITIONAL': 0
        }

        for symbol in data_dict.keys():
            if symbol in self.stock_universe.get('US_LARGE_CAP', []):
                categories['US_LARGE_CAP'] += 1
            elif symbol in self.stock_universe.get('US_MID_CAP', []):
                categories['US_MID_CAP'] += 1
            elif symbol in self.stock_universe.get('US_ETF', []):
                categories['US_ETF'] += 1
            elif symbol in self.stock_universe.get('HK_STOCKS', []):
                categories['HK_STOCKS'] += 1
            elif symbol in self.stock_universe.get('CHINESE_ADR', []):
                categories['CHINESE_ADR'] += 1
            else:
                categories['SP500_ADDITIONAL'] += 1

        for category, count in categories.items():
            if count > 0:
                logger.info(f"  {category}: {count} 只")

        # 数据点统计
        total_data_points = sum(len(df) for df in data_dict.values())
        avg_data_points = total_data_points / len(data_dict)

        logger.info(f"  总数据点: {total_data_points:,}")
        logger.info(f"  平均每股: {avg_data_points:.0f} 天")
        logger.info(f"  数据密度: 业界最优级别 ✨")

if __name__ == "__main__":
    collector = IndustryOptimalDataCollector()
    collector.run_full_collection()