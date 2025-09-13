"""
Technical Indicators Utilities
Author: Alvin
Description: Technical indicator calculations
"""

import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """
    Technical Indicators Calculator
    Implements various technical analysis indicators
    """
    
    @staticmethod
    def calculate_sma(prices: List[float], period: int) -> float:
        """Simple Moving Average"""
        if len(prices) < period:
            return 0
        return np.mean(prices[-period:])
    
    @staticmethod
    def calculate_ema(prices: List[float], period: int) -> float:
        """Exponential Moving Average"""
        if len(prices) < period:
            return prices[-1] if prices else 0
        
        multiplier = 2.0 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> float:
        """Relative Strength Index"""
        if len(prices) < period + 1:
            return 50
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        if len(gains) >= period:
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])
            
            if avg_loss == 0:
                return 100
            
            rs = avg_gain / avg_loss
            return 100 - (100 / (1 + rs))
        
        return 50
    
    @staticmethod
    def calculate_macd(prices: List[float], fast: int = 12, slow: int = 26) -> Dict[str, float]:
        """MACD (Moving Average Convergence Divergence)"""
        if len(prices) < slow:
            return {'macd': 0, 'signal': 0, 'histogram': 0}
        
        ema_fast = TechnicalIndicators.calculate_ema(prices, fast)
        ema_slow = TechnicalIndicators.calculate_ema(prices, slow)
        macd_line = ema_fast - ema_slow
        
        # Signal line (9-period EMA of MACD)
        macd_history = [macd_line]  # Simplified for single point
        signal_line = TechnicalIndicators.calculate_ema(macd_history, 9)
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def calculate_bollinger_bands(prices: List[float], period: int = 20, std_dev: float = 2) -> Dict[str, float]:
        """Bollinger Bands"""
        if len(prices) < period:
            return {'upper': 0, 'middle': 0, 'lower': 0, 'bandwidth': 0, 'percent_b': 0}
        
        recent_prices = prices[-period:]
        sma = np.mean(recent_prices)
        std = np.std(recent_prices)
        
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        current_price = prices[-1]
        
        bandwidth = (upper - lower) / sma if sma > 0 else 0
        percent_b = (current_price - lower) / (upper - lower) if (upper - lower) > 0 else 0.5
        
        return {
            'upper': upper,
            'middle': sma,
            'lower': lower,
            'bandwidth': bandwidth,
            'percent_b': percent_b
        }
    
    @staticmethod
    def calculate_atr(high_prices: List[float], low_prices: List[float], 
                     close_prices: List[float], period: int = 14) -> float:
        """Average True Range"""
        if len(high_prices) < period + 1:
            return 0
        
        true_ranges = []
        for i in range(1, len(high_prices)):
            tr1 = high_prices[i] - low_prices[i]
            tr2 = abs(high_prices[i] - close_prices[i-1])
            tr3 = abs(low_prices[i] - close_prices[i-1])
            true_ranges.append(max(tr1, tr2, tr3))
        
        return np.mean(true_ranges[-period:]) if true_ranges else 0
    
    @staticmethod
    def calculate_stochastic(high_prices: List[float], low_prices: List[float], 
                           close_prices: List[float], period: int = 14) -> Dict[str, float]:
        """Stochastic Oscillator"""
        if len(high_prices) < period:
            return {'k_percent': 50, 'd_percent': 50}
        
        recent_highs = high_prices[-period:]
        recent_lows = low_prices[-period:]
        current_close = close_prices[-1]
        
        highest_high = max(recent_highs)
        lowest_low = min(recent_lows)
        
        if highest_high == lowest_low:
            k_percent = 50
        else:
            k_percent = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100
        
        # Simplified D% calculation
        d_percent = k_percent  # In practice, this would be a moving average of K%
        
        return {
            'k_percent': k_percent,
            'd_percent': d_percent
        }
    
    @staticmethod
    def calculate_williams_r(high_prices: List[float], low_prices: List[float], 
                           close_prices: List[float], period: int = 14) -> float:
        """Williams %R"""
        if len(high_prices) < period:
            return -50
        
        recent_highs = high_prices[-period:]
        recent_lows = low_prices[-period:]
        current_close = close_prices[-1]
        
        highest_high = max(recent_highs)
        lowest_low = min(recent_lows)
        
        if highest_high == lowest_low:
            return -50
        
        williams_r = ((highest_high - current_close) / (highest_high - lowest_low)) * -100
        return williams_r
    
    @staticmethod
    def calculate_commodity_channel_index(high_prices: List[float], low_prices: List[float], 
                                        close_prices: List[float], period: int = 20) -> float:
        """Commodity Channel Index"""
        if len(high_prices) < period:
            return 0
        
        # Calculate Typical Price
        typical_prices = []
        for i in range(len(high_prices)):
            tp = (high_prices[i] + low_prices[i] + close_prices[i]) / 3
            typical_prices.append(tp)
        
        recent_tp = typical_prices[-period:]
        sma_tp = np.mean(recent_tp)
        
        # Calculate Mean Deviation
        mean_deviation = np.mean([abs(tp - sma_tp) for tp in recent_tp])
        
        if mean_deviation == 0:
            return 0
        
        current_tp = typical_prices[-1]
        cci = (current_tp - sma_tp) / (0.015 * mean_deviation)
        
        return cci
    
    def calculate_all_indicators(self, ohlcv_data: List[Dict]) -> Dict[str, float]:
        """Calculate all technical indicators for given OHLCV data"""
        if len(ohlcv_data) < 20:
            return self._get_default_indicators()
        
        # Extract price arrays
        opens = [d['open'] for d in ohlcv_data]
        highs = [d['high'] for d in ohlcv_data]
        lows = [d['low'] for d in ohlcv_data]
        closes = [d['close'] for d in ohlcv_data]
        volumes = [d['volume'] for d in ohlcv_data]
        
        indicators = {}
        
        # Moving Averages
        indicators['MA5'] = self.calculate_sma(closes, 5)
        indicators['MA10'] = self.calculate_sma(closes, 10)
        indicators['MA20'] = self.calculate_sma(closes, 20)
        indicators['MA50'] = self.calculate_sma(closes, 50)
        
        # Exponential Moving Averages
        indicators['EMA12'] = self.calculate_ema(closes, 12)
        indicators['EMA26'] = self.calculate_ema(closes, 26)
        
        # Oscillators
        indicators['RSI'] = self.calculate_rsi(closes)
        
        # MACD
        macd_data = self.calculate_macd(closes)
        indicators.update(macd_data)
        
        # Bollinger Bands
        bb_data = self.calculate_bollinger_bands(closes)
        indicators['BB_UPPER'] = bb_data['upper']
        indicators['BB_MIDDLE'] = bb_data['middle']
        indicators['BB_LOWER'] = bb_data['lower']
        indicators['BB_WIDTH'] = bb_data['bandwidth']
        indicators['BB_PERCENT'] = bb_data['percent_b']
        
        # Volatility
        indicators['ATR'] = self.calculate_atr(highs, lows, closes)
        
        # Stochastic
        stoch_data = self.calculate_stochastic(highs, lows, closes)
        indicators['STOCH_K'] = stoch_data['k_percent']
        indicators['STOCH_D'] = stoch_data['d_percent']
        
        # Williams %R
        indicators['WILLIAMS_R'] = self.calculate_williams_r(highs, lows, closes)
        
        # CCI
        indicators['CCI'] = self.calculate_commodity_channel_index(highs, lows, closes)
        
        # Price position and volatility
        indicators['PRICE_POSITION'] = self._calculate_price_position(closes, 20)
        indicators['VOLATILITY'] = self._calculate_volatility(closes, 20)
        
        # Volume analysis
        indicators['VOLUME_RATIO'] = self._calculate_volume_ratio(volumes)
        
        return indicators
    
    def _calculate_price_position(self, closes: List[float], period: int) -> float:
        """Calculate price position within recent range"""
        if len(closes) < period:
            return 0.5
        
        recent_closes = closes[-period:]
        high_price = max(recent_closes)
        low_price = min(recent_closes)
        current_price = closes[-1]
        
        if high_price == low_price:
            return 0.5
        
        return (current_price - low_price) / (high_price - low_price)
    
    def _calculate_volatility(self, closes: List[float], period: int) -> float:
        """Calculate price volatility"""
        if len(closes) < period:
            return 0
        
        recent_closes = closes[-period:]
        returns = np.diff(recent_closes) / recent_closes[:-1]
        return np.std(returns)
    
    def _calculate_volume_ratio(self, volumes: List[float]) -> float:
        """Calculate volume ratio compared to average"""
        if len(volumes) < 20:
            return 1.0
        
        current_volume = volumes[-1]
        avg_volume = np.mean(volumes[-20:-1])
        
        return current_volume / avg_volume if avg_volume > 0 else 1.0
    
    def _get_default_indicators(self) -> Dict[str, float]:
        """Default indicators when insufficient data"""
        return {
            'MA5': 100, 'MA10': 100, 'MA20': 100, 'MA50': 100,
            'EMA12': 100, 'EMA26': 100,
            'RSI': 50, 'macd': 0, 'signal': 0, 'histogram': 0,
            'BB_UPPER': 105, 'BB_MIDDLE': 100, 'BB_LOWER': 95,
            'BB_WIDTH': 0.1, 'BB_PERCENT': 0.5,
            'ATR': 2, 'STOCH_K': 50, 'STOCH_D': 50,
            'WILLIAMS_R': -50, 'CCI': 0,
            'PRICE_POSITION': 0.5, 'VOLATILITY': 0.02, 'VOLUME_RATIO': 1.0
        }
