"""
Feature Engineering Utilities
Author: Alvin
Description: Advanced feature engineering for trading signals
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Feature Engineering Class - Strategy Pattern
    Handles all feature preparation and technical indicator calculations
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.trading_config = config.get('trading', {})
        
    def prepare_features(self, current_data: Dict, indicators: Dict, history: List[Dict]) -> Dict[str, float]:
        """
        Main feature preparation method
        Uses Template Method pattern
        """
        try:
            features = {}
            
            # Basic price features
            features.update(self._extract_price_features(current_data, indicators))
            
            # Technical indicator features
            features.update(self._extract_technical_features(indicators))
            
            # Historical pattern features
            features.update(self._extract_pattern_features(history))
            
            # Market timing features
            features.update(self._extract_timing_features())
            
            # Composite signal features
            features.update(self._extract_composite_features(features))
            
            return features
            
        except Exception as e:
            logger.error(f"Feature preparation error: {e}")
            return self._get_default_features()
    
    def _extract_price_features(self, current_data: Dict, indicators: Dict) -> Dict[str, float]:
        """Extract price-based features"""
        features = {}
        
        ma5 = indicators.get('MA5', 0)
        ma10 = indicators.get('MA10', 0)
        ma20 = indicators.get('MA20', 0)
        current_price = current_data.get('close', 0)
        
        # MA relative position and trend
        features['ma5_ratio'] = (current_price - ma5) / ma5 if ma5 > 0 else 0
        features['ma10_ratio'] = (current_price - ma10) / ma10 if ma10 > 0 else 0
        features['ma20_ratio'] = (current_price - ma20) / ma20 if ma20 > 0 else 0
        features['ma_slope'] = (ma5 - ma20) / ma20 if ma20 > 0 else 0
        features['ma_convergence'] = abs(ma5 - ma10) / ma10 if ma10 > 0 else 0
        
        return features
    
    def _extract_technical_features(self, indicators: Dict) -> Dict[str, float]:
        """Extract technical indicator features"""
        features = {}
        
        # RSI features
        rsi = indicators.get('RSI', 50)
        features['rsi'] = rsi / 100.0
        features['rsi_oversold'] = 1 if rsi < self.trading_config.get('rsi_oversold', 30) else 0
        features['rsi_overbought'] = 1 if rsi > self.trading_config.get('rsi_overbought', 70) else 0
        features['rsi_neutral'] = 1 if 40 <= rsi <= 60 else 0
        features['rsi_extreme'] = 1 if (rsi < self.trading_config.get('rsi_extreme_low', 20) or 
                                       rsi > self.trading_config.get('rsi_extreme_high', 80)) else 0
        
        # MACD features
        macd = indicators.get('MACD', 0)
        current_price = indicators.get('current_price', 100)
        features['macd'] = macd / current_price if current_price > 0 else 0
        features['macd_bullish'] = 1 if macd > 0 else 0
        features['macd_strength'] = abs(macd) / current_price if current_price > 0 else 0
        
        # Volatility and position features
        features['price_position'] = indicators.get('PRICE_POSITION', 0.5)
        features['volatility'] = indicators.get('VOLATILITY', 0)
        features['high_volatility'] = 1 if features['volatility'] > 0.02 else 0
        features['low_volatility'] = 1 if features['volatility'] < 0.005 else 0
        
        # Volume features
        volume_ratio = indicators.get('VOLUME_RATIO', 1)
        features['volume_ratio'] = min(volume_ratio, 5.0)
        features['high_volume'] = 1 if volume_ratio > 2 else 0
        features['low_volume'] = 1 if volume_ratio < 0.5 else 0
        features['volume_surge'] = 1 if volume_ratio > 3 else 0
        
        # ATR features
        atr = indicators.get('ATR', 0)
        features['atr_ratio'] = atr / current_price if current_price > 0 else 0
        features['high_atr'] = 1 if (atr / current_price) > 0.02 else 0
        
        return features
    
    def _extract_pattern_features(self, history: List[Dict]) -> Dict[str, float]:
        """Extract historical pattern features"""
        features = {}
        
        if len(history) >= 10:
            recent_closes = [h.get('close', 0) for h in history[-10:]]
            recent_volumes = [h.get('volume', 0) for h in history[-10:]]
            
            features['price_trend_5'] = self._calculate_trend(recent_closes[-5:])
            features['price_trend_10'] = self._calculate_trend(recent_closes)
            features['consecutive_up'] = self._count_consecutive_direction(recent_closes, 'up')
            features['consecutive_down'] = self._count_consecutive_direction(recent_closes, 'down')
            
            # Volume trend
            features['volume_trend'] = self._calculate_trend(recent_volumes[-5:])
            
            # Price momentum
            features['momentum_3'] = self._calculate_momentum(recent_closes, 3)
            features['momentum_5'] = self._calculate_momentum(recent_closes, 5)
        else:
            # Default values for insufficient data
            default_pattern_features = {
                'price_trend_5': 0, 'price_trend_10': 0, 'consecutive_up': 0,
                'consecutive_down': 0, 'volume_trend': 0, 'momentum_3': 0, 'momentum_5': 0
            }
            features.update(default_pattern_features)
        
        return features
    
    def _extract_timing_features(self) -> Dict[str, float]:
        """Extract market timing features"""
        features = {}
        now = datetime.now()
        
        features['morning'] = 1 if 9 <= now.hour <= 11 else 0
        features['afternoon'] = 1 if 13 <= now.hour <= 15 else 0
        features['near_close'] = 1 if now.hour >= 14 and now.minute >= 30 else 0
        features['market_open'] = 1 if now.hour == 9 and now.minute <= 30 else 0
        
        return features
    
    def _extract_composite_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Extract composite signal features"""
        composite = {}
        
        # Bullish signal strength
        bullish_signals = sum([
            features.get('ma5_ratio', 0) > 0.01,
            features.get('rsi_oversold', 0),
            features.get('macd_bullish', 0),
            features.get('high_volume', 0) and features.get('price_trend_5', 0) > 0,
            features.get('price_position', 0.5) < 0.3,
            features.get('consecutive_up', 0) >= 2
        ])
        
        # Bearish signal strength
        bearish_signals = sum([
            features.get('ma5_ratio', 0) < -0.01,
            features.get('rsi_overbought', 0),
            not features.get('macd_bullish', 0),
            features.get('high_volume', 0) and features.get('price_trend_5', 0) < 0,
            features.get('price_position', 0.5) > 0.7,
            features.get('consecutive_down', 0) >= 2
        ])
        
        composite['bullish_strength'] = bullish_signals / 6.0
        composite['bearish_strength'] = bearish_signals / 6.0
        composite['signal_divergence'] = abs(composite['bullish_strength'] - composite['bearish_strength'])
        
        # Risk indicators
        composite['risk_level'] = min(1.0, features.get('volatility', 0) * 50 + features.get('atr_ratio', 0) * 25)
        composite['trend_strength'] = abs(features.get('ma_slope', 0)) + abs(features.get('momentum_5', 0))
        
        return composite
    
    def _calculate_trend(self, prices: List[float]) -> float:
        """Calculate price trend slope"""
        if len(prices) < 2:
            return 0
        
        x = np.arange(len(prices))
        try:
            slope = np.polyfit(x, prices, 1)[0]
            return slope / prices[0] if prices[0] > 0 else 0
        except:
            return 0
    
    def _count_consecutive_direction(self, prices: List[float], direction: str) -> int:
        """Count consecutive up/down movements"""
        if len(prices) < 2:
            return 0
        
        count = 0
        for i in range(len(prices) - 1, 0, -1):
            if direction == 'up' and prices[i] > prices[i-1]:
                count += 1
            elif direction == 'down' and prices[i] < prices[i-1]:
                count += 1
            else:
                break
        
        return count
    
    def _calculate_momentum(self, prices: List[float], period: int) -> float:
        """Calculate price momentum"""
        if len(prices) < period + 1:
            return 0
        
        try:
            return (prices[-1] - prices[-(period+1)]) / prices[-(period+1)]
        except (IndexError, ZeroDivisionError):
            return 0
    
    def _get_default_features(self) -> Dict[str, float]:
        """Return default feature set when calculation fails"""
        return {
            'ma5_ratio': 0, 'ma10_ratio': 0, 'ma20_ratio': 0, 'ma_slope': 0, 'ma_convergence': 0,
            'rsi': 0.5, 'rsi_oversold': 0, 'rsi_overbought': 0, 'rsi_neutral': 1, 'rsi_extreme': 0,
            'macd': 0, 'macd_bullish': 0, 'macd_strength': 0,
            'price_position': 0.5, 'volatility': 0, 'high_volatility': 0, 'low_volatility': 0,
            'volume_ratio': 1, 'high_volume': 0, 'low_volume': 0, 'volume_surge': 0,
            'atr_ratio': 0, 'high_atr': 0,
            'price_trend_5': 0, 'price_trend_10': 0, 'consecutive_up': 0, 'consecutive_down': 0,
            'volume_trend': 0, 'momentum_3': 0, 'momentum_5': 0,
            'morning': 0, 'afternoon': 0, 'near_close': 0, 'market_open': 0,
            'bullish_strength': 0, 'bearish_strength': 0, 'signal_divergence': 0,
            'risk_level': 0.5, 'trend_strength': 0
        }
