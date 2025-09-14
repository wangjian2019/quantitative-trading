"""
Backtest Service
Author: Alvin
Description: Comprehensive backtesting service with performance analysis
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

from models.ai_strategy import AITradingStrategy
from utils.technical_indicators import TechnicalIndicators

logger = logging.getLogger(__name__)

class BacktestService:
    """
    Backtest Service - Service Layer Pattern
    Handles backtesting operations and performance analysis
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.backtest_config = config.get('backtest', {})
        self.technical_indicators = TechnicalIndicators()
        
    def run_backtest(self, historical_data: List[Dict], strategy: AITradingStrategy) -> Dict[str, Any]:
        """
        Run comprehensive backtest analysis
        """
        try:
            logger.info("Starting backtest analysis...")
            
            # Initialize backtest parameters
            initial_capital = self.backtest_config.get('initial_capital', 10000)
            capital = initial_capital
            position = 0
            trades = []
            daily_returns = []
            portfolio_values = [initial_capital]
            
            # Split data for training and testing
            split_ratio = self.backtest_config.get('train_test_split', 0.5)
            split_point = int(len(historical_data) * split_ratio)
            train_data = historical_data[:split_point]
            test_data = historical_data[split_point:]
            
            # Train strategy on first half
            if not strategy.is_trained:
                logger.info("Training strategy for backtest...")
                strategy.train_models(train_data)
            
            # Run backtest on second half
            for i in range(20, len(test_data) - 1):
                current_data = test_data[i]
                history_subset = test_data[:i+1]
                
                # Calculate indicators
                ohlcv_subset = history_subset[-50:] if len(history_subset) >= 50 else history_subset
                indicators = self.technical_indicators.calculate_all_indicators(ohlcv_subset)
                indicators['current_price'] = current_data['close']
                
                # Get signal
                signal_result = strategy.generate_signal(current_data, indicators, history_subset[-20:])
                action = signal_result['action']
                confidence = signal_result['confidence']
                
                price = current_data['close']
                
                # Execute trades with high confidence
                high_confidence_only = self.backtest_config.get('high_confidence_only', True)
                min_confidence = 0.7 if high_confidence_only else 0.5
                
                if confidence > min_confidence:
                    trade_result = self._execute_backtest_trade(
                        action, price, capital, position, current_data, signal_result
                    )
                    
                    if trade_result:
                        capital = trade_result['capital']
                        position = trade_result['position']
                        if trade_result['trade']:
                            trades.append(trade_result['trade'])
                
                # Calculate portfolio value
                portfolio_value = capital + (position * price if position > 0 else 0)
                portfolio_values.append(portfolio_value)
                
                # Calculate daily return
                if len(portfolio_values) > 1:
                    daily_return = (portfolio_values[-1] - portfolio_values[-2]) / portfolio_values[-2]
                    daily_returns.append(daily_return)
            
            # Calculate final metrics
            final_price = test_data[-1]['close']
            final_value = capital + (position * final_price if position > 0 else 0)
            
            result = self._calculate_performance_metrics(
                initial_capital, final_value, trades, daily_returns, portfolio_values
            )
            
            result.update({
                'trades': trades[-20:] if len(trades) > 20 else trades,  # Last 20 trades
                'total_trades': len(trades),
                'test_period_days': len(test_data),
                'training_period_days': len(train_data)
            })
            
            logger.info(f"Backtest completed: {result['total_return']:.2%} total return")
            return result
            
        except Exception as e:
            logger.error(f"Backtest execution error: {e}")
            return {
                'error': f'Backtest failed: {str(e)}',
                'total_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0,
                'total_trades': 0, 'win_rate': 0
            }
    
    def _execute_backtest_trade(self, action: str, price: float, capital: float, 
                              position: float, current_data: Dict, signal_result: Dict) -> Optional[Dict]:
        """Execute a single backtest trade"""
        try:
            if action == 'BUY' and position == 0 and capital > 0:
                # Buy with available capital
                position_size = capital / price
                
                trade = {
                    'timestamp': current_data.get('timestamp', datetime.now().isoformat()),
                    'action': 'BUY',
                    'price': price,
                    'quantity': position_size,
                    'confidence': signal_result['confidence'],
                    'reason': signal_result['reason']
                }
                
                return {
                    'capital': 0,
                    'position': position_size,
                    'trade': trade
                }
                
            elif action == 'SELL' and position > 0:
                # Sell all position
                trade_value = position * price
                pnl = trade_value - (position * signal_result.get('entry_price', price))
                
                trade = {
                    'timestamp': current_data.get('timestamp', datetime.now().isoformat()),
                    'action': 'SELL',
                    'price': price,
                    'quantity': position,
                    'confidence': signal_result['confidence'],
                    'reason': signal_result['reason'],
                    'pnl': pnl
                }
                
                return {
                    'capital': trade_value,
                    'position': 0,
                    'trade': trade
                }
            
            return {
                'capital': capital,
                'position': position,
                'trade': None
            }
            
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            return None
    
    def _calculate_performance_metrics(self, initial_capital: float, final_value: float,
                                     trades: List[Dict], daily_returns: List[float],
                                     portfolio_values: List[float]) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        metrics = {}
        
        # Basic returns
        total_return = (final_value - initial_capital) / initial_capital
        metrics['initial_capital'] = initial_capital
        metrics['final_value'] = final_value
        metrics['total_return'] = total_return
        
        # Annualized return (assuming daily data)
        years = len(daily_returns) / 252 if daily_returns else 1
        annualized_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        metrics['annualized_return'] = annualized_return
        
        # Trade statistics
        if trades:
            winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
            losing_trades = [t for t in trades if t.get('pnl', 0) < 0]
            
            metrics['win_rate'] = len(winning_trades) / len(trades) if trades else 0
            metrics['total_trades'] = len(trades)
            metrics['winning_trades'] = len(winning_trades)
            metrics['losing_trades'] = len(losing_trades)
            
            if winning_trades:
                metrics['avg_win'] = np.mean([t['pnl'] for t in winning_trades])
                metrics['max_win'] = max([t['pnl'] for t in winning_trades])
            
            if losing_trades:
                metrics['avg_loss'] = np.mean([t['pnl'] for t in losing_trades])
                metrics['max_loss'] = min([t['pnl'] for t in losing_trades])
            
            # Profit factor
            total_wins = sum([t.get('pnl', 0) for t in winning_trades])
            total_losses = abs(sum([t.get('pnl', 0) for t in losing_trades]))
            metrics['profit_factor'] = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Risk metrics
        if daily_returns:
            metrics['volatility'] = np.std(daily_returns) * np.sqrt(252)  # Annualized
            
            # Sharpe ratio (assuming 2% risk-free rate)
            risk_free_rate = 0.02
            excess_return = annualized_return - risk_free_rate
            metrics['sharpe_ratio'] = excess_return / metrics['volatility'] if metrics['volatility'] > 0 else 0
            
            # Maximum drawdown
            metrics['max_drawdown'] = self._calculate_max_drawdown(portfolio_values)
            
            # Sortino ratio (downside deviation)
            downside_returns = [r for r in daily_returns if r < 0]
            if downside_returns:
                downside_deviation = np.std(downside_returns) * np.sqrt(252)
                metrics['sortino_ratio'] = excess_return / downside_deviation if downside_deviation > 0 else 0
            else:
                metrics['sortino_ratio'] = float('inf')
        
        return metrics
    
    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown"""
        if not portfolio_values:
            return 0
        
        peak = portfolio_values[0]
        max_drawdown = 0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
