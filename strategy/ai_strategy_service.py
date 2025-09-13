"""
Complete Python AI Trading Service
Author: Alvin
Description: AI-powered trading strategy service with machine learning models
Requirements: pip install flask pandas numpy scikit-learn joblib requests
"""

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
from datetime import datetime, timedelta
import logging
import os
import json

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

class AdvancedAIStrategy:
    """
    Advanced AI Strategy Engine
    Author: Alvin
    Uses ensemble machine learning models for trading signal generation
    """

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.model_performance = {}
        self.is_trained = False

        # Multi-model ensemble
        self.base_models = {
            'rf': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            'gb': GradientBoostingClassifier(n_estimators=100, max_depth=6, random_state=42),
            'lr': LogisticRegression(random_state=42, max_iter=1000)
        }

        # Model weights for ensemble
        self.model_weights = {'rf': 0.4, 'gb': 0.4, 'lr': 0.2}

    def prepare_features(self, current_data, indicators, history):
        """
        Prepare feature vector based on trading experience
        Author: Alvin
        """
        try:
            features = {}

            # 1. Basic technical indicator features
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

            # 2. RSI and overbought/oversold conditions
            rsi = indicators.get('RSI', 50)
            features['rsi'] = rsi / 100.0  # Normalize to 0-1
            features['rsi_oversold'] = 1 if rsi < 30 else 0
            features['rsi_overbought'] = 1 if rsi > 70 else 0
            features['rsi_neutral'] = 1 if 40 <= rsi <= 60 else 0
            features['rsi_extreme'] = 1 if rsi < 20 or rsi > 80 else 0

            # 3. MACD trend analysis
            macd = indicators.get('MACD', 0)
            features['macd'] = macd / current_price if current_price > 0 else 0
            features['macd_bullish'] = 1 if macd > 0 else 0
            features['macd_strength'] = abs(macd) / current_price if current_price > 0 else 0

            # 4. Price position and volatility
            price_position = indicators.get('PRICE_POSITION', 0.5)
            volatility = indicators.get('VOLATILITY', 0)
            features['price_position'] = price_position
            features['volatility'] = volatility
            features['high_volatility'] = 1 if volatility > 0.02 else 0
            features['low_volatility'] = 1 if volatility < 0.005 else 0

            # 5. Volume analysis
            volume_ratio = indicators.get('VOLUME_RATIO', 1)
            features['volume_ratio'] = min(volume_ratio, 5.0)  # Cap extreme values
            features['high_volume'] = 1 if volume_ratio > 2 else 0
            features['low_volume'] = 1 if volume_ratio < 0.5 else 0
            features['volume_surge'] = 1 if volume_ratio > 3 else 0

            # 6. ATR and risk measurement
            atr = indicators.get('ATR', 0)
            features['atr_ratio'] = atr / current_price if current_price > 0 else 0
            features['high_atr'] = 1 if (atr / current_price) > 0.02 else 0

            # 7. Historical price patterns
            if len(history) >= 10:
                recent_closes = [h.get('close', 0) for h in history[-10:]]
                recent_volumes = [h.get('volume', 0) for h in history[-10:]]

                features['price_trend_5'] = self.calculate_trend(recent_closes[-5:])
                features['price_trend_10'] = self.calculate_trend(recent_closes)
                features['consecutive_up'] = self.count_consecutive_direction(recent_closes, 'up')
                features['consecutive_down'] = self.count_consecutive_direction(recent_closes, 'down')

                # Volume trend
                features['volume_trend'] = self.calculate_trend(recent_volumes[-5:])

                # Price momentum
                features['momentum_3'] = (recent_closes[-1] - recent_closes[-4]) / recent_closes[-4] if len(recent_closes) >= 4 else 0
                features['momentum_5'] = (recent_closes[-1] - recent_closes[-6]) / recent_closes[-6] if len(recent_closes) >= 6 else 0
            else:
                features['price_trend_5'] = 0
                features['price_trend_10'] = 0
                features['consecutive_up'] = 0
                features['consecutive_down'] = 0
                features['volume_trend'] = 0
                features['momentum_3'] = 0
                features['momentum_5'] = 0

            # 8. Market timing features
            now = datetime.now()
            features['morning'] = 1 if 9 <= now.hour <= 11 else 0
            features['afternoon'] = 1 if 13 <= now.hour <= 15 else 0
            features['near_close'] = 1 if now.hour >= 14 and now.minute >= 30 else 0
            features['market_open'] = 1 if now.hour == 9 and now.minute <= 30 else 0

            # 9. Comprehensive signal strength
            bullish_signals = sum([
                features['ma5_ratio'] > 0.01,
                features['rsi_oversold'],
                features['macd_bullish'],
                features['high_volume'] and features['price_trend_5'] > 0,
                features['price_position'] < 0.3,  # Low position
                features['consecutive_up'] >= 2
            ])

            bearish_signals = sum([
                features['ma5_ratio'] < -0.01,
                features['rsi_overbought'],
                not features['macd_bullish'],
                features['high_volume'] and features['price_trend_5'] < 0,
                features['price_position'] > 0.7,  # High position
                features['consecutive_down'] >= 2
            ])

            features['bullish_strength'] = bullish_signals / 6.0
            features['bearish_strength'] = bearish_signals / 6.0
            features['signal_divergence'] = abs(features['bullish_strength'] - features['bearish_strength'])

            # 10. Risk indicators
            features['risk_level'] = min(1.0, features['volatility'] * 50 + features['atr_ratio'] * 25)
            features['trend_strength'] = abs(features['ma_slope']) + abs(features['momentum_5'])

            return features

        except Exception as e:
            logging.error(f"Feature preparation error: {e}")
            return self.get_default_features()

    def calculate_trend(self, prices):
        """Calculate price trend slope"""
        if len(prices) < 2:
            return 0

        x = np.arange(len(prices))
        try:
            slope = np.polyfit(x, prices, 1)[0]
            return slope / prices[0] if prices[0] > 0 else 0
        except:
            return 0

    def count_consecutive_direction(self, prices, direction):
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

    def get_default_features(self):
        """Return default feature set when calculation fails"""
        default_features = {
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
        return default_features

    def train_models(self, historical_data):
        """
        Train ensemble models
        Author: Alvin
        """
        try:
            if len(historical_data) < 100:
                logging.warning("Insufficient historical data for training")
                return False

            # Prepare training data
            features_list = []
            labels = []

            for i in range(50, len(historical_data) - 1):  # Skip first 50 for indicators
                current = historical_data[i]
                history = historical_data[:i+1]

                # Calculate basic indicators for this point
                closes = [h['close'] for h in history[-50:]]
                indicators = self.calculate_basic_indicators(closes, current)

                # Prepare features
                features = self.prepare_features(current, indicators, history[-20:])

                # Create label based on future price movement
                future_price = historical_data[i + 1]['close']
                current_price = current['close']

                price_change = (future_price - current_price) / current_price

                # Multi-class classification: 0=SELL, 1=HOLD, 2=BUY
                if price_change > 0.015:  # 1.5% up
                    label = 2  # BUY
                elif price_change < -0.015:  # 1.5% down
                    label = 0  # SELL
                else:
                    label = 1  # HOLD

                features_list.append(list(features.values()))
                labels.append(label)

            if not features_list:
                return False

            # Convert to arrays
            X = np.array(features_list)
            y = np.array(labels)

            # Store feature columns
            self.feature_columns = list(features.keys())

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train models
            for model_name, model in self.base_models.items():
                logging.info(f"Training model: {model_name}")

                # Train model
                model.fit(X_train_scaled, y_train)

                # Evaluate
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)

                # Store model and performance
                self.models[model_name] = model
                self.model_performance[model_name] = accuracy

                logging.info(f"{model_name} accuracy: {accuracy:.3f}")

            # Store scaler
            self.scalers['main'] = scaler
            self.is_trained = True

            # Save models
            self.save_models()

            logging.info("Model training completed successfully")
            return True

        except Exception as e:
            logging.error(f"Model training error: {e}")
            return False

    def calculate_basic_indicators(self, closes, current_data):
        """Calculate basic technical indicators"""
        indicators = {}

        if len(closes) >= 20:
            indicators['MA5'] = np.mean(closes[-5:])
            indicators['MA10'] = np.mean(closes[-10:])
            indicators['MA20'] = np.mean(closes[-20:])

            # Simple RSI calculation
            deltas = np.diff(closes)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)

            if len(gains) >= 14:
                avg_gain = np.mean(gains[-14:])
                avg_loss = np.mean(losses[-14:])
                rs = avg_gain / avg_loss if avg_loss > 0 else 100
                indicators['RSI'] = 100 - (100 / (1 + rs))
            else:
                indicators['RSI'] = 50

            # Simple MACD
            ema12 = closes[-1]  # Simplified
            ema26 = np.mean(closes[-26:]) if len(closes) >= 26 else closes[-1]
            indicators['MACD'] = ema12 - ema26

            # Price position
            high_20 = max(closes[-20:])
            low_20 = min(closes[-20:])
            indicators['PRICE_POSITION'] = (closes[-1] - low_20) / (high_20 - low_20) if high_20 > low_20 else 0.5

            # Volatility
            returns = np.diff(closes) / closes[:-1]
            indicators['VOLATILITY'] = np.std(returns[-20:]) if len(returns) >= 20 else 0

        else:
            # Default values for insufficient data
            indicators = {
                'MA5': closes[-1] if closes else 100,
                'MA10': closes[-1] if closes else 100,
                'MA20': closes[-1] if closes else 100,
                'RSI': 50,
                'MACD': 0,
                'PRICE_POSITION': 0.5,
                'VOLATILITY': 0.01
            }

        # Add volume and ATR placeholders
        indicators['VOLUME_RATIO'] = current_data.get('volume', 1000) / 1000
        indicators['ATR'] = indicators['VOLATILITY'] * closes[-1] if closes else 1

        return indicators

    def generate_signal(self, current_data, indicators, history):
        """
        Generate trading signal using ensemble models
        Author: Alvin
        """
        try:
            # Use simple strategy if models not trained
            if not self.is_trained:
                return self.simple_strategy(current_data, indicators)

            # Prepare features
            features = self.prepare_features(current_data, indicators, history)

            # Convert to array
            feature_array = np.array([list(features.values())]).reshape(1, -1)

            # Scale features
            feature_scaled = self.scalers['main'].transform(feature_array)

            # Get predictions from all models
            predictions = {}
            probabilities = {}

            for model_name, model in self.models.items():
                pred = model.predict(feature_scaled)[0]
                prob = model.predict_proba(feature_scaled)[0]

                predictions[model_name] = pred
                probabilities[model_name] = prob

            # Ensemble prediction using weighted voting
            ensemble_prob = np.zeros(3)  # 3 classes: SELL, HOLD, BUY

            for model_name, prob in probabilities.items():
                weight = self.model_weights[model_name]
                ensemble_prob += weight * prob

            # Get final prediction
            final_prediction = np.argmax(ensemble_prob)
            confidence = np.max(ensemble_prob)

            # Convert to action
            action_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
            action = action_map[final_prediction]

            # Generate explanation
            reason = self.generate_explanation(features, action, confidence)

            return {
                'action': action,
                'confidence': float(confidence),
                'reason': reason,
                'metadata': {
                    'model_predictions': predictions,
                    'ensemble_probabilities': ensemble_prob.tolist(),
                    'key_features': self.get_key_features(features),
                    'market_regime': self.detect_market_regime(features)
                }
            }

        except Exception as e:
            logging.error(f"Signal generation error: {e}")
            return self.simple_strategy(current_data, indicators)

    def simple_strategy(self, current_data, indicators):
        """Simple fallback strategy when ML models are not available"""
        try:
            rsi = indicators.get('RSI', 50)
            ma5 = indicators.get('MA5', 0)
            ma20 = indicators.get('MA20', 0)
            current_price = current_data.get('close', 0)

            # Simple RSI + MA strategy
            if rsi < 30 and current_price > ma5:
                return {
                    'action': 'BUY',
                    'confidence': 0.6,
                    'reason': 'RSI oversold with price above MA5'
                }
            elif rsi > 70 and current_price < ma20:
                return {
                    'action': 'SELL',
                    'confidence': 0.6,
                    'reason': 'RSI overbought with price below MA20'
                }
            else:
                return {
                    'action': 'HOLD',
                    'confidence': 0.5,
                    'reason': 'No clear signal from simple strategy'
                }

        except Exception as e:
            return {
                'action': 'HOLD',
                'confidence': 0.0,
                'reason': f'Strategy error: {str(e)}'
            }

    def generate_explanation(self, features, action, confidence):
        """Generate human-readable explanation for the trading signal"""
        explanations = []

        # Trend analysis
        if features['ma_slope'] > 0.01:
            explanations.append("Strong upward trend")
        elif features['ma_slope'] < -0.01:
            explanations.append("Strong downward trend")

        # RSI condition
        if features['rsi_oversold']:
            explanations.append("RSI oversold condition")
        elif features['rsi_overbought']:
            explanations.append("RSI overbought condition")

        # Volume analysis
        if features['high_volume']:
            explanations.append("High volume activity")

        # Momentum
        if features['momentum_5'] > 0.02:
            explanations.append("Strong positive momentum")
        elif features['momentum_5'] < -0.02:
            explanations.append("Strong negative momentum")

        # Signal strength
        if features['bullish_strength'] > 0.7:
            explanations.append("Multiple bullish signals")
        elif features['bearish_strength'] > 0.7:
            explanations.append("Multiple bearish signals")

        if not explanations:
            explanations.append("Mixed signals from technical indicators")

        return f"{action} signal with {confidence:.1%} confidence: " + ", ".join(explanations)

    def get_key_features(self, features):
        """Get the most important features for the current signal"""
        # Sort features by absolute value to find most significant
        sorted_features = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)
        return dict(sorted_features[:5])

    def detect_market_regime(self, features):
        """Detect current market regime"""
        if features['volatility'] > 0.02:
            return "High Volatility"
        elif features['trend_strength'] > 0.015:
            return "Trending Market"
        elif features['volume_ratio'] < 0.7:
            return "Low Volume"
        else:
            return "Normal Market"

    def save_models(self):
        """Save trained models to disk"""
        try:
            if not os.path.exists('models'):
                os.makedirs('models')

            for model_name, model in self.models.items():
                joblib.dump(model, f'models/{model_name}_model.pkl')

            joblib.dump(self.scalers['main'], 'models/scaler.pkl')

            # Save metadata
            metadata = {
                'feature_columns': self.feature_columns,
                'model_performance': self.model_performance,
                'is_trained': self.is_trained
            }

            with open('models/metadata.json', 'w') as f:
                json.dump(metadata, f)

            logging.info("Models saved successfully")

        except Exception as e:
            logging.error(f"Error saving models: {e}")

    def load_models(self):
        """Load trained models from disk"""
        try:
            if not os.path.exists('models/metadata.json'):
                return False

            # Load metadata
            with open('models/metadata.json', 'r') as f:
                metadata = json.load(f)

            self.feature_columns = metadata['feature_columns']
            self.model_performance = metadata['model_performance']
            self.is_trained = metadata['is_trained']

            # Load models
            for model_name in self.base_models.keys():
                if os.path.exists(f'models/{model_name}_model.pkl'):
                    self.models[model_name] = joblib.load(f'models/{model_name}_model.pkl')

            # Load scaler
            if os.path.exists('models/scaler.pkl'):
                self.scalers['main'] = joblib.load('models/scaler.pkl')

            logging.info("Models loaded successfully")
            return True

        except Exception as e:
            logging.error(f"Error loading models: {e}")
            return False

# Global strategy instance
strategy = AdvancedAIStrategy()

@app.route('/get_signal', methods=['POST'])
def get_signal():
    """
    Get trading signal endpoint
    Author: Alvin
    """
    try:
        data = request.json

        # Extract data
        current_data = data.get('current_data', {})
        indicators = data.get('indicators', {})
        history = data.get('history', [])

        # Generate signal
        result = strategy.generate_signal(current_data, indicators, history)

        return jsonify(result)

    except Exception as e:
        logging.error(f"Get signal error: {e}")
        return jsonify({
            'action': 'HOLD',
            'confidence': 0.0,
            'reason': f'Service error: {str(e)}'
        }), 500

@app.route('/train_model', methods=['POST'])
def train_model():
    """
    Train model endpoint
    Author: Alvin
    """
    try:
        data = request.json
        historical_data = data.get('historical_data', [])

        if len(historical_data) < 100:
            return jsonify({
                'success': False,
                'message': 'Insufficient historical data, need at least 100 records'
            }), 400

        success = strategy.train_models(historical_data)

        return jsonify({
            'success': success,
            'message': 'Model training completed' if success else 'Model training failed',
            'performance': strategy.model_performance if success else {}
        })

    except Exception as e:
        logging.error(f"Training error: {e}")
        return jsonify({
            'success': False,
            'message': f'Training error: {str(e)}'
        }), 500

@app.route('/backtest', methods=['POST'])
def backtest():
    """
    Backtest endpoint
    Author: Alvin
    """
    try:
        data = request.json
        historical_data = data.get('data', [])

        if len(historical_data) < 50:
            return jsonify({
                'error': 'Insufficient historical data'
            }), 400

        # Run backtest
        result = run_backtest(historical_data)

        return jsonify(result)

    except Exception as e:
        logging.error(f"Backtest error: {e}")
        return jsonify({
            'error': f'Backtest error: {str(e)}'
        }), 500

def run_backtest(historical_data):
    """
    Run backtest simulation
    Author: Alvin
    """
    try:
        # Initialize backtest parameters
        initial_capital = 10000
        capital = initial_capital
        position = 0
        trades = []
        daily_returns = []

        # Split data for training and testing
        split_point = len(historical_data) // 2
        train_data = historical_data[:split_point]
        test_data = historical_data[split_point:]

        # Train model on first half
        if not strategy.is_trained:
            strategy.train_models(train_data)

        # Run backtest on second half
        for i in range(20, len(test_data) - 1):  # Skip first 20 for indicators
            current_data = test_data[i]
            history_subset = test_data[:i+1]

            # Calculate indicators
            closes = [h['close'] for h in history_subset[-50:]]
            indicators = strategy.calculate_basic_indicators(closes, current_data)

            # Get signal
            signal_result = strategy.generate_signal(current_data, indicators, history_subset[-20:])
            action = signal_result['action']
            confidence = signal_result['confidence']

            price = current_data['close']

            # Execute trades only with high confidence
            if confidence > 0.7:
                if action == 'BUY' and position == 0 and capital > 0:
                    # Buy
                    position = capital / price
                    capital = 0
                    trades.append({
                        'timestamp': current_data.get('timestamp', i),
                        'action': 'BUY',
                        'price': price,
                        'quantity': position,
                        'confidence': confidence
                    })

                elif action == 'SELL' and position > 0:
                    # Sell
                    capital = position * price
                    trades.append({
                        'timestamp': current_data.get('timestamp', i),
                        'action': 'SELL',
                        'price': price,
                        'quantity': position,
                        'confidence': confidence
                    })
                    position = 0

            # Calculate daily return
            if i > 20:
                prev_price = test_data[i-1]['close']
                daily_return = (price - prev_price) / prev_price
                daily_returns.append(daily_return)

        # Calculate final value
        final_price = test_data[-1]['close']
        final_value = capital if position == 0 else position * final_price

        # Calculate performance metrics
        total_return = (final_value - initial_capital) / initial_capital

        # Calculate other metrics
        if len(trades) >= 2:
            returns = []
            for i in range(1, len(trades), 2):
                if i < len(trades) and trades[i]['action'] == 'SELL':
                    buy_price = trades[i-1]['price']
                    sell_price = trades[i]['price']
                    ret = (sell_price - buy_price) / buy_price
                    returns.append(ret)

            win_rate = len([r for r in returns if r > 0]) / len(returns) if returns else 0
            avg_return = np.mean(returns) if returns else 0
            sharpe_ratio = avg_return / np.std(returns) * np.sqrt(252) if returns and np.std(returns) > 0 else 0

            # Calculate maximum drawdown
            portfolio_values = [initial_capital]
            current_value = initial_capital

            for i, trade in enumerate(trades):
                if trade['action'] == 'BUY':
                    current_value = trade['quantity'] * trade['price']
                else:
                    current_value = trade['quantity'] * trade['price']
                portfolio_values.append(current_value)

            peak = initial_capital
            max_drawdown = 0
            for value in portfolio_values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
        else:
            win_rate = 0
            avg_return = 0
            sharpe_ratio = 0
            max_drawdown = 0

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': len(trades),
            'win_rate': win_rate,
            'final_value': final_value,
            'trades': trades[-10:] if len(trades) > 10 else trades,  # Return last 10 trades
            'summary': {
                'initial_capital': initial_capital,
                'final_value': final_value,
                'profit_loss': final_value - initial_capital,
                'roi_percent': total_return * 100,
                'avg_trade_return': avg_return if 'avg_return' in locals() else 0,
                'total_days': len(test_data),
                'trading_days': len([t for t in trades if t['action'] == 'BUY'])
            }
        }

    except Exception as e:
        logging.error(f"Backtest execution error: {e}")
        return {
            'error': f'Backtest execution failed: {str(e)}',
            'total_return': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'total_trades': 0,
            'win_rate': 0
        }

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    Author: Alvin
    """
    return jsonify({
        'status': 'healthy',
        'model_trained': strategy.is_trained,
        'models_available': list(strategy.models.keys()),
        'service': 'AI Trading Strategy Service',
        'author': 'Alvin',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/model_info', methods=['GET'])
def model_info():
    """
    Get model information endpoint
    Author: Alvin
    """
    return jsonify({
        'is_trained': strategy.is_trained,
        'models': list(strategy.models.keys()),
        'performance': strategy.model_performance,
        'feature_count': len(strategy.feature_columns),
        'features': strategy.feature_columns[:10] if strategy.feature_columns else [],  # First 10 features
        'model_weights': strategy.model_weights
    })

@app.route('/retrain', methods=['POST'])
def retrain_models():
    """
    Retrain models with new data
    Author: Alvin
    """
    try:
        data = request.json
        new_data = data.get('new_data', [])

        if len(new_data) < 200:
            return jsonify({
                'success': False,
                'message': 'Need at least 200 data points for retraining'
            }), 400

        # Clear existing models
        strategy.models = {}
        strategy.scalers = {}
        strategy.is_trained = False

        # Retrain with new data
        success = strategy.train_models(new_data)

        return jsonify({
            'success': success,
            'message': 'Models retrained successfully' if success else 'Retraining failed',
            'new_performance': strategy.model_performance if success else {}
        })

    except Exception as e:
        logging.error(f"Retraining error: {e}")
        return jsonify({
            'success': False,
            'message': f'Retraining error: {str(e)}'
        }), 500

@app.route('/feature_importance', methods=['GET'])
def get_feature_importance():
    """
    Get feature importance from trained models
    Author: Alvin
    """
    try:
        if not strategy.is_trained:
            return jsonify({
                'error': 'Models not trained yet'
            }), 400

        feature_importance = {}

        for model_name, model in strategy.models.items():
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                feature_importance[model_name] = dict(zip(strategy.feature_columns, importance))

        # Calculate average importance across models
        if feature_importance:
            avg_importance = {}
            for feature in strategy.feature_columns:
                scores = [feature_importance[model][feature] for model in feature_importance.keys()
                          if feature in feature_importance[model]]
                avg_importance[feature] = np.mean(scores) if scores else 0

            # Sort by importance
            sorted_importance = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)

            return jsonify({
                'average_importance': dict(sorted_importance),
                'model_specific': feature_importance,
                'top_features': sorted_importance[:10]
            })
        else:
            return jsonify({
                'error': 'No feature importance available'
            }), 400

    except Exception as e:
        logging.error(f"Feature importance error: {e}")
        return jsonify({
            'error': f'Error getting feature importance: {str(e)}'
        }), 500

def generate_sample_data(num_points=500):
    """
    Generate sample historical data for testing
    Author: Alvin
    """
    np.random.seed(42)

    data = []
    base_price = 100.0

    for i in range(num_points):
        # Simulate price movement with some trend and noise
        trend = 0.001 if i < num_points // 2 else -0.0005
        noise = np.random.normal(0, 0.02)

        price_change = trend + noise
        base_price *= (1 + price_change)

        # Generate OHLC data
        open_price = base_price
        high_price = base_price * (1 + abs(np.random.normal(0, 0.01)))
        low_price = base_price * (1 - abs(np.random.normal(0, 0.01)))
        close_price = base_price
        volume = int(np.random.normal(10000, 3000))

        data.append({
            'timestamp': (datetime.now() - timedelta(days=num_points-i)).isoformat(),
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': max(volume, 1000)
        })

    return data

@app.route('/generate_sample_data', methods=['GET'])
def get_sample_data():
    """
    Generate sample data for testing
    Author: Alvin
    """
    try:
        num_points = request.args.get('points', 500, type=int)
        data = generate_sample_data(num_points)

        return jsonify({
            'data': data,
            'count': len(data),
            'message': f'Generated {len(data)} sample data points'
        })

    except Exception as e:
        return jsonify({
            'error': f'Error generating sample data: {str(e)}'
        }), 500

@app.route('/quick_test', methods=['POST'])
def quick_test():
    """
    Quick test endpoint with sample data
    Author: Alvin
    """
    try:
        # Generate sample data
        sample_data = generate_sample_data(300)

        # Train model
        success = strategy.train_models(sample_data)

        if not success:
            return jsonify({
                'error': 'Failed to train model with sample data'
            }), 500

        # Test signal generation
        test_data = sample_data[-1]
        test_indicators = {
            'MA5': 100.5,
            'MA10': 100.2,
            'MA20': 99.8,
            'RSI': 45.5,
            'MACD': 0.3,
            'PRICE_POSITION': 0.6,
            'VOLATILITY': 0.015,
            'VOLUME_RATIO': 1.2,
            'ATR': 2.1
        }

        signal = strategy.generate_signal(test_data, test_indicators, sample_data[-20:])

        # Run mini backtest
        backtest_result = run_backtest(sample_data)

        return jsonify({
            'training_success': success,
            'model_performance': strategy.model_performance,
            'sample_signal': signal,
            'backtest_summary': {
                'total_return': backtest_result.get('total_return', 0),
                'total_trades': backtest_result.get('total_trades', 0),
                'win_rate': backtest_result.get('win_rate', 0)
            },
            'message': 'Quick test completed successfully'
        })

    except Exception as e:
        logging.error(f"Quick test error: {e}")
        return jsonify({
            'error': f'Quick test failed: {str(e)}'
        }), 500

if __name__ == '__main__':
    print("="*60)
    print("AI Trading Strategy Service")
    print("Author: Alvin")
    print("="*60)
    print("Starting Python AI service...")
    print("Required packages: flask pandas numpy scikit-learn joblib")
    print("Install with: pip install flask pandas numpy scikit-learn joblib")
    print("="*60)

    # Try to load existing models
    if strategy.load_models():
        print("✓ Existing models loaded successfully")
    else:
        print("ℹ No existing models found - will train on first use")

    print("Service endpoints:")
    print("  POST /get_signal       - Get trading signal")
    print("  POST /train_model      - Train ML models")
    print("  POST /backtest         - Run backtest")
    print("  POST /retrain          - Retrain models")
    print("  POST /quick_test       - Quick test with sample data")
    print("  GET  /health           - Health check")
    print("  GET  /model_info       - Model information")
    print("  GET  /feature_importance - Feature importance")
    print("  GET  /generate_sample_data - Generate test data")
    print("="*60)

    # Start the Flask application
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)