"""
纯AI模型服务 - 只负责策略计算
Author: Alvin
不包含通知功能，专注于AI模型推理
"""

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import warnings
from datetime import datetime, timedelta
import logging
import os
import json
from config import config

warnings.filterwarnings('ignore')

# Configure logging
log_config = config.get_logging_config()
import logging.handlers

os.makedirs('logs', exist_ok=True)

file_handler = logging.handlers.RotatingFileHandler(
    log_config['file_path'], 
    maxBytes=log_config['max_file_size'],
    backupCount=log_config['backup_count']
)

logging.basicConfig(
    level=getattr(logging, log_config['level']),
    format=log_config['format'],
    handlers=[
        logging.StreamHandler(),
        file_handler
    ]
)

app = Flask(__name__)

class AIModelService:
    """
    纯AI模型服务类
    只负责策略计算，不处理通知
    """

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.model_performance = {}
        self.is_trained = False
        self.config = config.get_model_config()
        self.trading_config = config.get_trading_config()

        # Multi-model ensemble
        self.base_models = {
            'rf': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            'gb': GradientBoostingClassifier(n_estimators=100, max_depth=6, random_state=42),
            'lr': LogisticRegression(random_state=42, max_iter=1000)
        }

        # Model weights for ensemble
        self.model_weights = self.config['ensemble_weights']
        
        # Create model save directory
        os.makedirs(self.config['save_path'], exist_ok=True)

    def prepare_features(self, current_data, indicators, history):
        """准备特征向量"""
        try:
            features = {}

            # 基本技术指标特征
            ma5 = indicators.get('MA5', 0)
            ma10 = indicators.get('MA10', 0)
            ma20 = indicators.get('MA20', 0)
            current_price = current_data.get('close', 0)

            # MA相对位置和趋势
            features['ma5_ratio'] = (current_price - ma5) / ma5 if ma5 > 0 else 0
            features['ma10_ratio'] = (current_price - ma10) / ma10 if ma10 > 0 else 0
            features['ma20_ratio'] = (current_price - ma20) / ma20 if ma20 > 0 else 0
            features['ma_slope'] = (ma5 - ma20) / ma20 if ma20 > 0 else 0
            features['ma_convergence'] = abs(ma5 - ma10) / ma10 if ma10 > 0 else 0

            # RSI和超买超卖条件
            rsi = indicators.get('RSI', 50)
            features['rsi'] = rsi / 100.0
            features['rsi_oversold'] = 1 if rsi < self.trading_config['rsi_oversold'] else 0
            features['rsi_overbought'] = 1 if rsi > self.trading_config['rsi_overbought'] else 0
            features['rsi_neutral'] = 1 if 40 <= rsi <= 60 else 0
            features['rsi_extreme'] = 1 if rsi < self.trading_config['rsi_extreme_low'] or rsi > self.trading_config['rsi_extreme_high'] else 0

            # MACD趋势分析
            macd = indicators.get('MACD', 0)
            features['macd'] = macd / current_price if current_price > 0 else 0
            features['macd_bullish'] = 1 if macd > 0 else 0
            features['macd_strength'] = abs(macd) / current_price if current_price > 0 else 0

            # 价格位置和波动率
            price_position = indicators.get('PRICE_POSITION', 0.5)
            volatility = indicators.get('VOLATILITY', 0)
            features['price_position'] = price_position
            features['volatility'] = volatility
            features['high_volatility'] = 1 if volatility > 0.02 else 0
            features['low_volatility'] = 1 if volatility < 0.005 else 0

            # 成交量分析
            volume_ratio = indicators.get('VOLUME_RATIO', 1)
            features['volume_ratio'] = min(volume_ratio, 5.0)
            features['high_volume'] = 1 if volume_ratio > 2 else 0
            features['low_volume'] = 1 if volume_ratio < 0.5 else 0
            features['volume_surge'] = 1 if volume_ratio > 3 else 0

            # ATR和风险测量
            atr = indicators.get('ATR', 0)
            features['atr_ratio'] = atr / current_price if current_price > 0 else 0
            features['high_atr'] = 1 if (atr / current_price) > 0.02 else 0

            # 历史价格模式
            if len(history) >= 10:
                recent_closes = [h.get('close', 0) for h in history[-10:]]
                recent_volumes = [h.get('volume', 0) for h in history[-10:]]

                features['price_trend_5'] = self.calculate_trend(recent_closes[-5:])
                features['price_trend_10'] = self.calculate_trend(recent_closes)
                features['consecutive_up'] = self.count_consecutive_direction(recent_closes, 'up')
                features['consecutive_down'] = self.count_consecutive_direction(recent_closes, 'down')

                features['volume_trend'] = self.calculate_trend(recent_volumes[-5:])
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

            # 市场时间特征
            now = datetime.now()
            features['morning'] = 1 if 9 <= now.hour <= 11 else 0
            features['afternoon'] = 1 if 13 <= now.hour <= 15 else 0
            features['near_close'] = 1 if now.hour >= 14 and now.minute >= 30 else 0
            features['market_open'] = 1 if now.hour == 9 and now.minute <= 30 else 0

            # 综合信号强度
            bullish_signals = sum([
                features['ma5_ratio'] > 0.01,
                features['rsi_oversold'],
                features['macd_bullish'],
                features['high_volume'] and features['price_trend_5'] > 0,
                features['price_position'] < 0.3,
                features['consecutive_up'] >= 2
            ])

            bearish_signals = sum([
                features['ma5_ratio'] < -0.01,
                features['rsi_overbought'],
                not features['macd_bullish'],
                features['high_volume'] and features['price_trend_5'] < 0,
                features['price_position'] > 0.7,
                features['consecutive_down'] >= 2
            ])

            features['bullish_strength'] = bullish_signals / 6.0
            features['bearish_strength'] = bearish_signals / 6.0
            features['signal_divergence'] = abs(features['bullish_strength'] - features['bearish_strength'])

            # 风险指标
            features['risk_level'] = min(1.0, features['volatility'] * 50 + features['atr_ratio'] * 25)
            features['trend_strength'] = abs(features['ma_slope']) + abs(features['momentum_5'])

            return features

        except Exception as e:
            logging.error(f"Feature preparation error: {e}")
            return self.get_default_features()

    def calculate_trend(self, prices):
        """计算价格趋势斜率"""
        if len(prices) < 2:
            return 0
        x = np.arange(len(prices))
        try:
            slope = np.polyfit(x, prices, 1)[0]
            return slope / prices[0] if prices[0] > 0 else 0
        except:
            return 0

    def count_consecutive_direction(self, prices, direction):
        """计算连续上涨/下跌次数"""
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
        """返回默认特征集"""
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

    def generate_signal(self, current_data, indicators, history):
        """生成交易信号"""
        try:
            # 如果模型未训练，使用简单策略
            if not self.is_trained:
                return self.simple_strategy(current_data, indicators)

            # 准备特征
            features = self.prepare_features(current_data, indicators, history)
            feature_array = np.array([list(features.values())]).reshape(1, -1)
            feature_scaled = self.scalers['main'].transform(feature_array)

            # 获取所有模型的预测
            predictions = {}
            probabilities = {}

            for model_name, model in self.models.items():
                pred = model.predict(feature_scaled)[0]
                prob = model.predict_proba(feature_scaled)[0]
                predictions[model_name] = pred
                probabilities[model_name] = prob

            # 集成预测
            ensemble_prob = np.zeros(3)  # SELL, HOLD, BUY
            for model_name, prob in probabilities.items():
                weight = self.model_weights[model_name]
                ensemble_prob += weight * prob

            final_prediction = np.argmax(ensemble_prob)
            confidence = np.max(ensemble_prob)

            action_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
            action = action_map[final_prediction]

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
        """简单回退策略"""
        try:
            rsi = indicators.get('RSI', 50)
            ma5 = indicators.get('MA5', 0)
            ma20 = indicators.get('MA20', 0)
            current_price = current_data.get('close', 0)

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
        """生成信号解释"""
        explanations = []
        if features['ma_slope'] > 0.01:
            explanations.append("强上升趋势")
        elif features['ma_slope'] < -0.01:
            explanations.append("强下降趋势")
        
        if features['rsi_oversold']:
            explanations.append("RSI超卖")
        elif features['rsi_overbought']:
            explanations.append("RSI超买")
        
        if features['high_volume']:
            explanations.append("高成交量")
        
        if features['momentum_5'] > 0.02:
            explanations.append("强正动量")
        elif features['momentum_5'] < -0.02:
            explanations.append("强负动量")
        
        if not explanations:
            explanations.append("技术指标混合信号")
        
        return f"{action}信号 置信度{confidence:.1%}: " + ", ".join(explanations)

    def get_key_features(self, features):
        """获取关键特征"""
        sorted_features = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)
        return dict(sorted_features[:5])

    def detect_market_regime(self, features):
        """检测市场状态"""
        if features['volatility'] > 0.02:
            return "高波动"
        elif features['trend_strength'] > 0.015:
            return "趋势市场"
        elif features['volume_ratio'] < 0.7:
            return "低成交量"
        else:
            return "正常市场"

    def train_models(self, historical_data):
        """训练模型"""
        try:
            if len(historical_data) < self.config['min_training_data']:
                logging.warning(f"历史数据不足: {len(historical_data)}")
                return False

            features_list = []
            labels = []

            for i in range(50, len(historical_data) - 1):
                current = historical_data[i]
                history = historical_data[:i+1]
                closes = [h['close'] for h in history[-50:]]
                indicators = self.calculate_basic_indicators(closes, current)
                features = self.prepare_features(current, indicators, history[-20:])
                
                future_price = historical_data[i + 1]['close']
                current_price = current['close']
                price_change = (future_price - current_price) / current_price
                threshold = self.trading_config['price_change_threshold']

                if price_change > threshold:
                    label = 2  # BUY
                elif price_change < -threshold:
                    label = 0  # SELL
                else:
                    label = 1  # HOLD

                features_list.append(list(features.values()))
                labels.append(label)

            if not features_list:
                return False

            X = np.array(features_list)
            y = np.array(labels)
            self.feature_columns = list(features.keys())

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            for model_name, model in self.base_models.items():
                logging.info(f"训练模型: {model_name}")
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                self.models[model_name] = model
                self.model_performance[model_name] = accuracy
                logging.info(f"{model_name} 准确率: {accuracy:.3f}")

            self.scalers['main'] = scaler
            self.is_trained = True
            self.save_models()
            logging.info("模型训练完成")
            return True

        except Exception as e:
            logging.error(f"模型训练错误: {e}")
            return False

    def calculate_basic_indicators(self, closes, current_data):
        """计算基本技术指标"""
        indicators = {}
        if len(closes) >= 20:
            indicators['MA5'] = np.mean(closes[-5:])
            indicators['MA10'] = np.mean(closes[-10:])
            indicators['MA20'] = np.mean(closes[-20:])

            # 简单RSI计算
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

            # 简单MACD
            ema12 = closes[-1]
            ema26 = np.mean(closes[-26:]) if len(closes) >= 26 else closes[-1]
            indicators['MACD'] = ema12 - ema26

            # 价格位置
            high_20 = max(closes[-20:])
            low_20 = min(closes[-20:])
            indicators['PRICE_POSITION'] = (closes[-1] - low_20) / (high_20 - low_20) if high_20 > low_20 else 0.5

            # 波动率
            returns = np.diff(closes) / closes[:-1]
            indicators['VOLATILITY'] = np.std(returns[-20:]) if len(returns) >= 20 else 0
        else:
            indicators = {
                'MA5': closes[-1] if closes else 100,
                'MA10': closes[-1] if closes else 100,
                'MA20': closes[-1] if closes else 100,
                'RSI': 50,
                'MACD': 0,
                'PRICE_POSITION': 0.5,
                'VOLATILITY': 0.01
            }

        indicators['VOLUME_RATIO'] = current_data.get('volume', 1000) / 1000
        indicators['ATR'] = indicators['VOLATILITY'] * closes[-1] if closes else 1
        return indicators

    def save_models(self):
        """保存模型"""
        try:
            save_path = self.config['save_path']
            for model_name, model in self.models.items():
                joblib.dump(model, f'{save_path}/{model_name}_model.pkl')
            joblib.dump(self.scalers['main'], f'{save_path}/scaler.pkl')
            
            metadata = {
                'feature_columns': self.feature_columns,
                'model_performance': self.model_performance,
                'is_trained': self.is_trained,
                'config_snapshot': {
                    'ensemble_weights': self.model_weights,
                    'price_change_threshold': self.trading_config['price_change_threshold'],
                    'min_confidence': self.trading_config['min_confidence']
                }
            }
            
            with open(f'{save_path}/metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logging.info(f"模型已保存到 {save_path}")
        except Exception as e:
            logging.error(f"保存模型错误: {e}")

    def load_models(self):
        """加载模型"""
        try:
            save_path = self.config['save_path']
            metadata_path = f'{save_path}/metadata.json'
            
            if not os.path.exists(metadata_path):
                logging.info(f"未找到现有模型: {save_path}")
                return False

            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            self.feature_columns = metadata['feature_columns']
            self.model_performance = metadata['model_performance']
            self.is_trained = metadata['is_trained']

            for model_name in self.base_models.keys():
                model_path = f'{save_path}/{model_name}_model.pkl'
                if os.path.exists(model_path):
                    self.models[model_name] = joblib.load(model_path)
                    logging.info(f"已加载模型: {model_name}")

            scaler_path = f'{save_path}/scaler.pkl'
            if os.path.exists(scaler_path):
                self.scalers['main'] = joblib.load(scaler_path)
                logging.info("已加载缩放器")

            logging.info(f"模型加载成功: {save_path}")
            return True

        except Exception as e:
            logging.error(f"加载模型错误: {e}")
            return False


# 全局模型实例
ai_model = AIModelService()

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'model_trained': ai_model.is_trained,
        'models_available': list(ai_model.models.keys()),
        'service': 'AI Model Service',
        'author': 'Alvin',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/get_signal', methods=['POST'])
def get_signal():
    """获取交易信号"""
    try:
        data = request.json
        current_data = data.get('current_data', {})
        indicators = data.get('indicators', {})
        history = data.get('history', [])

        result = ai_model.generate_signal(current_data, indicators, history)
        return jsonify(result)

    except Exception as e:
        logging.error(f"获取信号错误: {e}")
        return jsonify({
            'action': 'HOLD',
            'confidence': 0.0,
            'reason': f'服务错误: {str(e)}'
        }), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """获取模型信息"""
    return jsonify({
        'is_trained': ai_model.is_trained,
        'models': list(ai_model.models.keys()),
        'performance': ai_model.model_performance,
        'feature_count': len(ai_model.feature_columns),
        'features': ai_model.feature_columns[:10] if ai_model.feature_columns else [],
        'model_weights': ai_model.model_weights
    })

@app.route('/train_model', methods=['POST'])
def train_model():
    """训练模型"""
    try:
        data = request.json
        historical_data = data.get('historical_data', [])

        if len(historical_data) < 100:
            return jsonify({
                'success': False,
                'message': '历史数据不足，需要至少100条记录'
            }), 400

        success = ai_model.train_models(historical_data)
        return jsonify({
            'success': success,
            'message': '模型训练完成' if success else '模型训练失败',
            'performance': ai_model.model_performance if success else {}
        })

    except Exception as e:
        logging.error(f"训练错误: {e}")
        return jsonify({
            'success': False,
            'message': f'训练错误: {str(e)}'
        }), 500

if __name__ == '__main__':
    print("="*60)
    print("AI模型服务")
    print("Author: Alvin")
    print("="*60)
    print("启动Python AI模型服务...")
    print("只负责策略计算，不包含通知功能")
    print("="*60)

    # 尝试加载现有模型
    if ai_model.load_models():
        print("✓ 现有模型加载成功")
        print(f"✓ 模型性能: {ai_model.model_performance}")
    else:
        print("ℹ 未找到现有模型 - 首次使用时将训练")

    print("服务端点:")
    print("  POST /get_signal    - 获取交易信号")
    print("  POST /train_model   - 训练模型")
    print("  GET  /health        - 健康检查")
    print("  GET  /model_info    - 模型信息")
    print("="*60)

    # 获取Flask配置
    flask_config = config.get_flask_config()
    print(f"启动Flask服务器: {flask_config['host']}:{flask_config['port']}")
    print(f"调试模式: {flask_config['debug']}")
    print("="*60)

    app.run(
        host=flask_config['host'], 
        port=flask_config['port'], 
        debug=flask_config['debug'], 
        threaded=flask_config['threaded']
    )
