"""
AI Trading Strategy Service Configuration
Author: Alvin
Description: Configuration management for the AI trading service
"""

import os
import json
from typing import Dict, Any

class Config:
    """Configuration management class"""
    
    def __init__(self):
        self.config = self._load_default_config()
        self._load_from_env()
        self._load_from_file()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration"""
        return {
            # Flask服务配置
            'flask': {
                'host': '0.0.0.0',
                'port': 5001,  # 改用5001端口避免AirPlay冲突
                'debug': False,
                'threaded': True
            },
            
            # 模型配置
            'model': {
                'save_path': 'models',
                'auto_save': True,
                'min_training_data': 100,
                'feature_importance_threshold': 0.01,
                'ensemble_weights': {
                    'rf': 0.4,
                    'gb': 0.4,
                    'lr': 0.2
                }
            },
            
            # 数据处理配置
            'data': {
                'max_history_points': 100,
                'min_indicator_periods': 20,
                'volatility_window': 20,
                'volume_ratio_window': 20
            },
            
            # 交易信号配置
            'trading': {
                'min_confidence': 0.6,
                'high_confidence': 0.8,
                'price_change_threshold': 0.015,  # 1.5%
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'rsi_extreme_low': 20,
                'rsi_extreme_high': 80
            },
            
            # 回测配置
            'backtest': {
                'initial_capital': 10000,
                'min_data_points': 50,
                'train_test_split': 0.5,
                'high_confidence_only': True
            },
            
            # 日志配置
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file_path': 'logs/ai_service.log',
                'max_file_size': 10 * 1024 * 1024,  # 10MB
                'backup_count': 5
            },
            
            # 性能配置
            'performance': {
                'cache_indicators': True,
                'parallel_training': True,
                'max_workers': 4,
                'batch_size': 1000
            }
        }
    
    def _load_from_env(self):
        """Load configuration from environment variables"""
        env_mappings = {
            'FLASK_HOST': ['flask', 'host'],
            'FLASK_PORT': ['flask', 'port'],
            'FLASK_DEBUG': ['flask', 'debug'],
            'MODEL_SAVE_PATH': ['model', 'save_path'],
            'MIN_CONFIDENCE': ['trading', 'min_confidence'],
            'LOG_LEVEL': ['logging', 'level']
        }
        
        for env_var, config_path in env_mappings.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                # Convert to appropriate type
                if env_var in ['FLASK_PORT']:
                    value = int(value)
                elif env_var in ['FLASK_DEBUG']:
                    value = value.lower() in ['true', '1', 'yes']
                elif env_var in ['MIN_CONFIDENCE']:
                    value = float(value)
                
                # Set nested config value
                current = self.config
                for key in config_path[:-1]:
                    current = current[key]
                current[config_path[-1]] = value
    
    def _load_from_file(self):
        """Load configuration from config.json if exists"""
        config_file = 'config.json'
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    file_config = json.load(f)
                self._merge_config(self.config, file_config)
            except Exception as e:
                print(f"Warning: Failed to load config file {config_file}: {e}")
    
    def _merge_config(self, base_config: Dict, new_config: Dict):
        """Recursively merge configuration dictionaries"""
        for key, value in new_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._merge_config(base_config[key], value)
            else:
                base_config[key] = value
    
    def get(self, key_path: str, default=None):
        """Get configuration value by dot notation path"""
        keys = key_path.split('.')
        current = self.config
        
        try:
            for key in keys:
                current = current[key]
            return current
        except KeyError:
            return default
    
    def set(self, key_path: str, value):
        """Set configuration value by dot notation path"""
        keys = key_path.split('.')
        current = self.config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def save_to_file(self, filename='config.json'):
        """Save current configuration to file"""
        try:
            with open(filename, 'w') as f:
                json.dump(self.config, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving config to {filename}: {e}")
            return False
    
    def get_flask_config(self):
        """Get Flask-specific configuration"""
        return self.config['flask']
    
    def get_model_config(self):
        """Get model-specific configuration"""
        return self.config['model']
    
    def get_trading_config(self):
        """Get trading-specific configuration"""
        return self.config['trading']
    
    def get_logging_config(self):
        """Get logging-specific configuration"""
        return self.config['logging']

# Global configuration instance
config = Config()
