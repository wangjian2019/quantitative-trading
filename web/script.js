// AI量化交易平台 JavaScript - by Alvin

class TradingPlatformUI {
    constructor() {
        this.currentTab = 'dashboard';
        this.refreshInterval = null;
        this.apiBaseUrl = 'http://localhost:5000';
        this.javaApiUrl = 'http://localhost:8080';
        this.charts = {};
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.startTimeDisplay();
        this.startAutoRefresh();
        this.loadDashboard();
        this.checkSystemHealth();
    }

    setupEventListeners() {
        // Tab switching
        document.querySelectorAll('.tab-button').forEach(button => {
            button.addEventListener('click', (e) => {
                const tab = e.target.closest('.tab-button').dataset.tab;
                this.switchTab(tab);
            });
        });

        // Settings range input
        const minConfidenceRange = document.getElementById('minConfidence');
        const minConfidenceValue = document.getElementById('minConfidenceValue');
        
        if (minConfidenceRange && minConfidenceValue) {
            minConfidenceRange.addEventListener('input', (e) => {
                minConfidenceValue.textContent = Math.round(e.target.value * 100) + '%';
            });
        }

        // Stock confidence range input
        const stockMinConfidenceRange = document.getElementById('stockMinConfidence');
        const stockMinConfidenceValue = document.getElementById('stockMinConfidenceValue');
        
        if (stockMinConfidenceRange && stockMinConfidenceValue) {
            stockMinConfidenceRange.addEventListener('input', (e) => {
                stockMinConfidenceValue.textContent = Math.round(e.target.value * 100) + '%';
            });
        }
    }

    switchTab(tabName) {
        // Update buttons
        document.querySelectorAll('.tab-button').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');

        // Update content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        document.getElementById(tabName).classList.add('active');

        this.currentTab = tabName;

        // Load tab-specific data
        switch (tabName) {
            case 'dashboard':
                this.loadDashboard();
                break;
            case 'portfolio':
                this.loadPortfolio();
                break;
            case 'signals':
                this.loadSignals();
                break;
            case 'models':
                this.loadModels();
                break;
            case 'stocks':
                this.loadStocks();
                break;
            case 'backtest':
                this.loadBacktestHistory();
                break;
        }
    }

    async loadDashboard() {
        try {
            // Load health status
            const healthResponse = await fetch(`${this.apiBaseUrl}/health`);
            if (healthResponse.ok) {
                const health = await healthResponse.json();
                this.updateSystemStatus(health);
                this.updateHealthMetrics(health);
            }

            // Load recent signals
            this.loadRecentSignals();

        } catch (error) {
            console.error('Failed to load dashboard:', error);
            this.showToast('仪表板加载失败', 'error');
        }
    }

    async loadRecentSignals() {
        try {
            // Simulate recent signals for demo
            const signals = [
                {
                    symbol: 'AAPL',
                    action: 'BUY',
                    price: 175.32,
                    confidence: 0.85,
                    reason: 'RSI oversold with strong momentum',
                    timestamp: new Date().toISOString()
                },
                {
                    symbol: 'TSLA',
                    action: 'SELL', 
                    price: 250.15,
                    confidence: 0.78,
                    reason: 'Overbought conditions detected',
                    timestamp: new Date(Date.now() - 300000).toISOString()
                }
            ];

            this.updateLatestSignals(signals);
        } catch (error) {
            console.error('Failed to load recent signals:', error);
        }
    }

    async loadPortfolio() {
        try {
            // Simulate portfolio data
            const portfolio = {
                totalValue: 125000,
                totalReturn: 0.25,
                todayPnL: 1250,
                positions: [
                    { symbol: 'AAPL', shares: 100, avgCost: 150.0, currentPrice: 175.32, value: 17532 },
                    { symbol: 'TSLA', shares: 50, avgCost: 220.0, currentPrice: 250.15, value: 12507.5 },
                    { symbol: 'MSFT', shares: 80, avgCost: 300.0, currentPrice: 350.67, value: 28053.6 }
                ]
            };

            this.updatePortfolio(portfolio);
            this.createPortfolioChart(portfolio.positions);

        } catch (error) {
            console.error('Failed to load portfolio:', error);
            this.showToast('投资组合加载失败', 'error');
        }
    }

    async loadSignals() {
        try {
            // Simulate signals history
            const signals = [
                {
                    symbol: 'AAPL', action: 'BUY', price: 175.32, confidence: 0.85,
                    reason: 'RSI oversold with strong momentum', timestamp: new Date().toISOString()
                },
                {
                    symbol: 'TSLA', action: 'SELL', price: 250.15, confidence: 0.78,
                    reason: 'Overbought conditions detected', timestamp: new Date(Date.now() - 300000).toISOString()
                },
                {
                    symbol: 'MSFT', action: 'HOLD', price: 350.67, confidence: 0.65,
                    reason: 'Mixed signals from indicators', timestamp: new Date(Date.now() - 600000).toISOString()
                }
            ];

            this.updateSignalsHistory(signals);

        } catch (error) {
            console.error('Failed to load signals:', error);
            this.showToast('信号历史加载失败', 'error');
        }
    }

    async loadModels() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/model_info`);
            if (response.ok) {
                const modelInfo = await response.json();
                this.updateModelStatus(modelInfo);
                
                if (modelInfo.is_trained) {
                    this.loadFeatureImportance();
                }
            }
        } catch (error) {
            console.error('Failed to load models:', error);
            this.showToast('模型信息加载失败', 'error');
        }
    }

    async loadStocks() {
        try {
            // Load stock configuration from portfolio.json
            const response = await fetch('/portfolio.json');
            if (response.ok) {
                const portfolioData = await response.json();
                this.updateStocksList(portfolioData.symbols);
                this.loadStockQuotes(portfolioData.symbols);
            }
        } catch (error) {
            console.error('Failed to load stocks:', error);
            this.showToast('股票配置加载失败', 'error');
        }
    }

    async loadFeatureImportance() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/models/feature-importance`);
            if (response.ok) {
                const importance = await response.json();
                this.createFeatureImportanceChart(importance.top_features || []);
            }
        } catch (error) {
            console.error('Failed to load feature importance:', error);
        }
    }

    updateSystemStatus(health) {
        const indicator = document.getElementById('systemStatus');
        if (health.status === 'healthy') {
            indicator.className = 'status-indicator';
            indicator.innerHTML = '<i class="fas fa-circle"></i><span>系统正常</span>';
        } else {
            indicator.className = 'status-indicator offline';
            indicator.innerHTML = '<i class="fas fa-circle"></i><span>系统异常</span>';
        }
    }

    updateHealthMetrics(health) {
        const healthDiv = document.getElementById('healthStatus');
        healthDiv.innerHTML = `
            <div class="health-metrics">
                <div class="metric">
                    <div class="metric-value">95.2%</div>
                    <div class="metric-label">信号成功率</div>
                </div>
                <div class="metric">
                    <div class="metric-value">87.8%</div>
                    <div class="metric-label">交易成功率</div>
                </div>
                <div class="metric">
                    <div class="metric-value">4</div>
                    <div class="metric-label">活跃线程</div>
                </div>
                <div class="metric">
                    <div class="metric-value">256MB</div>
                    <div class="metric-label">内存使用</div>
                </div>
            </div>
        `;
    }

    updateLatestSignals(signals) {
        const signalsDiv = document.getElementById('latestSignals');
        if (!signals || signals.length === 0) {
            signalsDiv.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-signal fa-2x"></i>
                    <h4>暂无交易信号</h4>
                    <p>系统正在分析市场数据，请稍候</p>
                </div>
            `;
            return;
        }

        const signalsHtml = signals.map(signal => `
            <div class="signal-card ${signal.action.toLowerCase()}">
                <div class="signal-header">
                    <span class="signal-symbol">${signal.symbol}</span>
                    <span class="signal-action ${signal.action.toLowerCase()}">${signal.action}</span>
                </div>
                <div class="signal-details">
                    <div class="signal-detail-item">
                        <span class="signal-detail-label">价格</span>
                        <span class="signal-detail-value">$${signal.price.toFixed(2)}</span>
                    </div>
                    <div class="signal-detail-item">
                        <span class="signal-detail-label">置信度</span>
                        <span class="signal-detail-value">${(signal.confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div class="signal-detail-item" style="grid-column: 1 / -1;">
                        <span class="signal-detail-label">理由</span>
                        <span class="signal-detail-value">${signal.reason}</span>
                    </div>
                </div>
            </div>
        `).join('');

        signalsDiv.innerHTML = signalsHtml;
    }

    updatePortfolio(portfolio) {
        const overviewDiv = document.getElementById('portfolioOverview');
        const totalReturnClass = portfolio.totalReturn > 0 ? 'text-success' : 'text-danger';
        const todayPnLClass = portfolio.todayPnL > 0 ? 'text-success' : 'text-danger';

        overviewDiv.innerHTML = `
            <div class="portfolio-summary">
                <div class="portfolio-metric">
                    <div class="metric-value">$${portfolio.totalValue.toLocaleString()}</div>
                    <div class="metric-label">总资产</div>
                </div>
                <div class="portfolio-metric">
                    <div class="metric-value ${totalReturnClass}">${(portfolio.totalReturn * 100).toFixed(2)}%</div>
                    <div class="metric-label">总收益率</div>
                </div>
                <div class="portfolio-metric">
                    <div class="metric-value ${todayPnLClass}">$${portfolio.todayPnL.toFixed(2)}</div>
                    <div class="metric-label">今日盈亏</div>
                </div>
            </div>
        `;

        // Update positions detail
        const positionsDiv = document.getElementById('positionsDetail');
        const positionsHtml = portfolio.positions.map(pos => {
            const pnl = (pos.currentPrice - pos.avgCost) * pos.shares;
            const pnlPercent = ((pos.currentPrice - pos.avgCost) / pos.avgCost) * 100;
            const pnlClass = pnl > 0 ? 'text-success' : 'text-danger';

            return `
                <div class="position-row">
                    <div class="position-symbol">${pos.symbol}</div>
                    <div class="position-shares">${pos.shares}</div>
                    <div class="position-price">$${pos.currentPrice.toFixed(2)}</div>
                    <div class="position-value">$${pos.value.toFixed(2)}</div>
                    <div class="position-pnl ${pnlClass}">
                        $${pnl.toFixed(2)} (${pnlPercent.toFixed(1)}%)
                    </div>
                </div>
            `;
        }).join('');

        positionsDiv.innerHTML = `
            <div class="positions-table">
                <div class="position-header">
                    <div>标的</div>
                    <div>股数</div>
                    <div>当前价</div>
                    <div>市值</div>
                    <div>盈亏</div>
                </div>
                ${positionsHtml}
            </div>
        `;
    }

    createPortfolioChart(positions) {
        const ctx = document.getElementById('portfolioChart');
        if (!ctx) return;

        if (this.charts.portfolio) {
            this.charts.portfolio.destroy();
        }

        const labels = positions.map(pos => pos.symbol);
        const data = positions.map(pos => pos.value);
        const colors = [
            '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', 
            '#9966FF', '#FF9F40', '#FF6384', '#C9CBCF'
        ];

        this.charts.portfolio = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: labels,
                datasets: [{
                    data: data,
                    backgroundColor: colors.slice(0, labels.length),
                    borderWidth: 2,
                    borderColor: '#fff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            padding: 20,
                            usePointStyle: true
                        }
                    }
                }
            }
        });
    }

    async runBacktest() {
        const button = document.getElementById('runBacktestBtn');
        const resultsDiv = document.getElementById('backtestResults');

        // Update button state
        button.disabled = true;
        button.innerHTML = '<i class="fas fa-spinner fa-spin"></i><span>运行中...</span>';
        
        // Show loading state
        resultsDiv.innerHTML = `
            <div class="loading">
                <i class="fas fa-spinner fa-spin"></i>
                <span>正在运行3年历史回测分析，请稍候...</span>
            </div>
        `;

        try {
            const response = await fetch(`${this.apiBaseUrl}/api/backtest/quick`, { 
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            if (response.ok) {
                const result = await response.json();
                this.displayBacktestResults(result.backtest_summary);
                this.showToast('回测分析完成', 'success');
            } else {
                throw new Error('回测请求失败');
            }
        } catch (error) {
            console.error('Backtest failed:', error);
            resultsDiv.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-exclamation-triangle fa-2x text-danger"></i>
                    <h4>回测分析失败</h4>
                    <p>请检查网络连接和服务状态，然后重试</p>
                </div>
            `;
            this.showToast('回测分析失败', 'error');
        } finally {
            button.disabled = false;
            button.innerHTML = '<i class="fas fa-play"></i><span>运行3年历史回测</span>';
        }
    }

    async runQuickBacktest() {
        const button = document.getElementById('quickBacktestBtn');
        
        button.disabled = true;
        button.innerHTML = '<i class="fas fa-spinner fa-spin"></i><span>快速回测中...</span>';

        try {
            const response = await fetch(`${this.apiBaseUrl}/api/backtest/quick`, { 
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            if (response.ok) {
                const result = await response.json();
                this.displayBacktestResults(result.backtest_summary);
                this.showToast('快速回测完成', 'success');
            }
        } catch (error) {
            console.error('Quick backtest failed:', error);
            this.showToast('快速回测失败', 'error');
        } finally {
            button.disabled = false;
            button.innerHTML = '<i class="fas fa-bolt"></i><span>快速回测</span>';
        }
    }

    displayBacktestResults(result) {
        const resultsDiv = document.getElementById('backtestResults');
        
        const performanceRating = this.getPerformanceRating(result.total_return);
        const ratingClass = result.total_return > 0.1 ? 'text-success' : 
                           result.total_return > 0 ? 'text-warning' : 'text-danger';

        resultsDiv.innerHTML = `
            <div class="backtest-summary fade-in">
                <h4>📈 回测分析结果</h4>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-label">总收益率</div>
                        <div class="metric-value ${result.total_return > 0 ? 'positive' : 'negative'}">
                            ${(result.total_return * 100).toFixed(2)}%
                        </div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">夏普比率</div>
                        <div class="metric-value">${result.sharpe_ratio?.toFixed(2) || '0.00'}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">最大回撤</div>
                        <div class="metric-value negative">${(result.max_drawdown * 100).toFixed(2)}%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">胜率</div>
                        <div class="metric-value">${(result.win_rate * 100).toFixed(1)}%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">交易次数</div>
                        <div class="metric-value">${result.total_trades || 0}</div>
                    </div>
                </div>
                <div class="performance-summary">
                    <p><strong>策略评价:</strong> <span class="${ratingClass}">${performanceRating}</span></p>
                    <p><strong>分析时间:</strong> ${new Date().toLocaleString()}</p>
                </div>
            </div>
        `;
    }

    updateSignalsHistory(signals) {
        const signalsDiv = document.getElementById('signalsHistory');
        
        if (!signals || signals.length === 0) {
            signalsDiv.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-signal fa-2x"></i>
                    <h4>暂无信号历史</h4>
                    <p>系统运行后将显示交易信号历史</p>
                </div>
            `;
            return;
        }

        const signalsHtml = signals.map(signal => `
            <div class="signal-card ${signal.action.toLowerCase()} slide-in">
                <div class="signal-header">
                    <span class="signal-symbol">${signal.symbol}</span>
                    <span class="signal-action ${signal.action.toLowerCase()}">${signal.action}</span>
                </div>
                <div class="signal-details">
                    <div class="signal-detail-item">
                        <span class="signal-detail-label">价格</span>
                        <span class="signal-detail-value">$${signal.price.toFixed(2)}</span>
                    </div>
                    <div class="signal-detail-item">
                        <span class="signal-detail-label">置信度</span>
                        <span class="signal-detail-value">${(signal.confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div class="signal-detail-item" style="grid-column: 1 / -1;">
                        <span class="signal-detail-label">理由</span>
                        <span class="signal-detail-value">${signal.reason}</span>
                    </div>
                    <div class="signal-detail-item" style="grid-column: 1 / -1;">
                        <span class="signal-detail-label">时间</span>
                        <span class="signal-detail-value">${new Date(signal.timestamp).toLocaleString()}</span>
                    </div>
                </div>
            </div>
        `).join('');

        signalsDiv.innerHTML = signalsHtml;
    }

    updateModelStatus(modelInfo) {
        const statusDiv = document.getElementById('modelStatus');
        
        statusDiv.innerHTML = `
            <div class="model-info">
                <div class="model-status-item">
                    <span class="label">训练状态:</span>
                    <span class="value ${modelInfo.is_trained ? 'text-success' : 'text-warning'}">
                        ${modelInfo.is_trained ? '✅ 已训练' : '⚠️ 未训练'}
                    </span>
                </div>
                <div class="model-status-item">
                    <span class="label">可用模型:</span>
                    <span class="value">${modelInfo.models?.join(', ') || '无'}</span>
                </div>
                <div class="model-status-item">
                    <span class="label">特征数量:</span>
                    <span class="value">${modelInfo.feature_count || 0}</span>
                </div>
                <div class="model-status-item">
                    <span class="label">模型权重:</span>
                    <span class="value">RF: ${(modelInfo.model_weights?.rf * 100 || 0).toFixed(0)}%, GB: ${(modelInfo.model_weights?.gb * 100 || 0).toFixed(0)}%, LR: ${(modelInfo.model_weights?.lr * 100 || 0).toFixed(0)}%</span>
                </div>
            </div>
        `;
    }

    createFeatureImportanceChart(topFeatures) {
        const ctx = document.getElementById('featureChart');
        if (!ctx || !topFeatures) return;

        if (this.charts.features) {
            this.charts.features.destroy();
        }

        const labels = topFeatures.map(f => f[0]);
        const data = topFeatures.map(f => f[1]);

        this.charts.features = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels.slice(0, 10),
                datasets: [{
                    label: '特征重要性',
                    data: data.slice(0, 10),
                    backgroundColor: 'rgba(102, 126, 234, 0.6)',
                    borderColor: 'rgba(102, 126, 234, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: '重要性分数'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: '特征名称'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    }

    async checkSystemHealth() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/health`);
            const health = await response.json();
            this.updateSystemStatus(health);
        } catch (error) {
            const indicator = document.getElementById('systemStatus');
            indicator.className = 'status-indicator offline';
            indicator.innerHTML = '<i class="fas fa-circle"></i><span>连接失败</span>';
        }
    }

    startTimeDisplay() {
        const updateTime = () => {
            const timeDiv = document.getElementById('currentTime');
            if (timeDiv) {
                timeDiv.textContent = new Date().toLocaleString('zh-CN');
            }
        };
        
        updateTime();
        setInterval(updateTime, 1000);
    }

    startAutoRefresh() {
        this.refreshInterval = setInterval(() => {
            if (this.currentTab === 'dashboard') {
                this.loadDashboard();
            }
            this.checkSystemHealth();
        }, 30000); // Refresh every 30 seconds
    }

    getPerformanceRating(totalReturn) {
        if (totalReturn > 0.2) return '✅ 策略表现优秀！';
        if (totalReturn > 0.1) return '👍 策略表现良好';
        if (totalReturn > 0) return '⚠️ 策略表现一般，需要优化';
        return '❌ 策略表现不佳，需要重新评估';
    }

    showToast(message, type = 'info') {
        const container = document.getElementById('toastContainer');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        
        const icon = type === 'success' ? 'check-circle' : 
                    type === 'error' ? 'exclamation-circle' : 
                    type === 'warning' ? 'exclamation-triangle' : 'info-circle';
        
        toast.innerHTML = `
            <i class="fas fa-${icon}"></i>
            <span>${message}</span>
        `;
        
        container.appendChild(toast);
        
        // Show toast
        setTimeout(() => toast.classList.add('show'), 100);
        
        // Remove toast after 3 seconds
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => container.removeChild(toast), 300);
        }, 3000);
    }

    showModal(title, content) {
        document.getElementById('modalTitle').textContent = title;
        document.getElementById('modalBody').innerHTML = content;
        document.getElementById('modal').classList.add('show');
    }

    closeModal() {
        document.getElementById('modal').classList.remove('show');
    }

    // Utility functions
    async refreshHealth() {
        await this.checkSystemHealth();
        this.showToast('健康状态已刷新', 'success');
    }

    async refreshRealTimeData() {
        // Simulate real-time data refresh
        this.showToast('实时数据已刷新', 'success');
    }

    async refreshSignals() {
        await this.loadRecentSignals();
        this.showToast('信号数据已刷新', 'success');
    }

    saveSettings() {
        // Collect form data
        const settings = {
            dataInterval: document.getElementById('dataInterval').value,
            strategyInterval: document.getElementById('strategyInterval').value,
            minConfidence: document.getElementById('minConfidence').value,
            notificationEmail: document.getElementById('notificationEmail').value,
            emailNotifications: document.getElementById('emailNotifications').checked,
            wechatNotifications: document.getElementById('wechatNotifications').checked,
            wechatWebhook: document.getElementById('wechatWebhook').value
        };

        // Save to localStorage
        localStorage.setItem('tradingPlatformSettings', JSON.stringify(settings));
        this.showToast('设置已保存', 'success');
    }

    resetSettings() {
        // Reset to defaults
        document.getElementById('dataInterval').value = '60';
        document.getElementById('strategyInterval').value = '300';
        document.getElementById('minConfidence').value = '0.6';
        document.getElementById('minConfidenceValue').textContent = '60%';
        
        this.showToast('设置已重置为默认值', 'info');
    }

    showAbout() {
        this.showModal('关于系统', `
            <div class="about-content">
                <h4>🚀 AI量化交易平台 v2.0</h4>
                <p><strong>作者:</strong> Alvin</p>
                <p><strong>架构:</strong> Java + Python 微服务架构</p>
                <p><strong>特性:</strong></p>
                <ul>
                    <li>🤖 多模型AI集成学习</li>
                    <li>📊 50+技术指标分析</li>
                    <li>🛡️ 智能风险管理</li>
                    <li>📈 专业级回测分析</li>
                    <li>📧 实时通知提醒</li>
                    <li>🎨 现代化Web界面</li>
                </ul>
                <p><strong>免责声明:</strong> 本系统仅供学习研究使用，投资有风险，决策需谨慎。</p>
            </div>
        `);
    }

    updateStocksList(stocks) {
        const stocksDiv = document.getElementById('stocksList');
        
        if (!stocks || stocks.length === 0) {
            stocksDiv.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-chart-bar fa-2x"></i>
                    <h4>暂无监控股票</h4>
                    <p>请添加股票到监控列表</p>
                </div>
            `;
            return;
        }

        const stocksHtml = stocks.map((stock, index) => `
            <div class="stock-item">
                <div class="stock-info">
                    <div class="stock-symbol">${stock.symbol}</div>
                    <div class="stock-name">${stock.name}</div>
                    <div class="stock-sector">${stock.sector}</div>
                </div>
                <div class="stock-details">
                    <span class="stock-type ${stock.type}">${stock.type.toUpperCase()}</span>
                    <span class="stock-weight">权重: ${(stock.weight * 100).toFixed(1)}%</span>
                    <span class="stock-priority priority-${stock.priority}">${stock.priority}</span>
                    <span class="stock-confidence">置信度: ${(stock.min_confidence * 100).toFixed(0)}%</span>
                </div>
                <div class="stock-notes">${stock.notes}</div>
                <div class="stock-actions">
                    <button class="btn btn-sm btn-secondary" onclick="editStock(${index})">
                        <i class="fas fa-edit"></i>
                    </button>
                    <button class="btn btn-sm btn-danger" onclick="removeStock(${index})">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            </div>
        `).join('');

        stocksDiv.innerHTML = stocksHtml;
    }

    async loadStockQuotes(stocks) {
        const quotesDiv = document.getElementById('stockQuotes');
        
        // Simulate stock quotes
        const quotes = stocks.map(stock => ({
            symbol: stock.symbol,
            price: Math.random() * 200 + 50,
            change: (Math.random() - 0.5) * 10,
            changePercent: (Math.random() - 0.5) * 0.1,
            volume: Math.floor(Math.random() * 1000000) + 100000
        }));

        const quotesHtml = quotes.map(quote => {
            const changeClass = quote.change > 0 ? 'positive' : 'negative';
            const changeIcon = quote.change > 0 ? 'fa-arrow-up' : 'fa-arrow-down';
            
            return `
                <div class="quote-item">
                    <div class="quote-symbol">${quote.symbol}</div>
                    <div class="quote-price">$${quote.price.toFixed(2)}</div>
                    <div class="quote-change ${changeClass}">
                        <i class="fas ${changeIcon}"></i>
                        ${quote.change.toFixed(2)} (${(quote.changePercent * 100).toFixed(2)}%)
                    </div>
                    <div class="quote-volume">成交量: ${quote.volume.toLocaleString()}</div>
                </div>
            `;
        }).join('');

        quotesDiv.innerHTML = quotesHtml;
    }

    addStock() {
        const symbol = document.getElementById('stockSymbol').value.trim().toUpperCase();
        const name = document.getElementById('stockName').value.trim();
        const type = document.getElementById('stockType').value;
        const sector = document.getElementById('stockSector').value.trim();
        const weight = parseFloat(document.getElementById('stockWeight').value) / 100;
        const priority = document.getElementById('stockPriority').value;
        const minConfidence = parseFloat(document.getElementById('stockMinConfidence').value);
        const notes = document.getElementById('stockNotes').value.trim();

        if (!symbol || !name) {
            this.showToast('请填写股票代码和名称', 'warning');
            return;
        }

        const newStock = {
            symbol,
            name,
            type,
            sector,
            weight,
            priority,
            min_confidence: minConfidence,
            notes
        };

        // In a real implementation, this would save to backend
        console.log('Adding stock:', newStock);
        this.showToast(`已添加 ${symbol} 到监控列表`, 'success');
        
        // Clear form
        document.getElementById('stockSymbol').value = '';
        document.getElementById('stockName').value = '';
        document.getElementById('stockSector').value = '';
        document.getElementById('stockNotes').value = '';
        document.getElementById('stockWeight').value = '5';
        document.getElementById('stockPriority').value = 'medium';
        document.getElementById('stockMinConfidence').value = '0.7';
        document.getElementById('stockMinConfidenceValue').textContent = '70%';
        
        // Reload stocks list
        this.loadStocks();
    }

    showHelp() {
        this.showModal('使用帮助', `
            <div class="help-content">
                <h4>📖 使用指南</h4>
                <div class="help-section">
                    <h5>🎯 仪表板</h5>
                    <p>查看系统健康状态、实时数据和最新交易信号</p>
                </div>
                <div class="help-section">
                    <h5>📊 投资组合</h5>
                    <p>监控投资组合表现、资产分布和持仓详情</p>
                </div>
                <div class="help-section">
                    <h5>📈 回测分析</h5>
                    <p>运行历史数据回测，评估AI策略表现</p>
                </div>
                <div class="help-section">
                    <h5>🔔 交易信号</h5>
                    <p>查看AI生成的买入、卖出、持有信号历史</p>
                </div>
                <div class="help-section">
                    <h5>📊 股票配置</h5>
                    <p>管理监控股票列表，添加删除股票，查看实时行情</p>
                </div>
                <div class="help-section">
                    <h5>🧠 AI模型</h5>
                    <p>管理机器学习模型，查看特征重要性</p>
                </div>
                <div class="help-section">
                    <h5>⚙️ 设置</h5>
                    <p>配置交易参数、通知设置和系统选项</p>
                </div>
            </div>
        `);
    }
}

// Global functions
function refreshHealth() {
    if (window.tradingUI) {
        window.tradingUI.refreshHealth();
    }
}

function refreshRealTimeData() {
    if (window.tradingUI) {
        window.tradingUI.refreshRealTimeData();
    }
}

function refreshSignals() {
    if (window.tradingUI) {
        window.tradingUI.refreshSignals();
    }
}

function runBacktest() {
    if (window.tradingUI) {
        window.tradingUI.runBacktest();
    }
}

function runQuickBacktest() {
    if (window.tradingUI) {
        window.tradingUI.runQuickBacktest();
    }
}

function saveSettings() {
    if (window.tradingUI) {
        window.tradingUI.saveSettings();
    }
}

function resetSettings() {
    if (window.tradingUI) {
        window.tradingUI.resetSettings();
    }
}

function showAbout() {
    if (window.tradingUI) {
        window.tradingUI.showAbout();
    }
}

function showHelp() {
    if (window.tradingUI) {
        window.tradingUI.showHelp();
    }
}

function closeModal() {
    if (window.tradingUI) {
        window.tradingUI.closeModal();
    }
}

function addStock() {
    if (window.tradingUI) {
        window.tradingUI.addStock();
    }
}

function refreshStockList() {
    if (window.tradingUI) {
        window.tradingUI.loadStocks();
        window.tradingUI.showToast('股票列表已刷新', 'success');
    }
}

function refreshQuotes() {
    if (window.tradingUI) {
        window.tradingUI.loadStocks();
        window.tradingUI.showToast('行情数据已刷新', 'success');
    }
}

function exportStockConfig() {
    if (window.tradingUI) {
        window.tradingUI.showToast('股票配置导出功能开发中', 'info');
    }
}

function editStock(index) {
    if (window.tradingUI) {
        window.tradingUI.showToast(`编辑股票 #${index + 1} 功能开发中`, 'info');
    }
}

function removeStock(index) {
    if (window.tradingUI) {
        if (confirm('确定要删除这个股票吗？')) {
            window.tradingUI.showToast(`已删除股票 #${index + 1}`, 'success');
            // In real implementation, would remove from backend and reload
            window.tradingUI.loadStocks();
        }
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.tradingUI = new TradingPlatformUI();
});
