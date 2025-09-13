package com.alvin.quantitative.trading.platform.data;

import com.alvin.quantitative.trading.platform.core.KlineData;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Market Data Manager, Practical Quantitative Trading System - Java Core
 * Author: Alvin Description: A hybrid Java+Python automated trading system for
 * retail investors, Manages real-time market data and calculates technical
 * indicators
 */
public class MarketDataManager {
	private final Map<String, Queue<KlineData>> dataBuffers;
	private final Map<String, Map<String, Double>> technicalIndicators;
	private final int maxBufferSize;

	public MarketDataManager(int maxBufferSize) {
		this.dataBuffers = new ConcurrentHashMap<>();
		this.technicalIndicators = new ConcurrentHashMap<>();
		this.maxBufferSize = maxBufferSize;
	}

	public void addKlineData(String symbol, KlineData data) {
		dataBuffers.computeIfAbsent(symbol, k -> new LinkedList<>()).add(data);

		Queue<KlineData> buffer = dataBuffers.get(symbol);
		while (buffer.size() > maxBufferSize) {
			buffer.poll();
		}

		// Real-time technical indicator calculation
		updateTechnicalIndicators(symbol, buffer);
	}

	private void updateTechnicalIndicators(String symbol, Queue<KlineData> data) {
		if (data.size() < 20)
			return;

		List<Double> closes = data.stream().map(KlineData::getClose).collect(ArrayList::new, ArrayList::add,
				ArrayList::addAll);

		Map<String, Double> indicators = technicalIndicators.computeIfAbsent(symbol, k -> new HashMap<>());

		// Calculate various indicators
		indicators.put("MA5", calculateSMA(closes, 5));
		indicators.put("MA10", calculateSMA(closes, 10));
		indicators.put("MA20", calculateSMA(closes, 20));
		indicators.put("RSI", calculateRSI(closes, 14));
		indicators.put("MACD", calculateMACD(closes));
		indicators.put("ATR", calculateATR(data));
		indicators.put("VOLUME_RATIO", calculateVolumeRatio(data));

		// Price position related indicators
		double currentPrice = closes.get(closes.size() - 1);
		indicators.put("PRICE_POSITION", calculatePricePosition(closes, 20));
		indicators.put("VOLATILITY", calculateVolatility(closes, 20));
	}

	public Map<String, Double> getIndicators(String symbol) {
		return technicalIndicators.getOrDefault(symbol, new HashMap<>());
	}

	public List<KlineData> getRecentData(String symbol, int count) {
		Queue<KlineData> buffer = dataBuffers.get(symbol);
		if (buffer == null)
			return new ArrayList<>();

		return new ArrayList<>(buffer).stream().skip(Math.max(0, buffer.size() - count)).collect(ArrayList::new,
				ArrayList::add, ArrayList::addAll);
	}

	// Technical indicator calculation methods
	private double calculateSMA(List<Double> prices, int period) {
		if (prices.size() < period)
			return 0;
		return prices.subList(prices.size() - period, prices.size()).stream().mapToDouble(Double::doubleValue).average()
				.orElse(0);
	}

	private double calculateRSI(List<Double> prices, int period) {
		if (prices.size() < period + 1)
			return 50;

		double totalGain = 0, totalLoss = 0;
		for (int i = prices.size() - period; i < prices.size(); i++) {
			double change = prices.get(i) - prices.get(i - 1);
			if (change > 0)
				totalGain += change;
			else
				totalLoss -= change;
		}

		double avgGain = totalGain / period;
		double avgLoss = totalLoss / period;

		if (avgLoss == 0)
			return 100;
		return 100 - (100 / (1 + avgGain / avgLoss));
	}

	private double calculateMACD(List<Double> prices) {
		// Simplified MACD calculation
		double ema12 = calculateEMA(prices, 12);
		double ema26 = calculateEMA(prices, 26);
		return ema12 - ema26;
	}

	private double calculateEMA(List<Double> prices, int period) {
		if (prices.isEmpty())
			return 0;
		double multiplier = 2.0 / (period + 1);
		double ema = prices.get(Math.max(0, prices.size() - period));

		for (int i = Math.max(1, prices.size() - period + 1); i < prices.size(); i++) {
			ema = (prices.get(i) * multiplier) + (ema * (1 - multiplier));
		}
		return ema;
	}

	private double calculateATR(Queue<KlineData> data) {
		if (data.size() < 2)
			return 0;

		List<KlineData> dataList = new ArrayList<>(data);
		double atr = 0;
		int count = Math.min(14, dataList.size() - 1);

		for (int i = dataList.size() - count; i < dataList.size(); i++) {
			KlineData current = dataList.get(i);
			KlineData previous = dataList.get(i - 1);

			double tr1 = current.getHigh() - current.getLow();
			double tr2 = Math.abs(current.getHigh() - previous.getClose());
			double tr3 = Math.abs(current.getLow() - previous.getClose());

			atr += Math.max(tr1, Math.max(tr2, tr3));
		}

		return atr / count;
	}

	private double calculateVolumeRatio(Queue<KlineData> data) {
		List<KlineData> dataList = new ArrayList<>(data);
		if (dataList.size() < 20)
			return 1.0;

		long currentVolume = dataList.get(dataList.size() - 1).getVolume();
		double avgVolume = dataList.stream().skip(dataList.size() - 20).mapToLong(KlineData::getVolume).average()
				.orElse(1);

		return currentVolume / avgVolume;
	}

	private double calculatePricePosition(List<Double> prices, int period) {
		if (prices.size() < period)
			return 0.5;

		List<Double> recentPrices = prices.subList(prices.size() - period, prices.size());
		double high = recentPrices.stream().mapToDouble(Double::doubleValue).max().orElse(0);
		double low = recentPrices.stream().mapToDouble(Double::doubleValue).min().orElse(0);
		double current = prices.get(prices.size() - 1);

		return (high - low) == 0 ? 0.5 : (current - low) / (high - low);
	}

	private double calculateVolatility(List<Double> prices, int period) {
		if (prices.size() < period)
			return 0;

		List<Double> recentPrices = prices.subList(prices.size() - period, prices.size());
		double mean = recentPrices.stream().mapToDouble(Double::doubleValue).average().orElse(0);

		double variance = recentPrices.stream().mapToDouble(price -> Math.pow(price - mean, 2)).average().orElse(0);

		return Math.sqrt(variance) / mean;
	}
}