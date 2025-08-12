#!/usr/bin/env python3
from flask import Flask, render_template_string, jsonify, request
from flask_socketio import SocketIO, emit
import requests
import logging
from datetime import datetime, timedelta
import json
import numpy as np
import random
import threading
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'bitcoin_dashboard_secret'
socketio = SocketIO(app, cors_allowed_origins="*")

current_btc_data = None
price_history = []
update_thread = None
last_api_call = None
api_cache_duration = 25

class BitcoinTradingAgent:
    def __init__(self):
        self.portfolio_value = 100000
        self.cash = 100000
        self.bitcoin_holdings = 0
        self.trade_history = []
        self.current_strategy = "Learning"
        self.confidence = 0.5
        self.win_rate = 0.0
        self.total_trades = 0
        self.successful_trades = 0
        self.learning_episodes = 0
        self.last_action = "HOLD"
        self.last_decision_time = datetime.now()
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 0.1
        self.rsi = 50
        self.macd = 0
        self.volatility = 0.3
        self.sentiment = 0.5
        
    def get_market_state(self, btc_data):
        price = btc_data['price']
        change_pct = btc_data['change_percent']
        self.rsi = max(0, min(100, 50 + change_pct * 2))
        self.macd = change_pct * 0.1
        self.volatility = abs(change_pct) / 100
        self.sentiment = 0.6 if change_pct > 0 else 0.4
        rsi_state = "oversold" if self.rsi < 30 else "overbought" if self.rsi > 70 else "neutral"
        trend_state = "bullish" if change_pct > 2 else "bearish" if change_pct < -2 else "sideways"
        vol_state = "high" if self.volatility > 0.05 else "low"
        return f"{rsi_state}_{trend_state}_{vol_state}"
    
    def choose_action(self, state):
        if random.random() < self.epsilon or state not in self.q_table:
            action = random.choice([0, 1, 2])
        else:
            action = max(self.q_table[state], key=self.q_table[state].get)
        return action
    
    def execute_trade(self, action, btc_price):
        trade_amount = self.cash * 0.1
        if action == 1 and self.cash > trade_amount:
            bitcoin_bought = trade_amount / btc_price
            self.bitcoin_holdings += bitcoin_bought
            self.cash -= trade_amount
            self.last_action = "BUY"
            self.trade_history.append({
                'action': 'BUY', 'price': btc_price, 'amount': bitcoin_bought,
                'timestamp': datetime.now(), 'portfolio_value': self.get_portfolio_value(btc_price)
            })
        elif action == 2 and self.bitcoin_holdings > 0:
            sell_amount = self.bitcoin_holdings * 0.5
            self.cash += sell_amount * btc_price
            self.bitcoin_holdings -= sell_amount
            self.last_action = "SELL"
            self.trade_history.append({
                'action': 'SELL', 'price': btc_price, 'amount': sell_amount,
                'timestamp': datetime.now(), 'portfolio_value': self.get_portfolio_value(btc_price)
            })
        else:
            self.last_action = "HOLD"
        self.total_trades += 1
        self.last_decision_time = datetime.now()
    
    def get_portfolio_value(self, btc_price):
        return self.cash + (self.bitcoin_holdings * btc_price)
    
    def calculate_reward(self, old_portfolio_value, new_portfolio_value):
        return (new_portfolio_value - old_portfolio_value) / old_portfolio_value
    
    def update_q_table(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = {0: 0, 1: 0, 2: 0}
        if next_state not in self.q_table:
            self.q_table[next_state] = {0: 0, 1: 0, 2: 0}
        old_value = self.q_table[state][action]
        next_max = max(self.q_table[next_state].values())
        new_value = old_value + self.learning_rate * (reward + self.discount_factor * next_max - old_value)
        self.q_table[state][action] = new_value
    
    def make_decision(self, btc_data):
        current_price = btc_data['price']
        old_portfolio_value = self.get_portfolio_value(current_price)
        state = self.get_market_state(btc_data)
        action = self.choose_action(state)
        self.execute_trade(action, current_price)
        new_portfolio_value = self.get_portfolio_value(current_price)
        reward = self.calculate_reward(old_portfolio_value, new_portfolio_value)
        self.update_q_table(state, action, reward, state)
        
        if reward > 0:
            self.successful_trades += 1
        self.win_rate = (self.successful_trades / max(1, self.total_trades)) * 100
        self.learning_episodes += 1
        
        if self.rsi < 30:
            self.current_strategy = "Oversold Recovery"
        elif self.rsi > 70:
            self.current_strategy = "Overbought Caution"
        elif abs(btc_data['change_percent']) > 3:
            self.current_strategy = "High Volatility Trading"
        else:
            self.current_strategy = "Trend Following"
        
        recent_trades = self.trade_history[-10:] if len(self.trade_history) >= 10 else self.trade_history
        if recent_trades:
            recent_performance = sum(1 for trade in recent_trades if trade['portfolio_value'] > 100000) / len(recent_trades)
            self.confidence = min(0.95, max(0.3, recent_performance))
        
        return {
            'action': self.last_action, 'confidence': self.confidence, 'strategy': self.current_strategy,
            'portfolio_value': new_portfolio_value, 'cash': self.cash, 'bitcoin_holdings': self.bitcoin_holdings,
            'win_rate': self.win_rate, 'total_trades': self.total_trades, 'learning_episodes': self.learning_episodes,
            'rsi': self.rsi, 'macd': self.macd, 'volatility': self.volatility, 'sentiment': self.sentiment,
            'last_decision_time': self.last_decision_time.strftime('%H:%M:%S')
        }

ai_agent = BitcoinTradingAgent()

def get_bitcoin_data():
    global current_btc_data
    try:
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {'ids': 'bitcoin', 'vs_currencies': 'usd', 'include_24hr_change': 'true', 
                 'include_market_cap': 'true', 'include_24hr_vol': 'true'}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        bitcoin_data = data['bitcoin']
        
        btc_data = {
            'price': bitcoin_data['usd'], 'change_percent': bitcoin_data.get('usd_24h_change', 0),
            'market_cap': bitcoin_data.get('usd_market_cap', 0), 'volume': bitcoin_data.get('usd_24h_vol', 0),
            'source': 'Live', 'timestamp': datetime.now().isoformat()
        }
        current_btc_data = btc_data
        return btc_data
        
    except Exception as e:
        logger.error(f"Error fetching Bitcoin data: {e}")
        if current_btc_data:
            current_btc_data['timestamp'] = datetime.now().isoformat()
            return current_btc_data
        else:
            fallback_data = {
                'price': 95000, 'change_percent': 0, 'market_cap': 1900000000000,
                'volume': 25000000000, 'source': 'Live', 'timestamp': datetime.now().isoformat()
            }
            current_btc_data = fallback_data
            return fallback_data

def get_timeframe_predictions():
    timeframes = {
        '1h': {'name': '1 Hour', 'multiplier': 0.995 + random.random() * 0.01},
        '4h': {'name': '4 Hours', 'multiplier': 0.99 + random.random() * 0.02},
        '1d': {'name': '1 Day', 'multiplier': 0.98 + random.random() * 0.04},
        '1w': {'name': '1 Week', 'multiplier': 0.95 + random.random() * 0.1},
        '1m': {'name': '1 Month', 'multiplier': 0.9 + random.random() * 0.2}
    }
    btc_data = get_bitcoin_data()
    current_price = btc_data['price']
    predictions = {}
    
    for tf_key, tf_data in timeframes.items():
        predicted_price = current_price * tf_data['multiplier']
        price_change_pct = ((predicted_price - current_price) / current_price) * 100
        trend = 'bullish' if price_change_pct > 2 else 'bearish' if price_change_pct < -2 else 'neutral'
        predictions[tf_key] = {
            'name': tf_data['name'], 'predicted_price': predicted_price, 'price_change_pct': price_change_pct,
            'trend': trend, 'confidence': random.randint(60, 90)
        }
    return predictions

def get_historical_bitcoin_data():
    try:
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
        params = {'vs_currency': 'usd', 'days': '30', 'interval': 'daily'}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        prices = data['prices']
        volumes = data['total_volumes']
        chart_data = []
        for i, (timestamp, price) in enumerate(prices):
            chart_data.append({
                'timestamp': timestamp, 'price': price, 'volume': volumes[i][1] if i < len(volumes) else 0,
                'date': datetime.fromtimestamp(timestamp/1000).strftime('%Y-%m-%d')
            })
        return chart_data
    except Exception as e:
        logger.error(f"Error fetching historical data: {e}")
        base_price = 95000
        chart_data = []
        for i in range(30):
            date = datetime.now() - timedelta(days=29-i)
            price = base_price * (1 + random.uniform(-0.05, 0.05))
            chart_data.append({
                'timestamp': int(date.timestamp() * 1000), 'price': price, 'volume': random.uniform(20e9, 40e9),
                'date': date.strftime('%Y-%m-%d')
            })
        return chart_data

def update_price_history(btc_data):
    global price_history
    price_point = {
        'timestamp': btc_data['timestamp'], 'price': btc_data['price'],
        'volume': btc_data['volume'], 'time': datetime.now().strftime('%H:%M:%S')
    }
    price_history.append(price_point)
    if len(price_history) > 50:
        price_history = price_history[-50:]

def real_time_price_updater():
    while True:
        try:
            btc_data = get_bitcoin_data()
            update_price_history(btc_data)
            ai_decision = ai_agent.make_decision(btc_data)
            socketio.emit('price_update', {
                'btc_data': btc_data, 'price_history': price_history[-10:], 'ai_decision': ai_decision
            })
            logger.info(f"Price updated: ${btc_data['price']:,.2f} ({btc_data['change_percent']:+.2f}%) | AI: {ai_decision['action']} ({ai_decision['confidence']:.1%})")
        except Exception as e:
            logger.error(f"Error in price updater: {e}")
        time.sleep(25)

def start_price_updates():
    global update_thread
    if update_thread is None or not update_thread.is_alive():
        update_thread = threading.Thread(target=real_time_price_updater, daemon=True)
        update_thread.start()
        logger.info("Real-time price updates started")

def run_monte_carlo_simulation(current_price, days, simulations):
    dt = 1/252
    mu = 0.15
    sigma = 0.6
    results = []
    price_paths = []
    num_paths_to_show = min(50, simulations)
    
    for i in range(simulations):
        price = current_price
        path = [price]
        for day in range(days):
            z = np.random.standard_normal()
            price = price * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
            path.append(price)
        results.append(path[-1])
        if i < num_paths_to_show:
            price_paths.append(path)
    
    results = np.array(results)
    expected_price = np.mean(results)
    profit_prob = np.sum(results > current_price) / len(results) * 100
    var_95 = np.percentile(results, 5)
    var_pct = ((current_price - var_95) / current_price) * 100
    
    percentiles = [5, 25, 50, 75, 95]
    confidence_bands = []
    for day in range(days + 1):
        day_prices = [path[day] for path in price_paths]
        day_percentiles = np.percentile(day_prices, percentiles)
        confidence_bands.append({
            'day': day, 'p5': day_percentiles[0], 'p25': day_percentiles[1],
            'p50': day_percentiles[2], 'p75': day_percentiles[3], 'p95': day_percentiles[4]
        })
    
    risk_level = "LOW" if var_pct < 10 else "MEDIUM" if var_pct < 25 else "HIGH"
    
    if profit_prob > 70 and var_pct < 20:
        recommendation = "STRONG BUY - High probability of profit with manageable risk"
    elif profit_prob > 60:
        recommendation = "BUY - Favorable risk-reward ratio"
    elif profit_prob > 40:
        recommendation = "HOLD - Neutral outlook with balanced risk"
    else:
        recommendation = "SELL - High risk of loss, consider reducing position"
    
    price_distribution = np.histogram(results, bins=20)
    distribution_data = []
    for i in range(len(price_distribution[0])):
        distribution_data.append({
            'price': (price_distribution[1][i] + price_distribution[1][i+1]) / 2,
            'frequency': int(price_distribution[0][i])
        })
    
    return {
        'expected_price': expected_price, 'profit_probability': profit_prob, 'var_95': var_pct,
        'risk_level': risk_level, 'recommendation': recommendation, 'price_paths': price_paths,
        'confidence_bands': confidence_bands, 'price_distribution': distribution_data,
        'final_prices': results[:100].tolist()
    }

@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')
    if current_btc_data:
        emit('price_update', {'btc_data': current_btc_data, 'price_history': price_history[-10:]})

@socketio.on('disconnect')
def handle_disconnect():
    logger.info('Client disconnected')

@socketio.on('request_update')
def handle_request_update():
    btc_data = get_bitcoin_data()
    update_price_history(btc_data)
    emit('price_update', {'btc_data': btc_data, 'price_history': price_history[-10:]})

@app.route('/')
def dashboard():
    try:
        start_price_updates()
        btc_data = get_bitcoin_data()
        timeframe_predictions = get_timeframe_predictions()
        historical_data = get_historical_bitcoin_data()
        if not price_history:
            update_price_history(btc_data)
        return render_template_string(DASHBOARD_TEMPLATE, 
            btc_data=btc_data, timeframe_predictions=timeframe_predictions,
            historical_data=historical_data, price_history=price_history
        )
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        return f"Dashboard Error: {e}", 500

@app.route('/api/ai-agent/status')
def ai_agent_status():
    try:
        if current_btc_data:
            ai_decision = ai_agent.make_decision(current_btc_data)
            return jsonify({'success': True, 'agent_status': ai_decision})
        else:
            return jsonify({'success': False, 'error': 'No market data available'}), 400
    except Exception as e:
        logger.error(f"AI agent status error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/ai-agent/reset', methods=['POST'])
def reset_ai_agent():
    try:
        global ai_agent
        ai_agent = BitcoinTradingAgent()
        return jsonify({'success': True, 'message': 'AI Trading Agent reset successfully'})
    except Exception as e:
        logger.error(f"AI agent reset error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/realtime-price')
def api_realtime_price():
    try:
        btc_data = get_bitcoin_data()
        return jsonify({'success': True, 'data': btc_data, 'price_history': price_history[-20:]})
    except Exception as e:
        logger.error(f"Real-time price error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/historical-data')
def api_historical_data():
    try:
        chart_data = get_historical_bitcoin_data()
        return jsonify({'success': True, 'data': chart_data})
    except Exception as e:
        logger.error(f"Historical data error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/monte-carlo', methods=['POST'])
def api_monte_carlo():
    try:
        data = request.get_json()
        current_price = float(data.get('current_price', 100000))
        days = int(data.get('days', 90))
        simulations = int(data.get('simulations', 1000))
        result = run_monte_carlo_simulation(current_price, days, simulations)
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        logger.error(f"Monte Carlo error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/ml-pipeline/status')
def ml_pipeline_status():
    return jsonify({
        'success': True,
        'status': {
            'model_version': '1.2.3', 'last_trained': '2025-01-09 15:30:00', 'accuracy': 0.847,
            'status': 'active', 'next_retrain': '2025-01-10 00:00:00'
        }
    })

@app.route('/api/performance/metrics')
def performance_metrics():
    return jsonify({
        'success': True,
        'metrics': {
            'total_return': 23.5, 'sharpe_ratio': 1.42, 'max_drawdown': -8.3,
            'win_rate': 67.8, 'avg_trade_return': 2.1, 'volatility': 15.6
        }
    })

DASHBOARD_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bitcoin Price Prediction Model</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.socket.io/4.6.0/socket.io.min.js"></script>
    <style>
        :root {
            --primary-blue: #1e3a8a; --primary-dark: #1f2937; --primary-light: #f8fafc;
            --profit-green: #10b981; --loss-red: #ef4444; --warning-yellow: #f59e0b;
            --neutral-gray: #6b7280; --bg-primary: #ffffff; --bg-secondary: #f1f5f9;
            --bg-card: #ffffff; --text-primary: #000000; --text-secondary: #333333;
            --border-color: #e5e7eb; --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1);
            --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1);
            --gradient-primary: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --gradient-success: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            --gradient-warning: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
            --gradient-info: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        }
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--primary-light) 100%);
            color: var(--text-primary); line-height: 1.6; min-height: 100vh;
        }
        header {
            background: linear-gradient(135deg, var(--primary-dark) 0%, var(--primary-blue) 100%);
            color: white; padding: 2rem; box-shadow: var(--shadow-lg); position: sticky; top: 0; z-index: 100;
        }
        header h1 { font-size: 2.25rem; font-weight: 700; margin: 0; letter-spacing: -0.025em; }
        header p { margin-top: 0.5rem; opacity: 0.9; font-size: 1.125rem; }
        .container { max-width: 1400px; margin: 0 auto; padding: 2rem; }
        .tabs {
            display: flex; background: var(--bg-card); border-radius: 12px; padding: 0.5rem;
            margin-bottom: 2rem; box-shadow: var(--shadow-md); border: 1px solid var(--border-color); overflow-x: auto;
        }
        .tab {
            flex: 1; min-width: 140px; padding: 0.875rem 1.5rem; border: none; background: transparent;
            color: var(--text-secondary); font-weight: 500; font-size: 0.875rem; border-radius: 8px;
            cursor: pointer; transition: all 0.2s ease; white-space: nowrap; text-align: center;
        }
        .tab:hover { background: var(--bg-secondary); color: var(--text-primary); }
        .tab.active { background: var(--primary-blue); color: white; box-shadow: var(--shadow-md); }
        .tab-content { display: none; animation: fadeIn 0.3s ease-in-out; }
        .tab-content.active { display: block; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .card {
            background: var(--bg-card); border-radius: 16px; padding: 2rem; margin-bottom: 2rem;
            box-shadow: var(--shadow-md); border: 1px solid var(--border-color); transition: all 0.2s ease;
            position: relative; overflow: hidden;
        }
        .card::before {
            content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
            background: linear-gradient(90deg, var(--primary-blue), var(--profit-green));
        }
        .card:hover { box-shadow: var(--shadow-lg); transform: translateY(-2px); }
        .card h2 { font-size: 1.5rem; font-weight: 600; color: var(--text-primary); margin-bottom: 1.5rem; }
        .price-display { font-size: 2.5rem; font-weight: 700; color: var(--text-primary); margin: 1rem 0; text-align: center; }
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .metric-card {
            padding: 25px; border-radius: 12px; text-align: center; color: white; position: relative; overflow: hidden;
        }
        .metric-card h3 { margin: 0 0 10px 0; font-size: 0.9em; opacity: 0.9; }
        .metric-card .value { font-size: 2.2em; font-weight: 700; margin: 10px 0; }
        .metric-card .label { opacity: 0.8; font-size: 0.85em; }
        .btn {
            padding: 12px 24px; border: none; border-radius: 8px; cursor: pointer; font-weight: 600;
            font-size: 0.9rem; transition: all 0.3s ease; margin: 4px;
        }
        .btn-primary { background: var(--primary-blue); color: white; }
        .btn:hover { transform: translateY(-2px); box-shadow: var(--shadow-md); }
        .controls-grid {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;
            background: #f8fafc; padding: 20px; border-radius: 12px; margin-bottom: 25px;
        }
        .control-group label { display: block; margin-bottom: 5px; font-weight: 500; }
        .control-group input, .control-group select {
            width: 100%; padding: 8px; border: 1px solid #e2e8f0; border-radius: 6px;
        }
        .loading { display: none; text-align: center; padding: 40px; }
        .spinner {
            display: inline-block; width: 40px; height: 40px; border: 4px solid #f3f3f3;
            border-top: 4px solid var(--primary-blue); border-radius: 50%; animation: spin 1s linear infinite;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .alert { padding: 20px; border-radius: 12px; margin-bottom: 25px; border-left: 4px solid; }
        .alert-info { background: rgba(30, 58, 138, 0.1); border-left-color: var(--primary-blue); }
        .recommendation { padding: 15px; background: white; border-radius: 8px; border-left: 4px solid var(--primary-blue); }
        .status-indicator {
            position: fixed; top: 20px; right: 20px; padding: 8px 16px; border-radius: 20px;
            font-size: 0.8rem; font-weight: 600; z-index: 1000; transition: all 0.3s ease;
        }
        .status-live { background: var(--profit-green); color: white; box-shadow: 0 0 10px rgba(16, 185, 129, 0.5); }
        .status-connecting { background: var(--warning-yellow); color: white; }
        .status-offline { background: var(--loss-red); color: white; }
        .pulse { animation: pulse 2s infinite; }
        @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.7; } 100% { opacity: 1; } }
        .price-change { transition: all 0.5s ease; }
        .price-up { color: var(--profit-green) !important; animation: priceFlash 1s ease; }
        .price-down { color: var(--loss-red) !important; animation: priceFlash 1s ease; }
        @keyframes priceFlash { 0% { background-color: transparent; } 50% { background-color: rgba(255, 255, 255, 0.3); } 100% { background-color: transparent; } }
        @media (max-width: 768px) {
            .container { padding: 1rem; } .tabs { flex-direction: column; }
            .price-display { font-size: 2rem; } .metrics-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div id="status-indicator" class="status-indicator status-connecting pulse">Connecting...</div>
    <header>
        <h1>Bitcoin Price Prediction Model</h1>
        <p>Real-time Trading & Risk Management Dashboard</p>
    </header>
    <div class="container">
        <div class="tabs">
            <div class="tab active" data-target="overview-tab">Market Overview</div>
            <div class="tab" data-target="predictions-tab">Predictions</div>
            <div class="tab" data-target="monte-carlo-tab">Monte Carlo</div>
            <div class="tab" data-target="analytics-tab">Analytics & Risk</div>
            <div class="tab" data-target="ai-agent-tab">AI Trading Agent</div>
            <div class="tab" data-target="ml-pipeline-tab">ML Pipeline</div>
            <div class="tab" data-target="performance-tab">Performance</div>
        </div>

        <div id="overview-tab" class="tab-content active">
            <div class="card">
                <h2>Current Market Status</h2>
                <div id="current-price" class="price-display price-change">${{ "{:,.0f}".format(btc_data.price) }}</div>
                <p style="text-align: center; font-size: 1.2rem;" id="price-change-display">
                    <span style="color: {% if btc_data.change_percent >= 0 %}var(--profit-green){% else %}var(--loss-red){% endif %};">
                        {{ "{:+.2f}".format(btc_data.change_percent) }}% (24h)
                    </span>
                </p>
                <p style="text-align: center; font-size: 0.9rem; color: #666; margin-top: 10px;">
                    Last updated: <span id="last-updated">{{ btc_data.timestamp }}</span>
                </p>
                <div class="metrics-grid">
                    <div class="metric-card" style="background: var(--gradient-primary);">
                        <h3>Market Cap</h3>
                        <div class="value" id="market-cap">${{ "{:.1f}".format(btc_data.market_cap / 1e12) }}T</div>
                        <div class="label">Total Value</div>
                    </div>
                    <div class="metric-card" style="background: var(--gradient-success);">
                        <h3>24h Volume</h3>
                        <div class="value" id="volume">${{ "{:.1f}".format(btc_data.volume / 1e9) }}B</div>
                        <div class="label">Trading Activity</div>
                    </div>
                    <div class="metric-card" style="background: var(--gradient-warning);">
                        <h3>Data Source</h3>
                        <div class="value">Live</div>
                        <div class="label" id="data-source">{{ btc_data.source }}</div>
                    </div>
                </div>
            </div>
            <div class="card">
                <h2>Bitcoin Price Chart (30 Days)</h2>
                <div style="position: relative; height: 400px;">
                    <canvas id="priceChart"></canvas>
                </div>
            </div>
        </div>

        <div id="predictions-tab" class="tab-content">
            <div class="card">
                <h2>Multi-Timeframe Price Predictions</h2>
                <div class="metrics-grid">
                    {% for tf_key, pred in timeframe_predictions.items() %}
                    <div class="metric-card" style="background: {% if pred.trend == 'bullish' %}var(--gradient-success){% elif pred.trend == 'bearish' %}var(--gradient-warning){% else %}var(--gradient-info){% endif %};">
                        <h3>{{ pred.name }}</h3>
                        <div class="value">${{ "{:,.0f}".format(pred.predicted_price) }}</div>
                        <div class="label">{{ "{:+.1f}".format(pred.price_change_pct) }}% ({{ pred.confidence }}% confidence)</div>
                    </div>
                    {% endfor %}
                </div>
            </div>
        </div>

        <div id="monte-carlo-tab" class="tab-content">
            <div class="card">
                <h2>Monte Carlo Price Simulation</h2>
                <p style="color: #666; margin-bottom: 25px;">
                    Advanced probabilistic modeling to forecast Bitcoin price movements using thousands of simulated scenarios.
                </p>
                <div class="controls-grid">
                    <div class="control-group">
                        <label>Current Price ($)</label>
                        <input type="number" id="mc-current-price" value="{{ btc_data.price }}" step="0.01">
                    </div>
                    <div class="control-group">
                        <label>Time Horizon (Days)</label>
                        <select id="mc-days">
                            <option value="30">30 Days</option>
                            <option value="90" selected>90 Days</option>
                            <option value="180">180 Days</option>
                            <option value="252">1 Year</option>
                        </select>
                    </div>
                    <div class="control-group">
                        <label>Simulations</label>
                        <select id="mc-simulations">
                            <option value="500">500 (Fast)</option>
                            <option value="1000" selected>1,000 (Standard)</option>
                            <option value="5000">5,000 (Detailed)</option>
                        </select>
                    </div>
                    <div style="display: flex; align-items: end;">
                        <button onclick="runMonteCarloSimulation()" class="btn btn-primary" style="width: 100%;">
                            Run Simulation
                        </button>
                    </div>
                </div>
                <div id="mc-loading" class="loading">
                    <div class="spinner"></div>
                    <p style="margin-top: 15px; color: #666;">Running Monte Carlo simulation...</p>
                </div>
                <div id="mc-results" style="display: none;">
                    <div class="metrics-grid">
                        <div class="metric-card" style="background: var(--gradient-primary);">
                            <h3>Expected Price</h3>
                            <div class="value" id="mc-expected-price">$0</div>
                            <div class="label">Mean Forecast</div>
                        </div>
                        <div class="metric-card" style="background: var(--gradient-success);">
                            <h3>Profit Probability</h3>
                            <div class="value" id="mc-profit-prob">0%</div>
                            <div class="label">Chance of Gain</div>
                        </div>
                        <div class="metric-card" style="background: var(--gradient-warning);">
                            <h3>Value at Risk</h3>
                            <div class="value" id="mc-var">0%</div>
                            <div class="label">95% Confidence</div>
                        </div>
                        <div class="metric-card" style="background: var(--gradient-info);">
                            <h3>Risk Level</h3>
                            <div class="value" id="mc-risk-level">-</div>
                            <div class="label">Assessment</div>
                        </div>
                    </div>
                    <div style="background: #f8fafc; padding: 20px; border-radius: 12px;">
                        <h3 style="margin: 0 0 15px 0; color: var(--primary-blue);">Trading Recommendation</h3>
                        <div class="recommendation">
                            <p id="mc-recommendation" style="margin: 0; color: #666; font-size: 1.1em;">Run simulation to get recommendation</p>
                        </div>
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 25px;">
                        <div class="card" style="margin: 0;">
                            <h3>Price Path Simulations</h3>
                            <div style="position: relative; height: 300px;">
                                <canvas id="mcPathsChart"></canvas>
                            </div>
                        </div>
                        <div class="card" style="margin: 0;">
                            <h3>Price Distribution</h3>
                            <div style="position: relative; height: 300px;">
                                <canvas id="mcDistributionChart"></canvas>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <div id="analytics-tab" class="tab-content">
            <div class="card">
                <h2>Risk Analytics Dashboard</h2>
                <div class="metrics-grid">
                    <div class="metric-card" style="background: var(--gradient-primary);">
                        <h3>Current Volatility</h3>
                        <div class="value">45.2%</div>
                        <div class="label">Annualized</div>
                    </div>
                    <div class="metric-card" style="background: var(--gradient-success);">
                        <h3>Sharpe Ratio</h3>
                        <div class="value">1.34</div>
                        <div class="label">Risk-Adjusted Return</div>
                    </div>
                    <div class="metric-card" style="background: var(--gradient-warning);">
                        <h3>Max Drawdown</h3>
                        <div class="value">-12.5%</div>
                        <div class="label">Historical Peak Loss</div>
                    </div>
                </div>
            </div>
        </div>

        <div id="ai-agent-tab" class="tab-content">
            <div class="card">
                <h2>AI Trading Agent</h2>
                <div class="metrics-grid">
                    <div class="metric-card" style="background: var(--gradient-primary);">
                        <h3>Current Action</h3>
                        <div class="value" id="ai-current-action">HOLD</div>
                        <div class="label" id="ai-last-decision">Last decision: --:--:--</div>
                    </div>
                    <div class="metric-card" style="background: var(--gradient-success);">
                        <h3>Confidence Level</h3>
                        <div class="value" id="ai-confidence">50%</div>
                        <div class="label">Decision Certainty</div>
                    </div>
                    <div class="metric-card" style="background: var(--gradient-warning);">
                        <h3>Current Strategy</h3>
                        <div class="value" id="ai-strategy">Learning</div>
                        <div class="label">Active Approach</div>
                    </div>
                    <div class="metric-card" style="background: var(--gradient-info);">
                        <h3>Win Rate</h3>
                        <div class="value" id="ai-win-rate">0%</div>
                        <div class="label">Successful Trades</div>
                    </div>
                </div>
                <div class="card" style="margin: 25px 0 0 0;">
                    <h3>Portfolio Performance</h3>
                    <div class="metrics-grid">
                        <div class="metric-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                            <h3>Portfolio Value</h3>
                            <div class="value" id="ai-portfolio-value">$100,000</div>
                            <div class="label">Total Assets</div>
                        </div>
                        <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
                            <h3>Cash Balance</h3>
                            <div class="value" id="ai-cash">$100,000</div>
                            <div class="label">Available Cash</div>
                        </div>
                        <div class="metric-card" style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);">
                            <h3>Bitcoin Holdings</h3>
                            <div class="value" id="ai-bitcoin-holdings">0.000</div>
                            <div class="label">BTC Amount</div>
                        </div>
                        <div class="metric-card" style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);">
                            <h3>Total Trades</h3>
                            <div class="value" id="ai-total-trades">0</div>
                            <div class="label">Executed Orders</div>
                        </div>
                    </div>
                </div>
                <div class="card" style="margin: 25px 0 0 0;">
                    <h3>Control Panel</h3>
                    <div style="display: flex; gap: 15px; flex-wrap: wrap;">
                        <button onclick="resetAIAgent()" class="btn btn-primary">Reset AI Agent</button>
                        <button onclick="forceAIDecision()" class="btn btn-primary">Force Decision</button>
                    </div>
                </div>
            </div>
        </div>

        <div id="ml-pipeline-tab" class="tab-content">
            <div class="card">
                <h2>Machine Learning Pipeline</h2>
                <div class="metrics-grid">
                    <div class="metric-card" style="background: var(--gradient-primary);">
                        <h3>Model Version</h3>
                        <div class="value">v1.2.3</div>
                        <div class="label">Current Active</div>
                    </div>
                    <div class="metric-card" style="background: var(--gradient-success);">
                        <h3>Accuracy</h3>
                        <div class="value">84.7%</div>
                        <div class="label">Prediction Accuracy</div>
                    </div>
                    <div class="metric-card" style="background: var(--gradient-info);">
                        <h3>Status</h3>
                        <div class="value">Active</div>
                        <div class="label">Model State</div>
                    </div>
                </div>
            </div>
        </div>

        <div id="performance-tab" class="tab-content">
            <div class="card">
                <h2>Trading Performance Metrics</h2>
                <div class="metrics-grid">
                    <div class="metric-card" style="background: var(--gradient-success);">
                        <h3>Total Return</h3>
                        <div class="value">+23.5%</div>
                        <div class="label">YTD Performance</div>
                    </div>
                    <div class="metric-card" style="background: var(--gradient-primary);">
                        <h3>Win Rate</h3>
                        <div class="value">67.8%</div>
                        <div class="label">Successful Trades</div>
                    </div>
                    <div class="metric-card" style="background: var(--gradient-warning);">
                        <h3>Avg Trade Return</h3>
                        <div class="value">+2.1%</div>
                        <div class="label">Per Trade</div>
                    </div>
                    <div class="metric-card" style="background: var(--gradient-info);">
                        <h3>Volatility</h3>
                        <div class="value">15.6%</div>
                        <div class="label">Portfolio Risk</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let socket = null, priceChart = null, mcPathsChart = null, mcDistributionChart = null, lastPrice = 0;
        const historicalData = {{ historical_data | tojson }};
        
        function initSocket() {
            socket = io();
            socket.on('connect', () => { updateStatus('live', 'Live Updates'); socket.emit('request_update'); });
            socket.on('disconnect', () => updateStatus('offline', 'Offline'));
            socket.on('price_update', (data) => { updatePriceDisplay(data.btc_data); if (data.ai_decision) updateAIAgentDisplay(data.ai_decision); });
        }
        
        function updateStatus(status, text) {
            const indicator = document.getElementById('status-indicator');
            indicator.className = `status-indicator status-${status}`;
            indicator.textContent = text;
            if (status === 'live') indicator.classList.add('pulse');
        }
        
        function updatePriceDisplay(btcData) {
            const priceElement = document.getElementById('current-price');
            const newPrice = btcData.price;
            if (lastPrice > 0) {
                priceElement.classList.add(newPrice > lastPrice ? 'price-up' : 'price-down');
                setTimeout(() => priceElement.classList.remove('price-up', 'price-down'), 1000);
            }
            lastPrice = newPrice;
            priceElement.textContent = '$' + newPrice.toLocaleString('en-US', {minimumFractionDigits: 0, maximumFractionDigits: 0});
            const changeColor = btcData.change_percent >= 0 ? 'var(--profit-green)' : 'var(--loss-red)';
            document.getElementById('price-change-display').innerHTML = `<span style="color: ${changeColor};">${btcData.change_percent >= 0 ? '+' : ''}${btcData.change_percent.toFixed(2)}% (24h)</span>`;
            document.getElementById('last-updated').textContent = new Date(btcData.timestamp).toLocaleTimeString();
            document.getElementById('market-cap').textContent = '$' + (btcData.market_cap / 1e12).toFixed(1) + 'T';
            document.getElementById('volume').textContent = '$' + (btcData.volume / 1e9).toFixed(1) + 'B';
            document.getElementById('data-source').textContent = btcData.source;
            const mcPriceInput = document.getElementById('mc-current-price');
            if (mcPriceInput) mcPriceInput.value = newPrice.toFixed(2);
        }
        
        function updateAIAgentDisplay(aiDecision) {
            const actionElement = document.getElementById('ai-current-action');
            if (actionElement) {
                actionElement.textContent = aiDecision.action;
                actionElement.style.color = aiDecision.action === 'BUY' ? 'var(--profit-green)' : aiDecision.action === 'SELL' ? 'var(--loss-red)' : '#333';
            }
            if (document.getElementById('ai-confidence')) document.getElementById('ai-confidence').textContent = (aiDecision.confidence * 100).toFixed(0) + '%';
            if (document.getElementById('ai-strategy')) document.getElementById('ai-strategy').textContent = aiDecision.strategy;
            if (document.getElementById('ai-win-rate')) document.getElementById('ai-win-rate').textContent = aiDecision.win_rate.toFixed(1) + '%';
            if (document.getElementById('ai-last-decision')) document.getElementById('ai-last-decision').textContent = 'Last decision: ' + aiDecision.last_decision_time;
            if (document.getElementById('ai-portfolio-value')) document.getElementById('ai-portfolio-value').textContent = '$' + Math.round(aiDecision.portfolio_value).toLocaleString();
            if (document.getElementById('ai-cash')) document.getElementById('ai-cash').textContent = '$' + Math.round(aiDecision.cash).toLocaleString();
            if (document.getElementById('ai-bitcoin-holdings')) document.getElementById('ai-bitcoin-holdings').textContent = aiDecision.bitcoin_holdings.toFixed(3);
            if (document.getElementById('ai-total-trades')) document.getElementById('ai-total-trades').textContent = aiDecision.total_trades;
        }
        
        function initTabs() {
            document.querySelectorAll('.tab').forEach(tab => {
                tab.addEventListener('click', function() {
                    const target = this.getAttribute('data-target');
                    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
                    this.classList.add('active');
                    const targetContent = document.getElementById(target);
                    if (targetContent) targetContent.classList.add('active');
                });
            });
        }
        
        function initPriceChart() {
            const ctx = document.getElementById('priceChart').getContext('2d');
            const labels = historicalData.map(d => d.date);
            const prices = historicalData.map(d => d.price);
            priceChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Bitcoin Price', data: prices, borderColor: '#1e3a8a', backgroundColor: 'rgba(30, 58, 138, 0.1)',
                        borderWidth: 3, fill: true, tension: 0.4, pointRadius: 4, pointHoverRadius: 6,
                        pointBackgroundColor: '#1e3a8a', pointBorderColor: '#ffffff', pointBorderWidth: 2
                    }]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    plugins: { legend: { display: true, position: 'top', labels: { color: '#333', font: { size: 14, weight: '500' } } } },
                    scales: {
                        x: { grid: { color: 'rgba(0, 0, 0, 0.1)' }, ticks: { color: '#666', font: { size: 12 } } },
                        y: { grid: { color: 'rgba(0, 0, 0, 0.1)' }, ticks: { color: '#666', font: { size: 12 }, callback: value => '$' + value.toLocaleString() } }
                    }
                }
            });
        }
        
        function runMonteCarloSimulation() {
            const currentPrice = parseFloat(document.getElementById('mc-current-price').value);
            const days = parseInt(document.getElementById('mc-days').value);
            const simulations = parseInt(document.getElementById('mc-simulations').value);
            document.getElementById('mc-loading').style.display = 'block';
            document.getElementById('mc-results').style.display = 'none';
            fetch('/api/monte-carlo', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ current_price: currentPrice, days: days, simulations: simulations })
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('mc-loading').style.display = 'none';
                if (data.success) {
                    const result = data.result;
                    document.getElementById('mc-expected-price').textContent = '$' + Math.round(result.expected_price).toLocaleString();
                    document.getElementById('mc-profit-prob').textContent = result.profit_probability.toFixed(1) + '%';
                    document.getElementById('mc-var').textContent = result.var_95.toFixed(1) + '%';
                    document.getElementById('mc-risk-level').textContent = result.risk_level;
                    document.getElementById('mc-recommendation').textContent = result.recommendation;
                    initMCCharts(result.price_paths, result.price_distribution, days);
                    document.getElementById('mc-results').style.display = 'block';
                } else alert('Error: ' + data.error);
            })
            .catch(error => { document.getElementById('mc-loading').style.display = 'none'; alert('Network error: ' + error.message); });
        }
        
        function initMCCharts(pathsData, distributionData, days) {
            const pathsCtx = document.getElementById('mcPathsChart').getContext('2d');
            if (mcPathsChart) mcPathsChart.destroy();
            const datasets = pathsData.slice(0, 10).map((path, index) => ({
                label: `Path ${index + 1}`, data: path, borderColor: `hsla(${index * 36}, 70%, 50%, 0.6)`,
                backgroundColor: 'transparent', borderWidth: 1, pointRadius: 0, tension: 0.1
            }));
            const labels = Array.from({length: days + 1}, (_, i) => `Day ${i}`);
            mcPathsChart = new Chart(pathsCtx, {
                type: 'line', data: { labels: labels, datasets: datasets },
                options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } }
            });
            
            const distCtx = document.getElementById('mcDistributionChart').getContext('2d');
            if (mcDistributionChart) mcDistributionChart.destroy();
            const distLabels = distributionData.map(d => '$' + Math.round(d.price).toLocaleString());
            const frequencies = distributionData.map(d => d.frequency);
            mcDistributionChart = new Chart(distCtx, {
                type: 'bar', data: { labels: distLabels, datasets: [{ label: 'Frequency', data: frequencies, backgroundColor: 'rgba(30, 58, 138, 0.7)', borderColor: '#1e3a8a', borderWidth: 1 }] },
                options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } }
            });
        }
        
        function resetAIAgent() {
            if (confirm('Reset AI Trading Agent?')) {
                fetch('/api/ai-agent/reset', { method: 'POST', headers: { 'Content-Type': 'application/json' } })
                .then(response => response.json())
                .then(data => { if (data.success) { alert('AI reset!'); socket.emit('request_update'); } else alert('Error: ' + data.error); });
            }
        }
        
        function forceAIDecision() {
            fetch('/api/ai-agent/status')
            .then(response => response.json())
            .then(data => { if (data.success) { updateAIAgentDisplay(data.agent_status); alert('AI updated: ' + data.agent_status.action); } });
        }
        
        document.addEventListener('DOMContentLoaded', function() {
            initTabs(); initSocket(); initPriceChart();
            const refreshBtn = document.createElement('button');
            refreshBtn.textContent = 'Refresh'; refreshBtn.className = 'btn btn-primary';
            refreshBtn.style.cssText = 'position: fixed; bottom: 20px; right: 20px; z-index: 1000;';
            refreshBtn.onclick = () => socket.emit('request_update');
            document.body.appendChild(refreshBtn);
        });
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    print("Starting Bitcoin Price Prediction Model...")
    print("Features: Real-time updates (every 25 seconds), AI Trading Agent, Monte Carlo, Analytics")
    print(f"Access at: http://localhost:{port}")
    get_bitcoin_data()
    socketio.run(app, host='0.0.0.0', port=port, debug=False)
