
import os
import json
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
import MetaTrader5 as mt5

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mt5_strategy.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("mt5_api")

# MT5 timeframe mapping
TIMEFRAMES = {
    "1m": mt5.TIMEFRAME_M1,
    "5m": mt5.TIMEFRAME_M5,
    "15m": mt5.TIMEFRAME_M15,
    "30m": mt5.TIMEFRAME_M30,
    "1h": mt5.TIMEFRAME_H1,
    "4h": mt5.TIMEFRAME_H4,
    "1d": mt5.TIMEFRAME_D1,
    "1w": mt5.TIMEFRAME_W1,
    "1M": mt5.TIMEFRAME_MN1
}

class MT5Connection:
    """Handles connection with MetaTrader 5 terminal"""
    
    def __init__(self):
        self.connected = False
        self.account_info = None
    
    def connect(self, account_id: str, password: str, server: str, terminal_id: Optional[str] = None) -> bool:
        """Connect to MT5 terminal with provided credentials"""
        try:
            # Initialize MT5 connection
            if not mt5.initialize(
                login=int(account_id),
                password=password,
                server=server,
                path=terminal_id
            ):
                error = mt5.last_error()
                logger.error(f"MT5 initialization failed: {error}")
                return False
            
            # Authenticate
            if not mt5.login(
                login=int(account_id),
                password=password,
                server=server
            ):
                error = mt5.last_error()
                logger.error(f"MT5 login failed: {error}")
                mt5.shutdown()
                return False
            
            self.account_info = mt5.account_info()
            if self.account_info is None:
                logger.error("Failed to get account info")
                mt5.shutdown()
                return False
                
            self.connected = True
            logger.info(f"Connected to MT5: {self.account_info.server}, Account: {self.account_info.login}")
            return True
            
        except Exception as e:
            logger.error(f"MT5 connection error: {str(e)}")
            self.disconnect()
            return False
    
    def disconnect(self) -> None:
        """Disconnect from MT5 terminal"""
        if mt5.shutdown():
            self.connected = False
            self.account_info = None
            logger.info("Disconnected from MT5")
    
    def is_connected(self) -> bool:
        """Check if MT5 is connected"""
        return self.connected and mt5.terminal_info() is not None
    
    def get_account_info(self) -> Dict:
        """Return account information"""
        if not self.is_connected():
            raise ConnectionError("MT5 is not connected")
        
        account_info = mt5.account_info()
        if account_info is None:
            raise ConnectionError("Failed to get account info")
        
        # Convert named tuple to dictionary
        return {
            'balance': account_info.balance,
            'equity': account_info.equity,
            'profit': account_info.profit,
            'margin': account_info.margin,
            'margin_level': account_info.margin_level,
            'leverage': account_info.leverage,
            'currency': account_info.currency
        }


class DataFetcher:
    """Fetches historical and live price data from MT5"""
    
    def __init__(self, connection: MT5Connection):
        self.connection = connection
    
    def get_historical_data(self, symbol: str, timeframe: str, bars_count: int = 500) -> pd.DataFrame:
        """Fetch historical OHLC data from MT5"""
        if not self.connection.is_connected():
            raise ConnectionError("MT5 is not connected")
        
        # Map timeframe string to MT5 timeframe constant
        mt5_timeframe = TIMEFRAMES.get(timeframe)
        if mt5_timeframe is None:
            raise ValueError(f"Invalid timeframe: {timeframe}")
        
        # Get historical data from MT5
        rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, bars_count)
        if rates is None or len(rates) == 0:
            error = mt5.last_error()
            logger.error(f"Failed to get historical data for {symbol}: {error}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        return df
    
    def get_current_price(self, symbol: str) -> Dict:
        """Get current price for a symbol"""
        if not self.connection.is_connected():
            raise ConnectionError("MT5 is not connected")
        
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            error = mt5.last_error()
            logger.error(f"Failed to get current price for {symbol}: {error}")
            raise ValueError(f"Failed to get tick data for {symbol}")
        
        return {
            'bid': tick.bid,
            'ask': tick.ask,
            'last': tick.last,
            'volume': tick.volume,
            'time': datetime.fromtimestamp(tick.time),
            'spread': tick.ask - tick.bid
        }


class StrategyCalculator:
    """Implements the calculation logic for the trading strategy"""
    
    def calculate_rsi(self, price_data: pd.DataFrame, window: int = 14) -> np.ndarray:
        """Calculate Relative Strength Index (RSI)"""
        # Get price differences
        diff = price_data['close'].diff()
        
        # Get gain and loss
        gain = diff.where(diff > 0, 0)
        loss = -diff.where(diff < 0, 0)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        # First RSI value
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.values
    
    def calculate_correlation(self, df1: pd.DataFrame, df2: pd.DataFrame, window: int = 50) -> np.ndarray:
        """Calculate rolling correlation between two price series"""
        # Ensure the dataframes have the same index
        common_index = df1.index.intersection(df2.index)
        if len(common_index) < window:
            raise ValueError(f"Not enough data points for correlation calculation (need {window}, got {len(common_index)})")
        
        df1 = df1.loc[common_index]
        df2 = df2.loc[common_index]
        
        # Calculate the percentage changes
        returns1 = df1['close'].pct_change()
        returns2 = df2['close'].pct_change()
        
        # Calculate rolling correlation
        correlation = returns1.rolling(window=window).corr(returns2)
        
        return correlation.values
    
    def check_entry_conditions(
        self,
        correlation: float,
        pair1_rsi: float,
        pair2_rsi: float,
        correlation_threshold: float,
        rsi_overbought: float,
        rsi_oversold: float
    ) -> Tuple[bool, Optional[Dict]]:
        """Check if entry conditions are met for the strategy"""
        # Check correlation condition
        if correlation > correlation_threshold:
            return False, None
        
        # Check RSI conditions for entry signal
        if pair1_rsi > rsi_overbought and pair2_rsi < rsi_oversold:
            # Pair 1 is overbought and Pair 2 is oversold
            return True, {
                'pair1_direction': 'SELL',
                'pair2_direction': 'BUY'
            }
        elif pair1_rsi < rsi_oversold and pair2_rsi > rsi_overbought:
            # Pair 1 is oversold and Pair 2 is overbought
            return True, {
                'pair1_direction': 'BUY',
                'pair2_direction': 'SELL'
            }
        
        # Both pairs are either overbought or oversold
        return False, None
    
    def check_exit_conditions(
        self, 
        correlation: float, 
        correlation_threshold: float,
        pair1_profit: float,
        pair2_profit: float
    ) -> bool:
        """Check if exit conditions are met for the strategy"""
        # Check correlation condition
        if correlation < correlation_threshold:
            return False
        
        # Check combined profit condition
        total_profit = pair1_profit + pair2_profit
        if total_profit <= 0:
            return False
        
        # Both conditions are met, exit the trades
        return True


class TradeManager:
    """Manages trade execution and monitoring"""
    
    def __init__(self, connection: MT5Connection):
        self.connection = connection
    
    def open_trade(
        self, 
        symbol: str, 
        direction: str, 
        lot_size: float,
        magic_number: int,
        comment: str = ""
    ) -> Optional[int]:
        """Open a new trade with specified parameters"""
        if not self.connection.is_connected():
            raise ConnectionError("MT5 is not connected")
        
        # Define the trade request
        trade_type = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL
        price = self.get_trade_price(symbol, trade_type)
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": trade_type,
            "price": price,
            "deviation": 10,  # max price deviation in points
            "magic": magic_number,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        
        # Execute the trade
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Trade execution failed: {result.comment} (Code: {result.retcode})")
            return None
        
        logger.info(f"Trade opened: {symbol} {direction} {lot_size} lots, Ticket: {result.order}")
        return result.order
    
    def close_trade(self, ticket: int) -> bool:
        """Close an open trade by ticket number"""
        if not self.connection.is_connected():
            raise ConnectionError("MT5 is not connected")
        
        # Get the position details
        position = mt5.positions_get(ticket=ticket)
        if position is None or len(position) == 0:
            logger.error(f"Position not found: Ticket {ticket}")
            return False
        
        position = position[0]
        
        # Define the trade request to close position
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY,  # opposite of original trade
            "position": ticket,
            "price": self.get_trade_price(position.symbol, 
                                          mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY),
            "deviation": 10,
            "magic": position.magic,
            "comment": "Close position",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        
        # Execute the trade
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Failed to close trade: {result.comment} (Code: {result.retcode})")
            return False
        
        logger.info(f"Trade closed: Ticket {ticket}, Symbol {position.symbol}")
        return True
    
    def get_trade_price(self, symbol: str, trade_type: int) -> float:
        """Get the appropriate price for a trade (ask for buy, bid for sell)"""
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise ValueError(f"Failed to get price for {symbol}")
        
        # For buy orders, use ask price; for sell orders, use bid price
        if trade_type == mt5.ORDER_TYPE_BUY:
            return tick.ask
        else:
            return tick.bid
    
    def get_open_positions(self, magic_number: Optional[int] = None) -> List[Dict]:
        """Get all open positions, optionally filtered by magic number"""
        if not self.connection.is_connected():
            raise ConnectionError("MT5 is not connected")
        
        # Get all open positions
        if magic_number is not None:
            positions = mt5.positions_get(magic=magic_number)
        else:
            positions = mt5.positions_get()
        
        if positions is None:
            error = mt5.last_error()
            logger.error(f"Failed to get open positions: {error}")
            return []
        
        # Convert to list of dictionaries
        result = []
        for pos in positions:
            result.append({
                'ticket': pos.ticket,
                'symbol': pos.symbol,
                'direction': 'BUY' if pos.type == 0 else 'SELL',
                'volume': pos.volume,
                'open_price': pos.price_open,
                'current_price': pos.price_current,
                'profit': pos.profit,
                'magic': pos.magic,
                'comment': pos.comment,
                'open_time': datetime.fromtimestamp(pos.time).isoformat()
            })
        
        return result
    
    def calculate_position_profit(self, ticket: int) -> float:
        """Calculate current profit for a position"""
        if not self.connection.is_connected():
            raise ConnectionError("MT5 is not connected")
        
        # Get the position details
        position = mt5.positions_get(ticket=ticket)
        if position is None or len(position) == 0:
            logger.error(f"Position not found: Ticket {ticket}")
            return 0.0
        
        return position[0].profit


class StrategyManager:
    """Manages the strategy execution and monitoring"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.connection = MT5Connection()
        self.data_fetcher = None
        self.trade_manager = None
        self.calculator = StrategyCalculator()
        self.last_entry_time = None
        self.active_trades = {
            'pair1': None,
            'pair2': None
        }
    
    def initialize(self) -> bool:
        """Initialize the strategy"""
        # Connect to MT5
        if not self.connection.connect(
            account_id=self.config['mt5_account_id'],
            password=self.config['mt5_password'],
            server=self.config['mt5_server'],
            terminal_id=self.config.get('mt5_terminal_id')
        ):
            return False
        
        # Initialize data fetcher and trade manager
        self.data_fetcher = DataFetcher(self.connection)
        self.trade_manager = TradeManager(self.connection)
        
        return True
    
    def run_strategy_check(self) -> Dict:
        """Run a single check of the strategy conditions"""
        if not self.connection.is_connected():
            return {'status': 'error', 'message': 'MT5 is not connected'}
        
        try:
            # Get the latest data
            pair1_data = self.data_fetcher.get_historical_data(
                self.config['pair1'],
                self.config['timeframe'],
                self.config['correlation_window'] + 50  # Extra data for calculations
            )
            
            pair2_data = self.data_fetcher.get_historical_data(
                self.config['pair2'],
                self.config['timeframe'],
                self.config['correlation_window'] + 50  # Extra data for calculations
            )
            
            if pair1_data.empty or pair2_data.empty:
                return {'status': 'error', 'message': 'Failed to fetch price data'}
            
            # Calculate indicators
            pair1_rsi = self.calculator.calculate_rsi(pair1_data, self.config['rsi_window'])
            pair2_rsi = self.calculator.calculate_rsi(pair2_data, self.config['rsi_window'])
            correlation = self.calculator.calculate_correlation(
                pair1_data, pair2_data, self.config['correlation_window']
            )
            
            # Get the latest values
            latest_correlation = correlation[-1] if len(correlation) > 0 else None
            latest_pair1_rsi = pair1_rsi[-1] if len(pair1_rsi) > 0 else None
            latest_pair2_rsi = pair2_rsi[-1] if len(pair2_rsi) > 0 else None
            
            if latest_correlation is None or latest_pair1_rsi is None or latest_pair2_rsi is None:
                return {
                    'status': 'error', 
                    'message': 'Not enough data to calculate indicators'
                }
            
            # Check if we're in a trade
            if self.active_trades['pair1'] is not None and self.active_trades['pair2'] is not None:
                # Check exit conditions
                pair1_profit = self.trade_manager.calculate_position_profit(self.active_trades['pair1'])
                pair2_profit = self.trade_manager.calculate_position_profit(self.active_trades['pair2'])
                
                should_exit = self.calculator.check_exit_conditions(
                    latest_correlation,
                    self.config['correlation_exit_threshold'],
                    pair1_profit,
                    pair2_profit
                )
                
                if should_exit:
                    # Close the trades
                    self.trade_manager.close_trade(self.active_trades['pair1'])
                    self.trade_manager.close_trade(self.active_trades['pair2'])
                    
                    # Reset active trades
                    self.active_trades = {'pair1': None, 'pair2': None}
                    
                    return {
                        'status': 'exit',
                        'correlation': latest_correlation,
                        'pair1_rsi': latest_pair1_rsi,
                        'pair2_rsi': latest_pair2_rsi,
                        'pair1_profit': pair1_profit,
                        'pair2_profit': pair2_profit,
                        'total_profit': pair1_profit + pair2_profit
                    }
                
                return {
                    'status': 'monitoring',
                    'correlation': latest_correlation,
                    'pair1_rsi': latest_pair1_rsi,
                    'pair2_rsi': latest_pair2_rsi,
                    'pair1_profit': pair1_profit,
                    'pair2_profit': pair2_profit,
                    'total_profit': pair1_profit + pair2_profit
                }
            
            else:
                # Check if we're in cooldown period
                if self.last_entry_time is not None:
                    cooldown_duration = timedelta(hours=self.config['cooldown_period'])
                    if datetime.now() - self.last_entry_time < cooldown_duration:
                        return {
                            'status': 'cooldown',
                            'correlation': latest_correlation,
                            'pair1_rsi': latest_pair1_rsi,
                            'pair2_rsi': latest_pair2_rsi,
                            'cooldown_remaining': str(self.last_entry_time + cooldown_duration - datetime.now())
                        }
                
                # Check entry conditions
                should_enter, signal = self.calculator.check_entry_conditions(
                    latest_correlation,
                    latest_pair1_rsi,
                    latest_pair2_rsi,
                    self.config['correlation_entry_threshold'],
                    self.config['rsi_overbought'],
                    self.config['rsi_oversold']
                )
                
                if should_enter and signal:
                    # Enter trades
                    magic_number = int(time.time())
                    
                    # Open pair1 trade
                    pair1_ticket = self.trade_manager.open_trade(
                        self.config['pair1'],
                        signal['pair1_direction'],
                        self.config['lotsize_pair1'],
                        magic_number,
                        f"Strategy: {self.config['name']}, Pair1"
                    )
                    
                    # Open pair2 trade
                    pair2_ticket = self.trade_manager.open_trade(
                        self.config['pair2'],
                        signal['pair2_direction'],
                        self.config['lotsize_pair2'],
                        magic_number,
                        f"Strategy: {self.config['name']}, Pair2"
                    )
                    
                    if pair1_ticket and pair2_ticket:
                        # Update active trades
                        self.active_trades = {
                            'pair1': pair1_ticket,
                            'pair2': pair2_ticket
                        }
                        self.last_entry_time = datetime.now()
                        
                        return {
                            'status': 'entry',
                            'correlation': latest_correlation,
                            'pair1_rsi': latest_pair1_rsi,
                            'pair2_rsi': latest_pair2_rsi,
                            'pair1_direction': signal['pair1_direction'],
                            'pair2_direction': signal['pair2_direction'],
                            'pair1_ticket': pair1_ticket,
                            'pair2_ticket': pair2_ticket
                        }
                    
                    # If one of the trades failed, close the other one
                    if pair1_ticket and not pair2_ticket:
                        self.trade_manager.close_trade(pair1_ticket)
                    elif pair2_ticket and not pair1_ticket:
                        self.trade_manager.close_trade(pair2_ticket)
                    
                    return {
                        'status': 'error',
                        'message': 'Failed to open one or both trades'
                    }
                
                return {
                    'status': 'waiting',
                    'correlation': latest_correlation,
                    'pair1_rsi': latest_pair1_rsi,
                    'pair2_rsi': latest_pair2_rsi,
                    'correlation_threshold': self.config['correlation_entry_threshold'],
                    'rsi_overbought': self.config['rsi_overbought'],
                    'rsi_oversold': self.config['rsi_oversold']
                }
            
        except Exception as e:
            logger.error(f"Strategy check error: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def close_all_trades(self) -> bool:
        """Close all open trades for this strategy"""
        if not self.connection.is_connected():
            return False
        
        # If we have active trades, close them
        if self.active_trades['pair1'] is not None:
            self.trade_manager.close_trade(self.active_trades['pair1'])
            self.active_trades['pair1'] = None
        
        if self.active_trades['pair2'] is not None:
            self.trade_manager.close_trade(self.active_trades['pair2'])
            self.active_trades['pair2'] = None
        
        return True
    
    def shutdown(self) -> None:
        """Shutdown the strategy and disconnect from MT5"""
        if self.connection.is_connected():
            self.close_all_trades()
            self.connection.disconnect()


class BacktestEngine:
    """Implements backtesting functionality for the strategy"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.calculator = StrategyCalculator()
        self.trades = []
        self.trade_id_counter = 1
    
    def run_backtest(
        self, 
        pair1_data: pd.DataFrame, 
        pair2_data: pd.DataFrame, 
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict:
        """Run a backtest for the strategy"""
        # Ensure the data has datetime index
        if not isinstance(pair1_data.index, pd.DatetimeIndex):
            raise ValueError("pair1_data must have DatetimeIndex")
        if not isinstance(pair2_data.index, pd.DatetimeIndex):
            raise ValueError("pair2_data must have DatetimeIndex")
        
        # Filter by date range if provided
        if start_date:
            start_datetime = pd.to_datetime(start_date)
            pair1_data = pair1_data[pair1_data.index >= start_datetime]
            pair2_data = pair2_data[pair2_data.index >= start_datetime]
        
        if end_date:
            end_datetime = pd.to_datetime(end_date)
            pair1_data = pair1_data[pair1_data.index <= end_datetime]
            pair2_data = pair2_data[pair2_data.index <= end_datetime]
        
        # Find common timeframe
        common_index = pair1_data.index.intersection(pair2_data.index)
        pair1_data = pair1_data.loc[common_index]
        pair2_data = pair2_data.loc[common_index]
        
        # Ensure we have enough data
        min_required = max(self.config['correlation_window'], self.config['rsi_window']) + 10
        if len(pair1_data) < min_required or len(pair2_data) < min_required:
            raise ValueError(f"Not enough data for backtest. Need at least {min_required} bars.")
        
        # Calculate indicators
        pair1_rsi = self.calculator.calculate_rsi(pair1_data, self.config['rsi_window'])
        pair2_rsi = self.calculator.calculate_rsi(pair2_data, self.config['rsi_window'])
        correlation = self.calculator.calculate_correlation(
            pair1_data, pair2_data, self.config['correlation_window']
        )
        
        # Create a DataFrame for the backtest
        backtest_df = pd.DataFrame(index=common_index)
        backtest_df['pair1_close'] = pair1_data['close']
        backtest_df['pair2_close'] = pair2_data['close']
        
        # Add indicators starting from the point we have enough data
        start_idx = max(self.config['correlation_window'], self.config['rsi_window'])
        backtest_df['correlation'] = pd.Series(index=common_index[start_idx:], data=correlation[start_idx:])
        backtest_df['pair1_rsi'] = pd.Series(index=common_index, data=pair1_rsi)
        backtest_df['pair2_rsi'] = pd.Series(index=common_index, data=pair2_rsi)
        
        # Add trading signals
        backtest_df['entry_signal'] = False
        backtest_df['exit_signal'] = False
        backtest_df['pair1_direction'] = None
        backtest_df['pair2_direction'] = None
        
        # Initialize trading state
        in_trade = False
        last_entry_time = None
        active_trades = {
            'pair1': None,
            'pair2': None
        }
        self.trades = []
        
        # Run the backtest
        for idx, row in backtest_df.iterrows():
            # Skip rows with NaN for indicators
            if (np.isnan(row['correlation']) or 
                np.isnan(row['pair1_rsi']) or 
                np.isnan(row['pair2_rsi'])):
                continue
            
            # If in a trade, check exit conditions
            if in_trade:
                # Get current trade P&L
                pair1_trade = next((t for t in self.trades if t['id'] == active_trades['pair1']), None)
                pair2_trade = next((t for t in self.trades if t['id'] == active_trades['pair2']), None)
                
                if pair1_trade and pair2_trade:
                    # Calculate current profit
                    pair1_entry_price = pair1_trade['entry_price']
                    pair2_entry_price = pair2_trade['entry_price']
                    pair1_current_price = row['pair1_close']
                    pair2_current_price = row['pair2_close']
                    
                    # Profit calculation depends on direction
                    if pair1_trade['direction'] == 'BUY':
                        pair1_profit = (pair1_current_price - pair1_entry_price) * self.config['lotsize_pair1'] * 100000
                    else:
                        pair1_profit = (pair1_entry_price - pair1_current_price) * self.config['lotsize_pair1'] * 100000
                    
                    if pair2_trade['direction'] == 'BUY':
                        pair2_profit = (pair2_current_price - pair2_entry_price) * self.config['lotsize_pair2'] * 100000
                    else:
                        pair2_profit = (pair2_entry_price - pair2_current_price) * self.config['lotsize_pair2'] * 100000
                    
                    total_profit = pair1_profit + pair2_profit
                    
                    # Check exit conditions
                    should_exit = self.calculator.check_exit_conditions(
                        row['correlation'],
                        self.config['correlation_exit_threshold'],
                        pair1_profit,
                        pair2_profit
                    )
                    
                    if should_exit:
                        # Mark exit signal
                        backtest_df.at[idx, 'exit_signal'] = True
                        
                        # Close trades
                        pair1_trade['exit_time'] = idx
                        pair1_trade['exit_price'] = pair1_current_price
                        pair1_trade['profit'] = pair1_profit
                        pair1_trade['exit_correlation'] = row['correlation']
                        pair1_trade['exit_rsi'] = row['pair1_rsi']
                        
                        pair2_trade['exit_time'] = idx
                        pair2_trade['exit_price'] = pair2_current_price
                        pair2_trade['profit'] = pair2_profit
                        pair2_trade['exit_correlation'] = row['correlation']
                        pair2_trade['exit_rsi'] = row['pair2_rsi']
                        
                        # Reset trade state
                        in_trade = False
                        active_trades = {'pair1': None, 'pair2': None}
            
            # If not in a trade, check entry conditions
            else:
                # Check cooldown period
                if last_entry_time is not None:
                    cooldown_duration = pd.Timedelta(hours=self.config['cooldown_period'])
                    if idx - last_entry_time < cooldown_duration:
                        continue
                
                # Check entry conditions
                should_enter, signal = self.calculator.check_entry_conditions(
                    row['correlation'],
                    row['pair1_rsi'],
                    row['pair2_rsi'],
                    self.config['correlation_entry_threshold'],
                    self.config['rsi_overbought'],
                    self.config['rsi_oversold']
                )
                
                if should_enter and signal:
                    # Mark entry signal
                    backtest_df.at[idx, 'entry_signal'] = True
                    backtest_df.at[idx, 'pair1_direction'] = signal['pair1_direction']
                    backtest_df.at[idx, 'pair2_direction'] = signal['pair2_direction']
                    
                    # Create trades
                    pair1_id = self.trade_id_counter
                    self.trade_id_counter += 1
                    pair2_id = self.trade_id_counter
                    self.trade_id_counter += 1
                    
                    self.trades.append({
                        'id': pair1_id,
                        'symbol': self.config['pair1'],
                        'direction': signal['pair1_direction'],
                        'entry_time': idx,
                        'entry_price': row['pair1_close'],
                        'lot_size': self.config['lotsize_pair1'],
                        'entry_correlation': row['correlation'],
                        'entry_rsi': row['pair1_rsi'],
                        'exit_time': None,
                        'exit_price': None,
                        'profit': None,
                        'exit_correlation': None,
                        'exit_rsi': None
                    })
                    
                    self.trades.append({
                        'id': pair2_id,
                        'symbol': self.config['pair2'],
                        'direction': signal['pair2_direction'],
                        'entry_time': idx,
                        'entry_price': row['pair2_close'],
                        'lot_size': self.config['lotsize_pair2'],
                        'entry_correlation': row['correlation'],
                        'entry_rsi': row['pair2_rsi'],
                        'exit_time': None,
                        'exit_price': None,
                        'profit': None,
                        'exit_correlation': None,
                        'exit_rsi': None
                    })
                    
                    # Update trade state
                    in_trade = True
                    active_trades = {
                        'pair1': pair1_id,
                        'pair2': pair2_id
                    }
                    last_entry_time = idx
        
        # Close any remaining open trades at the end of the test period
        if in_trade:
            last_idx = backtest_df.index[-1]
            pair1_trade = next((t for t in self.trades if t['id'] == active_trades['pair1']), None)
            pair2_trade = next((t for t in self.trades if t['id'] == active_trades['pair2']), None)
            
            if pair1_trade and pair2_trade:
                # Get last prices
                pair1_last_price = backtest_df.loc[last_idx, 'pair1_close']
                pair2_last_price = backtest_df.loc[last_idx, 'pair2_close']
                
                # Calculate final profit
                if pair1_trade['direction'] == 'BUY':
                    pair1_profit = (pair1_last_price - pair1_trade['entry_price']) * self.config['lotsize_pair1'] * 100000
                else:
                    pair1_profit = (pair1_trade['entry_price'] - pair1_last_price) * self.config['lotsize_pair1'] * 100000
                
                if pair2_trade['direction'] == 'BUY':
                    pair2_profit = (pair2_last_price - pair2_trade['entry_price']) * self.config['lotsize_pair2'] * 100000
                else:
                    pair2_profit = (pair2_trade['entry_price'] - pair2_last_price) * self.config['lotsize_pair2'] * 100000
                
                # Close trades
                pair1_trade['exit_time'] = last_idx
                pair1_trade['exit_price'] = pair1_last_price
                pair1_trade['profit'] = pair1_profit
                pair1_trade['exit_correlation'] = backtest_df.loc[last_idx, 'correlation']
                pair1_trade['exit_rsi'] = backtest_df.loc[last_idx, 'pair1_rsi']
                
                pair2_trade['exit_time'] = last_idx
                pair2_trade['exit_price'] = pair2_last_price
                pair2_trade['profit'] = pair2_profit
                pair2_trade['exit_correlation'] = backtest_df.loc[last_idx, 'correlation']
                pair2_trade['exit_rsi'] = backtest_df.loc[last_idx, 'pair2_rsi']
        
        # Calculate performance metrics
        metrics = self.calculate_performance_metrics()
        
        return {
            'backtest_df': backtest_df,
            'trades': self.trades,
            'metrics': metrics
        }
    
    def calculate_performance_metrics(self) -> Dict:
        """Calculate performance metrics from backtest results"""
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'average_trade_length': 0,
                'total_profit': 0
            }
        
        # Calculate metrics only for completed trades
        completed_trades = [t for t in self.trades if t['exit_time'] is not None]
        if not completed_trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'average_trade_length': 0,
                'total_profit': 0
            }
        
        # Get total number of trades
        unique_trades = set()
        for trade in completed_trades:
            trade_key = (trade['entry_time'], trade['symbol'])
            unique_trades.add(trade_key)
        total_trades = len(unique_trades) // 2  # Divide by 2 because we always open pairs
        
        # Calculate profit metrics
        profits = [t['profit'] for t in completed_trades if t['profit'] is not None]
        total_profit = sum(profits)
        winning_trades = [p for p in profits if p > 0]
        losing_trades = [p for p in profits if p <= 0]
        
        # Win rate
        win_rate = len(winning_trades) / len(profits) if profits else 0
        
        # Profit factor
        profit_factor = sum(winning_trades) / abs(sum(losing_trades)) if sum(losing_trades) != 0 else float('inf')
        
        # Calculate Sharpe ratio
        if len(profits) > 1:
            mean_return = np.mean(profits)
            std_return = np.std(profits)
            sharpe_ratio = mean_return / std_return if std_return != 0 else 0
        else:
            sharpe_ratio = 0
        
        # Calculate max drawdown
        # This is a simplified approach - in practice you'd want to track equity curve
        if len(profits) > 0:
            cumulative_profits = np.cumsum(profits)
            max_drawdown = 0
            peak = cumulative_profits[0]
            
            for profit in cumulative_profits:
                if profit > peak:
                    peak = profit
                drawdown = peak - profit
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
                    
            # Convert to percentage if there was any positive peak
            if peak > 0:
                max_drawdown = (max_drawdown / peak) * 100
        else:
            max_drawdown = 0
        
        # Calculate average trade length
        trade_durations = []
        for trade in completed_trades:
            if trade['entry_time'] is not None and trade['exit_time'] is not None:
                duration = trade['exit_time'] - trade['entry_time']
                trade_durations.append(duration.total_seconds() / 3600)  # Convert to hours
        
        avg_trade_length = np.mean(trade_durations) if trade_durations else 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'average_trade_length': avg_trade_length,
            'total_profit': total_profit
        }


# API Endpoints can be created with Flask or FastAPI
# Example FastAPI implementation:
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uuid

app = FastAPI()

# Store active strategies
active_strategies = {}

# MT5 Connection
mt5_connection = MT5Connection()

# Models
class MT5ConnectionRequest(BaseModel):
    accountId: str
    password: str
    server: str
    terminalId: str = None

class StrategyConfigRequest(BaseModel):
    name: str
    pair1: str
    pair2: str
    timeframe: str
    correlation_window: int = 50
    rsi_window: int = 14
    rsi_overbought: float = 60.0
    rsi_oversold: float = 40.0
    correlation_entry_threshold: float = -0.3
    correlation_exit_threshold: float = 0.7
    cooldown_period: float = 24.0
    lotsize_pair1: float = 0.1
    lotsize_pair2: float = 0.01

# Connect to MT5
@app.post("/api/connect")
async def connect_to_mt5(request: MT5ConnectionRequest):
    if mt5_connection.is_connected():
        mt5_connection.disconnect()
        
    success = mt5_connection.connect(
        account_id=request.accountId,
        password=request.password,
        server=request.server,
        terminal_id=request.terminalId
    )
    
    if not success:
        raise HTTPException(status_code=400, detail="Failed to connect to MT5")
    
    return {"status": "connected", "account_info": mt5_connection.get_account_info()}

# Create a new strategy
@app.post("/api/strategies")
async def create_strategy(request: StrategyConfigRequest, background_tasks: BackgroundTasks):
    if not mt5_connection.is_connected():
        raise HTTPException(status_code=400, detail="MT5 is not connected")
    
    strategy_id = str(uuid.uuid4())
    
    # Prepare strategy config
    config = {
        "id": strategy_id,
        "name": request.name,
        "pair1": request.pair1,
        "pair2": request.pair2,
        "timeframe": request.timeframe,
        "correlation_window": request.correlation_window,
        "rsi_window": request.rsi_window,
        "rsi_overbought": request.rsi_overbought,
        "rsi_oversold": request.rsi_oversold,
        "correlation_entry_threshold": request.correlation_entry_threshold,
        "correlation_exit_threshold": request.correlation_exit_threshold,
        "cooldown_period": request.cooldown_period,
        "lotsize_pair1": request.lotsize_pair1,
        "lotsize_pair2": request.lotsize_pair2,
        "mt5_account_id": mt5_connection.account_info.login,
        "mt5_password": "********",  # Don't store actual password
        "mt5_server": mt5_connection.account_info.server
    }
    
    # Create strategy
    strategy = StrategyManager(config)
    success = strategy.initialize()
    
    if not success:
        raise HTTPException(status_code=400, detail="Failed to initialize strategy")
    
    # Store active strategy
    active_strategies[strategy_id] = {
        "config": config,
        "manager": strategy,
        "status": "initialized"
    }
    
    # Run strategy check in background
    background_tasks.add_task(run_strategy_check, strategy_id)
    
    return {"strategy_id": strategy_id}

# Get all strategies
@app.get("/api/strategies")
async def get_strategies():
    strategies = []
    for strategy_id, strategy_data in active_strategies.items():
        strategies.append({
            "id": strategy_id,
            "name": strategy_data["config"]["name"],
            "pair1": strategy_data["config"]["pair1"],
            "pair2": strategy_data["config"]["pair2"],
            "status": strategy_data["status"]
        })
    
    return strategies

# Get a specific strategy
@app.get("/api/strategies/{strategy_id}")
async def get_strategy(strategy_id: str):
    if strategy_id not in active_strategies:
        raise HTTPException(status_code=404, detail="Strategy not found")
    
    strategy_data = active_strategies[strategy_id]
    
    return {
        "id": strategy_id,
        "config": strategy_data["config"],
        "status": strategy_data["status"]
    }

# Update strategy
@app.put("/api/strategies/{strategy_id}")
async def update_strategy(strategy_id: str, request: StrategyConfigRequest):
    if strategy_id not in active_strategies:
        raise HTTPException(status_code=404, detail="Strategy not found")
    
    # Get existing strategy
    strategy_data = active_strategies[strategy_id]
    
    # Stop the strategy
    strategy_data["manager"].shutdown()
    
    # Update config
    strategy_data["config"].update({
        "name": request.name,
        "pair1": request.pair1,
        "pair2": request.pair2,
        "timeframe": request.timeframe,
        "correlation_window": request.correlation_window,
        "rsi_window": request.rsi_window,
        "rsi_overbought": request.rsi_overbought,
        "rsi_oversold": request.rsi_oversold,
        "correlation_entry_threshold": request.correlation_entry_threshold,
        "correlation_exit_threshold": request.correlation_exit_threshold,
        "cooldown_period": request.cooldown_period,
        "lotsize_pair1": request.lotsize_pair1,
        "lotsize_pair2": request.lotsize_pair2
    })
    
    # Reinitialize strategy
    strategy = StrategyManager(strategy_data["config"])
    success = strategy.initialize()
    
    if not success:
        raise HTTPException(status_code=400, detail="Failed to reinitialize strategy")
    
    # Update active strategy
    active_strategies[strategy_id]["manager"] = strategy
    active_strategies[strategy_id]["status"] = "initialized"
    
    return {"status": "updated"}

# Delete strategy
@app.delete("/api/strategies/{strategy_id}")
async def delete_strategy(strategy_id: str):
    if strategy_id not in active_strategies:
        raise HTTPException(status_code=404, detail="Strategy not found")
    
    # Shutdown strategy
    active_strategies[strategy_id]["manager"].shutdown()
    
    # Remove strategy
    del active_strategies[strategy_id]
    
    return {"status": "deleted"}

# Run strategy check
async def run_strategy_check(strategy_id: str):
    if strategy_id not in active_strategies:
        return
    
    strategy_data = active_strategies[strategy_id]
    
    try:
        # Update status
        active_strategies[strategy_id]["status"] = "checking"
        
        # Run strategy check
        result = strategy_data["manager"].run_strategy_check()
        
        # Update status
        active_strategies[strategy_id]["status"] = result["status"]
        active_strategies[strategy_id]["last_check"] = {
            "time": datetime.now().isoformat(),
            "result": result
        }
    except Exception as e:
        # Update status
        active_strategies[strategy_id]["status"] = "error"
        active_strategies[strategy_id]["last_check"] = {
            "time": datetime.now().isoformat(),
            "error": str(e)
        }
"""
