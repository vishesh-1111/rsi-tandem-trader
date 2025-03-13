
# MT5 Rolling Correlation Strategy API

This is the backend API for the Rolling Correlation Trading Strategy. It implements the connection to MT5, the strategy logic, and trade execution.

## Setup

1. Install MetaTrader 5 on your machine
2. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Make sure MetaTrader 5 is running
4. Start the API server:
   ```
   cd api
   python fast_api_server.py
   ```

## API Endpoints

- `/api/connect` - Connect to MT5
- `/api/disconnect` - Disconnect from MT5
- `/api/connection_status` - Check MT5 connection status
- `/api/strategies` - Create or list strategies
- `/api/strategies/{strategy_id}` - Get, update, or delete a strategy
- `/api/strategies/{strategy_id}/start` - Start a strategy
- `/api/strategies/{strategy_id}/stop` - Stop a strategy
- `/api/backtest` - Run a backtest
- `/api/pairs` - Get available currency pairs
- `/api/data/historical` - Get historical data for a pair

## Architecture

The backend is structured with the following main components:

1. **MT5Connection** - Handles authentication and connection to MT5
2. **DataFetcher** - Retrieves historical and live price data
3. **StrategyCalculator** - Implements correlation and RSI logic
4. **TradeManager** - Executes and monitors trades
5. **StrategyManager** - Manages the strategy execution
6. **BacktestEngine** - Implements backtesting functionality
7. **FastAPI Server** - Exposes REST endpoints for the frontend

## Strategy Logic

The rolling correlation strategy enters trades when:
1. Correlation between two pairs is below the entry threshold
2. One pair is overbought (RSI > threshold) and the other is oversold (RSI < threshold)

The strategy exits trades when:
1. Correlation between the pairs exceeds the exit threshold
2. The combined trade is profitable

## Testing

To run tests:
```
pytest
```
