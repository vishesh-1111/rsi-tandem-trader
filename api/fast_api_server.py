
import os
import uuid
import json
import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from mt5_api import MT5Connection, DataFetcher, StrategyCalculator, TradeManager, StrategyManager, BacktestEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api_server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("api_server")

app = FastAPI(title="Rolling Correlation Strategy API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
mt5_connection = MT5Connection()
active_strategies = {}  # Store active strategy instances
strategy_tasks = {}  # Store background tasks for strategies

# Pydantic models
class MT5ConnectionRequest(BaseModel):
    account_id: str
    password: str
    server: str
    terminal_id: Optional[str] = None

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

class BacktestRequest(BaseModel):
    strategy_config: StrategyConfigRequest
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    pair1_file: str  # Path to CSV file with pair1 historical data
    pair2_file: str  # Path to CSV file with pair2 historical data

# Background task to check strategy periodically
async def strategy_monitor(strategy_id: str):
    """Background task that periodically checks strategy conditions"""
    while strategy_id in active_strategies:
        try:
            strategy_data = active_strategies[strategy_id]
            
            # Check if strategy is active
            if strategy_data["status"] != "active":
                await asyncio.sleep(10)
                continue
            
            # Run strategy check
            result = strategy_data["manager"].run_strategy_check()
            
            # Update status
            active_strategies[strategy_id]["status"] = result["status"]
            active_strategies[strategy_id]["last_check"] = {
                "time": datetime.now().isoformat(),
                "result": result
            }
            
            # Log result
            logger.info(f"Strategy {strategy_id} check result: {result['status']}")
            
            # Wait before next check
            await asyncio.sleep(10)  # Check every 10 seconds
        except Exception as e:
            logger.error(f"Error in strategy monitor for {strategy_id}: {str(e)}")
            
            # Update status
            if strategy_id in active_strategies:
                active_strategies[strategy_id]["status"] = "error"
                active_strategies[strategy_id]["last_check"] = {
                    "time": datetime.now().isoformat(),
                    "error": str(e)
                }
            
            await asyncio.sleep(30)  # Wait longer on error

# API endpoints
@app.post("/api/connect")
async def connect_to_mt5(request: MT5ConnectionRequest):
    """Connect to MT5 terminal"""
    try:
        # Disconnect if already connected
        if mt5_connection.is_connected():
            mt5_connection.disconnect()
        
        # Connect to MT5
        success = mt5_connection.connect(
            account_id=request.account_id,
            password=request.password,
            server=request.server,
            terminal_id=request.terminal_id
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to connect to MT5")
        
        # Get account info
        account_info = mt5_connection.get_account_info()
        
        return {
            "status": "connected",
            "account_info": account_info
        }
    except Exception as e:
        logger.error(f"MT5 connection error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/connection_status")
async def connection_status():
    """Check MT5 connection status"""
    try:
        is_connected = mt5_connection.is_connected()
        
        response = {
            "connected": is_connected
        }
        
        if is_connected:
            response["account_info"] = mt5_connection.get_account_info()
        
        return response
    except Exception as e:
        logger.error(f"Error checking connection status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/disconnect")
async def disconnect_from_mt5():
    """Disconnect from MT5 terminal"""
    try:
        # Stop all strategies
        for strategy_id in list(active_strategies.keys()):
            await stop_strategy(strategy_id)
        
        # Disconnect from MT5
        mt5_connection.disconnect()
        
        return {"status": "disconnected"}
    except Exception as e:
        logger.error(f"MT5 disconnection error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/strategies")
async def create_strategy(request: StrategyConfigRequest, background_tasks: BackgroundTasks):
    """Create a new strategy"""
    try:
        if not mt5_connection.is_connected():
            raise HTTPException(status_code=400, detail="MT5 is not connected")
        
        # Generate a unique ID for the strategy
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
            "status": "initialized",
            "created_at": datetime.now().isoformat(),
            "last_check": None
        }
        
        # Start background task to monitor strategy
        task = asyncio.create_task(strategy_monitor(strategy_id))
        strategy_tasks[strategy_id] = task
        
        return {
            "strategy_id": strategy_id,
            "status": "initialized"
        }
    except Exception as e:
        logger.error(f"Error creating strategy: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/strategies")
async def get_strategies():
    """Get all strategies"""
    try:
        strategies = []
        for strategy_id, strategy_data in active_strategies.items():
            strategies.append({
                "id": strategy_id,
                "name": strategy_data["config"]["name"],
                "pair1": strategy_data["config"]["pair1"],
                "pair2": strategy_data["config"]["pair2"],
                "status": strategy_data["status"],
                "created_at": strategy_data["created_at"],
                "last_check": strategy_data["last_check"]
            })
        
        return strategies
    except Exception as e:
        logger.error(f"Error getting strategies: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/strategies/{strategy_id}")
async def get_strategy(strategy_id: str):
    """Get a specific strategy"""
    try:
        if strategy_id not in active_strategies:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        strategy_data = active_strategies[strategy_id]
        
        return {
            "id": strategy_id,
            "config": strategy_data["config"],
            "status": strategy_data["status"],
            "created_at": strategy_data["created_at"],
            "last_check": strategy_data["last_check"]
        }
    except Exception as e:
        logger.error(f"Error getting strategy {strategy_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/strategies/{strategy_id}")
async def update_strategy(strategy_id: str, request: StrategyConfigRequest):
    """Update a strategy"""
    try:
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
        
        return {
            "status": "updated",
            "id": strategy_id
        }
    except Exception as e:
        logger.error(f"Error updating strategy {strategy_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/strategies/{strategy_id}/start")
async def start_strategy(strategy_id: str):
    """Start a strategy"""
    try:
        if strategy_id not in active_strategies:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        # Update status
        active_strategies[strategy_id]["status"] = "active"
        
        return {
            "status": "started",
            "id": strategy_id
        }
    except Exception as e:
        logger.error(f"Error starting strategy {strategy_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/strategies/{strategy_id}/stop")
async def stop_strategy(strategy_id: str):
    """Stop a strategy"""
    try:
        if strategy_id not in active_strategies:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        # Update status
        active_strategies[strategy_id]["status"] = "stopped"
        
        # Close all open trades
        active_strategies[strategy_id]["manager"].close_all_trades()
        
        return {
            "status": "stopped",
            "id": strategy_id
        }
    except Exception as e:
        logger.error(f"Error stopping strategy {strategy_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/strategies/{strategy_id}")
async def delete_strategy(strategy_id: str):
    """Delete a strategy"""
    try:
        if strategy_id not in active_strategies:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        # Shutdown strategy
        active_strategies[strategy_id]["manager"].shutdown()
        
        # Cancel background task
        if strategy_id in strategy_tasks:
            strategy_tasks[strategy_id].cancel()
            del strategy_tasks[strategy_id]
        
        # Remove strategy
        del active_strategies[strategy_id]
        
        return {
            "status": "deleted",
            "id": strategy_id
        }
    except Exception as e:
        logger.error(f"Error deleting strategy {strategy_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/backtest")
async def run_backtest(request: BacktestRequest):
    """Run a backtest"""
    try:
        # Load historical data from CSV files
        pair1_data = pd.read_csv(request.pair1_file)
        pair2_data = pd.read_csv(request.pair2_file)
        
        # Convert time column to datetime and set as index
        pair1_data['time'] = pd.to_datetime(pair1_data['time'])
        pair2_data['time'] = pd.to_datetime(pair2_data['time'])
        pair1_data.set_index('time', inplace=True)
        pair2_data.set_index('time', inplace=True)
        
        # Create backtest engine
        config = {
            "name": request.strategy_config.name,
            "pair1": request.strategy_config.pair1,
            "pair2": request.strategy_config.pair2,
            "timeframe": request.strategy_config.timeframe,
            "correlation_window": request.strategy_config.correlation_window,
            "rsi_window": request.strategy_config.rsi_window,
            "rsi_overbought": request.strategy_config.rsi_overbought,
            "rsi_oversold": request.strategy_config.rsi_oversold,
            "correlation_entry_threshold": request.strategy_config.correlation_entry_threshold,
            "correlation_exit_threshold": request.strategy_config.correlation_exit_threshold,
            "cooldown_period": request.strategy_config.cooldown_period,
            "lotsize_pair1": request.strategy_config.lotsize_pair1,
            "lotsize_pair2": request.strategy_config.lotsize_pair2
        }
        
        backtest_engine = BacktestEngine(config)
        
        # Run backtest
        backtest_result = backtest_engine.run_backtest(
            pair1_data,
            pair2_data,
            request.start_date,
            request.end_date
        )
        
        # Convert trades to JSON-serializable format
        trades = []
        for trade in backtest_result['trades']:
            trade_copy = trade.copy()
            
            # Convert datetime objects to strings
            if isinstance(trade_copy['entry_time'], pd.Timestamp):
                trade_copy['entry_time'] = trade_copy['entry_time'].isoformat()
            if isinstance(trade_copy['exit_time'], pd.Timestamp) and trade_copy['exit_time'] is not None:
                trade_copy['exit_time'] = trade_copy['exit_time'].isoformat()
            
            trades.append(trade_copy)
        
        # Return results
        return {
            "metrics": backtest_result['metrics'],
            "trades": trades,
            "strategy_config": config
        }
    except Exception as e:
        logger.error(f"Backtest error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/pairs")
async def get_available_pairs():
    """Get available currency pairs from MT5"""
    try:
        if not mt5_connection.is_connected():
            raise HTTPException(status_code=400, detail="MT5 is not connected")
        
        import MetaTrader5 as mt5
        
        # Get symbols
        symbols = mt5.symbols_get()
        if symbols is None:
            raise HTTPException(status_code=500, detail="Failed to get symbols from MT5")
        
        # Filter forex pairs
        forex_pairs = [s.name for s in symbols if s.path.startswith("Forex")]
        
        return forex_pairs
    except Exception as e:
        logger.error(f"Error getting pairs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/data/historical")
async def get_historical_data(pair: str, timeframe: str, bars: int = 500):
    """Get historical data for a pair"""
    try:
        if not mt5_connection.is_connected():
            raise HTTPException(status_code=400, detail="MT5 is not connected")
        
        # Create data fetcher
        data_fetcher = DataFetcher(mt5_connection)
        
        # Get historical data
        data = data_fetcher.get_historical_data(pair, timeframe, bars)
        
        # Convert to list of dictionaries
        records = data.reset_index().to_dict(orient='records')
        
        # Convert datetime objects to strings
        for record in records:
            if isinstance(record['time'], pd.Timestamp):
                record['time'] = record['time'].isoformat()
        
        return records
    except Exception as e:
        logger.error(f"Error getting historical data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# On startup
@app.on_event("startup")
async def startup_event():
    logger.info("API Server starting up")

# On shutdown
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("API Server shutting down")
    
    # Stop all strategies
    for strategy_id in list(active_strategies.keys()):
        try:
            active_strategies[strategy_id]["manager"].shutdown()
        except Exception as e:
            logger.error(f"Error shutting down strategy {strategy_id}: {str(e)}")
    
    # Cancel all tasks
    for task in strategy_tasks.values():
        task.cancel()
    
    # Disconnect from MT5
    if mt5_connection.is_connected():
        mt5_connection.disconnect()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
