
export type TimeFrame = '1m' | '5m' | '15m' | '30m' | '1h' | '4h' | '1d' | '1w' | '1M';

export interface StrategyConfig {
  id: string;
  name: string;
  pair1: string;
  pair2: string;
  timeframe: TimeFrame;
  correlation_window: number;
  rsi_window: number;
  rsi_overbought: number;
  rsi_oversold: number;
  correlation_entry_threshold: number;
  correlation_exit_threshold: number;
  cooldown_period: number;
  lotsize_pair1: number;
  lotsize_pair2: number;
  isActive: boolean;
}

export interface Trade {
  id: string;
  strategyId: string;
  symbol: string;
  ticket: number;
  direction: 'BUY' | 'SELL';
  openTime: string;
  closeTime?: string;
  openPrice: number;
  closePrice?: number;
  lotSize: number;
  profit?: number;
  magicNumber: number;
  entryCorrelation: number;
  entryRsi: number;
  exitCorrelation?: number;
  exitRsi?: number;
}

export interface MT5Credentials {
  accountId: string;
  password: string;
  server: string;
  terminalId?: string;
}

export interface CorrelationData {
  timestamp: string;
  value: number;
}

export interface RsiData {
  timestamp: string;
  pair1: number;
  pair2: number;
}

export interface PerformanceMetrics {
  totalTrades: number;
  winRate: number;
  profitFactor: number;
  sharpeRatio: number;
  maxDrawdown: number;
  averageTradeLength: number;
  totalProfit: number;
}
