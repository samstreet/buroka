export interface User {
  user_id: string
  username: string
  email: string
  full_name?: string
  roles: string[]
  created_at: string
  last_login?: string
  is_active: boolean
}

export interface Pattern {
  id: string
  symbol: string
  patternType: string
  patternName: string
  direction: "bullish" | "bearish" | "neutral"
  confidence: number
  timeframe: string
  detectedAt: string
  entryPrice?: number
  targetPrice?: number
  stopLoss?: number
  status: "active" | "completed" | "failed"
}

export interface MarketData {
  symbol: string
  timestamp: string
  open: number
  high: number
  low: number
  close: number
  volume: number
}

export interface ApiResponse<T> {
  data: T
  message?: string
  error?: string
  status: number
}