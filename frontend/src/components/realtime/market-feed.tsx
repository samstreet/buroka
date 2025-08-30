"use client"

import { useState, useEffect, useRef } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  TrendingUp,
  TrendingDown,
  Activity,
  Pause,
  Play,
  Volume2,
  Clock,
  Wifi,
  WifiOff,
  AlertTriangle,
} from "lucide-react"
import { cn } from "@/lib/utils"

export interface MarketTick {
  id: string
  symbol: string
  price: number
  change: number
  changePercent: number
  volume: number
  bid: number
  ask: number
  bidSize: number
  askSize: number
  lastTradeTime: Date
  dayHigh: number
  dayLow: number
  dayOpen: number
  timestamp: Date
}

export interface MarketUpdate {
  type: "price" | "volume" | "trade" | "pattern" | "alert"
  data: any
  timestamp: Date
}

interface MarketFeedProps {
  symbols?: string[]
  maxUpdates?: number
  autoScroll?: boolean
  showFilters?: boolean
  onTickUpdate?: (tick: MarketTick) => void
  onPatternDetected?: (pattern: any) => void
}

export function MarketFeed({
  symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT"],
  maxUpdates = 100,
  autoScroll = true,
  showFilters = true,
  onTickUpdate,
  onPatternDetected,
}: MarketFeedProps) {
  const [isConnected, setIsConnected] = useState(false)
  const [isPaused, setIsPaused] = useState(false)
  const [updates, setUpdates] = useState<MarketUpdate[]>([])
  const [marketTicks, setMarketTicks] = useState<Map<string, MarketTick>>(new Map())
  const [selectedSymbol, setSelectedSymbol] = useState<string>("all")
  const [updateType, setUpdateType] = useState<string>("all")
  const [updateCount, setUpdateCount] = useState(0)
  const scrollAreaRef = useRef<HTMLDivElement>(null)
  const wsRef = useRef<WebSocket | null>(null)

  // Mock WebSocket connection
  useEffect(() => {
    if (isPaused) return

    // Simulate WebSocket connection
    setIsConnected(true)
    
    const generateMockUpdate = (): MarketUpdate => {
      const symbol = symbols[Math.floor(Math.random() * symbols.length)]
      const existingTick = marketTicks.get(symbol)
      const basePrice = existingTick?.price || (150 + Math.random() * 200)
      
      const change = (Math.random() - 0.5) * 5
      const newPrice = Math.max(1, basePrice + change)
      const changePercent = (change / basePrice) * 100
      
      const updateTypes = ["price", "volume", "trade", "pattern", "alert"]
      const type = updateTypes[Math.floor(Math.random() * updateTypes.length)] as any
      
      let data: any
      
      switch (type) {
        case "price":
          data = {
            symbol,
            oldPrice: basePrice,
            newPrice,
            change,
            changePercent,
            volume: Math.floor(Math.random() * 1000000),
          }
          break
        case "volume":
          data = {
            symbol,
            volume: Math.floor(Math.random() * 5000000),
            avgVolume: Math.floor(Math.random() * 2000000),
          }
          break
        case "trade":
          data = {
            symbol,
            price: newPrice,
            size: Math.floor(Math.random() * 1000) + 100,
            side: Math.random() > 0.5 ? "buy" : "sell",
          }
          break
        case "pattern":
          const patterns = ["Triangle", "Flag", "Head & Shoulders", "Double Top", "Double Bottom"]
          data = {
            symbol,
            patternType: patterns[Math.floor(Math.random() * patterns.length)],
            confidence: 0.7 + Math.random() * 0.3,
            direction: Math.random() > 0.5 ? "bullish" : "bearish",
          }
          break
        case "alert":
          const alerts = ["Price breakout", "Volume spike", "Moving average cross", "RSI oversold"]
          data = {
            symbol,
            alertType: alerts[Math.floor(Math.random() * alerts.length)],
            message: `${symbol} triggered ${alerts[Math.floor(Math.random() * alerts.length)].toLowerCase()}`,
          }
          break
      }
      
      // Update market tick
      const newTick: MarketTick = {
        id: `${symbol}-${Date.now()}`,
        symbol,
        price: newPrice,
        change,
        changePercent,
        volume: Math.floor(Math.random() * 1000000),
        bid: newPrice - 0.01,
        ask: newPrice + 0.01,
        bidSize: Math.floor(Math.random() * 1000) + 100,
        askSize: Math.floor(Math.random() * 1000) + 100,
        lastTradeTime: new Date(),
        dayHigh: Math.max(existingTick?.dayHigh || newPrice, newPrice),
        dayLow: Math.min(existingTick?.dayLow || newPrice, newPrice),
        dayOpen: existingTick?.dayOpen || newPrice,
        timestamp: new Date(),
      }
      
      setMarketTicks(prev => new Map(prev).set(symbol, newTick))
      onTickUpdate?.(newTick)
      
      if (type === "pattern") {
        onPatternDetected?.(data)
      }
      
      return {
        type,
        data,
        timestamp: new Date(),
      }
    }

    const interval = setInterval(() => {
      if (!isPaused) {
        const update = generateMockUpdate()
        
        setUpdates(prev => {
          const newUpdates = [update, ...prev.slice(0, maxUpdates - 1)]
          return newUpdates
        })
        
        setUpdateCount(prev => prev + 1)
      }
    }, 500 + Math.random() * 1500) // Random interval between 500ms and 2s

    return () => {
      clearInterval(interval)
      setIsConnected(false)
    }
  }, [isPaused, symbols, maxUpdates, marketTicks, onTickUpdate, onPatternDetected])

  // Auto scroll effect
  useEffect(() => {
    if (autoScroll && scrollAreaRef.current) {
      scrollAreaRef.current.scrollTop = 0
    }
  }, [updates, autoScroll])

  const toggleConnection = () => {
    setIsPaused(!isPaused)
  }

  const getUpdateIcon = (type: string) => {
    switch (type) {
      case "price":
        return <TrendingUp className="h-4 w-4 text-blue-600" />
      case "volume":
        return <Volume2 className="h-4 w-4 text-purple-600" />
      case "trade":
        return <Activity className="h-4 w-4 text-green-600" />
      case "pattern":
        return <TrendingUp className="h-4 w-4 text-orange-600" />
      case "alert":
        return <AlertTriangle className="h-4 w-4 text-red-600" />
      default:
        return <Clock className="h-4 w-4 text-gray-600" />
    }
  }

  const getUpdateMessage = (update: MarketUpdate) => {
    switch (update.type) {
      case "price":
        return `${update.data.symbol}: $${update.data.newPrice.toFixed(2)} (${update.data.changePercent >= 0 ? '+' : ''}${update.data.changePercent.toFixed(2)}%)`
      case "volume":
        return `${update.data.symbol}: Volume ${(update.data.volume / 1000000).toFixed(2)}M`
      case "trade":
        return `${update.data.symbol}: ${update.data.side.toUpperCase()} ${update.data.size} @ $${update.data.price.toFixed(2)}`
      case "pattern":
        return `${update.data.symbol}: ${update.data.patternType} pattern detected (${(update.data.confidence * 100).toFixed(0)}% confidence)`
      case "alert":
        return update.data.message
      default:
        return "Unknown update"
    }
  }

  const filteredUpdates = updates.filter(update => {
    if (selectedSymbol !== "all" && update.data.symbol !== selectedSymbol) {
      return false
    }
    if (updateType !== "all" && update.type !== updateType) {
      return false
    }
    return true
  })

  const marketTicksArray = Array.from(marketTicks.values()).sort((a, b) => 
    Math.abs(b.changePercent) - Math.abs(a.changePercent)
  )

  return (
    <div className="space-y-4">
      {/* Market Overview */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardDescription className="flex items-center gap-2">
              <Activity className="h-4 w-4" />
              Total Updates
            </CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">{updateCount.toLocaleString()}</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardDescription className="flex items-center gap-2">
              {isConnected && !isPaused ? (
                <Wifi className="h-4 w-4 text-green-600" />
              ) : (
                <WifiOff className="h-4 w-4 text-red-600" />
              )}
              Connection
            </CardDescription>
          </CardHeader>
          <CardContent>
            <p className={cn(
              "text-2xl font-bold",
              isConnected && !isPaused ? "text-green-600" : "text-red-600"
            )}>
              {isPaused ? "Paused" : isConnected ? "Live" : "Offline"}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardDescription className="flex items-center gap-2">
              <TrendingUp className="h-4 w-4" />
              Top Mover
            </CardDescription>
          </CardHeader>
          <CardContent>
            {marketTicksArray.length > 0 ? (
              <div>
                <p className="text-lg font-bold">{marketTicksArray[0].symbol}</p>
                <p className={cn(
                  "text-sm",
                  marketTicksArray[0].changePercent >= 0 ? "text-green-600" : "text-red-600"
                )}>
                  {marketTicksArray[0].changePercent >= 0 ? "+" : ""}
                  {marketTicksArray[0].changePercent.toFixed(2)}%
                </p>
              </div>
            ) : (
              <p className="text-lg font-bold">-</p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardDescription className="flex items-center gap-2">
              <Volume2 className="h-4 w-4" />
              Active Symbols
            </CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">{marketTicks.size}</p>
          </CardContent>
        </Card>
      </div>

      {/* Live Feed */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Market Tickers */}
        <Card className="lg:col-span-1">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center justify-between">
              <span>Live Tickers</span>
              <Button
                variant="ghost"
                size="icon"
                onClick={toggleConnection}
              >
                {isPaused ? <Play className="h-4 w-4" /> : <Pause className="h-4 w-4" />}
              </Button>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-96">
              <div className="space-y-2">
                {marketTicksArray.map((tick) => (
                  <div key={tick.symbol} className="flex items-center justify-between p-2 rounded-lg border">
                    <div>
                      <p className="font-medium">{tick.symbol}</p>
                      <p className="text-sm text-muted-foreground">
                        ${tick.price.toFixed(2)}
                      </p>
                    </div>
                    <div className="text-right">
                      <p className={cn(
                        "font-medium",
                        tick.changePercent >= 0 ? "text-green-600" : "text-red-600"
                      )}>
                        {tick.changePercent >= 0 ? "+" : ""}
                        {tick.changePercent.toFixed(2)}%
                      </p>
                      <p className="text-sm text-muted-foreground">
                        Vol: {(tick.volume / 1000000).toFixed(2)}M
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </ScrollArea>
          </CardContent>
        </Card>

        {/* Update Feed */}
        <Card className="lg:col-span-2">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle>Live Updates</CardTitle>
              <div className="flex items-center gap-2">
                {showFilters && (
                  <>
                    <Select value={selectedSymbol} onValueChange={setSelectedSymbol}>
                      <SelectTrigger className="w-32">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="all">All Symbols</SelectItem>
                        {symbols.map(symbol => (
                          <SelectItem key={symbol} value={symbol}>
                            {symbol}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>

                    <Select value={updateType} onValueChange={setUpdateType}>
                      <SelectTrigger className="w-32">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="all">All Types</SelectItem>
                        <SelectItem value="price">Price</SelectItem>
                        <SelectItem value="volume">Volume</SelectItem>
                        <SelectItem value="trade">Trade</SelectItem>
                        <SelectItem value="pattern">Pattern</SelectItem>
                        <SelectItem value="alert">Alert</SelectItem>
                      </SelectContent>
                    </Select>
                  </>
                )}
              </div>
            </div>
            <CardDescription>
              {filteredUpdates.length} updates • {isPaused ? "Paused" : "Live"}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-96" ref={scrollAreaRef}>
              <div className="space-y-2">
                {filteredUpdates.map((update, index) => (
                  <div key={`${update.timestamp.getTime()}-${index}`} className={cn(
                    "flex items-center gap-3 p-3 rounded-lg transition-all",
                    index === 0 && !isPaused ? "bg-accent animate-pulse" : "bg-muted/50"
                  )}>
                    <div className="flex items-center gap-2">
                      {getUpdateIcon(update.type)}
                      <Badge variant="outline" className="text-xs">
                        {update.type.toUpperCase()}
                      </Badge>
                    </div>
                    <div className="flex-1">
                      <p className="text-sm">{getUpdateMessage(update)}</p>
                      <p className="text-xs text-muted-foreground">
                        {update.timestamp.toLocaleTimeString()}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </ScrollArea>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}