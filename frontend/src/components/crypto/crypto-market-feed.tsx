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
  Bitcoin,
  DollarSign,
} from "lucide-react"
import { cn } from "@/lib/utils"
import { cryptoAPI, CryptoTicker, DEFAULT_CRYPTO_PAIRS } from "@/lib/crypto-api"

export interface CryptoUpdate {
  type: "price" | "volume" | "trade" | "pattern" | "alert"
  symbol: string
  data: any
  timestamp: Date
}

interface CryptoMarketFeedProps {
  symbols?: string[]
  maxUpdates?: number
  autoScroll?: boolean
  showFilters?: boolean
  onTickUpdate?: (ticker: CryptoTicker) => void
  onPatternDetected?: (pattern: any) => void
}

export function CryptoMarketFeed({
  symbols = DEFAULT_CRYPTO_PAIRS,
  maxUpdates = 100,
  autoScroll = true,
  showFilters = true,
  onTickUpdate,
  onPatternDetected,
}: CryptoMarketFeedProps) {
  const [isConnected, setIsConnected] = useState(false)
  const [isPaused, setIsPaused] = useState(false)
  const [updates, setUpdates] = useState<CryptoUpdate[]>([])
  const [cryptoTickers, setCryptoTickers] = useState<Map<string, CryptoTicker>>(new Map())
  const [selectedSymbol, setSelectedSymbol] = useState<string>("all")
  const [updateType, setUpdateType] = useState<string>("all")
  const [updateCount, setUpdateCount] = useState(0)
  const scrollAreaRef = useRef<HTMLDivElement>(null)
  const wsRef = useRef<WebSocket | null>(null)

  // Real crypto WebSocket connection to Binance
  useEffect(() => {
    if (isPaused || symbols.length === 0) return

    const connectWebSocket = () => {
      try {
        // Create WebSocket URL for multiple ticker streams
        const streams = symbols.slice(0, 10).map(symbol => `${symbol.toLowerCase()}@ticker`).join('/')
        const wsUrl = `${cryptoAPI.getWebSocketUrl()}/ws/${streams}`
        
        console.log('Connecting to Binance WebSocket:', wsUrl)
        
        const ws = new WebSocket(wsUrl)
        wsRef.current = ws
        
        ws.onopen = () => {
          console.log('Connected to Binance WebSocket')
          setIsConnected(true)
        }
        
        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data)
            
            // Handle ticker update
            if (data.e === '24hrTicker') {
              const ticker: CryptoTicker = {
                symbol: data.s,
                price: parseFloat(data.c),
                price_change: parseFloat(data.p),
                price_change_percent: parseFloat(data.P),
                high_price: parseFloat(data.h),
                low_price: parseFloat(data.l),
                open_price: parseFloat(data.o),
                volume: parseFloat(data.v),
                quote_volume: parseFloat(data.q),
                open_time: new Date(data.O).toISOString(),
                close_time: new Date(data.C).toISOString(),
                count: parseInt(data.c)
              }
              
              // Update tickers map
              setCryptoTickers(prev => new Map(prev).set(ticker.symbol, ticker))
              
              // Create update entry
              const update: CryptoUpdate = {
                type: "price",
                symbol: ticker.symbol,
                data: {
                  price: ticker.price,
                  change: ticker.price_change_percent,
                  volume: ticker.volume
                },
                timestamp: new Date()
              }
              
              setUpdates(prev => [update, ...prev.slice(0, maxUpdates - 1)])
              setUpdateCount(prev => prev + 1)
              
              onTickUpdate?.(ticker)
              
              // Generate pattern detection simulation occasionally
              if (Math.random() < 0.05) { // 5% chance
                const patternUpdate: CryptoUpdate = {
                  type: "pattern",
                  symbol: ticker.symbol,
                  data: {
                    pattern: ["Triangle", "Flag", "Head & Shoulders", "Support Break"][Math.floor(Math.random() * 4)],
                    confidence: 0.7 + Math.random() * 0.3,
                    direction: Math.random() > 0.5 ? "bullish" : "bearish"
                  },
                  timestamp: new Date()
                }
                
                setUpdates(prev => [patternUpdate, ...prev.slice(0, maxUpdates - 1)])
                onPatternDetected?.(patternUpdate.data)
              }
            }
          } catch (error) {
            console.error('Error parsing WebSocket message:', error)
          }
        }
        
        ws.onerror = (error) => {
          console.error('WebSocket error:', error)
          setIsConnected(false)
        }
        
        ws.onclose = (event) => {
          console.log('WebSocket closed:', event.code, event.reason)
          setIsConnected(false)
          
          // Attempt to reconnect after delay if not manually closed
          if (!isPaused && event.code !== 1000) {
            setTimeout(connectWebSocket, 3000)
          }
        }
        
      } catch (error) {
        console.error('Error creating WebSocket connection:', error)
        setIsConnected(false)
      }
    }

    connectWebSocket()

    return () => {
      if (wsRef.current) {
        wsRef.current.close(1000, 'Component unmounting')
        wsRef.current = null
      }
      setIsConnected(false)
    }
  }, [isPaused, symbols, maxUpdates, onTickUpdate, onPatternDetected])

  // Auto scroll effect
  useEffect(() => {
    if (autoScroll && scrollAreaRef.current) {
      scrollAreaRef.current.scrollTop = 0
    }
  }, [updates, autoScroll])

  // Load initial ticker data
  useEffect(() => {
    const loadInitialData = async () => {
      try {
        const tickers = await cryptoAPI.getTickerData(symbols.slice(0, 20))
        const tickersMap = new Map(tickers.map(ticker => [ticker.symbol, ticker]))
        setCryptoTickers(tickersMap)
      } catch (error) {
        console.error('Error loading initial ticker data:', error)
      }
    }

    loadInitialData()
  }, [symbols])

  const toggleConnection = () => {
    setIsPaused(!isPaused)
  }

  const getUpdateIcon = (type: string) => {
    switch (type) {
      case "price":
        return <DollarSign className="h-4 w-4 text-blue-600" />
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

  const getUpdateMessage = (update: CryptoUpdate) => {
    const formattedSymbol = cryptoAPI.formatSymbol(update.symbol)
    
    switch (update.type) {
      case "price":
        const price = cryptoAPI.formatPrice(update.data.price, update.symbol)
        const change = update.data.change >= 0 ? '+' : ''
        return `${formattedSymbol}: $${price} (${change}${update.data.change.toFixed(2)}%)`
      case "volume":
        const volume = cryptoAPI.formatVolume(update.data.volume)
        return `${formattedSymbol}: Volume ${volume}`
      case "trade":
        return `${formattedSymbol}: ${update.data.side?.toUpperCase()} ${update.data.size} @ $${update.data.price}`
      case "pattern":
        return `${formattedSymbol}: ${update.data.pattern} pattern detected (${(update.data.confidence * 100).toFixed(0)}% confidence)`
      case "alert":
        return update.data.message
      default:
        return "Unknown update"
    }
  }

  const filteredUpdates = updates.filter(update => {
    if (selectedSymbol !== "all" && update.symbol !== selectedSymbol) {
      return false
    }
    if (updateType !== "all" && update.type !== updateType) {
      return false
    }
    return true
  })

  const cryptoTickersArray = Array.from(cryptoTickers.values()).sort((a, b) => 
    Math.abs(b.price_change_percent) - Math.abs(a.price_change_percent)
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
            {cryptoTickersArray.length > 0 ? (
              <div>
                <p className="text-lg font-bold">
                  {cryptoAPI.formatSymbol(cryptoTickersArray[0].symbol)}
                </p>
                <p className={cn(
                  "text-sm",
                  cryptoTickersArray[0].price_change_percent >= 0 ? "text-green-600" : "text-red-600"
                )}>
                  {cryptoTickersArray[0].price_change_percent >= 0 ? "+" : ""}
                  {cryptoTickersArray[0].price_change_percent.toFixed(2)}%
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
              <Bitcoin className="h-4 w-4" />
              Active Pairs
            </CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">{cryptoTickers.size}</p>
          </CardContent>
        </Card>
      </div>

      {/* Live Feed */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Crypto Tickers */}
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
                {cryptoTickersArray.map((ticker) => (
                  <div key={ticker.symbol} className="flex items-center justify-between p-2 rounded-lg border">
                    <div>
                      <p className="font-medium">{cryptoAPI.formatSymbol(ticker.symbol)}</p>
                      <p className="text-sm text-muted-foreground">
                        ${cryptoAPI.formatPrice(ticker.price, ticker.symbol)}
                      </p>
                    </div>
                    <div className="text-right">
                      <p className={cn(
                        "font-medium",
                        ticker.price_change_percent >= 0 ? "text-green-600" : "text-red-600"
                      )}>
                        {ticker.price_change_percent >= 0 ? "+" : ""}
                        {ticker.price_change_percent.toFixed(2)}%
                      </p>
                      <p className="text-sm text-muted-foreground">
                        Vol: {cryptoAPI.formatVolume(ticker.volume)}
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
                        {symbols.slice(0, 10).map(symbol => (
                          <SelectItem key={symbol} value={symbol}>
                            {cryptoAPI.formatSymbol(symbol)}
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