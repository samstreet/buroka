"use client"

import { useState, useEffect, useCallback } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import {
  TrendingUp,
  TrendingDown,
  Volume2,
  Activity,
  Search,
  Filter,
  Play,
  Pause,
  Target,
  AlertTriangle,
  Eye,
  Star,
} from "lucide-react"
import { cn } from "@/lib/utils"
import { format } from "date-fns"

export interface ScannerResult {
  id: string
  symbol: string
  name: string
  price: number
  change: number
  changePercent: number
  volume: number
  avgVolume: number
  volumeRatio: number
  marketCap: number
  sector: string
  signals: ScannerSignal[]
  score: number
  lastUpdate: Date
}

export interface ScannerSignal {
  type: "pattern" | "technical" | "volume" | "price" | "news"
  name: string
  description: string
  strength: "weak" | "moderate" | "strong"
  direction: "bullish" | "bearish" | "neutral"
  confidence: number
  timestamp: Date
}

interface MarketScannerProps {
  maxResults?: number
  autoRefresh?: boolean
  refreshInterval?: number
  onResultClick?: (result: ScannerResult) => void
  onAddToWatchlist?: (symbol: string) => void
}

export function MarketScanner({
  maxResults = 100,
  autoRefresh = true,
  refreshInterval = 5000,
  onResultClick,
  onAddToWatchlist,
}: MarketScannerProps) {
  const [results, setResults] = useState<ScannerResult[]>([])
  const [isScanning, setIsScanning] = useState(true)
  const [searchQuery, setSearchQuery] = useState("")
  const [sectorFilter, setSectorFilter] = useState("all")
  const [signalFilter, setSignalFilter] = useState("all")
  const [sortBy, setSortBy] = useState<"score" | "change" | "volume" | "signals">("score")
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("desc")
  const [scanCount, setScanCount] = useState(0)
  const [lastScanTime, setLastScanTime] = useState<Date | null>(null)

  // Sample crypto universe
  const cryptoUniverse = [
    { symbol: "BTCUSDT", name: "Bitcoin", sector: "Layer 1", marketCap: 850000000000 },
    { symbol: "ETHUSDT", name: "Ethereum", sector: "Layer 1", marketCap: 280000000000 },
    { symbol: "BNBUSDT", name: "Binance Coin", sector: "Exchange Token", marketCap: 85000000000 },
    { symbol: "ADAUSDT", name: "Cardano", sector: "Layer 1", marketCap: 42000000000 },
    { symbol: "SOLUSDT", name: "Solana", sector: "Layer 1", marketCap: 38000000000 },
    { symbol: "DOGEUSDT", name: "Dogecoin", sector: "Meme Coin", marketCap: 28000000000 },
    { symbol: "XRPUSDT", name: "Ripple", sector: "Payment", marketCap: 35000000000 },
    { symbol: "DOTUSDT", name: "Polkadot", sector: "Interoperability", marketCap: 12000000000 },
    { symbol: "AVAXUSDT", name: "Avalanche", sector: "Layer 1", marketCap: 15000000000 },
    { symbol: "MATICUSDT", name: "Polygon", sector: "Layer 2", marketCap: 8500000000 },
    { symbol: "LINKUSDT", name: "Chainlink", sector: "Oracle", marketCap: 14000000000 },
    { symbol: "LTCUSDT", name: "Litecoin", sector: "Payment", marketCap: 7800000000 },
    { symbol: "UNIUSDT", name: "Uniswap", sector: "DeFi", marketCap: 6200000000 },
    { symbol: "ATOMUSDT", name: "Cosmos", sector: "Interoperability", marketCap: 4800000000 },
    { symbol: "VETUSDT", name: "VeChain", sector: "Supply Chain", marketCap: 3200000000 },
  ]

  const signalTypes = ["pattern", "technical", "volume", "price", "news"]
  const patternTypes = [
    "Triangle Breakout", "Flag Pattern", "Head & Shoulders", "Double Bottom",
    "Cup & Handle", "Wedge", "Channel Breakout", "Support Break", "Resistance Break"
  ]
  const technicalSignals = [
    "RSI Oversold", "RSI Overbought", "MACD Bullish Cross", "MACD Bearish Cross",
    "Moving Average Cross", "Bollinger Band Squeeze", "Volume Breakout"
  ]

  const generateSignals = useCallback((symbol: string): ScannerSignal[] => {
    const signals: ScannerSignal[] = []
    const numSignals = Math.floor(Math.random() * 4) + 1 // 1-4 signals

    for (let i = 0; i < numSignals; i++) {
      const type = signalTypes[Math.floor(Math.random() * signalTypes.length)] as any
      const strength = ["weak", "moderate", "strong"][Math.floor(Math.random() * 3)] as any
      const direction = Math.random() > 0.5 ? "bullish" : "bearish"
      const confidence = 0.6 + Math.random() * 0.4

      let name: string
      let description: string

      switch (type) {
        case "pattern":
          name = patternTypes[Math.floor(Math.random() * patternTypes.length)]
          description = `${name} pattern detected with ${strength} confirmation`
          break
        case "technical":
          name = technicalSignals[Math.floor(Math.random() * technicalSignals.length)]
          description = `${name} indicating ${direction} momentum`
          break
        case "volume":
          name = "Volume Spike"
          description = `Unusual volume activity - ${(1.5 + Math.random() * 2).toFixed(1)}x average`
          break
        case "price":
          name = direction === "bullish" ? "Price Breakout" : "Price Breakdown"
          description = `Price ${direction === "bullish" ? "broke above" : "broke below"} key level`
          break
        case "news":
          const newsTypes = ["Earnings Beat", "Analyst Upgrade", "Product Launch", "Partnership"]
          name = newsTypes[Math.floor(Math.random() * newsTypes.length)]
          description = `${name} could impact stock price`
          break
        default:
          name = "Unknown Signal"
          description = "Signal detected"
      }

      signals.push({
        type,
        name,
        description,
        strength,
        direction,
        confidence,
        timestamp: new Date(),
      })
    }

    return signals
  }, [])

  const generateScanResult = useCallback((crypto: typeof cryptoUniverse[0]): ScannerResult => {
    const basePrice = 100 + Math.random() * 200
    const change = (Math.random() - 0.5) * 10
    const changePercent = (change / basePrice) * 100
    const volume = Math.floor(Math.random() * 10000000) + 1000000
    const avgVolume = Math.floor(Math.random() * 5000000) + 2000000
    const volumeRatio = volume / avgVolume

    const signals = generateSignals(crypto.symbol)
    
    // Calculate score based on signals
    let score = 0
    signals.forEach(signal => {
      let signalScore = signal.confidence * 100
      
      // Weight by strength
      if (signal.strength === "strong") signalScore *= 1.5
      else if (signal.strength === "moderate") signalScore *= 1.2
      
      // Weight by type
      if (signal.type === "pattern") signalScore *= 1.3
      else if (signal.type === "technical") signalScore *= 1.2
      
      score += signalScore
    })
    
    // Normalize score
    score = Math.min(100, score / signals.length)

    return {
      id: `${crypto.symbol}-${Date.now()}`,
      symbol: crypto.symbol,
      name: crypto.name,
      price: basePrice,
      change,
      changePercent,
      volume,
      avgVolume,
      volumeRatio,
      marketCap: crypto.marketCap,
      sector: crypto.sector,
      signals,
      score,
      lastUpdate: new Date(),
    }
  }, [generateSignals])

  // Perform market scan
  const performScan = useCallback(() => {
    if (!isScanning) return

    const newResults: ScannerResult[] = []
    
    // Randomly select stocks to scan (simulate real scanner behavior)
    const cryptosToScan = cryptoUniverse
      .sort(() => Math.random() - 0.5)
      .slice(0, Math.floor(Math.random() * cryptoUniverse.length) + 5)

    cryptosToScan.forEach(crypto => {
      // Only include cryptos with signals (simulate filtering)
      if (Math.random() > 0.3) { // 70% chance to have signals
        const result = generateScanResult(crypto)
        if (result.signals.length > 0 && result.score > 30) { // Minimum quality threshold
          newResults.push(result)
        }
      }
    })

    // Sort by score initially
    newResults.sort((a, b) => b.score - a.score)

    setResults(newResults.slice(0, maxResults))
    setScanCount(prev => prev + 1)
    setLastScanTime(new Date())
  }, [isScanning, generateScanResult, maxResults])

  // Auto-refresh scanning
  useEffect(() => {
    if (!autoRefresh || !isScanning) return

    const interval = setInterval(performScan, refreshInterval)
    
    // Perform initial scan
    performScan()

    return () => clearInterval(interval)
  }, [autoRefresh, isScanning, refreshInterval, performScan])

  // Filter and sort results
  const filteredAndSortedResults = results
    .filter(result => {
      // Search filter
      if (searchQuery && !result.symbol.toLowerCase().includes(searchQuery.toLowerCase()) &&
          !result.name.toLowerCase().includes(searchQuery.toLowerCase())) {
        return false
      }
      
      // Sector filter
      if (sectorFilter !== "all" && result.sector !== sectorFilter) {
        return false
      }
      
      // Signal filter
      if (signalFilter !== "all") {
        const hasSignalType = result.signals.some(signal => signal.type === signalFilter)
        if (!hasSignalType) return false
      }
      
      return true
    })
    .sort((a, b) => {
      let aValue: number
      let bValue: number
      
      switch (sortBy) {
        case "change":
          aValue = a.changePercent
          bValue = b.changePercent
          break
        case "volume":
          aValue = a.volumeRatio
          bValue = b.volumeRatio
          break
        case "signals":
          aValue = a.signals.length
          bValue = b.signals.length
          break
        default:
          aValue = a.score
          bValue = b.score
      }
      
      return sortOrder === "desc" ? bValue - aValue : aValue - bValue
    })

  const getSignalIcon = (signal: ScannerSignal) => {
    switch (signal.type) {
      case "pattern":
        return <TrendingUp className="h-3 w-3" />
      case "technical":
        return <Activity className="h-3 w-3" />
      case "volume":
        return <Volume2 className="h-3 w-3" />
      case "price":
        return <Target className="h-3 w-3" />
      case "news":
        return <AlertTriangle className="h-3 w-3" />
      default:
        return <Activity className="h-3 w-3" />
    }
  }

  const getSignalColor = (signal: ScannerSignal) => {
    if (signal.direction === "bullish") return "text-green-600"
    if (signal.direction === "bearish") return "text-red-600"
    return "text-gray-600"
  }

  const getScoreColor = (score: number) => {
    if (score >= 80) return "text-green-600"
    if (score >= 60) return "text-yellow-600"
    return "text-red-600"
  }

  const sectors = [...new Set(cryptoUniverse.map(c => c.sector))]

  return (
    <div className="space-y-4">
      {/* Scanner Controls */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Market Scanner</CardTitle>
              <CardDescription>
                Real-time opportunities across {cryptoUniverse.length} crypto pairs • 
                Last scan: {lastScanTime ? format(lastScanTime, "HH:mm:ss") : "Never"} • 
                {filteredAndSortedResults.length} results
              </CardDescription>
            </div>
            <div className="flex items-center gap-2">
              <Badge variant={isScanning ? "default" : "secondary"}>
                Scan #{scanCount}
              </Badge>
              <Button
                variant={isScanning ? "destructive" : "default"}
                size="sm"
                onClick={() => setIsScanning(!isScanning)}
              >
                {isScanning ? <Pause className="mr-2 h-4 w-4" /> : <Play className="mr-2 h-4 w-4" />}
                {isScanning ? "Pause" : "Start"}
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="flex gap-2">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                type="search"
                placeholder="Search symbols or names..."
                className="pl-10"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>
            
            <Select value={sectorFilter} onValueChange={setSectorFilter}>
              <SelectTrigger className="w-48">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Sectors</SelectItem>
                {sectors.map(sector => (
                  <SelectItem key={sector} value={sector}>
                    {sector}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            
            <Select value={signalFilter} onValueChange={setSignalFilter}>
              <SelectTrigger className="w-32">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Signals</SelectItem>
                <SelectItem value="pattern">Patterns</SelectItem>
                <SelectItem value="technical">Technical</SelectItem>
                <SelectItem value="volume">Volume</SelectItem>
                <SelectItem value="price">Price</SelectItem>
                <SelectItem value="news">News</SelectItem>
              </SelectContent>
            </Select>
            
            <Select value={sortBy} onValueChange={(value: any) => setSortBy(value)}>
              <SelectTrigger className="w-32">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="score">Score</SelectItem>
                <SelectItem value="change">Change</SelectItem>
                <SelectItem value="volume">Volume</SelectItem>
                <SelectItem value="signals">Signals</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {/* Scanner Results */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle>Scan Results</CardTitle>
          <CardDescription>
            Live market opportunities sorted by {sortBy}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {filteredAndSortedResults.length === 0 ? (
            <div className="text-center py-8">
              <Search className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
              <h3 className="text-lg font-medium mb-2">No Results Found</h3>
              <p className="text-sm text-muted-foreground">
                {isScanning ? "Scanning for opportunities..." : "Start scanning to find market opportunities"}
              </p>
            </div>
          ) : (
            <ScrollArea className="h-96">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Symbol</TableHead>
                    <TableHead>Price</TableHead>
                    <TableHead>Change</TableHead>
                    <TableHead>Volume</TableHead>
                    <TableHead>Signals</TableHead>
                    <TableHead>Score</TableHead>
                    <TableHead>Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {filteredAndSortedResults.map((result) => (
                    <TableRow key={result.id} className="cursor-pointer hover:bg-accent">
                      <TableCell>
                        <div>
                          <p className="font-medium">{result.symbol}</p>
                          <p className="text-xs text-muted-foreground">{result.name}</p>
                        </div>
                      </TableCell>
                      <TableCell>
                        <div>
                          <p className="font-medium">${result.price.toFixed(2)}</p>
                          <Badge variant="outline" className="text-xs">
                            {result.sector}
                          </Badge>
                        </div>
                      </TableCell>
                      <TableCell>
                        <div className={cn(
                          "font-medium",
                          result.changePercent >= 0 ? "text-green-600" : "text-red-600"
                        )}>
                          {result.changePercent >= 0 ? "+" : ""}
                          {result.changePercent.toFixed(2)}%
                        </div>
                        <p className="text-xs text-muted-foreground">
                          ${result.change.toFixed(2)}
                        </p>
                      </TableCell>
                      <TableCell>
                        <div>
                          <p className="font-medium">{(result.volume / 1000000).toFixed(1)}M</p>
                          <p className="text-xs text-muted-foreground">
                            {result.volumeRatio.toFixed(1)}x avg
                          </p>
                        </div>
                      </TableCell>
                      <TableCell>
                        <div className="flex flex-wrap gap-1">
                          {result.signals.slice(0, 3).map((signal, index) => (
                            <Badge
                              key={index}
                              variant="outline"
                              className={cn("text-xs", getSignalColor(signal))}
                            >
                              <div className="flex items-center gap-1">
                                {getSignalIcon(signal)}
                                {signal.name}
                              </div>
                            </Badge>
                          ))}
                          {result.signals.length > 3 && (
                            <Badge variant="outline" className="text-xs">
                              +{result.signals.length - 3}
                            </Badge>
                          )}
                        </div>
                      </TableCell>
                      <TableCell>
                        <div className={cn("text-lg font-bold", getScoreColor(result.score))}>
                          {result.score.toFixed(0)}
                        </div>
                      </TableCell>
                      <TableCell>
                        <div className="flex gap-1">
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-8 w-8"
                            onClick={() => onResultClick?.(result)}
                          >
                            <Eye className="h-4 w-4" />
                          </Button>
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-8 w-8"
                            onClick={() => onAddToWatchlist?.(result.symbol)}
                          >
                            <Star className="h-4 w-4" />
                          </Button>
                        </div>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </ScrollArea>
          )}
        </CardContent>
      </Card>
    </div>
  )
}