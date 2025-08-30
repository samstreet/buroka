"use client"

import { useState, useEffect } from "react"
import { ProtectedRoute } from "@/components/auth/protected-route"
import { DashboardLayout } from "@/components/layout/dashboard-layout"
import { CryptoMarketFeed } from "@/components/crypto/crypto-market-feed"
import { CryptoRealtimeChart } from "@/components/crypto/crypto-realtime-chart"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { ErrorBoundary } from "@/components/error-boundary"
import {
  Bitcoin,
  BarChart3,
  Bell,
  Search,
  TrendingUp,
  TrendingDown,
  Wifi,
  Activity,
  DollarSign,
  Volume2,
} from "lucide-react"
import { cryptoAPI, CryptoTicker, MarketOverview, DEFAULT_CRYPTO_PAIRS } from "@/lib/crypto-api"

export default function CryptoPage() {
  const [selectedSymbol, setSelectedSymbol] = useState("BTCUSDT")
  const [selectedTab, setSelectedTab] = useState("overview")
  const [marketOverview, setMarketOverview] = useState<MarketOverview | null>(null)
  const [topMovers, setTopMovers] = useState<{ gainers: CryptoTicker[], losers: CryptoTicker[] }>({ gainers: [], losers: [] })
  const [loading, setLoading] = useState(true)

  // Load market overview data
  useEffect(() => {
    const loadMarketData = async () => {
      try {
        setLoading(true)
        
        // Load market overview
        const overview = await cryptoAPI.getMarketOverview()
        setMarketOverview(overview)
        
        // Load top gainers/losers ticker data
        const gainersLosers = await cryptoAPI.getGainersLosers(5)
        if (gainersLosers.gainers.length > 0 || gainersLosers.losers.length > 0) {
          const allSymbols = [...gainersLosers.gainers, ...gainersLosers.losers]
          const tickers = await cryptoAPI.getTickerData(allSymbols)
          
          const gainers = tickers.filter(t => gainersLosers.gainers.includes(t.symbol))
          const losers = tickers.filter(t => gainersLosers.losers.includes(t.symbol))
          
          setTopMovers({ gainers, losers })
        }
        
      } catch (error) {
        console.error('Error loading market data:', error)
      } finally {
        setLoading(false)
      }
    }

    loadMarketData()
  }, [])

  const handleTickUpdate = (ticker: CryptoTicker) => {
    // Update top movers if needed
    setTopMovers(prev => ({
      gainers: prev.gainers.map(t => t.symbol === ticker.symbol ? ticker : t),
      losers: prev.losers.map(t => t.symbol === ticker.symbol ? ticker : t)
    }))
  }

  const handleSymbolSelect = (symbol: string) => {
    setSelectedSymbol(symbol)
    setSelectedTab("charts")
  }

  if (loading) {
    return (
      <ProtectedRoute>
        <DashboardLayout>
          <div className="flex items-center justify-center min-h-[400px]">
            <div className="text-center">
              <Bitcoin className="h-12 w-12 animate-spin text-orange-500 mx-auto mb-4" />
              <p>Loading cryptocurrency data...</p>
            </div>
          </div>
        </DashboardLayout>
      </ProtectedRoute>
    )
  }

  return (
    <ProtectedRoute>
      <ErrorBoundary>
        <DashboardLayout>
          <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-3xl font-bold flex items-center gap-2">
                  <Bitcoin className="h-8 w-8 text-orange-500" />
                  Cryptocurrency Markets
                </h1>
                <p className="text-muted-foreground">
                  Real-time crypto trading data powered by Binance
                </p>
              </div>
              <div className="flex items-center gap-2">
                <Badge variant="outline" className="text-green-600">
                  <Wifi className="h-4 w-4 mr-1" />
                  Binance Live
                </Badge>
                <Select value={selectedSymbol} onValueChange={setSelectedSymbol}>
                  <SelectTrigger className="w-40">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {DEFAULT_CRYPTO_PAIRS.map(symbol => (
                      <SelectItem key={symbol} value={symbol}>
                        {cryptoAPI.formatSymbol(symbol)}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>

            {/* Market Overview Stats */}
            {marketOverview && (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <Card>
                  <CardHeader className="pb-2">
                    <CardDescription className="flex items-center gap-2">
                      <Bitcoin className="h-4 w-4" />
                      Active Pairs
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <p className="text-2xl font-bold">{marketOverview.total_pairs}</p>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="pb-2">
                    <CardDescription className="flex items-center gap-2">
                      <TrendingUp className="h-4 w-4" />
                      Top Gainer
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    {topMovers.gainers.length > 0 ? (
                      <div>
                        <p className="font-bold">
                          {cryptoAPI.formatSymbol(topMovers.gainers[0].symbol)}
                        </p>
                        <p className="text-green-600 text-sm">
                          +{topMovers.gainers[0].price_change_percent.toFixed(2)}%
                        </p>
                      </div>
                    ) : (
                      <p className="text-2xl font-bold">-</p>
                    )}
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="pb-2">
                    <CardDescription className="flex items-center gap-2">
                      <TrendingDown className="h-4 w-4" />
                      Top Loser
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    {topMovers.losers.length > 0 ? (
                      <div>
                        <p className="font-bold">
                          {cryptoAPI.formatSymbol(topMovers.losers[0].symbol)}
                        </p>
                        <p className="text-red-600 text-sm">
                          {topMovers.losers[0].price_change_percent.toFixed(2)}%
                        </p>
                      </div>
                    ) : (
                      <p className="text-2xl font-bold">-</p>
                    )}
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="pb-2">
                    <CardDescription className="flex items-center gap-2">
                      <Activity className="h-4 w-4" />
                      Intervals
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <p className="text-2xl font-bold">{marketOverview.supported_intervals.length}</p>
                  </CardContent>
                </Card>
              </div>
            )}

            {/* Tabs */}
            <Tabs value={selectedTab} onValueChange={setSelectedTab}>
              <TabsList className="grid w-full grid-cols-4">
                <TabsTrigger value="overview" className="flex items-center gap-2">
                  <Activity className="h-4 w-4" />
                  Overview
                </TabsTrigger>
                <TabsTrigger value="charts" className="flex items-center gap-2">
                  <BarChart3 className="h-4 w-4" />
                  Charts
                </TabsTrigger>
                <TabsTrigger value="movers" className="flex items-center gap-2">
                  <TrendingUp className="h-4 w-4" />
                  Movers
                </TabsTrigger>
                <TabsTrigger value="watchlist" className="flex items-center gap-2">
                  <Search className="h-4 w-4" />
                  Watchlist
                </TabsTrigger>
              </TabsList>

              <TabsContent value="overview" className="space-y-6">
                {/* Market Feed */}
                <CryptoMarketFeed
                  symbols={DEFAULT_CRYPTO_PAIRS.slice(0, 10)}
                  onTickUpdate={handleTickUpdate}
                  onPatternDetected={(pattern) => console.log("Pattern detected:", pattern)}
                />
              </TabsContent>

              <TabsContent value="charts" className="space-y-6">
                {/* Main Chart */}
                <CryptoRealtimeChart
                  symbol={selectedSymbol}
                  height={600}
                  showVolume={true}
                  showPatterns={true}
                  interval="1m"
                  maxDataPoints={500}
                />

                {/* Multiple mini charts */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {DEFAULT_CRYPTO_PAIRS.slice(0, 6).map(symbol => (
                    <CryptoRealtimeChart
                      key={symbol}
                      symbol={symbol}
                      height={200}
                      showVolume={false}
                      interval="5m"
                      maxDataPoints={100}
                    />
                  ))}
                </div>
              </TabsContent>

              <TabsContent value="movers" className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Top Gainers */}
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <TrendingUp className="h-5 w-5 text-green-600" />
                        Top Gainers
                      </CardTitle>
                      <CardDescription>
                        Biggest 24hr percentage gainers
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-3">
                        {topMovers.gainers.map((ticker, index) => (
                          <div 
                            key={ticker.symbol} 
                            className="flex items-center justify-between p-3 rounded-lg border cursor-pointer hover:bg-accent"
                            onClick={() => handleSymbolSelect(ticker.symbol)}
                          >
                            <div className="flex items-center gap-3">
                              <span className="text-lg font-bold text-muted-foreground">
                                #{index + 1}
                              </span>
                              <div>
                                <p className="font-medium">{cryptoAPI.formatSymbol(ticker.symbol)}</p>
                                <p className="text-sm text-muted-foreground">
                                  ${cryptoAPI.formatPrice(ticker.price, ticker.symbol)}
                                </p>
                              </div>
                            </div>
                            <div className="text-right">
                              <p className="font-bold text-green-600">
                                +{ticker.price_change_percent.toFixed(2)}%
                              </p>
                              <p className="text-sm text-muted-foreground">
                                Vol: {cryptoAPI.formatVolume(ticker.volume)}
                              </p>
                            </div>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>

                  {/* Top Losers */}
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <TrendingDown className="h-5 w-5 text-red-600" />
                        Top Losers
                      </CardTitle>
                      <CardDescription>
                        Biggest 24hr percentage losers
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-3">
                        {topMovers.losers.map((ticker, index) => (
                          <div 
                            key={ticker.symbol} 
                            className="flex items-center justify-between p-3 rounded-lg border cursor-pointer hover:bg-accent"
                            onClick={() => handleSymbolSelect(ticker.symbol)}
                          >
                            <div className="flex items-center gap-3">
                              <span className="text-lg font-bold text-muted-foreground">
                                #{index + 1}
                              </span>
                              <div>
                                <p className="font-medium">{cryptoAPI.formatSymbol(ticker.symbol)}</p>
                                <p className="text-sm text-muted-foreground">
                                  ${cryptoAPI.formatPrice(ticker.price, ticker.symbol)}
                                </p>
                              </div>
                            </div>
                            <div className="text-right">
                              <p className="font-bold text-red-600">
                                {ticker.price_change_percent.toFixed(2)}%
                              </p>
                              <p className="text-sm text-muted-foreground">
                                Vol: {cryptoAPI.formatVolume(ticker.volume)}
                              </p>
                            </div>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                </div>
              </TabsContent>

              <TabsContent value="watchlist" className="space-y-6">
                {/* Watchlist Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                  {DEFAULT_CRYPTO_PAIRS.map((symbol) => (
                    <Card 
                      key={symbol}
                      className="cursor-pointer hover:shadow-lg transition-shadow"
                      onClick={() => handleSymbolSelect(symbol)}
                    >
                      <CardHeader className="pb-2">
                        <CardTitle className="text-base flex items-center justify-between">
                          <span>{cryptoAPI.formatSymbol(symbol)}</span>
                          <Bitcoin className="h-4 w-4 text-orange-500" />
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-1">
                          <p className="text-sm text-muted-foreground">Loading...</p>
                          <div className="h-16 bg-muted/30 rounded animate-pulse"></div>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </TabsContent>
            </Tabs>
          </div>
        </DashboardLayout>
      </ErrorBoundary>
    </ProtectedRoute>
  )
}