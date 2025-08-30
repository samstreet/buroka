"use client"

import { useState } from "react"
import { ProtectedRoute } from "@/components/auth/protected-route"
import { DashboardLayout } from "@/components/layout/dashboard-layout"
import { PriceChart } from "@/components/charts/price-chart"
import { VolumeChart } from "@/components/charts/volume-chart"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { 
  TrendingUp, 
  TrendingDown, 
  Activity, 
  AlertCircle,
  Search,
  Filter,
  Download
} from "lucide-react"
import { ErrorBoundary } from "@/components/error-boundary"

const watchlist = [
  { symbol: "AAPL", name: "Apple Inc.", price: 178.32, change: 2.45, changePercent: 1.39 },
  { symbol: "GOOGL", name: "Alphabet Inc.", price: 138.21, change: -1.23, changePercent: -0.88 },
  { symbol: "MSFT", name: "Microsoft Corp.", price: 378.91, change: 5.67, changePercent: 1.52 },
  { symbol: "AMZN", name: "Amazon.com Inc.", price: 127.43, change: -0.89, changePercent: -0.69 },
  { symbol: "TSLA", name: "Tesla Inc.", price: 243.56, change: 8.92, changePercent: 3.80 },
]

const recentPatterns = [
  { id: "1", symbol: "AAPL", pattern: "Head & Shoulders", confidence: 0.85, time: "2 hours ago", direction: "bearish" },
  { id: "2", symbol: "GOOGL", pattern: "Triangle Breakout", confidence: 0.92, time: "4 hours ago", direction: "bullish" },
  { id: "3", symbol: "MSFT", pattern: "Double Bottom", confidence: 0.78, time: "6 hours ago", direction: "bullish" },
  { id: "4", symbol: "TSLA", pattern: "Flag Pattern", confidence: 0.88, time: "8 hours ago", direction: "bullish" },
]

export default function AnalysisPage() {
  const [selectedSymbol, setSelectedSymbol] = useState("AAPL")
  const [searchQuery, setSearchQuery] = useState("")

  return (
    <ProtectedRoute>
      <ErrorBoundary>
        <DashboardLayout>
          <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-3xl font-bold">Market Analysis</h1>
                <p className="text-muted-foreground">Real-time charts and pattern detection</p>
              </div>
              <div className="flex items-center gap-2">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                  <Input
                    type="search"
                    placeholder="Search symbols..."
                    className="pl-10 w-64"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                  />
                </div>
                <Button variant="outline">
                  <Filter className="mr-2 h-4 w-4" />
                  Filters
                </Button>
                <Button variant="outline">
                  <Download className="mr-2 h-4 w-4" />
                  Export
                </Button>
              </div>
            </div>

            {/* Main Content */}
            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
              {/* Left Sidebar - Watchlist */}
              <div className="lg:col-span-1 space-y-4">
                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-base">Watchlist</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2">
                    {watchlist.map((stock) => (
                      <button
                        key={stock.symbol}
                        onClick={() => setSelectedSymbol(stock.symbol)}
                        className={`w-full text-left p-3 rounded-lg transition-colors ${
                          selectedSymbol === stock.symbol
                            ? "bg-accent"
                            : "hover:bg-accent/50"
                        }`}
                      >
                        <div className="flex items-center justify-between">
                          <div>
                            <p className="font-medium">{stock.symbol}</p>
                            <p className="text-xs text-muted-foreground">{stock.name}</p>
                          </div>
                          <div className="text-right">
                            <p className="font-medium">${stock.price}</p>
                            <p className={`text-xs ${
                              stock.change >= 0 ? "text-green-600" : "text-red-600"
                            }`}>
                              {stock.change >= 0 ? "+" : ""}{stock.changePercent}%
                            </p>
                          </div>
                        </div>
                      </button>
                    ))}
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-base">Recent Patterns</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    {recentPatterns.map((pattern) => (
                      <div key={pattern.id} className="space-y-2">
                        <div className="flex items-center justify-between">
                          <Badge variant={pattern.direction === "bullish" ? "default" : "destructive"}>
                            {pattern.symbol}
                          </Badge>
                          <span className="text-xs text-muted-foreground">{pattern.time}</span>
                        </div>
                        <p className="text-sm font-medium">{pattern.pattern}</p>
                        <div className="flex items-center gap-2">
                          {pattern.direction === "bullish" ? (
                            <TrendingUp className="h-3 w-3 text-green-600" />
                          ) : (
                            <TrendingDown className="h-3 w-3 text-red-600" />
                          )}
                          <span className="text-xs">
                            Confidence: {Math.round(pattern.confidence * 100)}%
                          </span>
                        </div>
                      </div>
                    ))}
                  </CardContent>
                </Card>
              </div>

              {/* Right Side - Charts */}
              <div className="lg:col-span-3 space-y-4">
                <Tabs defaultValue="chart" className="w-full">
                  <TabsList className="grid w-full grid-cols-4">
                    <TabsTrigger value="chart">Chart</TabsTrigger>
                    <TabsTrigger value="patterns">Patterns</TabsTrigger>
                    <TabsTrigger value="indicators">Indicators</TabsTrigger>
                    <TabsTrigger value="news">News</TabsTrigger>
                  </TabsList>

                  <TabsContent value="chart" className="space-y-4">
                    <PriceChart 
                      symbol={selectedSymbol}
                      showVolume={true}
                      showIndicators={true}
                    />
                    <VolumeChart />
                  </TabsContent>

                  <TabsContent value="patterns" className="space-y-4">
                    <Card>
                      <CardHeader>
                        <CardTitle>Detected Patterns</CardTitle>
                        <CardDescription>
                          AI-detected chart patterns for {selectedSymbol}
                        </CardDescription>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-4">
                          {recentPatterns
                            .filter(p => p.symbol === selectedSymbol)
                            .map((pattern) => (
                              <div key={pattern.id} className="flex items-center justify-between p-4 border rounded-lg">
                                <div className="flex items-center gap-4">
                                  {pattern.direction === "bullish" ? (
                                    <TrendingUp className="h-8 w-8 text-green-600" />
                                  ) : (
                                    <TrendingDown className="h-8 w-8 text-red-600" />
                                  )}
                                  <div>
                                    <p className="font-medium">{pattern.pattern}</p>
                                    <p className="text-sm text-muted-foreground">
                                      Detected {pattern.time}
                                    </p>
                                  </div>
                                </div>
                                <div className="text-right">
                                  <Badge variant={pattern.confidence > 0.8 ? "default" : "secondary"}>
                                    {Math.round(pattern.confidence * 100)}% confidence
                                  </Badge>
                                  <p className="text-sm text-muted-foreground mt-1">
                                    {pattern.direction} signal
                                  </p>
                                </div>
                              </div>
                            ))}
                        </div>
                      </CardContent>
                    </Card>
                  </TabsContent>

                  <TabsContent value="indicators" className="space-y-4">
                    <Card>
                      <CardHeader>
                        <CardTitle>Technical Indicators</CardTitle>
                        <CardDescription>
                          Key indicators for {selectedSymbol}
                        </CardDescription>
                      </CardHeader>
                      <CardContent>
                        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                          {[
                            { name: "RSI", value: 65.4, status: "neutral" },
                            { name: "MACD", value: 2.3, status: "bullish" },
                            { name: "Stochastic", value: 78.2, status: "overbought" },
                            { name: "Moving Avg", value: 175.5, status: "bullish" },
                            { name: "Volume", value: "12.3M", status: "high" },
                            { name: "Volatility", value: "23.4%", status: "moderate" },
                          ].map((indicator) => (
                            <div key={indicator.name} className="space-y-2">
                              <p className="text-sm text-muted-foreground">{indicator.name}</p>
                              <p className="text-2xl font-bold">{indicator.value}</p>
                              <Badge variant="outline">{indicator.status}</Badge>
                            </div>
                          ))}
                        </div>
                      </CardContent>
                    </Card>
                  </TabsContent>

                  <TabsContent value="news" className="space-y-4">
                    <Card>
                      <CardHeader>
                        <CardTitle>Latest News</CardTitle>
                        <CardDescription>
                          Recent news and updates for {selectedSymbol}
                        </CardDescription>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-4">
                          {[
                            {
                              title: "Apple Reports Record Q4 Earnings",
                              source: "Reuters",
                              time: "2 hours ago",
                              sentiment: "positive",
                            },
                            {
                              title: "New iPhone 15 Sales Exceed Expectations",
                              source: "Bloomberg",
                              time: "5 hours ago",
                              sentiment: "positive",
                            },
                            {
                              title: "Apple Faces Regulatory Challenges in EU",
                              source: "WSJ",
                              time: "1 day ago",
                              sentiment: "negative",
                            },
                          ].map((news, index) => (
                            <div key={index} className="flex items-start gap-4 p-4 border rounded-lg">
                              <AlertCircle className="h-5 w-5 text-muted-foreground mt-0.5" />
                              <div className="flex-1">
                                <p className="font-medium">{news.title}</p>
                                <div className="flex items-center gap-2 mt-1">
                                  <span className="text-xs text-muted-foreground">{news.source}</span>
                                  <span className="text-xs text-muted-foreground">•</span>
                                  <span className="text-xs text-muted-foreground">{news.time}</span>
                                  <Badge 
                                    variant={news.sentiment === "positive" ? "default" : "destructive"}
                                    className="text-xs"
                                  >
                                    {news.sentiment}
                                  </Badge>
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </CardContent>
                    </Card>
                  </TabsContent>
                </Tabs>
              </div>
            </div>
          </div>
        </DashboardLayout>
      </ErrorBoundary>
    </ProtectedRoute>
  )
}