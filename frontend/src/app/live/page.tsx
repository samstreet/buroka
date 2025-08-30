"use client"

import { useState } from "react"
import { ProtectedRoute } from "@/components/auth/protected-route"
import { DashboardLayout } from "@/components/layout/dashboard-layout"
import { MarketFeed } from "@/components/realtime/market-feed"
import { PatternAlertsLive } from "@/components/realtime/pattern-alerts-live"
import { RealtimeChart } from "@/components/realtime/realtime-chart"
import { MarketScanner } from "@/components/realtime/market-scanner"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
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
  Activity,
  BarChart3,
  Bell,
  Search,
  TrendingUp,
  Wifi,
  Eye,
  Star,
} from "lucide-react"

export default function LivePage() {
  const [selectedSymbol, setSelectedSymbol] = useState("AAPL")
  const [selectedTab, setSelectedTab] = useState("overview")

  const watchlistSymbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META"]

  const handlePatternDetected = (pattern: any) => {
    console.log("Pattern detected:", pattern)
    // Could trigger notifications, alerts, etc.
  }

  const handleTickUpdate = (tick: any) => {
    console.log("Price update:", tick)
    // Could update other components, store in state, etc.
  }

  const handleScannerResultClick = (result: any) => {
    setSelectedSymbol(result.symbol)
    setSelectedTab("charts")
  }

  const handleAddToWatchlist = (symbol: string) => {
    console.log("Add to watchlist:", symbol)
    // Implement watchlist functionality
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
                  <Wifi className="h-8 w-8 text-green-600" />
                  Live Market Data
                </h1>
                <p className="text-muted-foreground">
                  Real-time market monitoring, pattern detection, and opportunity scanning
                </p>
              </div>
              <div className="flex items-center gap-2">
                <Select value={selectedSymbol} onValueChange={setSelectedSymbol}>
                  <SelectTrigger className="w-32">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {watchlistSymbols.map(symbol => (
                      <SelectItem key={symbol} value={symbol}>
                        {symbol}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>

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
                <TabsTrigger value="alerts" className="flex items-center gap-2">
                  <Bell className="h-4 w-4" />
                  Alerts
                </TabsTrigger>
                <TabsTrigger value="scanner" className="flex items-center gap-2">
                  <Search className="h-4 w-4" />
                  Scanner
                </TabsTrigger>
              </TabsList>

              <TabsContent value="overview" className="space-y-6">
                {/* Market Feed and Alerts Side by Side */}
                <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
                  <MarketFeed
                    symbols={watchlistSymbols}
                    onTickUpdate={handleTickUpdate}
                    onPatternDetected={handlePatternDetected}
                  />
                  
                  <PatternAlertsLive
                    maxAlerts={20}
                    showNotifications={true}
                    onAlertTriggered={(alert) => console.log("Alert triggered:", alert)}
                  />
                </div>

                {/* Quick Charts */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <RealtimeChart
                    symbol={selectedSymbol}
                    height={300}
                    showVolume={true}
                    updateInterval={2000}
                  />
                  
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <TrendingUp className="h-5 w-5" />
                        Market Movers
                      </CardTitle>
                      <CardDescription>
                        Top performers and biggest losers
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-3">
                        {[
                          { symbol: "NVDA", change: 5.67, changePercent: 12.3 },
                          { symbol: "TSLA", change: -8.21, changePercent: -3.4 },
                          { symbol: "META", change: 15.43, changePercent: 4.8 },
                          { symbol: "GOOGL", change: -3.21, changePercent: -2.1 },
                        ].map((mover, index) => (
                          <div key={mover.symbol} className="flex items-center justify-between p-2 rounded-lg border">
                            <div className="flex items-center gap-2">
                              <span className="font-medium">{mover.symbol}</span>
                            </div>
                            <div className={`text-right font-medium ${
                              mover.changePercent >= 0 ? "text-green-600" : "text-red-600"
                            }`}>
                              <p>{mover.changePercent >= 0 ? "+" : ""}{mover.changePercent.toFixed(2)}%</p>
                              <p className="text-sm">{mover.change >= 0 ? "+" : ""}${mover.change.toFixed(2)}</p>
                            </div>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                </div>
              </TabsContent>

              <TabsContent value="charts" className="space-y-6">
                {/* Full-size chart */}
                <RealtimeChart
                  symbol={selectedSymbol}
                  height={600}
                  showVolume={true}
                  showPatterns={true}
                  updateInterval={1000}
                  maxDataPoints={500}
                />

                {/* Multiple mini charts */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {watchlistSymbols.slice(0, 6).map(symbol => (
                    <RealtimeChart
                      key={symbol}
                      symbol={symbol}
                      height={200}
                      showVolume={false}
                      updateInterval={3000}
                      maxDataPoints={100}
                    />
                  ))}
                </div>
              </TabsContent>

              <TabsContent value="alerts" className="space-y-6">
                <PatternAlertsLive
                  maxAlerts={100}
                  showNotifications={true}
                  autoAcknowledge={false}
                  onAlertTriggered={(alert) => {
                    console.log("New alert:", alert)
                    // Could integrate with notification system
                  }}
                  onAlertAction={(alertId, action) => {
                    console.log("Alert action:", alertId, action)
                    // Handle alert actions
                  }}
                />
              </TabsContent>

              <TabsContent value="scanner" className="space-y-6">
                <MarketScanner
                  maxResults={50}
                  autoRefresh={true}
                  refreshInterval={10000}
                  onResultClick={handleScannerResultClick}
                  onAddToWatchlist={handleAddToWatchlist}
                />
              </TabsContent>
            </Tabs>
          </div>
        </DashboardLayout>
      </ErrorBoundary>
    </ProtectedRoute>
  )
}