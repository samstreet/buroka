"use client"

import { useState } from "react"
import { Pattern } from "./pattern-list"
import { PriceChart } from "@/components/charts/price-chart"
import { PatternOverlay, PatternData } from "@/components/charts/pattern-overlay"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  TrendingUp,
  TrendingDown,
  Clock,
  Activity,
  DollarSign,
  BarChart3,
  AlertCircle,
  BookOpen,
  Settings,
  Share2,
  Minus,
} from "lucide-react"
import { format } from "date-fns"

interface PatternDetailProps {
  pattern: Pattern
  historicalData?: any[]
  similarPatterns?: Pattern[]
  onBacktest?: () => void
  onCreateAlert?: () => void
  onShare?: () => void
}

export function PatternDetail({
  pattern,
  historicalData,
  similarPatterns = [],
  onBacktest,
  onCreateAlert,
  onShare,
}: PatternDetailProps) {
  const [selectedTab, setSelectedTab] = useState("overview")

  const getDirectionIcon = (direction: string) => {
    switch (direction) {
      case "bullish":
        return <TrendingUp className="h-5 w-5 text-green-600" />
      case "bearish":
        return <TrendingDown className="h-5 w-5 text-red-600" />
      default:
        return <Minus className="h-5 w-5 text-gray-600" />
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case "active":
        return "text-green-600"
      case "completed":
        return "text-blue-600"
      case "failed":
        return "text-red-600"
      default:
        return "text-gray-600"
    }
  }

  // Convert pattern to PatternData for overlay
  const patternData: PatternData = {
    id: pattern.id,
    type: pattern.type as any,
    startTime: Math.floor(pattern.startTime.getTime() / 1000) as any,
    endTime: Math.floor(pattern.endTime.getTime() / 1000) as any,
    startPrice: pattern.priceAtDetection,
    endPrice: pattern.currentPrice,
    confidence: pattern.confidence,
    direction: pattern.direction,
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div className="space-y-1">
          <div className="flex items-center gap-3">
            <h2 className="text-2xl font-bold">{pattern.symbol}</h2>
            <Badge variant="outline" className="text-lg">
              {pattern.type}
            </Badge>
            <div className="flex items-center gap-1">
              {getDirectionIcon(pattern.direction)}
              <span className="capitalize font-medium">{pattern.direction}</span>
            </div>
          </div>
          <p className="text-muted-foreground">
            Detected on {format(pattern.startTime, "MMM dd, yyyy HH:mm")}
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" onClick={onShare}>
            <Share2 className="mr-2 h-4 w-4" />
            Share
          </Button>
          <Button variant="outline" onClick={onCreateAlert}>
            <AlertCircle className="mr-2 h-4 w-4" />
            Set Alert
          </Button>
          <Button onClick={onBacktest}>
            <Activity className="mr-2 h-4 w-4" />
            Backtest
          </Button>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Confidence</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <p className="text-2xl font-bold">
                {Math.round(pattern.confidence * 100)}%
              </p>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-blue-600 h-2 rounded-full"
                  style={{ width: `${pattern.confidence * 100}%` }}
                />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Price Change</CardDescription>
          </CardHeader>
          <CardContent>
            <p className={`text-2xl font-bold ${
              pattern.priceChangePercent >= 0 ? "text-green-600" : "text-red-600"
            }`}>
              {pattern.priceChangePercent >= 0 ? "+" : ""}
              {pattern.priceChangePercent.toFixed(2)}%
            </p>
            <p className="text-sm text-muted-foreground mt-1">
              ${pattern.priceAtDetection.toFixed(2)} → ${pattern.currentPrice.toFixed(2)}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Volume</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">
              {(pattern.volume / 1000000).toFixed(2)}M
            </p>
            <p className="text-sm text-muted-foreground mt-1">
              24h Volume
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Status</CardDescription>
          </CardHeader>
          <CardContent>
            <p className={`text-2xl font-bold capitalize ${getStatusColor(pattern.status)}`}>
              {pattern.status}
            </p>
            <p className="text-sm text-muted-foreground mt-1">
              {pattern.timeframe} timeframe
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Tabs */}
      <Tabs value={selectedTab} onValueChange={setSelectedTab}>
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="technical">Technical</TabsTrigger>
          <TabsTrigger value="performance">Performance</TabsTrigger>
          <TabsTrigger value="similar">Similar</TabsTrigger>
          <TabsTrigger value="backtest">Backtest</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          {/* Chart with Pattern Overlay */}
          <Card>
            <CardHeader>
              <CardTitle>Pattern Visualization</CardTitle>
              <CardDescription>
                Interactive chart showing the detected pattern
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-[500px]">
                <PriceChart 
                  symbol={pattern.symbol}
                  data={historicalData}
                  height={450}
                  showVolume={false}
                  showIndicators={true}
                />
                {/* Pattern overlay would be added here */}
              </div>
            </CardContent>
          </Card>

          {/* Pattern Description */}
          <Card>
            <CardHeader>
              <CardTitle>Pattern Description</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <h4 className="font-medium mb-2">What is a {pattern.type}?</h4>
                <p className="text-sm text-muted-foreground">
                  A {pattern.type} pattern is a technical analysis formation that typically 
                  indicates a {pattern.direction} market sentiment. This pattern has been 
                  identified with {Math.round(pattern.confidence * 100)}% confidence based on 
                  historical price action and volume data.
                </p>
              </div>
              <div>
                <h4 className="font-medium mb-2">Key Characteristics</h4>
                <ul className="list-disc list-inside text-sm text-muted-foreground space-y-1">
                  <li>Formation period: {pattern.timeframe}</li>
                  <li>Direction: {pattern.direction}</li>
                  <li>Entry price: ${pattern.priceAtDetection.toFixed(2)}</li>
                  <li>Current price: ${pattern.currentPrice.toFixed(2)}</li>
                </ul>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="technical" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Technical Analysis</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                <div>
                  <p className="text-sm text-muted-foreground">Support Level</p>
                  <p className="text-lg font-medium">
                    ${(pattern.priceAtDetection * 0.98).toFixed(2)}
                  </p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Resistance Level</p>
                  <p className="text-lg font-medium">
                    ${(pattern.priceAtDetection * 1.02).toFixed(2)}
                  </p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Target Price</p>
                  <p className="text-lg font-medium">
                    ${(pattern.priceAtDetection * (pattern.direction === "bullish" ? 1.05 : 0.95)).toFixed(2)}
                  </p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Stop Loss</p>
                  <p className="text-lg font-medium">
                    ${(pattern.priceAtDetection * (pattern.direction === "bullish" ? 0.97 : 1.03)).toFixed(2)}
                  </p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Risk/Reward</p>
                  <p className="text-lg font-medium">1:2.5</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Volatility</p>
                  <p className="text-lg font-medium">23.4%</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Technical Indicators</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {[
                  { name: "RSI", value: 65.4, status: "Overbought", color: "text-orange-600" },
                  { name: "MACD", value: "Bullish Cross", status: "Buy Signal", color: "text-green-600" },
                  { name: "Moving Average", value: "Above 50 MA", status: "Bullish", color: "text-green-600" },
                  { name: "Bollinger Bands", value: "Near Upper", status: "Strong", color: "text-blue-600" },
                ].map((indicator) => (
                  <div key={indicator.name} className="flex items-center justify-between p-3 border rounded-lg">
                    <div>
                      <p className="font-medium">{indicator.name}</p>
                      <p className="text-sm text-muted-foreground">{indicator.value}</p>
                    </div>
                    <Badge variant="outline" className={indicator.color}>
                      {indicator.status}
                    </Badge>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="performance" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Historical Performance</CardTitle>
              <CardDescription>
                How this pattern has performed historically
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div>
                  <p className="text-sm text-muted-foreground">Success Rate</p>
                  <p className="text-2xl font-bold text-green-600">73%</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Avg. Return</p>
                  <p className="text-2xl font-bold">+12.5%</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Occurrences</p>
                  <p className="text-2xl font-bold">247</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Avg. Duration</p>
                  <p className="text-2xl font-bold">5.2 days</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Performance Distribution</CardTitle>
            </CardHeader>
            <CardContent>
              {/* Performance chart would go here */}
              <div className="h-64 flex items-center justify-center text-muted-foreground">
                Performance distribution chart
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="similar" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Similar Patterns</CardTitle>
              <CardDescription>
                Other patterns with similar characteristics
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {similarPatterns.slice(0, 5).map((similar) => (
                  <div key={similar.id} className="flex items-center justify-between p-3 border rounded-lg">
                    <div className="flex items-center gap-3">
                      <div>
                        <p className="font-medium">{similar.symbol}</p>
                        <p className="text-sm text-muted-foreground">
                          {similar.type} • {similar.timeframe}
                        </p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className={`font-medium ${
                        similar.priceChangePercent >= 0 ? "text-green-600" : "text-red-600"
                      }`}>
                        {similar.priceChangePercent >= 0 ? "+" : ""}
                        {similar.priceChangePercent.toFixed(2)}%
                      </p>
                      <p className="text-sm text-muted-foreground">
                        {Math.round(similar.confidence * 100)}% confidence
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="backtest" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Backtesting Configuration</CardTitle>
              <CardDescription>
                Test this pattern against historical data
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="text-center py-8">
                  <Activity className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                  <h3 className="text-lg font-medium mb-2">Run Backtest</h3>
                  <p className="text-sm text-muted-foreground mb-4">
                    Test this pattern's performance against historical data
                  </p>
                  <Button onClick={onBacktest}>
                    <Activity className="mr-2 h-4 w-4" />
                    Start Backtesting
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}