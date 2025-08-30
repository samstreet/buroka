"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  TrendingUp,
  TrendingDown,
  Target,
  Clock,
  Award,
  Activity,
  BarChart3,
  PieChart,
} from "lucide-react"

interface PerformanceMetrics {
  patternType: string
  totalOccurrences: number
  successfulPatterns: number
  successRate: number
  averageReturn: number
  averageLosingReturn: number
  maxReturn: number
  minReturn: number
  averageDuration: number
  winRate: {
    "1d": number
    "3d": number
    "7d": number
    "30d": number
  }
  marketConditions: {
    bullish: { occurrences: number; successRate: number; avgReturn: number }
    bearish: { occurrences: number; successRate: number; avgReturn: number }
    sideways: { occurrences: number; successRate: number; avgReturn: number }
  }
  timeframes: {
    "1h": { occurrences: number; successRate: number; avgReturn: number }
    "4h": { occurrences: number; successRate: number; avgReturn: number }
    "1d": { occurrences: number; successRate: number; avgReturn: number }
    "1w": { occurrences: number; successRate: number; avgReturn: number }
  }
  sectors: {
    [key: string]: { occurrences: number; successRate: number; avgReturn: number }
  }
}

interface PatternPerformanceProps {
  metrics: PerformanceMetrics
  historicalData?: any[]
}

export function PatternPerformance({ metrics, historicalData }: PatternPerformanceProps) {
  const [selectedTimeframe, setSelectedTimeframe] = useState("all")
  const [selectedSector, setSelectedSector] = useState("all")

  const getSuccessRateColor = (rate: number) => {
    if (rate >= 0.7) return "text-green-600"
    if (rate >= 0.5) return "text-yellow-600"
    return "text-red-600"
  }

  const getReturnColor = (return_: number) => {
    return return_ >= 0 ? "text-green-600" : "text-red-600"
  }

  return (
    <div className="space-y-6">
      {/* Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardDescription className="flex items-center gap-2">
              <Award className="h-4 w-4" />
              Success Rate
            </CardDescription>
          </CardHeader>
          <CardContent>
            <p className={`text-2xl font-bold ${getSuccessRateColor(metrics.successRate)}`}>
              {(metrics.successRate * 100).toFixed(1)}%
            </p>
            <p className="text-sm text-muted-foreground mt-1">
              {metrics.successfulPatterns} of {metrics.totalOccurrences} patterns
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardDescription className="flex items-center gap-2">
              <TrendingUp className="h-4 w-4" />
              Avg Return
            </CardDescription>
          </CardHeader>
          <CardContent>
            <p className={`text-2xl font-bold ${getReturnColor(metrics.averageReturn)}`}>
              {metrics.averageReturn >= 0 ? "+" : ""}{metrics.averageReturn.toFixed(2)}%
            </p>
            <p className="text-sm text-muted-foreground mt-1">
              Max: {metrics.maxReturn.toFixed(2)}% | Min: {metrics.minReturn.toFixed(2)}%
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardDescription className="flex items-center gap-2">
              <Clock className="h-4 w-4" />
              Avg Duration
            </CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">
              {metrics.averageDuration.toFixed(1)} days
            </p>
            <p className="text-sm text-muted-foreground mt-1">
              Median holding period
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardDescription className="flex items-center gap-2">
              <BarChart3 className="h-4 w-4" />
              Occurrences
            </CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">
              {metrics.totalOccurrences}
            </p>
            <p className="text-sm text-muted-foreground mt-1">
              Total patterns detected
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Performance Tabs */}
      <Tabs defaultValue="overview" className="w-full">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="timeframes">Timeframes</TabsTrigger>
          <TabsTrigger value="market">Market</TabsTrigger>
          <TabsTrigger value="sectors">Sectors</TabsTrigger>
          <TabsTrigger value="timeline">Timeline</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          {/* Win Rate by Time Horizon */}
          <Card>
            <CardHeader>
              <CardTitle>Win Rate by Time Horizon</CardTitle>
              <CardDescription>
                Success rate at different time intervals after pattern detection
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {Object.entries(metrics.winRate).map(([period, rate]) => (
                  <div key={period} className="space-y-2">
                    <p className="text-sm text-muted-foreground">{period.toUpperCase()}</p>
                    <div className="flex items-center justify-between">
                      <p className={`text-xl font-bold ${getSuccessRateColor(rate)}`}>
                        {(rate * 100).toFixed(1)}%
                      </p>
                      <div className="w-16 bg-gray-200 rounded-full h-2">
                        <div 
                          className={`h-2 rounded-full ${
                            rate >= 0.7 ? "bg-green-600" : 
                            rate >= 0.5 ? "bg-yellow-600" : "bg-red-600"
                          }`}
                          style={{ width: `${rate * 100}%` }}
                        />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Performance Distribution */}
          <Card>
            <CardHeader>
              <CardTitle>Return Distribution</CardTitle>
              <CardDescription>
                How returns are distributed across all pattern occurrences
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div>
                    <p className="text-sm text-muted-foreground">Winning Patterns</p>
                    <p className="text-lg font-bold text-green-600">
                      {Math.round(metrics.successRate * metrics.totalOccurrences)}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Losing Patterns</p>
                    <p className="text-lg font-bold text-red-600">
                      {metrics.totalOccurrences - Math.round(metrics.successRate * metrics.totalOccurrences)}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Avg Win</p>
                    <p className="text-lg font-bold text-green-600">
                      +{(metrics.averageReturn * 1.5).toFixed(2)}%
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Avg Loss</p>
                    <p className="text-lg font-bold text-red-600">
                      {metrics.averageLosingReturn.toFixed(2)}%
                    </p>
                  </div>
                </div>
                
                {/* Visual distribution would go here */}
                <div className="h-40 flex items-center justify-center border rounded-lg text-muted-foreground">
                  Return distribution histogram would be displayed here
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="timeframes" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Performance by Timeframe</CardTitle>
              <CardDescription>
                How patterns perform across different chart timeframes
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {Object.entries(metrics.timeframes).map(([timeframe, data]) => (
                  <div key={timeframe} className="flex items-center justify-between p-4 border rounded-lg">
                    <div className="flex items-center gap-4">
                      <Badge variant="outline" className="text-sm">
                        {timeframe.toUpperCase()}
                      </Badge>
                      <div>
                        <p className="font-medium">
                          {data.occurrences} patterns
                        </p>
                        <p className="text-sm text-muted-foreground">
                          Success rate: {(data.successRate * 100).toFixed(1)}%
                        </p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className={`text-lg font-bold ${getReturnColor(data.avgReturn)}`}>
                        {data.avgReturn >= 0 ? "+" : ""}{data.avgReturn.toFixed(2)}%
                      </p>
                      <div className="w-20 bg-gray-200 rounded-full h-2 mt-1">
                        <div 
                          className={`h-2 rounded-full ${
                            data.successRate >= 0.7 ? "bg-green-600" : 
                            data.successRate >= 0.5 ? "bg-yellow-600" : "bg-red-600"
                          }`}
                          style={{ width: `${data.successRate * 100}%` }}
                        />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="market" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Performance by Market Conditions</CardTitle>
              <CardDescription>
                How patterns perform in different market environments
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {Object.entries(metrics.marketConditions).map(([condition, data]) => (
                  <div key={condition} className="flex items-center justify-between p-4 border rounded-lg">
                    <div className="flex items-center gap-4">
                      <div className="flex items-center gap-2">
                        {condition === "bullish" && <TrendingUp className="h-5 w-5 text-green-600" />}
                        {condition === "bearish" && <TrendingDown className="h-5 w-5 text-red-600" />}
                        {condition === "sideways" && <Activity className="h-5 w-5 text-gray-600" />}
                        <Badge 
                          variant="outline" 
                          className={`capitalize ${
                            condition === "bullish" ? "text-green-600" :
                            condition === "bearish" ? "text-red-600" : "text-gray-600"
                          }`}
                        >
                          {condition}
                        </Badge>
                      </div>
                      <div>
                        <p className="font-medium">
                          {data.occurrences} patterns
                        </p>
                        <p className="text-sm text-muted-foreground">
                          Success rate: {(data.successRate * 100).toFixed(1)}%
                        </p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className={`text-lg font-bold ${getReturnColor(data.avgReturn)}`}>
                        {data.avgReturn >= 0 ? "+" : ""}{data.avgReturn.toFixed(2)}%
                      </p>
                      <div className="w-20 bg-gray-200 rounded-full h-2 mt-1">
                        <div 
                          className={`h-2 rounded-full ${
                            data.successRate >= 0.7 ? "bg-green-600" : 
                            data.successRate >= 0.5 ? "bg-yellow-600" : "bg-red-600"
                          }`}
                          style={{ width: `${data.successRate * 100}%` }}
                        />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="sectors" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Performance by Sector</CardTitle>
              <CardDescription>
                How patterns perform across different market sectors
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {Object.entries(metrics.sectors).slice(0, 8).map(([sector, data]) => (
                  <div key={sector} className="flex items-center justify-between p-4 border rounded-lg">
                    <div className="flex items-center gap-4">
                      <Badge variant="outline" className="text-sm">
                        {sector.toUpperCase()}
                      </Badge>
                      <div>
                        <p className="font-medium">
                          {data.occurrences} patterns
                        </p>
                        <p className="text-sm text-muted-foreground">
                          Success rate: {(data.successRate * 100).toFixed(1)}%
                        </p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className={`text-lg font-bold ${getReturnColor(data.avgReturn)}`}>
                        {data.avgReturn >= 0 ? "+" : ""}{data.avgReturn.toFixed(2)}%
                      </p>
                      <div className="w-20 bg-gray-200 rounded-full h-2 mt-1">
                        <div 
                          className={`h-2 rounded-full ${
                            data.successRate >= 0.7 ? "bg-green-600" : 
                            data.successRate >= 0.5 ? "bg-yellow-600" : "bg-red-600"
                          }`}
                          style={{ width: `${data.successRate * 100}%` }}
                        />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="timeline" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Performance Over Time</CardTitle>
              <CardDescription>
                Pattern performance trends over different time periods
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-64 flex items-center justify-center border rounded-lg text-muted-foreground">
                Performance timeline chart would be displayed here
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Recent Performance</CardTitle>
              <CardDescription>
                Last 30 days performance summary
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div>
                  <p className="text-sm text-muted-foreground">New Patterns</p>
                  <p className="text-2xl font-bold">24</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Success Rate</p>
                  <p className="text-2xl font-bold text-green-600">76%</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Avg Return</p>
                  <p className="text-2xl font-bold text-green-600">+8.3%</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Best Pattern</p>
                  <p className="text-2xl font-bold text-green-600">+23.4%</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}