"use client"

import { useState } from "react"
import { Pattern } from "./pattern-list"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
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
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  Play,
  Pause,
  Square,
  BarChart3,
  TrendingUp,
  TrendingDown,
  Target,
  Clock,
  DollarSign,
  Activity,
  Settings,
  Download,
  RefreshCw,
} from "lucide-react"
import { format, subDays } from "date-fns"

interface BacktestConfig {
  patternType?: string
  symbols: string[]
  startDate: Date
  endDate: Date
  initialCapital: number
  positionSize: number // percentage of capital per trade
  stopLoss: number // percentage
  takeProfit: number // percentage
  minConfidence: number
  timeframes: string[]
  maxPositions: number
  commission: number // percentage per trade
}

interface BacktestResult {
  id: string
  config: BacktestConfig
  status: "running" | "completed" | "failed"
  progress: number
  startTime: Date
  endTime?: Date
  results?: {
    totalTrades: number
    winningTrades: number
    losingTrades: number
    winRate: number
    totalReturn: number
    annualizedReturn: number
    maxDrawdown: number
    sharpeRatio: number
    profitFactor: number
    averageWin: number
    averageLoss: number
    maxWin: number
    maxLoss: number
    totalDays: number
    finalCapital: number
    trades: BacktestTrade[]
  }
  error?: string
}

interface BacktestTrade {
  id: string
  symbol: string
  patternType: string
  entryDate: Date
  exitDate?: Date
  entryPrice: number
  exitPrice?: number
  quantity: number
  pnl?: number
  pnlPercent?: number
  confidence: number
  status: "open" | "closed" | "stopped"
  stopLossPrice?: number
  takeProfitPrice?: number
}

interface PatternBacktestProps {
  pattern?: Pattern
  availableSymbols?: string[]
  onRunBacktest?: (config: BacktestConfig) => Promise<BacktestResult>
}

const defaultConfig: BacktestConfig = {
  symbols: ["AAPL"],
  startDate: subDays(new Date(), 365),
  endDate: new Date(),
  initialCapital: 10000,
  positionSize: 10,
  stopLoss: 5,
  takeProfit: 10,
  minConfidence: 0.7,
  timeframes: ["1d"],
  maxPositions: 5,
  commission: 0.1,
}

export function PatternBacktest({
  pattern,
  availableSymbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"],
  onRunBacktest,
}: PatternBacktestProps) {
  const [config, setConfig] = useState<BacktestConfig>({
    ...defaultConfig,
    patternType: pattern?.type,
    symbols: pattern ? [pattern.symbol] : defaultConfig.symbols,
  })
  const [currentTest, setCurrentTest] = useState<BacktestResult | null>(null)
  const [testHistory, setTestHistory] = useState<BacktestResult[]>([])
  const [selectedTab, setSelectedTab] = useState("config")

  const handleRunBacktest = async () => {
    if (!onRunBacktest) return

    const newTest: BacktestResult = {
      id: Math.random().toString(36).substr(2, 9),
      config: { ...config },
      status: "running",
      progress: 0,
      startTime: new Date(),
    }

    setCurrentTest(newTest)
    setSelectedTab("results")

    try {
      // Simulate progress
      const progressInterval = setInterval(() => {
        setCurrentTest(prev => {
          if (!prev || prev.status !== "running") return prev
          const newProgress = Math.min(prev.progress + Math.random() * 20, 95)
          return { ...prev, progress: newProgress }
        })
      }, 500)

      const result = await onRunBacktest(config)
      clearInterval(progressInterval)

      const completedTest = {
        ...result,
        status: "completed" as const,
        progress: 100,
        endTime: new Date(),
      }

      setCurrentTest(completedTest)
      setTestHistory(prev => [completedTest, ...prev.slice(0, 9)])
    } catch (error) {
      setCurrentTest(prev => prev ? {
        ...prev,
        status: "failed",
        error: error instanceof Error ? error.message : "Unknown error",
        endTime: new Date(),
      } : null)
    }
  }

  const handleStopBacktest = () => {
    setCurrentTest(prev => prev ? {
      ...prev,
      status: "failed",
      error: "Stopped by user",
      endTime: new Date(),
    } : null)
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case "running":
        return "text-blue-600"
      case "completed":
        return "text-green-600"
      case "failed":
        return "text-red-600"
      default:
        return "text-gray-600"
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Pattern Backtesting</h2>
          <p className="text-muted-foreground">
            {pattern ? `Testing ${pattern.type} pattern for ${pattern.symbol}` : 
             "Test pattern strategies against historical data"}
          </p>
        </div>
        <div className="flex gap-2">
          {currentTest?.status === "running" ? (
            <Button variant="destructive" onClick={handleStopBacktest}>
              <Square className="mr-2 h-4 w-4" />
              Stop Test
            </Button>
          ) : (
            <Button onClick={handleRunBacktest} disabled={!config.symbols.length}>
              <Play className="mr-2 h-4 w-4" />
              Run Backtest
            </Button>
          )}
        </div>
      </div>

      {/* Tabs */}
      <Tabs value={selectedTab} onValueChange={setSelectedTab}>
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="config">Configuration</TabsTrigger>
          <TabsTrigger value="results">Results</TabsTrigger>
          <TabsTrigger value="trades">Trades</TabsTrigger>
          <TabsTrigger value="history">History</TabsTrigger>
        </TabsList>

        <TabsContent value="config" className="space-y-6">
          {/* Basic Configuration */}
          <Card>
            <CardHeader>
              <CardTitle>Basic Configuration</CardTitle>
              <CardDescription>
                Set up the parameters for your backtest
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Pattern Type</Label>
                  <Select
                    value={config.patternType || ""}
                    onValueChange={(value) => 
                      setConfig({ ...config, patternType: value || undefined })
                    }
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select pattern type" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="">All Patterns</SelectItem>
                      <SelectItem value="Triangle">Triangle</SelectItem>
                      <SelectItem value="Flag">Flag</SelectItem>
                      <SelectItem value="Head & Shoulders">Head & Shoulders</SelectItem>
                      <SelectItem value="Double Top">Double Top</SelectItem>
                      <SelectItem value="Double Bottom">Double Bottom</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label>Symbols</Label>
                  <Input
                    value={config.symbols.join(", ")}
                    onChange={(e) => 
                      setConfig({
                        ...config,
                        symbols: e.target.value.split(",").map(s => s.trim()).filter(Boolean)
                      })
                    }
                    placeholder="e.g., AAPL, GOOGL, MSFT"
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Start Date</Label>
                  <Input
                    type="date"
                    value={config.startDate.toISOString().split('T')[0]}
                    onChange={(e) => 
                      setConfig({ ...config, startDate: new Date(e.target.value) })
                    }
                  />
                </div>

                <div className="space-y-2">
                  <Label>End Date</Label>
                  <Input
                    type="date"
                    value={config.endDate.toISOString().split('T')[0]}
                    onChange={(e) => 
                      setConfig({ ...config, endDate: new Date(e.target.value) })
                    }
                  />
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Trading Configuration */}
          <Card>
            <CardHeader>
              <CardTitle>Trading Parameters</CardTitle>
              <CardDescription>
                Configure position sizing and risk management
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-3 gap-4">
                <div className="space-y-2">
                  <Label>Initial Capital ($)</Label>
                  <Input
                    type="number"
                    value={config.initialCapital}
                    onChange={(e) => 
                      setConfig({ ...config, initialCapital: parseFloat(e.target.value) || 10000 })
                    }
                  />
                </div>

                <div className="space-y-2">
                  <Label>Position Size (%)</Label>
                  <Input
                    type="number"
                    min="1"
                    max="100"
                    value={config.positionSize}
                    onChange={(e) => 
                      setConfig({ ...config, positionSize: parseFloat(e.target.value) || 10 })
                    }
                  />
                </div>

                <div className="space-y-2">
                  <Label>Max Positions</Label>
                  <Input
                    type="number"
                    min="1"
                    value={config.maxPositions}
                    onChange={(e) => 
                      setConfig({ ...config, maxPositions: parseInt(e.target.value) || 5 })
                    }
                  />
                </div>
              </div>

              <div className="grid grid-cols-3 gap-4">
                <div className="space-y-2">
                  <Label>Stop Loss (%)</Label>
                  <Input
                    type="number"
                    min="0"
                    step="0.1"
                    value={config.stopLoss}
                    onChange={(e) => 
                      setConfig({ ...config, stopLoss: parseFloat(e.target.value) || 5 })
                    }
                  />
                </div>

                <div className="space-y-2">
                  <Label>Take Profit (%)</Label>
                  <Input
                    type="number"
                    min="0"
                    step="0.1"
                    value={config.takeProfit}
                    onChange={(e) => 
                      setConfig({ ...config, takeProfit: parseFloat(e.target.value) || 10 })
                    }
                  />
                </div>

                <div className="space-y-2">
                  <Label>Commission (%)</Label>
                  <Input
                    type="number"
                    min="0"
                    step="0.01"
                    value={config.commission}
                    onChange={(e) => 
                      setConfig({ ...config, commission: parseFloat(e.target.value) || 0.1 })
                    }
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Min Confidence</Label>
                  <Input
                    type="number"
                    min="0"
                    max="1"
                    step="0.1"
                    value={config.minConfidence}
                    onChange={(e) => 
                      setConfig({ ...config, minConfidence: parseFloat(e.target.value) || 0.7 })
                    }
                  />
                </div>

                <div className="space-y-2">
                  <Label>Timeframes</Label>
                  <Select
                    value={config.timeframes[0] || "1d"}
                    onValueChange={(value) => 
                      setConfig({ ...config, timeframes: [value] })
                    }
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="1h">1 Hour</SelectItem>
                      <SelectItem value="4h">4 Hours</SelectItem>
                      <SelectItem value="1d">1 Day</SelectItem>
                      <SelectItem value="1w">1 Week</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="results" className="space-y-6">
          {!currentTest ? (
            <Card>
              <CardContent className="text-center py-8">
                <BarChart3 className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                <h3 className="text-lg font-medium mb-2">No Backtest Running</h3>
                <p className="text-sm text-muted-foreground">
                  Configure and run a backtest to see results
                </p>
              </CardContent>
            </Card>
          ) : (
            <>
              {/* Test Status */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Activity className="h-5 w-5" />
                    Backtest Status
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center justify-between mb-4">
                    <div>
                      <p className={`font-medium ${getStatusColor(currentTest.status)}`}>
                        {currentTest.status.toUpperCase()}
                      </p>
                      <p className="text-sm text-muted-foreground">
                        Started: {format(currentTest.startTime, "PPp")}
                        {currentTest.endTime && (
                          <> • Completed: {format(currentTest.endTime, "PPp")}</>
                        )}
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="text-2xl font-bold">{Math.round(currentTest.progress)}%</p>
                      <p className="text-sm text-muted-foreground">Progress</p>
                    </div>
                  </div>
                  
                  {currentTest.status === "running" && (
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-blue-600 h-2 rounded-full transition-all"
                        style={{ width: `${currentTest.progress}%` }}
                      />
                    </div>
                  )}

                  {currentTest.error && (
                    <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg">
                      <p className="text-red-700 text-sm">{currentTest.error}</p>
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Results Summary */}
              {currentTest.results && (
                <>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <Card>
                      <CardHeader className="pb-2">
                        <CardDescription className="flex items-center gap-2">
                          <DollarSign className="h-4 w-4" />
                          Total Return
                        </CardDescription>
                      </CardHeader>
                      <CardContent>
                        <p className={`text-2xl font-bold ${
                          currentTest.results.totalReturn >= 0 ? "text-green-600" : "text-red-600"
                        }`}>
                          {currentTest.results.totalReturn >= 0 ? "+" : ""}
                          {currentTest.results.totalReturn.toFixed(2)}%
                        </p>
                        <p className="text-sm text-muted-foreground">
                          ${currentTest.results.finalCapital.toLocaleString()}
                        </p>
                      </CardContent>
                    </Card>

                    <Card>
                      <CardHeader className="pb-2">
                        <CardDescription className="flex items-center gap-2">
                          <Target className="h-4 w-4" />
                          Win Rate
                        </CardDescription>
                      </CardHeader>
                      <CardContent>
                        <p className="text-2xl font-bold">
                          {(currentTest.results.winRate * 100).toFixed(1)}%
                        </p>
                        <p className="text-sm text-muted-foreground">
                          {currentTest.results.winningTrades}/{currentTest.results.totalTrades} trades
                        </p>
                      </CardContent>
                    </Card>

                    <Card>
                      <CardHeader className="pb-2">
                        <CardDescription className="flex items-center gap-2">
                          <TrendingDown className="h-4 w-4" />
                          Max Drawdown
                        </CardDescription>
                      </CardHeader>
                      <CardContent>
                        <p className="text-2xl font-bold text-red-600">
                          -{currentTest.results.maxDrawdown.toFixed(2)}%
                        </p>
                        <p className="text-sm text-muted-foreground">
                          Worst period
                        </p>
                      </CardContent>
                    </Card>

                    <Card>
                      <CardHeader className="pb-2">
                        <CardDescription className="flex items-center gap-2">
                          <BarChart3 className="h-4 w-4" />
                          Sharpe Ratio
                        </CardDescription>
                      </CardHeader>
                      <CardContent>
                        <p className="text-2xl font-bold">
                          {currentTest.results.sharpeRatio.toFixed(2)}
                        </p>
                        <p className="text-sm text-muted-foreground">
                          Risk-adjusted return
                        </p>
                      </CardContent>
                    </Card>
                  </div>

                  {/* Performance Chart Placeholder */}
                  <Card>
                    <CardHeader>
                      <CardTitle>Equity Curve</CardTitle>
                      <CardDescription>
                        Portfolio value over time
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="h-64 flex items-center justify-center border rounded-lg text-muted-foreground">
                        Equity curve chart would be displayed here
                      </div>
                    </CardContent>
                  </Card>
                </>
              )}
            </>
          )}
        </TabsContent>

        <TabsContent value="trades" className="space-y-4">
          {currentTest?.results?.trades ? (
            <Card>
              <CardHeader>
                <CardTitle>Trade History</CardTitle>
                <CardDescription>
                  Detailed breakdown of all trades executed
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Symbol</TableHead>
                      <TableHead>Pattern</TableHead>
                      <TableHead>Entry Date</TableHead>
                      <TableHead>Exit Date</TableHead>
                      <TableHead>Entry Price</TableHead>
                      <TableHead>Exit Price</TableHead>
                      <TableHead>P&L</TableHead>
                      <TableHead>P&L %</TableHead>
                      <TableHead>Status</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {currentTest.results.trades.slice(0, 20).map((trade) => (
                      <TableRow key={trade.id}>
                        <TableCell className="font-medium">{trade.symbol}</TableCell>
                        <TableCell>{trade.patternType}</TableCell>
                        <TableCell>{format(trade.entryDate, "MMM dd, yyyy")}</TableCell>
                        <TableCell>
                          {trade.exitDate ? format(trade.exitDate, "MMM dd, yyyy") : "-"}
                        </TableCell>
                        <TableCell>${trade.entryPrice.toFixed(2)}</TableCell>
                        <TableCell>
                          {trade.exitPrice ? `$${trade.exitPrice.toFixed(2)}` : "-"}
                        </TableCell>
                        <TableCell className={
                          trade.pnl ? (trade.pnl >= 0 ? "text-green-600" : "text-red-600") : ""
                        }>
                          {trade.pnl ? `$${trade.pnl.toFixed(2)}` : "-"}
                        </TableCell>
                        <TableCell className={
                          trade.pnlPercent ? (trade.pnlPercent >= 0 ? "text-green-600" : "text-red-600") : ""
                        }>
                          {trade.pnlPercent ? 
                            `${trade.pnlPercent >= 0 ? "+" : ""}${trade.pnlPercent.toFixed(2)}%` : 
                            "-"
                          }
                        </TableCell>
                        <TableCell>
                          <Badge variant={
                            trade.status === "closed" ? "secondary" : 
                            trade.status === "open" ? "default" : "destructive"
                          }>
                            {trade.status}
                          </Badge>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          ) : (
            <Card>
              <CardContent className="text-center py-8">
                <Activity className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                <h3 className="text-lg font-medium mb-2">No Trades Available</h3>
                <p className="text-sm text-muted-foreground">
                  Run a backtest to see detailed trade history
                </p>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="history" className="space-y-4">
          {testHistory.length > 0 ? (
            <Card>
              <CardHeader>
                <CardTitle>Backtest History</CardTitle>
                <CardDescription>
                  Previously completed backtests
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {testHistory.map((test) => (
                    <div key={test.id} className="flex items-center justify-between p-4 border rounded-lg">
                      <div>
                        <p className="font-medium">
                          {test.config.patternType || "All Patterns"} - {test.config.symbols.join(", ")}
                        </p>
                        <p className="text-sm text-muted-foreground">
                          {format(test.startTime, "PPp")}
                          {test.results && (
                            <> • {test.results.totalTrades} trades • {(test.results.winRate * 100).toFixed(1)}% win rate</>
                          )}
                        </p>
                      </div>
                      <div className="text-right">
                        {test.results ? (
                          <p className={`font-medium ${
                            test.results.totalReturn >= 0 ? "text-green-600" : "text-red-600"
                          }`}>
                            {test.results.totalReturn >= 0 ? "+" : ""}
                            {test.results.totalReturn.toFixed(2)}%
                          </p>
                        ) : (
                          <Badge variant="destructive">Failed</Badge>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          ) : (
            <Card>
              <CardContent className="text-center py-8">
                <Clock className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                <h3 className="text-lg font-medium mb-2">No History Available</h3>
                <p className="text-sm text-muted-foreground">
                  Completed backtests will appear here
                </p>
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>
    </div>
  )
}