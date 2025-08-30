"use client"

import { useState } from "react"
import { ProtectedRoute } from "@/components/auth/protected-route"
import { DashboardLayout } from "@/components/layout/dashboard-layout"
import { PatternList, Pattern } from "@/components/patterns/pattern-list"
import { PatternDetail } from "@/components/patterns/pattern-detail"
import { PatternPerformance } from "@/components/patterns/pattern-performance"
import { PatternComparison } from "@/components/patterns/pattern-comparison"
import { PatternAlerts } from "@/components/patterns/pattern-alerts"
import { PatternBacktest } from "@/components/patterns/pattern-backtest"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { ErrorBoundary } from "@/components/error-boundary"
import {
  BarChart3,
  GitCompare,
  Bell,
  Activity,
  Eye,
  ArrowLeft,
} from "lucide-react"

// Mock data for patterns
const mockPatterns: Pattern[] = [
  {
    id: "1",
    symbol: "AAPL",
    type: "Head & Shoulders",
    timeframe: "1d",
    confidence: 0.85,
    direction: "bearish",
    startTime: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
    endTime: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000),
    priceAtDetection: 178.32,
    currentPrice: 174.85,
    priceChange: -3.47,
    priceChangePercent: -1.95,
    volume: 45000000,
    status: "active",
    accuracy: 0.73,
    successRate: 0.68,
  },
  {
    id: "2",
    symbol: "GOOGL",
    type: "Triangle Breakout",
    timeframe: "4h",
    confidence: 0.92,
    direction: "bullish",
    startTime: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000),
    endTime: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000),
    priceAtDetection: 138.21,
    currentPrice: 141.75,
    priceChange: 3.54,
    priceChangePercent: 2.56,
    volume: 28000000,
    status: "completed",
    accuracy: 0.89,
    successRate: 0.84,
  },
  {
    id: "3",
    symbol: "MSFT",
    type: "Double Bottom",
    timeframe: "1d",
    confidence: 0.78,
    direction: "bullish",
    startTime: new Date(Date.now() - 10 * 24 * 60 * 60 * 1000),
    endTime: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000),
    priceAtDetection: 378.91,
    currentPrice: 385.67,
    priceChange: 6.76,
    priceChangePercent: 1.78,
    volume: 32000000,
    status: "active",
    accuracy: 0.81,
    successRate: 0.75,
  },
  {
    id: "4",
    symbol: "TSLA",
    type: "Flag Pattern",
    timeframe: "1h",
    confidence: 0.88,
    direction: "bullish",
    startTime: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000),
    endTime: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000),
    priceAtDetection: 243.56,
    currentPrice: 251.42,
    priceChange: 7.86,
    priceChangePercent: 3.23,
    volume: 65000000,
    status: "completed",
    accuracy: 0.91,
    successRate: 0.87,
  },
  {
    id: "5",
    symbol: "AMZN",
    type: "Wedge",
    timeframe: "1d",
    confidence: 0.71,
    direction: "bearish",
    startTime: new Date(Date.now() - 8 * 24 * 60 * 60 * 1000),
    endTime: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000),
    priceAtDetection: 127.43,
    currentPrice: 124.89,
    priceChange: -2.54,
    priceChangePercent: -1.99,
    volume: 38000000,
    status: "failed",
    accuracy: 0.65,
    successRate: 0.58,
  },
]

// Mock performance metrics
const mockPerformanceMetrics = {
  patternType: "Head & Shoulders",
  totalOccurrences: 247,
  successfulPatterns: 180,
  successRate: 0.73,
  averageReturn: 12.5,
  averageLosingReturn: -6.8,
  maxReturn: 45.2,
  minReturn: -23.1,
  averageDuration: 5.2,
  winRate: {
    "1d": 0.78,
    "3d": 0.71,
    "7d": 0.68,
    "30d": 0.73,
  },
  marketConditions: {
    bullish: { occurrences: 82, successRate: 0.79, avgReturn: 15.3 },
    bearish: { occurrences: 95, successRate: 0.69, avgReturn: 9.8 },
    sideways: { occurrences: 70, successRate: 0.71, avgReturn: 11.2 },
  },
  timeframes: {
    "1h": { occurrences: 45, successRate: 0.67, avgReturn: 8.9 },
    "4h": { occurrences: 78, successRate: 0.74, avgReturn: 13.2 },
    "1d": { occurrences: 89, successRate: 0.76, avgReturn: 14.8 },
    "1w": { occurrences: 35, successRate: 0.71, avgReturn: 16.5 },
  },
  sectors: {
    tech: { occurrences: 89, successRate: 0.78, avgReturn: 15.2 },
    finance: { occurrences: 45, successRate: 0.69, avgReturn: 10.8 },
    healthcare: { occurrences: 32, successRate: 0.72, avgReturn: 12.1 },
    energy: { occurrences: 28, successRate: 0.66, avgReturn: 9.5 },
    retail: { occurrences: 25, successRate: 0.74, avgReturn: 13.8 },
    manufacturing: { occurrences: 18, successRate: 0.71, avgReturn: 11.9 },
    utilities: { occurrences: 10, successRate: 0.70, avgReturn: 8.7 },
  },
}

export default function PatternsPage() {
  const [selectedPattern, setSelectedPattern] = useState<Pattern | null>(null)
  const [isDetailDialogOpen, setIsDetailDialogOpen] = useState(false)
  const [selectedTab, setSelectedTab] = useState("list")
  const [comparisonPatterns, setComparisonPatterns] = useState<Pattern[]>([])

  const handlePatternClick = (pattern: Pattern) => {
    setSelectedPattern(pattern)
    setIsDetailDialogOpen(true)
  }

  const handleAlertClick = (pattern: Pattern) => {
    setSelectedTab("alerts")
    // Could pre-configure alert with this pattern
  }

  const handleBacktestClick = (pattern: Pattern) => {
    setSelectedTab("backtest")
    setSelectedPattern(pattern)
  }

  return (
    <ProtectedRoute>
      <ErrorBoundary>
        <DashboardLayout>
          <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-3xl font-bold">Pattern Analysis</h1>
                <p className="text-muted-foreground">
                  Analyze, compare, and backtest trading patterns
                </p>
              </div>
            </div>

            {/* Tabs */}
            <Tabs value={selectedTab} onValueChange={setSelectedTab}>
              <TabsList className="grid w-full grid-cols-5">
                <TabsTrigger value="list" className="flex items-center gap-2">
                  <Eye className="h-4 w-4" />
                  Patterns
                </TabsTrigger>
                <TabsTrigger value="performance" className="flex items-center gap-2">
                  <BarChart3 className="h-4 w-4" />
                  Performance
                </TabsTrigger>
                <TabsTrigger value="comparison" className="flex items-center gap-2">
                  <GitCompare className="h-4 w-4" />
                  Compare
                </TabsTrigger>
                <TabsTrigger value="alerts" className="flex items-center gap-2">
                  <Bell className="h-4 w-4" />
                  Alerts
                </TabsTrigger>
                <TabsTrigger value="backtest" className="flex items-center gap-2">
                  <Activity className="h-4 w-4" />
                  Backtest
                </TabsTrigger>
              </TabsList>

              <TabsContent value="list" className="space-y-4">
                <PatternList
                  patterns={mockPatterns}
                  onPatternClick={handlePatternClick}
                  onAlertClick={handleAlertClick}
                />
              </TabsContent>

              <TabsContent value="performance" className="space-y-4">
                <PatternPerformance metrics={mockPerformanceMetrics} />
              </TabsContent>

              <TabsContent value="comparison" className="space-y-4">
                <PatternComparison
                  availablePatterns={mockPatterns}
                  selectedPatterns={comparisonPatterns}
                  onPatternsChange={setComparisonPatterns}
                />
              </TabsContent>

              <TabsContent value="alerts" className="space-y-4">
                <PatternAlerts />
              </TabsContent>

              <TabsContent value="backtest" className="space-y-4">
                <PatternBacktest
                  pattern={selectedPattern || undefined}
                  availableSymbols={["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META", "NFLX"]}
                  onRunBacktest={async (config) => {
                    // Mock backtest results
                    await new Promise(resolve => setTimeout(resolve, 3000))
                    
                    return {
                      id: "test-1",
                      config,
                      status: "completed" as const,
                      progress: 100,
                      startTime: new Date(),
                      endTime: new Date(),
                      results: {
                        totalTrades: 45,
                        winningTrades: 32,
                        losingTrades: 13,
                        winRate: 0.711,
                        totalReturn: 15.4,
                        annualizedReturn: 18.7,
                        maxDrawdown: -8.3,
                        sharpeRatio: 1.42,
                        profitFactor: 2.1,
                        averageWin: 8.2,
                        averageLoss: -4.1,
                        maxWin: 23.5,
                        maxLoss: -12.8,
                        totalDays: 365,
                        finalCapital: 11540,
                        trades: Array.from({ length: 45 }, (_, i) => ({
                          id: `trade-${i}`,
                          symbol: config.symbols[i % config.symbols.length],
                          patternType: config.patternType || "Triangle",
                          entryDate: new Date(Date.now() - (45 - i) * 7 * 24 * 60 * 60 * 1000),
                          exitDate: new Date(Date.now() - (45 - i - 2) * 7 * 24 * 60 * 60 * 1000),
                          entryPrice: 100 + Math.random() * 100,
                          exitPrice: 100 + Math.random() * 120,
                          quantity: Math.floor(Math.random() * 100) + 10,
                          pnl: (Math.random() - 0.3) * 1000,
                          pnlPercent: (Math.random() - 0.3) * 20,
                          confidence: 0.7 + Math.random() * 0.3,
                          status: Math.random() > 0.1 ? "closed" as const : "open" as const,
                        })),
                      },
                    }
                  }}
                />
              </TabsContent>
            </Tabs>
          </div>

          {/* Pattern Detail Dialog */}
          <Dialog open={isDetailDialogOpen} onOpenChange={setIsDetailDialogOpen}>
            <DialogContent className="max-w-6xl max-h-[90vh] overflow-y-auto">
              <DialogHeader>
                <div className="flex items-center justify-between">
                  <DialogTitle>Pattern Details</DialogTitle>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => setIsDetailDialogOpen(false)}
                  >
                    <ArrowLeft className="h-4 w-4" />
                  </Button>
                </div>
              </DialogHeader>
              {selectedPattern && (
                <PatternDetail
                  pattern={selectedPattern}
                  similarPatterns={mockPatterns.filter(p => 
                    p.id !== selectedPattern.id && p.type === selectedPattern.type
                  )}
                  onBacktest={() => {
                    setSelectedTab("backtest")
                    setIsDetailDialogOpen(false)
                  }}
                  onCreateAlert={() => {
                    setSelectedTab("alerts")
                    setIsDetailDialogOpen(false)
                  }}
                  onShare={() => {
                    // Implement share functionality
                    console.log("Sharing pattern:", selectedPattern)
                  }}
                />
              )}
            </DialogContent>
          </Dialog>
        </DashboardLayout>
      </ErrorBoundary>
    </ProtectedRoute>
  )
}