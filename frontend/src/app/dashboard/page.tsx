"use client"

import { ProtectedRoute } from "@/components/auth/protected-route"
import { DashboardLayout } from "@/components/layout/dashboard-layout"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { TrendingUp, TrendingDown, Activity, DollarSign, Bitcoin } from "lucide-react"
import { ErrorBoundary } from "@/components/error-boundary"
import { SuspenseWrapper } from "@/components/suspense-wrapper"
import { LoadingDashboard } from "@/components/ui/loading"
import { useState, useEffect } from "react"

function DashboardContent() {
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    // Simulate data loading
    const timer = setTimeout(() => {
      setIsLoading(false)
    }, 1000)
    return () => clearTimeout(timer)
  }, [])

  if (isLoading) {
    return <LoadingDashboard />
  }

  return (
    <div className="space-y-8">
          <div className="mb-8">
            <h1 className="text-3xl font-bold flex items-center gap-2">
              <Bitcoin className="h-8 w-8 text-orange-500" />
              Crypto Dashboard
            </h1>
            <p className="text-muted-foreground">Welcome to your cryptocurrency trading dashboard</p>
          </div>

          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Total Patterns</CardTitle>
                <TrendingUp className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">254</div>
                <p className="text-xs text-muted-foreground">+20.1% from last month</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Success Rate</CardTitle>
                <Activity className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">73.2%</div>
                <p className="text-xs text-muted-foreground">+2.5% from last week</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Active Alerts</CardTitle>
                <TrendingDown className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">12</div>
                <p className="text-xs text-muted-foreground">3 critical, 9 normal</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Portfolio Value</CardTitle>
                <DollarSign className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">$45,231</div>
                <p className="text-xs text-muted-foreground">+12.3% this month</p>
              </CardContent>
            </Card>
          </div>

          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-7 mt-4">
            <Card className="col-span-4">
              <CardHeader>
                <CardTitle>Recent Patterns</CardTitle>
                <CardDescription>Latest detected patterns across crypto pairs</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {[
                    { pattern: "Head and Shoulders", pair: "BTC/USDT", timeframe: "1H", confidence: 85, direction: "Bullish", time: "2 hours ago" },
                    { pattern: "Triangle Breakout", pair: "ETH/USDT", timeframe: "4H", confidence: 92, direction: "Bullish", time: "4 hours ago" },
                    { pattern: "Double Bottom", pair: "ADA/USDT", timeframe: "1D", confidence: 78, direction: "Bullish", time: "6 hours ago" },
                    { pattern: "Bull Flag", pair: "SOL/USDT", timeframe: "30M", confidence: 73, direction: "Bullish", time: "1 hour ago" },
                    { pattern: "Descending Triangle", pair: "DOGE/USDT", timeframe: "2H", confidence: 88, direction: "Bearish", time: "3 hours ago" },
                  ].map((item, i) => (
                    <div key={i} className="flex items-center justify-between p-4 border rounded-lg">
                      <div className="flex items-center space-x-4">
                        <div className="p-2 bg-accent rounded">
                          <TrendingUp className="h-4 w-4" />
                        </div>
                        <div>
                          <p className="font-medium">{item.pattern}</p>
                          <p className="text-sm text-muted-foreground">{item.pair} • {item.timeframe} • {item.confidence}% confidence</p>
                        </div>
                      </div>
                      <div className="text-right">
                        <p className={`font-medium ${item.direction === "Bullish" ? "text-green-600" : "text-red-600"}`}>{item.direction}</p>
                        <p className="text-sm text-muted-foreground">{item.time}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card className="col-span-3">
              <CardHeader>
                <CardTitle>Top Performers</CardTitle>
                <CardDescription>Best performing patterns this week</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {[
                    { name: "Triangle Breakout", success: 92, count: 24 },
                    { name: "Double Bottom", success: 87, count: 15 },
                    { name: "Bull Flag", success: 81, count: 31 },
                    { name: "Cup & Handle", success: 78, count: 12 },
                    { name: "Ascending Triangle", success: 75, count: 28 },
                  ].map((pattern) => (
                    <div key={pattern.name} className="space-y-2">
                      <div className="flex items-center justify-between">
                        <p className="text-sm font-medium">{pattern.name}</p>
                        <p className="text-sm text-muted-foreground">{pattern.success}%</p>
                      </div>
                      <div className="w-full bg-secondary rounded-full h-2">
                        <div
                          className="bg-primary h-2 rounded-full"
                          style={{ width: `${pattern.success}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
    </div>
  )
}

export default function DashboardPage() {
  return (
    <ProtectedRoute>
      <ErrorBoundary>
        <DashboardLayout>
          <SuspenseWrapper type="dashboard">
            <DashboardContent />
          </SuspenseWrapper>
        </DashboardLayout>
      </ErrorBoundary>
    </ProtectedRoute>
  )
}