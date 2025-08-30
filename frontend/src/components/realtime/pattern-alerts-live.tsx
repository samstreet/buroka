"use client"

import { useState, useEffect, useRef } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import {
  AlertTriangle,
  TrendingUp,
  TrendingDown,
  Bell,
  Eye,
  Clock,
  Target,
  Volume2,
  Activity,
  CheckCircle,
  X,
  Play,
  Pause,
} from "lucide-react"
import { cn } from "@/lib/utils"
import { format } from "date-fns"

export interface LivePatternAlert {
  id: string
  symbol: string
  patternType: string
  direction: "bullish" | "bearish" | "neutral"
  confidence: number
  price: number
  volume: number
  timeframe: string
  timestamp: Date
  priority: "low" | "medium" | "high" | "critical"
  status: "new" | "acknowledged" | "dismissed"
  expectedMove?: number
  stopLoss?: number
  takeProfit?: number
  description: string
  metadata?: {
    rsi?: number
    macd?: string
    volumeRatio?: number
    priceTarget?: number
  }
}

interface PatternAlertsLiveProps {
  maxAlerts?: number
  autoAcknowledge?: boolean
  showNotifications?: boolean
  priorityFilter?: string[]
  onAlertTriggered?: (alert: LivePatternAlert) => void
  onAlertAction?: (alertId: string, action: string) => void
}

export function PatternAlertsLive({
  maxAlerts = 50,
  autoAcknowledge = false,
  showNotifications = true,
  priorityFilter = ["low", "medium", "high", "critical"],
  onAlertTriggered,
  onAlertAction,
}: PatternAlertsLiveProps) {
  const [alerts, setAlerts] = useState<LivePatternAlert[]>([])
  const [selectedAlert, setSelectedAlert] = useState<LivePatternAlert | null>(null)
  const [isDetailDialogOpen, setIsDetailDialogOpen] = useState(false)
  const [isLive, setIsLive] = useState(true)
  const [alertCount, setAlertCount] = useState({ today: 0, total: 0 })
  const audioRef = useRef<HTMLAudioElement | null>(null)

  const symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT", "DOTUSDT"]
  const patternTypes = [
    "Triangle Breakout", "Flag Pattern", "Head & Shoulders", "Double Bottom", 
    "Double Top", "Cup & Handle", "Wedge", "Channel Breakout", "Support Break",
    "Resistance Break", "Moving Average Cross", "Volume Spike"
  ]

  // Generate mock live alerts
  useEffect(() => {
    if (!isLive) return

    const generateAlert = (): LivePatternAlert => {
      const symbol = symbols[Math.floor(Math.random() * symbols.length)]
      const patternType = patternTypes[Math.floor(Math.random() * patternTypes.length)]
      const direction = Math.random() > 0.5 ? "bullish" : "bearish"
      const confidence = 0.65 + Math.random() * 0.35
      const price = 100 + Math.random() * 200
      const priorityRandom = Math.random()
      
      let priority: "low" | "medium" | "high" | "critical"
      if (confidence > 0.9) priority = "critical"
      else if (confidence > 0.8) priority = "high"
      else if (confidence > 0.7) priority = "medium"
      else priority = "low"

      return {
        id: `alert-${Date.now()}-${Math.random()}`,
        symbol,
        patternType,
        direction,
        confidence,
        price,
        volume: Math.floor(Math.random() * 5000000) + 1000000,
        timeframe: ["1m", "5m", "15m", "1h", "4h", "1d"][Math.floor(Math.random() * 6)],
        timestamp: new Date(),
        priority,
        status: autoAcknowledge && priority === "low" ? "acknowledged" : "new",
        expectedMove: direction === "bullish" ? Math.random() * 8 + 2 : -(Math.random() * 8 + 2),
        stopLoss: direction === "bullish" ? price * 0.97 : price * 1.03,
        takeProfit: direction === "bullish" ? price * 1.05 : price * 0.95,
        description: `${patternType} pattern detected on ${symbol} with ${(confidence * 100).toFixed(0)}% confidence. ${direction === "bullish" ? "Potential upward" : "Potential downward"} movement expected.`,
        metadata: {
          rsi: 30 + Math.random() * 40,
          macd: Math.random() > 0.5 ? "Bullish Cross" : "Bearish Cross",
          volumeRatio: 1 + Math.random() * 3,
          priceTarget: direction === "bullish" ? price * (1.03 + Math.random() * 0.07) : price * (0.93 + Math.random() * 0.07),
        }
      }
    }

    const interval = setInterval(() => {
      // Generate alerts based on probability (lower for higher priority)
      const shouldGenerate = Math.random() < 0.3 // 30% chance every interval
      
      if (shouldGenerate) {
        const newAlert = generateAlert()
        
        setAlerts(prev => {
          const filtered = prev.slice(0, maxAlerts - 1)
          return [newAlert, ...filtered]
        })
        
        setAlertCount(prev => ({
          today: prev.today + 1,
          total: prev.total + 1
        }))

        onAlertTriggered?.(newAlert)

        // Play notification sound for high priority alerts
        if (showNotifications && (newAlert.priority === "high" || newAlert.priority === "critical")) {
          if (!audioRef.current) {
            audioRef.current = new Audio("data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmYbCjWJ1+/NeSsFJnfH8N+POAkTX7Xq7KlUFApGnt+yvmYbCzaJ2PHNeSsFJnjI8N+QOAkSX7Xq7KlUFApGnt+yv2YbCzaJ2PHNeSsFJnjH8N+QOAkSX7Xq7KlUFApGnt+yv2YbCzaJ2PHOeSsFJnjH8N+QOAkSX7Xq7KlUFApGnt+yv2YbCzaJ2PHOeSsFJnjH8N+QOAkSX7Xq7KlUFApGnt+yv2YbCzaJ2PHOeSsFJnjH8N+QOAkSX7Xq7KlUFApGnt+yv2YbCzaJ2PHOeSsFJnjH8N+QOAkSX7Xq7KlUFApGnt+yv2YbCzaJ2PHOeSsFJnjH8N+QOAkSX7Xq7KlUFApGnt+yv2YbCzaJ2PHOeSsFJnjH8N+QOAkSX7Xq7KlUFApGnt+yv2YbCzaJ2PHOeSsFJnjH8N+QOAkSX7Xq7KlUFApGnt+yv2YbCzaJ2PHOeSsFJnjH8N+QOAkSX7Xq7KlUFApGnt+yv2YbCzaJ2PHOeSsFJnjH8N+QOAkSX7Xq7KlUFApGnt+yv2YbCzaJ2PHOeSsFJnjH8N+QOAkSX7Xq7KlUFApGnt+yv2YbCzaJ2PHOeSsFJnjH8N+QOAkSX7Xq7KlUEBAAA=")
          }
          audioRef.current?.play().catch(() => {
            // Ignore autoplay restrictions
          })
        }
      }
    }, 2000 + Math.random() * 8000) // Random interval 2-10 seconds

    return () => clearInterval(interval)
  }, [isLive, maxAlerts, autoAcknowledge, showNotifications, onAlertTriggered])

  const handleAlertAction = (alertId: string, action: "acknowledge" | "dismiss" | "view") => {
    if (action === "view") {
      const alert = alerts.find(a => a.id === alertId)
      if (alert) {
        setSelectedAlert(alert)
        setIsDetailDialogOpen(true)
      }
    } else {
      setAlerts(prev => prev.map(alert => 
        alert.id === alertId 
          ? { ...alert, status: action === "acknowledge" ? "acknowledged" : "dismissed" }
          : alert
      ))
      onAlertAction?.(alertId, action)
    }
  }

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case "critical":
        return "text-red-600 bg-red-50 border-red-200"
      case "high":
        return "text-orange-600 bg-orange-50 border-orange-200"
      case "medium":
        return "text-yellow-600 bg-yellow-50 border-yellow-200"
      default:
        return "text-blue-600 bg-blue-50 border-blue-200"
    }
  }

  const getPriorityIcon = (priority: string) => {
    switch (priority) {
      case "critical":
        return <AlertTriangle className="h-4 w-4 text-red-600" />
      case "high":
        return <AlertTriangle className="h-4 w-4 text-orange-600" />
      case "medium":
        return <Bell className="h-4 w-4 text-yellow-600" />
      default:
        return <Bell className="h-4 w-4 text-blue-600" />
    }
  }

  const getDirectionIcon = (direction: string) => {
    switch (direction) {
      case "bullish":
        return <TrendingUp className="h-4 w-4 text-green-600" />
      case "bearish":
        return <TrendingDown className="h-4 w-4 text-red-600" />
      default:
        return <Activity className="h-4 w-4 text-gray-600" />
    }
  }

  const filteredAlerts = alerts.filter(alert => 
    priorityFilter.includes(alert.priority)
  )

  const newAlertsCount = filteredAlerts.filter(alert => alert.status === "new").length
  const acknowledgedCount = filteredAlerts.filter(alert => alert.status === "acknowledged").length

  return (
    <div className="space-y-4">
      {/* Alert Statistics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardDescription className="flex items-center gap-2">
              <Bell className="h-4 w-4" />
              New Alerts
            </CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-red-600">{newAlertsCount}</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardDescription className="flex items-center gap-2">
              <CheckCircle className="h-4 w-4" />
              Acknowledged
            </CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-green-600">{acknowledgedCount}</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardDescription className="flex items-center gap-2">
              <Clock className="h-4 w-4" />
              Today Total
            </CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">{alertCount.today}</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardDescription className="flex items-center gap-2">
              <Activity className="h-4 w-4" />
              Status
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setIsLive(!isLive)}
              >
                {isLive ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
              </Button>
              <p className={cn(
                "font-bold",
                isLive ? "text-green-600" : "text-gray-600"
              )}>
                {isLive ? "Live" : "Paused"}
              </p>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Live Alerts Feed */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span>Live Pattern Alerts</span>
            <Badge variant="outline" className="animate-pulse">
              {isLive ? "LIVE" : "PAUSED"}
            </Badge>
          </CardTitle>
          <CardDescription>
            Real-time pattern detection alerts • {filteredAlerts.length} active alerts
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ScrollArea className="h-96">
            <div className="space-y-3">
              {filteredAlerts.map((alert, index) => (
                <div
                  key={alert.id}
                  className={cn(
                    "p-4 rounded-lg border transition-all",
                    getPriorityColor(alert.priority),
                    alert.status === "new" && index < 3 ? "animate-pulse" : "",
                    alert.status === "dismissed" ? "opacity-50" : ""
                  )}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex items-start gap-3 flex-1">
                      <div className="flex items-center gap-2 mt-1">
                        {getPriorityIcon(alert.priority)}
                        {getDirectionIcon(alert.direction)}
                      </div>
                      
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-1">
                          <Badge variant="outline" className="font-bold">
                            {alert.symbol}
                          </Badge>
                          <Badge variant="secondary" className="text-xs">
                            {alert.patternType}
                          </Badge>
                          <Badge variant="outline" className="text-xs">
                            {alert.timeframe}
                          </Badge>
                          <Badge 
                            variant={alert.priority === "critical" ? "destructive" : 
                                   alert.priority === "high" ? "default" : "secondary"}
                            className="text-xs"
                          >
                            {alert.priority.toUpperCase()}
                          </Badge>
                        </div>
                        
                        <p className="text-sm mb-2">{alert.description}</p>
                        
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs text-muted-foreground">
                          <div>
                            <span className="font-medium">Price:</span> ${alert.price.toFixed(2)}
                          </div>
                          <div>
                            <span className="font-medium">Confidence:</span> {(alert.confidence * 100).toFixed(0)}%
                          </div>
                          <div>
                            <span className="font-medium">Expected:</span> 
                            <span className={cn(
                              alert.expectedMove && alert.expectedMove >= 0 ? "text-green-600" : "text-red-600"
                            )}>
                              {alert.expectedMove && alert.expectedMove >= 0 ? "+" : ""}
                              {alert.expectedMove?.toFixed(1)}%
                            </span>
                          </div>
                          <div>
                            <span className="font-medium">Volume:</span> {(alert.volume / 1000000).toFixed(1)}M
                          </div>
                        </div>
                        
                        <p className="text-xs text-muted-foreground mt-1">
                          {format(alert.timestamp, "HH:mm:ss")}
                        </p>
                      </div>
                    </div>
                    
                    <div className="flex gap-1 ml-2">
                      {alert.status === "new" && (
                        <>
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-8 w-8"
                            onClick={() => handleAlertAction(alert.id, "acknowledge")}
                          >
                            <CheckCircle className="h-4 w-4" />
                          </Button>
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-8 w-8"
                            onClick={() => handleAlertAction(alert.id, "dismiss")}
                          >
                            <X className="h-4 w-4" />
                          </Button>
                        </>
                      )}
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-8 w-8"
                        onClick={() => handleAlertAction(alert.id, "view")}
                      >
                        <Eye className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                </div>
              ))}
              
              {filteredAlerts.length === 0 && (
                <div className="text-center py-8">
                  <Bell className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                  <h3 className="text-lg font-medium mb-2">No Active Alerts</h3>
                  <p className="text-sm text-muted-foreground">
                    {isLive ? "Waiting for pattern detections..." : "Start live monitoring to see alerts"}
                  </p>
                </div>
              )}
            </div>
          </ScrollArea>
        </CardContent>
      </Card>

      {/* Alert Detail Dialog */}
      <Dialog open={isDetailDialogOpen} onOpenChange={setIsDetailDialogOpen}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              {selectedAlert && getPriorityIcon(selectedAlert.priority)}
              Pattern Alert Details
            </DialogTitle>
            <DialogDescription>
              {selectedAlert && `${selectedAlert.symbol} • ${selectedAlert.patternType}`}
            </DialogDescription>
          </DialogHeader>
          
          {selectedAlert && (
            <div className="space-y-4">
              {/* Overview */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <h4 className="font-medium mb-2">Pattern Information</h4>
                  <div className="space-y-1 text-sm">
                    <p><span className="font-medium">Symbol:</span> {selectedAlert.symbol}</p>
                    <p><span className="font-medium">Pattern:</span> {selectedAlert.patternType}</p>
                    <p><span className="font-medium">Direction:</span> 
                      <span className={cn(
                        "ml-1 capitalize",
                        selectedAlert.direction === "bullish" ? "text-green-600" : "text-red-600"
                      )}>
                        {selectedAlert.direction}
                      </span>
                    </p>
                    <p><span className="font-medium">Timeframe:</span> {selectedAlert.timeframe}</p>
                    <p><span className="font-medium">Confidence:</span> {(selectedAlert.confidence * 100).toFixed(0)}%</p>
                  </div>
                </div>
                
                <div>
                  <h4 className="font-medium mb-2">Trading Levels</h4>
                  <div className="space-y-1 text-sm">
                    <p><span className="font-medium">Current Price:</span> ${selectedAlert.price.toFixed(2)}</p>
                    <p><span className="font-medium">Target Price:</span> ${selectedAlert.metadata?.priceTarget?.toFixed(2)}</p>
                    <p><span className="font-medium">Stop Loss:</span> ${selectedAlert.stopLoss?.toFixed(2)}</p>
                    <p><span className="font-medium">Take Profit:</span> ${selectedAlert.takeProfit?.toFixed(2)}</p>
                    <p><span className="font-medium">Expected Move:</span> 
                      <span className={cn(
                        "ml-1",
                        selectedAlert.expectedMove && selectedAlert.expectedMove >= 0 ? "text-green-600" : "text-red-600"
                      )}>
                        {selectedAlert.expectedMove && selectedAlert.expectedMove >= 0 ? "+" : ""}
                        {selectedAlert.expectedMove?.toFixed(1)}%
                      </span>
                    </p>
                  </div>
                </div>
              </div>
              
              {/* Technical Indicators */}
              {selectedAlert.metadata && (
                <div>
                  <h4 className="font-medium mb-2">Technical Indicators</h4>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <p><span className="font-medium">RSI:</span> {selectedAlert.metadata.rsi?.toFixed(1)}</p>
                    <p><span className="font-medium">MACD:</span> {selectedAlert.metadata.macd}</p>
                    <p><span className="font-medium">Volume Ratio:</span> {selectedAlert.metadata.volumeRatio?.toFixed(2)}x</p>
                    <p><span className="font-medium">Volume:</span> {(selectedAlert.volume / 1000000).toFixed(1)}M</p>
                  </div>
                </div>
              )}
              
              {/* Actions */}
              <div className="flex gap-2 pt-4 border-t">
                <Button
                  onClick={() => handleAlertAction(selectedAlert.id, "acknowledge")}
                  disabled={selectedAlert.status !== "new"}
                >
                  <CheckCircle className="mr-2 h-4 w-4" />
                  Acknowledge
                </Button>
                <Button
                  variant="outline"
                  onClick={() => handleAlertAction(selectedAlert.id, "dismiss")}
                  disabled={selectedAlert.status === "dismissed"}
                >
                  <X className="mr-2 h-4 w-4" />
                  Dismiss
                </Button>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  )
}