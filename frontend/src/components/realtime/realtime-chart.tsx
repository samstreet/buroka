"use client"

import { useEffect, useRef, useState, useCallback } from "react"
import { createChart, ColorType, IChartApi, ISeriesApi, Time } from "lightweight-charts"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  Play,
  Pause,
  Square,
  Maximize2,
  Activity,
  TrendingUp,
  Wifi,
  WifiOff,
} from "lucide-react"
import { useTheme } from "next-themes"
import { cn } from "@/lib/utils"

export interface RealtimePriceData {
  time: Time
  open: number
  high: number
  low: number
  close: number
  volume?: number
}

export interface RealtimeVolumeData {
  time: Time
  value: number
  color?: string
}

interface RealtimeChartProps {
  symbol: string
  height?: number
  showVolume?: boolean
  showPatterns?: boolean
  maxDataPoints?: number
  updateInterval?: number
  onPriceUpdate?: (data: RealtimePriceData) => void
  onVolumeUpdate?: (data: RealtimeVolumeData) => void
}

export function RealtimeChart({
  symbol,
  height = 500,
  showVolume = true,
  showPatterns = true,
  maxDataPoints = 1000,
  updateInterval = 1000,
  onPriceUpdate,
  onVolumeUpdate,
}: RealtimeChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null)
  const chart = useRef<IChartApi | null>(null)
  const candlestickSeries = useRef<ISeriesApi<"Candlestick"> | null>(null)
  const volumeSeries = useRef<ISeriesApi<"Histogram"> | null>(null)
  const lineSeries = useRef<ISeriesApi<"Line"> | null>(null)
  
  const [isStreaming, setIsStreaming] = useState(true)
  const [isConnected, setIsConnected] = useState(false)
  const [chartType, setChartType] = useState<"candlestick" | "line">("candlestick")
  const [currentPrice, setCurrentPrice] = useState<number | null>(null)
  const [priceChange, setPriceChange] = useState<number | null>(null)
  const [priceChangePercent, setPriceChangePercent] = useState<number | null>(null)
  const [volume, setVolume] = useState<number | null>(null)
  const [dataCount, setDataCount] = useState(0)
  
  const priceDataRef = useRef<RealtimePriceData[]>([])
  const volumeDataRef = useRef<RealtimeVolumeData[]>([])
  const lastPriceRef = useRef<number | null>(null)
  const streamingIntervalRef = useRef<NodeJS.Timeout | null>(null)
  
  const { theme } = useTheme()

  // Initialize chart
  useEffect(() => {
    if (!chartContainerRef.current) return

    const handleResize = () => {
      chart.current?.applyOptions({ 
        width: chartContainerRef.current?.clientWidth || 0 
      })
    }

    // Create chart
    chart.current = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: theme === "dark" ? "#9CA3AF" : "#6B7280",
      },
      width: chartContainerRef.current.clientWidth,
      height: height,
      grid: {
        vertLines: {
          color: theme === "dark" ? "#1F2937" : "#E5E7EB",
        },
        horzLines: {
          color: theme === "dark" ? "#1F2937" : "#E5E7EB",
        },
      },
      rightPriceScale: {
        borderColor: theme === "dark" ? "#374151" : "#D1D5DB",
        scaleMargins: showVolume ? {
          top: 0.1,
          bottom: 0.3,
        } : {
          top: 0.1,
          bottom: 0.1,
        },
      },
      timeScale: {
        borderColor: theme === "dark" ? "#374151" : "#D1D5DB",
        timeVisible: true,
        secondsVisible: true,
      },
      crosshair: {
        mode: 1, // Normal crosshair
      },
    })

    // Add candlestick series
    if (chartType === "candlestick") {
      candlestickSeries.current = chart.current.addCandlestickSeries({
        upColor: "#10B981",
        downColor: "#EF4444",
        borderUpColor: "#10B981",
        borderDownColor: "#EF4444",
        wickUpColor: "#10B981",
        wickDownColor: "#EF4444",
      })
    } else {
      lineSeries.current = chart.current.addLineSeries({
        color: "#3B82F6",
        lineWidth: 2,
      })
    }

    // Add volume series if enabled
    if (showVolume) {
      volumeSeries.current = chart.current.addHistogramSeries({
        color: "#3B82F680",
        priceFormat: {
          type: "volume",
        },
        priceScaleId: "",
      })

      chart.current.priceScale("").applyOptions({
        scaleMargins: {
          top: 0.7,
          bottom: 0,
        },
      })
    }

    // Handle resize
    window.addEventListener("resize", handleResize)

    return () => {
      window.removeEventListener("resize", handleResize)
      chart.current?.remove()
    }
  }, [height, showVolume, theme, chartType])

  // Generate mock real-time data
  const generateRealtimeData = useCallback(() => {
    const now = Math.floor(Date.now() / 1000) as Time
    const basePrice = lastPriceRef.current || (150 + Math.random() * 100)
    
    // Generate realistic price movement
    const volatility = 0.02
    const trend = (Math.random() - 0.5) * 0.001
    const noise = (Math.random() - 0.5) * volatility
    const priceChange = basePrice * (trend + noise)
    
    const newPrice = Math.max(1, basePrice + priceChange)
    const open = lastPriceRef.current || newPrice
    const high = Math.max(open, newPrice) + Math.random() * basePrice * 0.005
    const low = Math.min(open, newPrice) - Math.random() * basePrice * 0.005
    const close = newPrice
    
    const newVolume = Math.floor(Math.random() * 1000000) + 100000
    
    const priceData: RealtimePriceData = {
      time: now,
      open,
      high,
      low,
      close,
      volume: newVolume,
    }
    
    const volumeData: RealtimeVolumeData = {
      time: now,
      value: newVolume,
      color: close >= open ? "#10B98180" : "#EF444480",
    }

    // Update refs
    lastPriceRef.current = close
    priceDataRef.current = [...priceDataRef.current, priceData].slice(-maxDataPoints)
    volumeDataRef.current = [...volumeDataRef.current, volumeData].slice(-maxDataPoints)

    // Update chart series
    if (chartType === "candlestick" && candlestickSeries.current) {
      candlestickSeries.current.update(priceData)
    } else if (chartType === "line" && lineSeries.current) {
      lineSeries.current.update({ time: now, value: close })
    }
    
    if (volumeSeries.current && showVolume) {
      volumeSeries.current.update(volumeData)
    }

    // Update state
    setCurrentPrice(close)
    if (priceDataRef.current.length > 1) {
      const previousClose = priceDataRef.current[priceDataRef.current.length - 2].close
      const change = close - previousClose
      const changePercent = (change / previousClose) * 100
      setPriceChange(change)
      setPriceChangePercent(changePercent)
    }
    setVolume(newVolume)
    setDataCount(priceDataRef.current.length)

    // Call callbacks
    onPriceUpdate?.(priceData)
    onVolumeUpdate?.(volumeData)
  }, [chartType, showVolume, maxDataPoints, onPriceUpdate, onVolumeUpdate])

  // Start/stop streaming
  useEffect(() => {
    if (isStreaming) {
      setIsConnected(true)
      streamingIntervalRef.current = setInterval(generateRealtimeData, updateInterval)
    } else {
      setIsConnected(false)
      if (streamingIntervalRef.current) {
        clearInterval(streamingIntervalRef.current)
        streamingIntervalRef.current = null
      }
    }

    return () => {
      if (streamingIntervalRef.current) {
        clearInterval(streamingIntervalRef.current)
      }
    }
  }, [isStreaming, updateInterval, generateRealtimeData])

  const toggleStreaming = () => {
    setIsStreaming(!isStreaming)
  }

  const clearChart = () => {
    priceDataRef.current = []
    volumeDataRef.current = []
    lastPriceRef.current = null
    setCurrentPrice(null)
    setPriceChange(null)
    setPriceChangePercent(null)
    setVolume(null)
    setDataCount(0)
    
    // Clear series data
    if (candlestickSeries.current) {
      candlestickSeries.current.setData([])
    }
    if (lineSeries.current) {
      lineSeries.current.setData([])
    }
    if (volumeSeries.current) {
      volumeSeries.current.setData([])
    }
  }

  const handleFullscreen = () => {
    if (chartContainerRef.current?.requestFullscreen) {
      chartContainerRef.current.requestFullscreen()
    }
  }

  const handleChartTypeChange = (type: string) => {
    setChartType(type as "candlestick" | "line")
    
    // Recreate chart with new type - this would trigger the useEffect above
    // For now, we'll just update the state and let the effect handle it
  }

  return (
    <Card className="overflow-hidden">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <div className="flex items-center gap-4">
          <div>
            <CardTitle className="text-lg font-semibold flex items-center gap-2">
              {symbol}
              {isConnected ? (
                <Wifi className="h-4 w-4 text-green-600" />
              ) : (
                <WifiOff className="h-4 w-4 text-red-600" />
              )}
            </CardTitle>
            {currentPrice && (
              <div className="flex items-center gap-2 mt-1">
                <span className="text-xl font-bold">
                  ${currentPrice.toFixed(2)}
                </span>
                {priceChange !== null && priceChangePercent !== null && (
                  <div className={cn(
                    "flex items-center gap-1 text-sm font-medium",
                    priceChange >= 0 ? "text-green-600" : "text-red-600"
                  )}>
                    {priceChange >= 0 ? (
                      <TrendingUp className="h-4 w-4" />
                    ) : (
                      <TrendingUp className="h-4 w-4 rotate-180" />
                    )}
                    {priceChange >= 0 ? "+" : ""}{priceChange.toFixed(2)} 
                    ({priceChangePercent >= 0 ? "+" : ""}{priceChangePercent.toFixed(2)}%)
                  </div>
                )}
              </div>
            )}
          </div>
          
          <div className="flex items-center gap-2">
            <Badge variant={isStreaming ? "default" : "secondary"} className="animate-pulse">
              {isStreaming ? "LIVE" : "PAUSED"}
            </Badge>
            {volume && (
              <Badge variant="outline">
                Vol: {(volume / 1000000).toFixed(2)}M
              </Badge>
            )}
            <Badge variant="outline">
              {dataCount} pts
            </Badge>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <Select value={chartType} onValueChange={handleChartTypeChange}>
            <SelectTrigger className="w-32">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="candlestick">Candlestick</SelectItem>
              <SelectItem value="line">Line</SelectItem>
            </SelectContent>
          </Select>
          
          <Button
            variant="outline"
            size="icon"
            onClick={toggleStreaming}
            className={cn(
              isStreaming ? "text-red-600 hover:text-red-700" : "text-green-600 hover:text-green-700"
            )}
          >
            {isStreaming ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
          </Button>
          
          <Button
            variant="outline"
            size="icon"
            onClick={clearChart}
          >
            <Square className="h-4 w-4" />
          </Button>
          
          <Button variant="ghost" size="icon" onClick={handleFullscreen}>
            <Maximize2 className="h-4 w-4" />
          </Button>
        </div>
      </CardHeader>
      
      <CardContent className="p-0">
        <div ref={chartContainerRef} className="w-full" />
        
        {/* Real-time status bar */}
        <div className="flex items-center justify-between p-2 bg-muted/50 text-xs text-muted-foreground">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-1">
              <Activity className="h-3 w-3" />
              <span>Status: {isConnected ? "Connected" : "Disconnected"}</span>
            </div>
            <div>
              Update Rate: {updateInterval}ms
            </div>
            <div>
              Max Points: {maxDataPoints}
            </div>
          </div>
          
          <div className="flex items-center gap-4">
            {currentPrice && (
              <>
                <div>
                  Last: ${currentPrice.toFixed(2)}
                </div>
                <div>
                  Time: {new Date().toLocaleTimeString()}
                </div>
              </>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}