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
  Bitcoin,
} from "lucide-react"
import { useTheme } from "next-themes"
import { cn } from "@/lib/utils"
import { cryptoAPI, CryptoKline, DEFAULT_CRYPTO_PAIRS, BINANCE_INTERVALS } from "@/lib/crypto-api"

export interface CryptoPriceData {
  time: Time
  open: number
  high: number
  low: number
  close: number
  volume?: number
}

export interface CryptoVolumeData {
  time: Time
  value: number
  color?: string
}

interface CryptoRealtimeChartProps {
  symbol: string
  height?: number
  showVolume?: boolean
  showPatterns?: boolean
  maxDataPoints?: number
  interval?: string
  onPriceUpdate?: (data: CryptoPriceData) => void
  onVolumeUpdate?: (data: CryptoVolumeData) => void
}

export function CryptoRealtimeChart({
  symbol,
  height = 500,
  showVolume = true,
  showPatterns = true,
  maxDataPoints = 1000,
  interval = "1m",
  onPriceUpdate,
  onVolumeUpdate,
}: CryptoRealtimeChartProps) {
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
  const [selectedInterval, setSelectedInterval] = useState(interval)
  
  const priceDataRef = useRef<CryptoPriceData[]>([])
  const volumeDataRef = useRef<CryptoVolumeData[]>([])
  const lastPriceRef = useRef<number | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  
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
        secondsVisible: selectedInterval.endsWith('m'),
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
        color: "#F59E0B",
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
  }, [height, showVolume, theme, chartType, selectedInterval])

  // Load historical data and set up WebSocket
  useEffect(() => {
    if (!symbol || !isStreaming) return

    const loadHistoricalData = async () => {
      try {
        console.log(`Loading historical data for ${symbol} (${selectedInterval})`)
        
        // Get historical kline data
        const klines = await cryptoAPI.getKlineData(symbol, selectedInterval, 500)
        
        if (klines.length === 0) {
          console.warn(`No historical data received for ${symbol}`)
          return
        }

        // Convert to chart format
        const chartData = klines.map(kline => ({
          time: Math.floor(new Date(kline.open_time).getTime() / 1000) as Time,
          open: kline.open_price,
          high: kline.high_price,
          low: kline.low_price,
          close: kline.close_price,
          volume: kline.volume
        }))

        // Set data to chart
        if (chartType === "candlestick" && candlestickSeries.current) {
          candlestickSeries.current.setData(chartData)
        } else if (chartType === "line" && lineSeries.current) {
          const lineData = chartData.map(d => ({ time: d.time, value: d.close }))
          lineSeries.current.setData(lineData)
        }

        if (showVolume && volumeSeries.current) {
          const volumeData = chartData.map(d => ({
            time: d.time,
            value: d.volume || 0,
            color: d.close >= d.open ? "#10B98180" : "#EF444480",
          }))
          volumeSeries.current.setData(volumeData)
        }

        // Update refs and state
        priceDataRef.current = chartData
        const lastKline = klines[klines.length - 1]
        if (lastKline) {
          setCurrentPrice(lastKline.close_price)
          lastPriceRef.current = lastKline.close_price
          setDataCount(chartData.length)
          
          if (klines.length > 1) {
            const prevClose = klines[klines.length - 2].close_price
            const change = lastKline.close_price - prevClose
            const changePercent = (change / prevClose) * 100
            setPriceChange(change)
            setPriceChangePercent(changePercent)
          }
        }

        // Fit content
        chart.current?.timeScale().fitContent()

        console.log(`Loaded ${chartData.length} data points for ${symbol}`)
        
      } catch (error) {
        console.error(`Error loading historical data for ${symbol}:`, error)
      }
    }

    const connectWebSocket = () => {
      try {
        const streamName = cryptoAPI.createKlineStreamName(symbol, selectedInterval)
        const wsUrl = `${cryptoAPI.getWebSocketUrl()}/ws/${streamName}`
        
        console.log(`Connecting to Binance kline WebSocket: ${wsUrl}`)
        
        const ws = new WebSocket(wsUrl)
        wsRef.current = ws
        
        ws.onopen = () => {
          console.log(`Connected to Binance kline stream for ${symbol}`)
          setIsConnected(true)
        }
        
        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data)
            
            if (data.k) {
              const kline = data.k
              const newData: CryptoPriceData = {
                time: Math.floor(kline.t / 1000) as Time,
                open: parseFloat(kline.o),
                high: parseFloat(kline.h),
                low: parseFloat(kline.l),
                close: parseFloat(kline.c),
                volume: parseFloat(kline.v)
              }
              
              // Update chart
              if (chartType === "candlestick" && candlestickSeries.current) {
                candlestickSeries.current.update(newData)
              } else if (chartType === "line" && lineSeries.current) {
                lineSeries.current.update({ time: newData.time, value: newData.close })
              }
              
              if (showVolume && volumeSeries.current) {
                const volumeData: CryptoVolumeData = {
                  time: newData.time,
                  value: newData.volume || 0,
                  color: newData.close >= newData.open ? "#10B98180" : "#EF444480",
                }
                volumeSeries.current.update(volumeData)
                setVolume(volumeData.value)
                onVolumeUpdate?.(volumeData)
              }
              
              // Update state
              setCurrentPrice(newData.close)
              if (lastPriceRef.current) {
                const change = newData.close - lastPriceRef.current
                const changePercent = (change / lastPriceRef.current) * 100
                setPriceChange(change)
                setPriceChangePercent(changePercent)
              }
              lastPriceRef.current = newData.close
              
              // Update data ref if kline is closed
              if (kline.x) {
                priceDataRef.current = [...priceDataRef.current, newData].slice(-maxDataPoints)
                setDataCount(priceDataRef.current.length)
              }
              
              onPriceUpdate?.(newData)
            }
          } catch (error) {
            console.error('Error parsing kline WebSocket message:', error)
          }
        }
        
        ws.onerror = (error) => {
          console.error('Kline WebSocket error:', error)
          setIsConnected(false)
        }
        
        ws.onclose = (event) => {
          console.log('Kline WebSocket closed:', event.code, event.reason)
          setIsConnected(false)
          
          // Attempt to reconnect if not manually closed
          if (isStreaming && event.code !== 1000) {
            setTimeout(connectWebSocket, 3000)
          }
        }
        
      } catch (error) {
        console.error('Error creating kline WebSocket connection:', error)
        setIsConnected(false)
      }
    }

    // Load historical data first, then connect WebSocket
    loadHistoricalData().then(() => {
      if (isStreaming) {
        connectWebSocket()
      }
    })

    return () => {
      if (wsRef.current) {
        wsRef.current.close(1000, 'Component cleanup')
        wsRef.current = null
      }
      setIsConnected(false)
    }
  }, [symbol, selectedInterval, isStreaming, chartType, showVolume, maxDataPoints, onPriceUpdate, onVolumeUpdate])

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
  }

  const handleIntervalChange = (newInterval: string) => {
    setSelectedInterval(newInterval)
  }

  return (
    <Card className="overflow-hidden">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <div className="flex items-center gap-4">
          <div>
            <CardTitle className="text-lg font-semibold flex items-center gap-2">
              <Bitcoin className="h-5 w-5" />
              {cryptoAPI.formatSymbol(symbol)}
              {isConnected ? (
                <Wifi className="h-4 w-4 text-green-600" />
              ) : (
                <WifiOff className="h-4 w-4 text-red-600" />
              )}
            </CardTitle>
            {currentPrice && (
              <div className="flex items-center gap-2 mt-1">
                <span className="text-xl font-bold">
                  ${cryptoAPI.formatPrice(currentPrice, symbol)}
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
                    {priceChange >= 0 ? "+" : ""}{cryptoAPI.formatPrice(priceChange, symbol)} 
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
                Vol: {cryptoAPI.formatVolume(volume)}
              </Badge>
            )}
            <Badge variant="outline">
              {dataCount} pts
            </Badge>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <Select value={selectedInterval} onValueChange={handleIntervalChange}>
            <SelectTrigger className="w-20">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {BINANCE_INTERVALS.map(int => (
                <SelectItem key={int} value={int}>
                  {int}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

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
              <span>Binance: {isConnected ? "Connected" : "Disconnected"}</span>
            </div>
            <div>
              Interval: {selectedInterval}
            </div>
            <div>
              Max Points: {maxDataPoints}
            </div>
          </div>
          
          <div className="flex items-center gap-4">
            {currentPrice && (
              <>
                <div>
                  Last: ${cryptoAPI.formatPrice(currentPrice, symbol)}
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