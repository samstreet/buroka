"use client"

import { useEffect, useRef, useState } from "react"
import { createChart, ColorType, IChartApi, ISeriesApi, Time, CrosshairMode } from "lightweight-charts"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { 
  Select, 
  SelectContent, 
  SelectItem, 
  SelectTrigger, 
  SelectValue 
} from "@/components/ui/select"
import { Maximize2, TrendingUp, BarChart3, Activity } from "lucide-react"
import { useTheme } from "next-themes"

interface PriceData {
  time: Time
  open: number
  high: number
  low: number
  close: number
  volume?: number
}

interface PriceChartProps {
  data?: PriceData[]
  symbol?: string
  height?: number
  showVolume?: boolean
  showIndicators?: boolean
}

const timeframes = [
  { value: "1m", label: "1 Min" },
  { value: "5m", label: "5 Min" },
  { value: "15m", label: "15 Min" },
  { value: "30m", label: "30 Min" },
  { value: "1h", label: "1 Hour" },
  { value: "4h", label: "4 Hours" },
  { value: "1d", label: "1 Day" },
  { value: "1w", label: "1 Week" },
]

const chartTypes = [
  { value: "candlestick", label: "Candlestick", icon: TrendingUp },
  { value: "line", label: "Line", icon: Activity },
  { value: "bar", label: "Bar", icon: BarChart3 },
]

export function PriceChart({ 
  data = generateMockData(),
  symbol = "AAPL",
  height = 500,
  showVolume = true,
  showIndicators = true
}: PriceChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null)
  const chart = useRef<IChartApi | null>(null)
  const candlestickSeries = useRef<ISeriesApi<"Candlestick"> | null>(null)
  const volumeSeries = useRef<ISeriesApi<"Histogram"> | null>(null)
  const [selectedTimeframe, setSelectedTimeframe] = useState("1h")
  const [selectedChartType, setSelectedChartType] = useState("candlestick")
  const { theme } = useTheme()

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
      crosshair: {
        mode: CrosshairMode.Normal,
      },
      rightPriceScale: {
        borderColor: theme === "dark" ? "#374151" : "#D1D5DB",
      },
      timeScale: {
        borderColor: theme === "dark" ? "#374151" : "#D1D5DB",
        timeVisible: true,
        secondsVisible: false,
      },
    })

    // Add candlestick series
    candlestickSeries.current = chart.current.addCandlestickSeries({
      upColor: "#10B981",
      downColor: "#EF4444",
      borderUpColor: "#10B981",
      borderDownColor: "#EF4444",
      wickUpColor: "#10B981",
      wickDownColor: "#EF4444",
    })

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

    // Set data
    candlestickSeries.current.setData(data)
    
    if (showVolume && volumeSeries.current) {
      const volumeData = data.map(d => ({
        time: d.time,
        value: d.volume || Math.random() * 1000000,
        color: d.close >= d.open ? "#10B98180" : "#EF444480",
      }))
      volumeSeries.current.setData(volumeData)
    }

    // Fit content
    chart.current.timeScale().fitContent()

    // Handle resize
    window.addEventListener("resize", handleResize)

    return () => {
      window.removeEventListener("resize", handleResize)
      chart.current?.remove()
    }
  }, [data, height, showVolume, theme])

  const handleTimeframeChange = (timeframe: string) => {
    setSelectedTimeframe(timeframe)
    // Here you would fetch new data based on timeframe
    // For now, we'll just update the chart with mock data
    const newData = generateMockData()
    candlestickSeries.current?.setData(newData)
    chart.current?.timeScale().fitContent()
  }

  const handleChartTypeChange = (type: string) => {
    setSelectedChartType(type)
    // Here you would change the series type
    // For simplicity, we're keeping it as candlestick
  }

  const handleFullscreen = () => {
    if (chartContainerRef.current?.requestFullscreen) {
      chartContainerRef.current.requestFullscreen()
    }
  }

  const addSMA = () => {
    if (!chart.current || !data) return

    const sma = calculateSMA(data, 20)
    const lineSeries = chart.current.addLineSeries({
      color: "#3B82F6",
      lineWidth: 2,
      title: "SMA 20",
    })
    lineSeries.setData(sma)
  }

  const addEMA = () => {
    if (!chart.current || !data) return

    const ema = calculateEMA(data, 20)
    const lineSeries = chart.current.addLineSeries({
      color: "#8B5CF6",
      lineWidth: 2,
      title: "EMA 20",
    })
    lineSeries.setData(ema)
  }

  const addBollingerBands = () => {
    if (!chart.current || !data) return

    const { upper, middle, lower } = calculateBollingerBands(data, 20, 2)
    
    const upperBand = chart.current.addLineSeries({
      color: "#F59E0B",
      lineWidth: 1,
      title: "BB Upper",
    })
    upperBand.setData(upper)

    const middleBand = chart.current.addLineSeries({
      color: "#6B7280",
      lineWidth: 1,
      title: "BB Middle",
    })
    middleBand.setData(middle)

    const lowerBand = chart.current.addLineSeries({
      color: "#F59E0B",
      lineWidth: 1,
      title: "BB Lower",
    })
    lowerBand.setData(lower)
  }

  return (
    <Card className="overflow-hidden">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <div className="flex items-center gap-4">
          <CardTitle className="text-lg font-semibold">{symbol}</CardTitle>
          <Select value={selectedTimeframe} onValueChange={handleTimeframeChange}>
            <SelectTrigger className="w-24">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {timeframes.map((tf) => (
                <SelectItem key={tf.value} value={tf.value}>
                  {tf.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="flex items-center gap-2">
          {showIndicators && (
            <>
              <Button variant="outline" size="sm" onClick={addSMA}>
                SMA
              </Button>
              <Button variant="outline" size="sm" onClick={addEMA}>
                EMA
              </Button>
              <Button variant="outline" size="sm" onClick={addBollingerBands}>
                BB
              </Button>
            </>
          )}
          <Button variant="ghost" size="icon" onClick={handleFullscreen}>
            <Maximize2 className="h-4 w-4" />
          </Button>
        </div>
      </CardHeader>
      <CardContent className="p-0">
        <div ref={chartContainerRef} className="w-full" />
      </CardContent>
    </Card>
  )
}

// Helper functions for indicators
function calculateSMA(data: PriceData[], period: number) {
  const sma = []
  for (let i = period - 1; i < data.length; i++) {
    let sum = 0
    for (let j = 0; j < period; j++) {
      sum += data[i - j].close
    }
    sma.push({
      time: data[i].time,
      value: sum / period,
    })
  }
  return sma
}

function calculateEMA(data: PriceData[], period: number) {
  const ema = []
  const multiplier = 2 / (period + 1)
  
  // Start with SMA for first value
  let sum = 0
  for (let i = 0; i < period; i++) {
    sum += data[i].close
  }
  let previousEMA = sum / period
  ema.push({ time: data[period - 1].time, value: previousEMA })

  // Calculate EMA for rest
  for (let i = period; i < data.length; i++) {
    const currentEMA = (data[i].close - previousEMA) * multiplier + previousEMA
    ema.push({ time: data[i].time, value: currentEMA })
    previousEMA = currentEMA
  }
  return ema
}

function calculateBollingerBands(data: PriceData[], period: number, stdDev: number) {
  const middle = calculateSMA(data, period)
  const upper = []
  const lower = []

  for (let i = 0; i < middle.length; i++) {
    const dataIndex = i + period - 1
    let sumSquaredDiff = 0
    
    for (let j = 0; j < period; j++) {
      const diff = data[dataIndex - j].close - middle[i].value
      sumSquaredDiff += diff * diff
    }
    
    const std = Math.sqrt(sumSquaredDiff / period)
    upper.push({ time: middle[i].time, value: middle[i].value + std * stdDev })
    lower.push({ time: middle[i].time, value: middle[i].value - std * stdDev })
  }

  return { upper, middle, lower }
}

// Generate mock data for demonstration
function generateMockData(): PriceData[] {
  const data: PriceData[] = []
  const basePrice = 150
  const now = Math.floor(Date.now() / 1000)
  
  for (let i = 0; i < 100; i++) {
    const time = (now - (100 - i) * 3600) as Time
    const volatility = 0.02
    const random = Math.random() - 0.5
    const open = basePrice + random * basePrice * volatility
    const close = open + (Math.random() - 0.5) * basePrice * volatility
    const high = Math.max(open, close) + Math.random() * basePrice * volatility * 0.5
    const low = Math.min(open, close) - Math.random() * basePrice * volatility * 0.5
    const volume = Math.floor(Math.random() * 1000000)
    
    data.push({ time, open, high, low, close, volume })
  }
  
  return data
}