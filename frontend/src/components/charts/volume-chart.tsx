"use client"

import { useEffect, useRef } from "react"
import { createChart, ColorType, IChartApi, Time } from "lightweight-charts"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useTheme } from "next-themes"

interface VolumeData {
  time: Time
  value: number
  color?: string
}

interface VolumeChartProps {
  data?: VolumeData[]
  height?: number
  title?: string
}

export function VolumeChart({ 
  data = generateMockVolumeData(),
  height = 200,
  title = "Volume"
}: VolumeChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null)
  const chart = useRef<IChartApi | null>(null)
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
      rightPriceScale: {
        borderColor: theme === "dark" ? "#374151" : "#D1D5DB",
        scaleMargins: {
          top: 0.1,
          bottom: 0.1,
        },
      },
      timeScale: {
        borderColor: theme === "dark" ? "#374151" : "#D1D5DB",
        timeVisible: true,
        secondsVisible: false,
      },
    })

    // Add histogram series for volume
    const volumeSeries = chart.current.addHistogramSeries({
      color: "#3B82F680",
      priceFormat: {
        type: "volume",
      },
    })

    // Set data
    volumeSeries.setData(data)

    // Fit content
    chart.current.timeScale().fitContent()

    // Handle resize
    window.addEventListener("resize", handleResize)

    return () => {
      window.removeEventListener("resize", handleResize)
      chart.current?.remove()
    }
  }, [data, height, theme])

  return (
    <Card className="overflow-hidden">
      <CardHeader className="pb-2">
        <CardTitle className="text-base font-medium">{title}</CardTitle>
      </CardHeader>
      <CardContent className="p-0">
        <div ref={chartContainerRef} className="w-full" />
      </CardContent>
    </Card>
  )
}

// Generate mock volume data for demonstration
function generateMockVolumeData(): VolumeData[] {
  const data: VolumeData[] = []
  const now = Math.floor(Date.now() / 1000)
  
  for (let i = 0; i < 100; i++) {
    const time = (now - (100 - i) * 3600) as Time
    const value = Math.floor(Math.random() * 1000000) + 100000
    const isPositive = Math.random() > 0.5
    const color = isPositive ? "#10B98180" : "#EF444480"
    
    data.push({ time, value, color })
  }
  
  return data
}