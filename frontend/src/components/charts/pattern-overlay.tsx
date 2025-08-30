"use client"

import { useEffect } from "react"
import { IChartApi, ISeriesApi, Time } from "lightweight-charts"

export interface PatternData {
  id: string
  type: "triangle" | "flag" | "wedge" | "channel" | "head-shoulders" | "double-top" | "double-bottom"
  startTime: Time
  endTime: Time
  startPrice: number
  endPrice: number
  confidence: number
  direction: "bullish" | "bearish" | "neutral"
  points?: { time: Time; price: number }[]
}

interface PatternOverlayProps {
  chart: IChartApi | null
  patterns: PatternData[]
  showLabels?: boolean
  highlightPattern?: string | null
}

export function PatternOverlay({ 
  chart, 
  patterns, 
  showLabels = true,
  highlightPattern
}: PatternOverlayProps) {
  useEffect(() => {
    if (!chart) return

    const markers: any[] = []
    const lines: ISeriesApi<"Line">[] = []

    patterns.forEach((pattern) => {
      const isHighlighted = pattern.id === highlightPattern
      const opacity = isHighlighted ? 1 : 0.6
      const lineWidth = isHighlighted ? 3 : 2

      // Color based on direction
      const color = pattern.direction === "bullish" 
        ? `rgba(16, 185, 129, ${opacity})`
        : pattern.direction === "bearish"
        ? `rgba(239, 68, 68, ${opacity})`
        : `rgba(107, 114, 128, ${opacity})`

      // Draw pattern outline
      if (pattern.points && pattern.points.length > 0) {
        const lineSeries = chart.addLineSeries({
          color: color,
          lineWidth: lineWidth,
          lineStyle: 2, // Dashed line
          crosshairMarkerVisible: false,
          priceLineVisible: false,
          lastValueVisible: false,
        })
        lineSeries.setData(pattern.points)
        lines.push(lineSeries)
      }

      // Add pattern markers
      if (showLabels) {
        markers.push({
          time: pattern.startTime,
          position: "aboveBar",
          color: color,
          shape: "circle",
          text: `${pattern.type} (${Math.round(pattern.confidence * 100)}%)`,
        })
      }
    })

    // Apply markers to the chart
    if (markers.length > 0 && chart) {
      const firstSeries = chart.getSeries()[0]
      if (firstSeries) {
        firstSeries.setMarkers(markers)
      }
    }

    // Cleanup function
    return () => {
      lines.forEach(line => chart.removeSeries(line))
    }
  }, [chart, patterns, showLabels, highlightPattern])

  return null
}

// Helper function to generate pattern points
export function generatePatternPoints(pattern: PatternData): { time: Time; price: number }[] {
  const points = []
  
  switch (pattern.type) {
    case "triangle":
      // Generate triangle pattern points
      points.push(
        { time: pattern.startTime, price: pattern.startPrice },
        { time: pattern.endTime, price: pattern.endPrice },
      )
      break
      
    case "flag":
      // Generate flag pattern points
      const flagMidTime = ((pattern.startTime as number) + (pattern.endTime as number)) / 2 as Time
      const flagMidPrice = (pattern.startPrice + pattern.endPrice) / 2
      points.push(
        { time: pattern.startTime, price: pattern.startPrice },
        { time: flagMidTime, price: flagMidPrice + 2 },
        { time: pattern.endTime, price: pattern.endPrice },
      )
      break
      
    case "head-shoulders":
      // Generate head and shoulders pattern points
      const hsQuarter = ((pattern.endTime as number) - (pattern.startTime as number)) / 4
      const time1 = (pattern.startTime as number) + hsQuarter as Time
      const time2 = (pattern.startTime as number) + hsQuarter * 2 as Time
      const time3 = (pattern.startTime as number) + hsQuarter * 3 as Time
      
      points.push(
        { time: pattern.startTime, price: pattern.startPrice },
        { time: time1, price: pattern.startPrice + 5 },
        { time: time2, price: pattern.startPrice + 10 },
        { time: time3, price: pattern.startPrice + 5 },
        { time: pattern.endTime, price: pattern.endPrice },
      )
      break
      
    case "double-top":
      // Generate double top pattern points
      const dtMid = ((pattern.startTime as number) + (pattern.endTime as number)) / 2 as Time
      const peak = Math.max(pattern.startPrice, pattern.endPrice) + 5
      points.push(
        { time: pattern.startTime, price: pattern.startPrice },
        { time: (pattern.startTime as number) + ((dtMid as number) - (pattern.startTime as number)) / 2 as Time, price: peak },
        { time: dtMid, price: pattern.startPrice - 2 },
        { time: (dtMid as number) + ((pattern.endTime as number) - (dtMid as number)) / 2 as Time, price: peak },
        { time: pattern.endTime, price: pattern.endPrice },
      )
      break
      
    case "double-bottom":
      // Generate double bottom pattern points
      const dbMid = ((pattern.startTime as number) + (pattern.endTime as number)) / 2 as Time
      const trough = Math.min(pattern.startPrice, pattern.endPrice) - 5
      points.push(
        { time: pattern.startTime, price: pattern.startPrice },
        { time: (pattern.startTime as number) + ((dbMid as number) - (pattern.startTime as number)) / 2 as Time, price: trough },
        { time: dbMid, price: pattern.startPrice + 2 },
        { time: (dbMid as number) + ((pattern.endTime as number) - (dbMid as number)) / 2 as Time, price: trough },
        { time: pattern.endTime, price: pattern.endPrice },
      )
      break
      
    default:
      // Default line pattern
      points.push(
        { time: pattern.startTime, price: pattern.startPrice },
        { time: pattern.endTime, price: pattern.endPrice },
      )
  }
  
  return points
}