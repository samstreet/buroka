"use client"

import { useState, useMemo } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
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
import {
  TrendingUp,
  TrendingDown,
  AlertCircle,
  Eye,
  Filter,
  Search,
  ChevronUp,
  ChevronDown,
  Minus,
} from "lucide-react"

export interface Pattern {
  id: string
  symbol: string
  type: string
  timeframe: string
  confidence: number
  direction: "bullish" | "bearish" | "neutral"
  startTime: Date
  endTime: Date
  priceAtDetection: number
  currentPrice: number
  priceChange: number
  priceChangePercent: number
  volume: number
  status: "active" | "completed" | "failed"
  accuracy?: number
  successRate?: number
}

interface PatternListProps {
  patterns: Pattern[]
  onPatternClick?: (pattern: Pattern) => void
  onAlertClick?: (pattern: Pattern) => void
  showActions?: boolean
}

type SortField = "confidence" | "priceChange" | "volume" | "time"
type SortOrder = "asc" | "desc"

export function PatternList({
  patterns,
  onPatternClick,
  onAlertClick,
  showActions = true,
}: PatternListProps) {
  const [searchQuery, setSearchQuery] = useState("")
  const [selectedType, setSelectedType] = useState<string>("all")
  const [selectedDirection, setSelectedDirection] = useState<string>("all")
  const [selectedTimeframe, setSelectedTimeframe] = useState<string>("all")
  const [selectedStatus, setSelectedStatus] = useState<string>("all")
  const [sortField, setSortField] = useState<SortField>("time")
  const [sortOrder, setSortOrder] = useState<SortOrder>("desc")

  // Get unique values for filters
  const patternTypes = useMemo(() => {
    const types = new Set(patterns.map(p => p.type))
    return Array.from(types).sort()
  }, [patterns])

  const timeframes = useMemo(() => {
    const tf = new Set(patterns.map(p => p.timeframe))
    return Array.from(tf).sort()
  }, [patterns])

  // Filter patterns
  const filteredPatterns = useMemo(() => {
    let filtered = patterns

    // Search filter
    if (searchQuery) {
      filtered = filtered.filter(p =>
        p.symbol.toLowerCase().includes(searchQuery.toLowerCase()) ||
        p.type.toLowerCase().includes(searchQuery.toLowerCase())
      )
    }

    // Type filter
    if (selectedType !== "all") {
      filtered = filtered.filter(p => p.type === selectedType)
    }

    // Direction filter
    if (selectedDirection !== "all") {
      filtered = filtered.filter(p => p.direction === selectedDirection)
    }

    // Timeframe filter
    if (selectedTimeframe !== "all") {
      filtered = filtered.filter(p => p.timeframe === selectedTimeframe)
    }

    // Status filter
    if (selectedStatus !== "all") {
      filtered = filtered.filter(p => p.status === selectedStatus)
    }

    // Sort
    filtered.sort((a, b) => {
      let aVal: any, bVal: any

      switch (sortField) {
        case "confidence":
          aVal = a.confidence
          bVal = b.confidence
          break
        case "priceChange":
          aVal = a.priceChangePercent
          bVal = b.priceChangePercent
          break
        case "volume":
          aVal = a.volume
          bVal = b.volume
          break
        case "time":
          aVal = a.startTime.getTime()
          bVal = b.startTime.getTime()
          break
      }

      if (sortOrder === "asc") {
        return aVal - bVal
      } else {
        return bVal - aVal
      }
    })

    return filtered
  }, [patterns, searchQuery, selectedType, selectedDirection, selectedTimeframe, selectedStatus, sortField, sortOrder])

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortOrder(sortOrder === "asc" ? "desc" : "asc")
    } else {
      setSortField(field)
      setSortOrder("desc")
    }
  }

  const getDirectionIcon = (direction: string) => {
    switch (direction) {
      case "bullish":
        return <TrendingUp className="h-4 w-4 text-green-600" />
      case "bearish":
        return <TrendingDown className="h-4 w-4 text-red-600" />
      default:
        return <Minus className="h-4 w-4 text-gray-600" />
    }
  }

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "active":
        return <Badge variant="default">Active</Badge>
      case "completed":
        return <Badge variant="secondary">Completed</Badge>
      case "failed":
        return <Badge variant="destructive">Failed</Badge>
      default:
        return <Badge variant="outline">{status}</Badge>
    }
  }

  const getSortIcon = (field: SortField) => {
    if (sortField !== field) return null
    return sortOrder === "asc" ? 
      <ChevronUp className="h-4 w-4" /> : 
      <ChevronDown className="h-4 w-4" />
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>Detected Patterns</CardTitle>
            <CardDescription>
              {filteredPatterns.length} patterns found
            </CardDescription>
          </div>
          <div className="flex items-center gap-2">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                type="search"
                placeholder="Search patterns..."
                className="pl-10 w-64"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>
            <Button variant="outline" size="icon">
              <Filter className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {/* Filters */}
        <div className="mb-4 flex gap-2">
          <Select value={selectedType} onValueChange={setSelectedType}>
            <SelectTrigger className="w-[150px]">
              <SelectValue placeholder="Pattern Type" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Types</SelectItem>
              {patternTypes.map(type => (
                <SelectItem key={type} value={type}>
                  {type}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <Select value={selectedDirection} onValueChange={setSelectedDirection}>
            <SelectTrigger className="w-[130px]">
              <SelectValue placeholder="Direction" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All</SelectItem>
              <SelectItem value="bullish">Bullish</SelectItem>
              <SelectItem value="bearish">Bearish</SelectItem>
              <SelectItem value="neutral">Neutral</SelectItem>
            </SelectContent>
          </Select>

          <Select value={selectedTimeframe} onValueChange={setSelectedTimeframe}>
            <SelectTrigger className="w-[120px]">
              <SelectValue placeholder="Timeframe" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All</SelectItem>
              {timeframes.map(tf => (
                <SelectItem key={tf} value={tf}>
                  {tf}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <Select value={selectedStatus} onValueChange={setSelectedStatus}>
            <SelectTrigger className="w-[130px]">
              <SelectValue placeholder="Status" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All</SelectItem>
              <SelectItem value="active">Active</SelectItem>
              <SelectItem value="completed">Completed</SelectItem>
              <SelectItem value="failed">Failed</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Pattern Table */}
        <div className="rounded-md border">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Symbol</TableHead>
                <TableHead>Pattern</TableHead>
                <TableHead>Direction</TableHead>
                <TableHead>Timeframe</TableHead>
                <TableHead 
                  className="cursor-pointer"
                  onClick={() => handleSort("confidence")}
                >
                  <div className="flex items-center gap-1">
                    Confidence
                    {getSortIcon("confidence")}
                  </div>
                </TableHead>
                <TableHead 
                  className="cursor-pointer"
                  onClick={() => handleSort("priceChange")}
                >
                  <div className="flex items-center gap-1">
                    Price Change
                    {getSortIcon("priceChange")}
                  </div>
                </TableHead>
                <TableHead 
                  className="cursor-pointer"
                  onClick={() => handleSort("volume")}
                >
                  <div className="flex items-center gap-1">
                    Volume
                    {getSortIcon("volume")}
                  </div>
                </TableHead>
                <TableHead>Status</TableHead>
                {showActions && <TableHead>Actions</TableHead>}
              </TableRow>
            </TableHeader>
            <TableBody>
              {filteredPatterns.map((pattern) => (
                <TableRow 
                  key={pattern.id}
                  className="cursor-pointer hover:bg-accent"
                  onClick={() => onPatternClick?.(pattern)}
                >
                  <TableCell className="font-medium">{pattern.symbol}</TableCell>
                  <TableCell>{pattern.type}</TableCell>
                  <TableCell>
                    <div className="flex items-center gap-2">
                      {getDirectionIcon(pattern.direction)}
                      <span className="capitalize">{pattern.direction}</span>
                    </div>
                  </TableCell>
                  <TableCell>{pattern.timeframe}</TableCell>
                  <TableCell>
                    <div className="flex items-center gap-2">
                      <div className="w-full max-w-[100px] bg-gray-200 rounded-full h-2">
                        <div 
                          className="bg-blue-600 h-2 rounded-full"
                          style={{ width: `${pattern.confidence * 100}%` }}
                        />
                      </div>
                      <span className="text-sm">
                        {Math.round(pattern.confidence * 100)}%
                      </span>
                    </div>
                  </TableCell>
                  <TableCell>
                    <span className={`font-medium ${
                      pattern.priceChangePercent >= 0 ? "text-green-600" : "text-red-600"
                    }`}>
                      {pattern.priceChangePercent >= 0 ? "+" : ""}
                      {pattern.priceChangePercent.toFixed(2)}%
                    </span>
                  </TableCell>
                  <TableCell>
                    {(pattern.volume / 1000000).toFixed(2)}M
                  </TableCell>
                  <TableCell>{getStatusBadge(pattern.status)}</TableCell>
                  {showActions && (
                    <TableCell onClick={(e) => e.stopPropagation()}>
                      <div className="flex gap-2">
                        <Button
                          variant="ghost"
                          size="icon"
                          onClick={() => onPatternClick?.(pattern)}
                        >
                          <Eye className="h-4 w-4" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="icon"
                          onClick={() => onAlertClick?.(pattern)}
                        >
                          <AlertCircle className="h-4 w-4" />
                        </Button>
                      </div>
                    </TableCell>
                  )}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </CardContent>
    </Card>
  )
}