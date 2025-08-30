"use client"

import { useState } from "react"
import { Pattern } from "./pattern-list"
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
  Plus,
  X,
  Search,
  BarChart3,
  Target,
  Clock,
  Minus,
} from "lucide-react"

interface PatternComparisonProps {
  availablePatterns: Pattern[]
  selectedPatterns?: Pattern[]
  onPatternsChange?: (patterns: Pattern[]) => void
  maxPatterns?: number
}

export function PatternComparison({
  availablePatterns,
  selectedPatterns = [],
  onPatternsChange,
  maxPatterns = 4,
}: PatternComparisonProps) {
  const [searchQuery, setSearchQuery] = useState("")
  const [showAddDialog, setShowAddDialog] = useState(false)

  const filteredAvailable = availablePatterns.filter(pattern =>
    !selectedPatterns.some(selected => selected.id === pattern.id) &&
    (pattern.symbol.toLowerCase().includes(searchQuery.toLowerCase()) ||
     pattern.type.toLowerCase().includes(searchQuery.toLowerCase()))
  )

  const addPattern = (pattern: Pattern) => {
    if (selectedPatterns.length < maxPatterns) {
      const newSelection = [...selectedPatterns, pattern]
      onPatternsChange?.(newSelection)
      setSearchQuery("")
      if (newSelection.length >= maxPatterns) {
        setShowAddDialog(false)
      }
    }
  }

  const removePattern = (patternId: string) => {
    const newSelection = selectedPatterns.filter(p => p.id !== patternId)
    onPatternsChange?.(newSelection)
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

  const getStatusColor = (status: string) => {
    switch (status) {
      case "active":
        return "text-green-600"
      case "completed":
        return "text-blue-600"
      case "failed":
        return "text-red-600"
      default:
        return "text-gray-600"
    }
  }

  const getComparisonWinner = (values: number[], index: number) => {
    const max = Math.max(...values)
    const min = Math.min(...values)
    if (values[index] === max && max !== min) return "winner"
    if (values[index] === min && max !== min) return "loser"
    return "neutral"
  }

  const getComparisonClass = (type: string) => {
    switch (type) {
      case "winner":
        return "bg-green-50 text-green-700 font-semibold"
      case "loser":
        return "bg-red-50 text-red-700"
      default:
        return ""
    }
  }

  if (selectedPatterns.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Pattern Comparison</CardTitle>
          <CardDescription>
            Select patterns to compare their performance and characteristics
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8">
            <BarChart3 className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
            <h3 className="text-lg font-medium mb-2">No Patterns Selected</h3>
            <p className="text-sm text-muted-foreground mb-4">
              Add patterns to start comparing their performance metrics
            </p>
            <Button onClick={() => setShowAddDialog(true)}>
              <Plus className="mr-2 h-4 w-4" />
              Add Pattern
            </Button>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header with controls */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Pattern Comparison</h2>
          <p className="text-muted-foreground">
            Comparing {selectedPatterns.length} patterns
          </p>
        </div>
        <div className="flex gap-2">
          {selectedPatterns.length < maxPatterns && (
            <Button 
              variant="outline" 
              onClick={() => setShowAddDialog(true)}
            >
              <Plus className="mr-2 h-4 w-4" />
              Add Pattern
            </Button>
          )}
        </div>
      </div>

      {/* Selected patterns overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {selectedPatterns.map((pattern) => (
          <Card key={pattern.id} className="relative">
            <Button
              variant="ghost"
              size="icon"
              className="absolute top-2 right-2 h-6 w-6"
              onClick={() => removePattern(pattern.id)}
            >
              <X className="h-4 w-4" />
            </Button>
            <CardHeader className="pb-2">
              <div className="flex items-center gap-2">
                <Badge variant="outline">{pattern.symbol}</Badge>
                {getDirectionIcon(pattern.direction)}
              </div>
            </CardHeader>
            <CardContent>
              <p className="font-medium mb-1">{pattern.type}</p>
              <p className="text-sm text-muted-foreground mb-2">
                {pattern.timeframe} • {Math.round(pattern.confidence * 100)}%
              </p>
              <p className={`text-lg font-bold ${
                pattern.priceChangePercent >= 0 ? "text-green-600" : "text-red-600"
              }`}>
                {pattern.priceChangePercent >= 0 ? "+" : ""}
                {pattern.priceChangePercent.toFixed(2)}%
              </p>
            </CardContent>
          </Card>
        ))}
        
        {/* Add pattern card */}
        {selectedPatterns.length < maxPatterns && (
          <Card 
            className="border-dashed cursor-pointer hover:bg-accent"
            onClick={() => setShowAddDialog(true)}
          >
            <CardContent className="flex items-center justify-center h-full min-h-[120px]">
              <div className="text-center">
                <Plus className="h-8 w-8 text-muted-foreground mx-auto mb-2" />
                <p className="text-sm text-muted-foreground">Add Pattern</p>
              </div>
            </CardContent>
          </Card>
        )}
      </div>

      {/* Comparison table */}
      <Card>
        <CardHeader>
          <CardTitle>Detailed Comparison</CardTitle>
          <CardDescription>
            Side-by-side comparison of key metrics
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Metric</TableHead>
                  {selectedPatterns.map(pattern => (
                    <TableHead key={pattern.id} className="text-center">
                      {pattern.symbol}
                      <br />
                      <span className="text-xs font-normal text-muted-foreground">
                        {pattern.type}
                      </span>
                    </TableHead>
                  ))}
                </TableRow>
              </TableHeader>
              <TableBody>
                {/* Confidence */}
                <TableRow>
                  <TableCell className="font-medium">Confidence</TableCell>
                  {selectedPatterns.map((pattern, index) => {
                    const confidences = selectedPatterns.map(p => p.confidence)
                    const winnerType = getComparisonWinner(confidences, index)
                    return (
                      <TableCell key={pattern.id} className={`text-center ${getComparisonClass(winnerType)}`}>
                        {Math.round(pattern.confidence * 100)}%
                      </TableCell>
                    )
                  })}
                </TableRow>

                {/* Price Change */}
                <TableRow>
                  <TableCell className="font-medium">Price Change</TableCell>
                  {selectedPatterns.map((pattern, index) => {
                    const changes = selectedPatterns.map(p => p.priceChangePercent)
                    const winnerType = getComparisonWinner(changes, index)
                    return (
                      <TableCell key={pattern.id} className={`text-center ${getComparisonClass(winnerType)}`}>
                        <span className={
                          pattern.priceChangePercent >= 0 ? "text-green-600" : "text-red-600"
                        }>
                          {pattern.priceChangePercent >= 0 ? "+" : ""}
                          {pattern.priceChangePercent.toFixed(2)}%
                        </span>
                      </TableCell>
                    )
                  })}
                </TableRow>

                {/* Volume */}
                <TableRow>
                  <TableCell className="font-medium">Volume</TableCell>
                  {selectedPatterns.map((pattern, index) => {
                    const volumes = selectedPatterns.map(p => p.volume)
                    const winnerType = getComparisonWinner(volumes, index)
                    return (
                      <TableCell key={pattern.id} className={`text-center ${getComparisonClass(winnerType)}`}>
                        {(pattern.volume / 1000000).toFixed(2)}M
                      </TableCell>
                    )
                  })}
                </TableRow>

                {/* Entry Price */}
                <TableRow>
                  <TableCell className="font-medium">Entry Price</TableCell>
                  {selectedPatterns.map(pattern => (
                    <TableCell key={pattern.id} className="text-center">
                      ${pattern.priceAtDetection.toFixed(2)}
                    </TableCell>
                  ))}
                </TableRow>

                {/* Current Price */}
                <TableRow>
                  <TableCell className="font-medium">Current Price</TableCell>
                  {selectedPatterns.map(pattern => (
                    <TableCell key={pattern.id} className="text-center">
                      ${pattern.currentPrice.toFixed(2)}
                    </TableCell>
                  ))}
                </TableRow>

                {/* Direction */}
                <TableRow>
                  <TableCell className="font-medium">Direction</TableCell>
                  {selectedPatterns.map(pattern => (
                    <TableCell key={pattern.id} className="text-center">
                      <div className="flex items-center justify-center gap-1">
                        {getDirectionIcon(pattern.direction)}
                        <span className="capitalize">{pattern.direction}</span>
                      </div>
                    </TableCell>
                  ))}
                </TableRow>

                {/* Status */}
                <TableRow>
                  <TableCell className="font-medium">Status</TableCell>
                  {selectedPatterns.map(pattern => (
                    <TableCell key={pattern.id} className="text-center">
                      <Badge 
                        variant={pattern.status === "active" ? "default" : 
                               pattern.status === "completed" ? "secondary" : "destructive"}
                      >
                        {pattern.status}
                      </Badge>
                    </TableCell>
                  ))}
                </TableRow>

                {/* Timeframe */}
                <TableRow>
                  <TableCell className="font-medium">Timeframe</TableCell>
                  {selectedPatterns.map(pattern => (
                    <TableCell key={pattern.id} className="text-center">
                      {pattern.timeframe}
                    </TableCell>
                  ))}
                </TableRow>
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>

      {/* Performance comparison charts */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card>
          <CardHeader>
            <CardTitle>Performance Chart</CardTitle>
            <CardDescription>
              Visual comparison of price movements
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-64 flex items-center justify-center border rounded-lg text-muted-foreground">
              Performance comparison chart would be displayed here
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Risk vs Return</CardTitle>
            <CardDescription>
              Scatter plot showing risk-return profile
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-64 flex items-center justify-center border rounded-lg text-muted-foreground">
              Risk vs return scatter plot would be displayed here
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Add pattern dialog */}
      {showAddDialog && (
        <Card className="fixed inset-4 z-50 bg-background border shadow-lg">
          <CardHeader className="flex flex-row items-center justify-between">
            <div>
              <CardTitle>Add Pattern to Comparison</CardTitle>
              <CardDescription>
                Select a pattern to add to your comparison
              </CardDescription>
            </div>
            <Button 
              variant="ghost" 
              size="icon"
              onClick={() => {
                setShowAddDialog(false)
                setSearchQuery("")
              }}
            >
              <X className="h-4 w-4" />
            </Button>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                type="search"
                placeholder="Search patterns..."
                className="pl-10"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>
            
            <div className="max-h-64 overflow-y-auto space-y-2">
              {filteredAvailable.slice(0, 10).map(pattern => (
                <div 
                  key={pattern.id}
                  className="flex items-center justify-between p-3 border rounded-lg cursor-pointer hover:bg-accent"
                  onClick={() => addPattern(pattern)}
                >
                  <div className="flex items-center gap-3">
                    <Badge variant="outline">{pattern.symbol}</Badge>
                    <div>
                      <p className="font-medium">{pattern.type}</p>
                      <p className="text-sm text-muted-foreground">
                        {pattern.timeframe} • {Math.round(pattern.confidence * 100)}%
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className={`font-medium ${
                      pattern.priceChangePercent >= 0 ? "text-green-600" : "text-red-600"
                    }`}>
                      {pattern.priceChangePercent >= 0 ? "+" : ""}
                      {pattern.priceChangePercent.toFixed(2)}%
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}