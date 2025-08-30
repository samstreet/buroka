"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { Textarea } from "@/components/ui/textarea"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import {
  Bell,
  Plus,
  Edit3,
  Trash2,
  Mail,
  MessageSquare,
  Smartphone,
  AlertCircle,
  CheckCircle,
  Clock,
  TrendingUp,
  TrendingDown,
} from "lucide-react"

interface PatternAlert {
  id: string
  name: string
  description?: string
  isActive: boolean
  triggers: {
    patternTypes: string[]
    symbols: string[]
    timeframes: string[]
    minConfidence: number
    direction?: "bullish" | "bearish" | "any"
    priceChange?: {
      operator: ">" | "<" | ">=" | "<="
      value: number
    }
  }
  notifications: {
    email: boolean
    sms: boolean
    push: boolean
    webhook?: string
  }
  cooldown: number // minutes
  createdAt: Date
  lastTriggered?: Date
  triggerCount: number
}

interface PatternAlertsProps {
  alerts?: PatternAlert[]
  onCreateAlert?: (alert: Omit<PatternAlert, "id" | "createdAt" | "triggerCount">) => void
  onUpdateAlert?: (id: string, alert: Partial<PatternAlert>) => void
  onDeleteAlert?: (id: string) => void
}

const defaultAlert: Omit<PatternAlert, "id" | "createdAt" | "triggerCount"> = {
  name: "",
  description: "",
  isActive: true,
  triggers: {
    patternTypes: [],
    symbols: [],
    timeframes: [],
    minConfidence: 0.7,
    direction: "any",
  },
  notifications: {
    email: true,
    sms: false,
    push: true,
  },
  cooldown: 60,
  lastTriggered: undefined,
}

export function PatternAlerts({
  alerts = [],
  onCreateAlert,
  onUpdateAlert,
  onDeleteAlert,
}: PatternAlertsProps) {
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false)
  const [editingAlert, setEditingAlert] = useState<PatternAlert | null>(null)
  const [formData, setFormData] = useState<typeof defaultAlert>(defaultAlert)

  const patternTypes = [
    "Triangle", "Flag", "Wedge", "Channel", "Head & Shoulders", 
    "Double Top", "Double Bottom", "Cup & Handle", "Ascending Triangle", "Descending Triangle"
  ]

  const timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]

  const handleCreateAlert = () => {
    setFormData(defaultAlert)
    setEditingAlert(null)
    setIsCreateDialogOpen(true)
  }

  const handleEditAlert = (alert: PatternAlert) => {
    setFormData({
      name: alert.name,
      description: alert.description,
      isActive: alert.isActive,
      triggers: alert.triggers,
      notifications: alert.notifications,
      cooldown: alert.cooldown,
      lastTriggered: alert.lastTriggered,
    })
    setEditingAlert(alert)
    setIsCreateDialogOpen(true)
  }

  const handleSaveAlert = () => {
    if (editingAlert) {
      onUpdateAlert?.(editingAlert.id, formData)
    } else {
      onCreateAlert?.(formData)
    }
    setIsCreateDialogOpen(false)
    setFormData(defaultAlert)
    setEditingAlert(null)
  }

  const handleToggleAlert = (id: string, isActive: boolean) => {
    onUpdateAlert?.(id, { isActive })
  }

  const formatLastTriggered = (date?: Date) => {
    if (!date) return "Never"
    const now = new Date()
    const diff = now.getTime() - date.getTime()
    const minutes = Math.floor(diff / 60000)
    
    if (minutes < 1) return "Just now"
    if (minutes < 60) return `${minutes}m ago`
    if (minutes < 1440) return `${Math.floor(minutes / 60)}h ago`
    return `${Math.floor(minutes / 1440)}d ago`
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Pattern Alerts</h2>
          <p className="text-muted-foreground">
            Set up notifications when specific patterns are detected
          </p>
        </div>
        <Button onClick={handleCreateAlert}>
          <Plus className="mr-2 h-4 w-4" />
          Create Alert
        </Button>
      </div>

      {/* Alerts summary */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Total Alerts</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">{alerts.length}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Active Alerts</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-green-600">
              {alerts.filter(a => a.isActive).length}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Triggered Today</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-blue-600">
              {alerts.filter(a => {
                if (!a.lastTriggered) return false
                const today = new Date()
                today.setHours(0, 0, 0, 0)
                return a.lastTriggered >= today
              }).length}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Total Triggers</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">
              {alerts.reduce((sum, alert) => sum + alert.triggerCount, 0)}
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Alerts table */}
      <Card>
        <CardHeader>
          <CardTitle>Alert Configuration</CardTitle>
          <CardDescription>
            Manage your pattern detection alerts
          </CardDescription>
        </CardHeader>
        <CardContent>
          {alerts.length === 0 ? (
            <div className="text-center py-8">
              <Bell className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
              <h3 className="text-lg font-medium mb-2">No Alerts Configured</h3>
              <p className="text-sm text-muted-foreground mb-4">
                Create your first alert to get notified when patterns are detected
              </p>
              <Button onClick={handleCreateAlert}>
                <Plus className="mr-2 h-4 w-4" />
                Create First Alert
              </Button>
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Name</TableHead>
                  <TableHead>Patterns</TableHead>
                  <TableHead>Symbols</TableHead>
                  <TableHead>Notifications</TableHead>
                  <TableHead>Last Triggered</TableHead>
                  <TableHead>Triggers</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {alerts.map((alert) => (
                  <TableRow key={alert.id}>
                    <TableCell>
                      <div>
                        <p className="font-medium">{alert.name}</p>
                        {alert.description && (
                          <p className="text-sm text-muted-foreground">
                            {alert.description}
                          </p>
                        )}
                      </div>
                    </TableCell>
                    <TableCell>
                      <div className="flex flex-wrap gap-1">
                        {alert.triggers.patternTypes.slice(0, 2).map(type => (
                          <Badge key={type} variant="outline" className="text-xs">
                            {type}
                          </Badge>
                        ))}
                        {alert.triggers.patternTypes.length > 2 && (
                          <Badge variant="outline" className="text-xs">
                            +{alert.triggers.patternTypes.length - 2}
                          </Badge>
                        )}
                      </div>
                    </TableCell>
                    <TableCell>
                      <div className="flex flex-wrap gap-1">
                        {alert.triggers.symbols.slice(0, 3).map(symbol => (
                          <Badge key={symbol} variant="secondary" className="text-xs">
                            {symbol}
                          </Badge>
                        ))}
                        {alert.triggers.symbols.length > 3 && (
                          <Badge variant="secondary" className="text-xs">
                            +{alert.triggers.symbols.length - 3}
                          </Badge>
                        )}
                      </div>
                    </TableCell>
                    <TableCell>
                      <div className="flex gap-1">
                        {alert.notifications.email && <Mail className="h-4 w-4 text-blue-600" />}
                        {alert.notifications.sms && <Smartphone className="h-4 w-4 text-green-600" />}
                        {alert.notifications.push && <Bell className="h-4 w-4 text-purple-600" />}
                      </div>
                    </TableCell>
                    <TableCell>
                      <span className="text-sm">
                        {formatLastTriggered(alert.lastTriggered)}
                      </span>
                    </TableCell>
                    <TableCell>
                      <Badge variant="outline">
                        {alert.triggerCount}
                      </Badge>
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center gap-2">
                        <Switch
                          checked={alert.isActive}
                          onCheckedChange={(checked) => handleToggleAlert(alert.id, checked)}
                        />
                        {alert.isActive ? (
                          <CheckCircle className="h-4 w-4 text-green-600" />
                        ) : (
                          <Clock className="h-4 w-4 text-gray-600" />
                        )}
                      </div>
                    </TableCell>
                    <TableCell>
                      <div className="flex gap-2">
                        <Button
                          variant="ghost"
                          size="icon"
                          onClick={() => handleEditAlert(alert)}
                        >
                          <Edit3 className="h-4 w-4" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="icon"
                          onClick={() => onDeleteAlert?.(alert.id)}
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      {/* Create/Edit Alert Dialog */}
      <Dialog open={isCreateDialogOpen} onOpenChange={setIsCreateDialogOpen}>
        <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>
              {editingAlert ? "Edit Alert" : "Create New Alert"}
            </DialogTitle>
            <DialogDescription>
              Configure when and how you want to be notified about pattern detections
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-6">
            {/* Basic Information */}
            <div className="space-y-4">
              <h3 className="text-lg font-medium">Basic Information</h3>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="name">Alert Name</Label>
                  <Input
                    id="name"
                    value={formData.name}
                    onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                    placeholder="e.g., AAPL Bullish Patterns"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="active">Active</Label>
                  <div className="flex items-center space-x-2">
                    <Switch
                      id="active"
                      checked={formData.isActive}
                      onCheckedChange={(checked) => setFormData({ ...formData, isActive: checked })}
                    />
                    <span className="text-sm">
                      {formData.isActive ? "Enabled" : "Disabled"}
                    </span>
                  </div>
                </div>
              </div>
              <div className="space-y-2">
                <Label htmlFor="description">Description (Optional)</Label>
                <Textarea
                  id="description"
                  value={formData.description}
                  onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                  placeholder="Brief description of this alert..."
                  rows={2}
                />
              </div>
            </div>

            {/* Trigger Conditions */}
            <div className="space-y-4">
              <h3 className="text-lg font-medium">Trigger Conditions</h3>
              
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Pattern Types</Label>
                  <Select>
                    <SelectTrigger>
                      <SelectValue placeholder="Select pattern types" />
                    </SelectTrigger>
                    <SelectContent>
                      {patternTypes.map(type => (
                        <SelectItem key={type} value={type}>
                          {type}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label>Direction</Label>
                  <Select
                    value={formData.triggers.direction}
                    onValueChange={(value: any) => 
                      setFormData({
                        ...formData,
                        triggers: { ...formData.triggers, direction: value }
                      })
                    }
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="any">Any Direction</SelectItem>
                      <SelectItem value="bullish">Bullish Only</SelectItem>
                      <SelectItem value="bearish">Bearish Only</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Symbols</Label>
                  <Input
                    placeholder="e.g., AAPL, GOOGL, MSFT"
                    value={formData.triggers.symbols.join(", ")}
                    onChange={(e) => 
                      setFormData({
                        ...formData,
                        triggers: {
                          ...formData.triggers,
                          symbols: e.target.value.split(",").map(s => s.trim()).filter(Boolean)
                        }
                      })
                    }
                  />
                </div>

                <div className="space-y-2">
                  <Label>Min Confidence</Label>
                  <Input
                    type="number"
                    min="0"
                    max="1"
                    step="0.1"
                    value={formData.triggers.minConfidence}
                    onChange={(e) => 
                      setFormData({
                        ...formData,
                        triggers: {
                          ...formData.triggers,
                          minConfidence: parseFloat(e.target.value) || 0.7
                        }
                      })
                    }
                  />
                </div>
              </div>
            </div>

            {/* Notification Settings */}
            <div className="space-y-4">
              <h3 className="text-lg font-medium">Notification Settings</h3>
              
              <div className="grid grid-cols-3 gap-4">
                <div className="flex items-center space-x-2">
                  <Switch
                    checked={formData.notifications.email}
                    onCheckedChange={(checked) => 
                      setFormData({
                        ...formData,
                        notifications: { ...formData.notifications, email: checked }
                      })
                    }
                  />
                  <Mail className="h-4 w-4" />
                  <span className="text-sm">Email</span>
                </div>

                <div className="flex items-center space-x-2">
                  <Switch
                    checked={formData.notifications.sms}
                    onCheckedChange={(checked) => 
                      setFormData({
                        ...formData,
                        notifications: { ...formData.notifications, sms: checked }
                      })
                    }
                  />
                  <Smartphone className="h-4 w-4" />
                  <span className="text-sm">SMS</span>
                </div>

                <div className="flex items-center space-x-2">
                  <Switch
                    checked={formData.notifications.push}
                    onCheckedChange={(checked) => 
                      setFormData({
                        ...formData,
                        notifications: { ...formData.notifications, push: checked }
                      })
                    }
                  />
                  <Bell className="h-4 w-4" />
                  <span className="text-sm">Push</span>
                </div>
              </div>

              <div className="space-y-2">
                <Label>Cooldown Period (minutes)</Label>
                <Input
                  type="number"
                  min="1"
                  value={formData.cooldown}
                  onChange={(e) => 
                    setFormData({
                      ...formData,
                      cooldown: parseInt(e.target.value) || 60
                    })
                  }
                />
                <p className="text-sm text-muted-foreground">
                  Minimum time between notifications for similar patterns
                </p>
              </div>
            </div>
          </div>

          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setIsCreateDialogOpen(false)}
            >
              Cancel
            </Button>
            <Button onClick={handleSaveAlert}>
              {editingAlert ? "Update Alert" : "Create Alert"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}