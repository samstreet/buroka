"use client"

import { useEffect, useRef, useState, useCallback } from "react"

export interface WebSocketMessage {
  type: string
  data: any
  timestamp: number
}

export interface WebSocketOptions {
  url: string
  protocols?: string | string[]
  reconnectAttempts?: number
  reconnectDelay?: number
  heartbeatInterval?: number
  onOpen?: (event: Event) => void
  onClose?: (event: CloseEvent) => void
  onError?: (event: Event) => void
  onMessage?: (message: WebSocketMessage) => void
}

export interface WebSocketState {
  socket: WebSocket | null
  isConnected: boolean
  isConnecting: boolean
  error: string | null
  lastMessage: WebSocketMessage | null
  messageHistory: WebSocketMessage[]
  reconnectCount: number
}

export function useWebSocket(options: WebSocketOptions) {
  const {
    url,
    protocols,
    reconnectAttempts = 5,
    reconnectDelay = 3000,
    heartbeatInterval = 30000,
    onOpen,
    onClose,
    onError,
    onMessage,
  } = options

  const [state, setState] = useState<WebSocketState>({
    socket: null,
    isConnected: false,
    isConnecting: false,
    error: null,
    lastMessage: null,
    messageHistory: [],
    reconnectCount: 0,
  })

  const socketRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const heartbeatIntervalRef = useRef<NodeJS.Timeout | null>(null)
  const reconnectCountRef = useRef(0)
  const messageHistoryRef = useRef<WebSocketMessage[]>([])

  const clearTimeouts = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = null
    }
    if (heartbeatIntervalRef.current) {
      clearInterval(heartbeatIntervalRef.current)
      heartbeatIntervalRef.current = null
    }
  }, [])

  const sendHeartbeat = useCallback(() => {
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      const heartbeatMessage: WebSocketMessage = {
        type: "heartbeat",
        data: { timestamp: Date.now() },
        timestamp: Date.now(),
      }
      socketRef.current.send(JSON.stringify(heartbeatMessage))
    }
  }, [])

  const startHeartbeat = useCallback(() => {
    if (heartbeatInterval > 0) {
      heartbeatIntervalRef.current = setInterval(sendHeartbeat, heartbeatInterval)
    }
  }, [heartbeatInterval, sendHeartbeat])

  const addToHistory = useCallback((message: WebSocketMessage) => {
    messageHistoryRef.current = [message, ...messageHistoryRef.current.slice(0, 99)] // Keep last 100 messages
  }, [])

  const connect = useCallback(() => {
    if (socketRef.current?.readyState === WebSocket.CONNECTING || 
        socketRef.current?.readyState === WebSocket.OPEN) {
      return
    }

    setState(prev => ({ ...prev, isConnecting: true, error: null }))

    try {
      const socket = new WebSocket(url, protocols)
      socketRef.current = socket

      socket.onopen = (event) => {
        console.log("WebSocket connected:", url)
        reconnectCountRef.current = 0
        setState(prev => ({
          ...prev,
          socket,
          isConnected: true,
          isConnecting: false,
          error: null,
          reconnectCount: 0,
        }))
        startHeartbeat()
        onOpen?.(event)
      }

      socket.onclose = (event) => {
        console.log("WebSocket closed:", event.code, event.reason)
        clearTimeouts()
        setState(prev => ({
          ...prev,
          socket: null,
          isConnected: false,
          isConnecting: false,
        }))
        onClose?.(event)

        // Attempt to reconnect if not a manual close
        if (event.code !== 1000 && reconnectCountRef.current < reconnectAttempts) {
          reconnectCountRef.current++
          console.log(`Attempting to reconnect (${reconnectCountRef.current}/${reconnectAttempts})...`)
          
          setState(prev => ({ ...prev, reconnectCount: reconnectCountRef.current }))
          
          reconnectTimeoutRef.current = setTimeout(() => {
            connect()
          }, reconnectDelay * reconnectCountRef.current) // Exponential backoff
        }
      }

      socket.onerror = (event) => {
        console.error("WebSocket error:", event)
        setState(prev => ({
          ...prev,
          error: "WebSocket connection error",
          isConnecting: false,
        }))
        onError?.(event)
      }

      socket.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data)
          
          // Skip heartbeat responses
          if (message.type === "heartbeat" || message.type === "pong") {
            return
          }
          
          addToHistory(message)
          setState(prev => ({
            ...prev,
            lastMessage: message,
            messageHistory: [message, ...prev.messageHistory.slice(0, 99)],
          }))
          onMessage?.(message)
        } catch (error) {
          console.error("Failed to parse WebSocket message:", error)
        }
      }
    } catch (error) {
      console.error("Failed to create WebSocket connection:", error)
      setState(prev => ({
        ...prev,
        error: "Failed to create WebSocket connection",
        isConnecting: false,
      }))
    }
  }, [url, protocols, reconnectAttempts, reconnectDelay, onOpen, onClose, onError, onMessage, startHeartbeat, addToHistory])

  const disconnect = useCallback(() => {
    clearTimeouts()
    if (socketRef.current) {
      socketRef.current.close(1000, "Manual disconnect")
      socketRef.current = null
    }
    setState(prev => ({
      ...prev,
      socket: null,
      isConnected: false,
      isConnecting: false,
    }))
  }, [clearTimeouts])

  const sendMessage = useCallback((message: Omit<WebSocketMessage, "timestamp">) => {
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      const fullMessage: WebSocketMessage = {
        ...message,
        timestamp: Date.now(),
      }
      socketRef.current.send(JSON.stringify(fullMessage))
      return true
    }
    return false
  }, [])

  const subscribe = useCallback((channel: string, symbol?: string) => {
    return sendMessage({
      type: "subscribe",
      data: { channel, symbol },
    })
  }, [sendMessage])

  const unsubscribe = useCallback((channel: string, symbol?: string) => {
    return sendMessage({
      type: "unsubscribe",
      data: { channel, symbol },
    })
  }, [sendMessage])

  // Auto-connect on mount
  useEffect(() => {
    connect()
    return () => {
      disconnect()
    }
  }, [connect, disconnect])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      clearTimeouts()
      if (socketRef.current) {
        socketRef.current.close()
      }
    }
  }, [clearTimeouts])

  return {
    ...state,
    connect,
    disconnect,
    sendMessage,
    subscribe,
    unsubscribe,
    isReady: state.isConnected && !state.isConnecting,
  }
}

// Mock WebSocket server for development
export class MockWebSocketServer {
  private clients: Set<WebSocket> = new Set()
  private intervals: Set<NodeJS.Timeout> = new Set()

  start() {
    // This would typically be a real WebSocket server
    // For now, we'll simulate it by creating mock data streams
    console.log("Mock WebSocket server started")
  }

  simulateMarketData() {
    const symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
    const patternTypes = ["Triangle", "Flag", "Head & Shoulders", "Double Bottom"]

    const interval = setInterval(() => {
      const mockMessages = [
        {
          type: "price_update",
          data: {
            symbol: symbols[Math.floor(Math.random() * symbols.length)],
            price: 150 + Math.random() * 100,
            change: (Math.random() - 0.5) * 5,
            volume: Math.floor(Math.random() * 1000000),
          },
          timestamp: Date.now(),
        },
        {
          type: "pattern_detected",
          data: {
            symbol: symbols[Math.floor(Math.random() * symbols.length)],
            pattern: patternTypes[Math.floor(Math.random() * patternTypes.length)],
            confidence: 0.7 + Math.random() * 0.3,
            direction: Math.random() > 0.5 ? "bullish" : "bearish",
          },
          timestamp: Date.now(),
        },
        {
          type: "volume_spike",
          data: {
            symbol: symbols[Math.floor(Math.random() * symbols.length)],
            volume: Math.floor(Math.random() * 5000000),
            volumeRatio: 2 + Math.random() * 3,
          },
          timestamp: Date.now(),
        },
      ]

      // Simulate sending to connected clients
      this.clients.forEach(client => {
        if (client.readyState === WebSocket.OPEN) {
          const message = mockMessages[Math.floor(Math.random() * mockMessages.length)]
          try {
            client.send(JSON.stringify(message))
          } catch (error) {
            console.error("Failed to send mock message:", error)
          }
        }
      })
    }, 1000 + Math.random() * 2000)

    this.intervals.add(interval)
  }

  addClient(client: WebSocket) {
    this.clients.add(client)
    
    client.onclose = () => {
      this.clients.delete(client)
    }
  }

  stop() {
    this.intervals.forEach(interval => clearInterval(interval))
    this.intervals.clear()
    this.clients.clear()
  }
}

// Custom hook for market data WebSocket
export function useMarketDataWebSocket(symbols: string[] = []) {
  const [marketData, setMarketData] = useState<Map<string, any>>(new Map())
  const [patterns, setPatterns] = useState<any[]>([])
  const [alerts, setAlerts] = useState<any[]>([])

  const handleMessage = useCallback((message: WebSocketMessage) => {
    switch (message.type) {
      case "price_update":
        setMarketData(prev => {
          const newMap = new Map(prev)
          newMap.set(message.data.symbol, message.data)
          return newMap
        })
        break
      
      case "pattern_detected":
        setPatterns(prev => [message.data, ...prev.slice(0, 49)])
        break
      
      case "alert":
        setAlerts(prev => [message.data, ...prev.slice(0, 99)])
        break
    }
  }, [])

  // For development, we'll use a mock URL
  // In production, this would be your actual WebSocket server
  const websocket = useWebSocket({
    url: "ws://localhost:8080/market-data", // This would be your real WebSocket URL
    onMessage: handleMessage,
    reconnectAttempts: 5,
    reconnectDelay: 3000,
    heartbeatInterval: 30000,
  })

  // Subscribe to symbols when connection is ready
  useEffect(() => {
    if (websocket.isReady) {
      symbols.forEach(symbol => {
        websocket.subscribe("price", symbol)
        websocket.subscribe("patterns", symbol)
      })
    }
  }, [websocket.isReady, symbols, websocket])

  return {
    ...websocket,
    marketData: Array.from(marketData.values()),
    patterns,
    alerts,
    subscribeTo: websocket.subscribe,
    unsubscribeFrom: websocket.unsubscribe,
  }
}