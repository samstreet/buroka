import { Suspense, ReactNode } from "react"
import { LoadingCard, LoadingDashboard, LoadingPage } from "@/components/ui/loading"

interface SuspenseWrapperProps {
  children: ReactNode
  fallback?: ReactNode
  type?: "page" | "card" | "dashboard" | "custom"
  message?: string
}

export function SuspenseWrapper({ 
  children, 
  fallback,
  type = "card",
  message 
}: SuspenseWrapperProps) {
  
  const getFallback = () => {
    if (fallback) return fallback
    
    switch (type) {
      case "page":
        return <LoadingPage message={message} />
      case "dashboard":
        return <LoadingDashboard />
      case "card":
        return <LoadingCard message={message} />
      default:
        return <LoadingCard message={message} />
    }
  }

  return (
    <Suspense fallback={getFallback()}>
      {children}
    </Suspense>
  )
}