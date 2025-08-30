"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import {
  LayoutDashboard,
  TrendingUp,
  BarChart3,
  Bell,
  Settings,
  FileText,
  Activity,
  Zap,
  Database,
  ChevronLeft,
  ChevronRight,
  Wifi,
  GitCompare,
  Bitcoin,
  Coins,
  LineChart,
} from "lucide-react"
import { cn } from "@/lib/utils"
import { useState } from "react"

const menuItems = [
  {
    title: "Dashboard",
    href: "/dashboard",
    icon: LayoutDashboard,
  },
  {
    title: "Crypto Markets",
    href: "/crypto",
    icon: Bitcoin,
  },
  {
    title: "Live Trading",
    href: "/live",
    icon: Wifi,
  },
  {
    title: "Chart Analysis",
    href: "/analysis",
    icon: LineChart,
  },
  {
    title: "Patterns",
    href: "/patterns",
    icon: GitCompare,
  },
  {
    title: "Indicators",
    href: "/indicators",
    icon: Activity,
  },
  {
    title: "Alerts",
    href: "/alerts",
    icon: Bell,
  },
  {
    title: "Backtest",
    href: "/backtest",
    icon: Zap,
  },
  {
    title: "Portfolio",
    href: "/portfolio",
    icon: Coins,
  },
  {
    title: "Reports",
    href: "/reports",
    icon: FileText,
  },
  {
    title: "Settings",
    href: "/settings",
    icon: Settings,
  },
]

export function Sidebar() {
  const pathname = usePathname()
  const [isCollapsed, setIsCollapsed] = useState(false)

  return (
    <aside
      className={cn(
        "sticky top-0 h-screen border-r bg-background transition-all duration-300",
        isCollapsed ? "w-16" : "w-64"
      )}
    >
      <div className="flex h-full flex-col">
        <div className="flex items-center justify-end p-4">
          <button
            onClick={() => setIsCollapsed(!isCollapsed)}
            className="p-2 hover:bg-accent rounded-lg transition-colors"
            aria-label={isCollapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            {isCollapsed ? <ChevronRight size={20} /> : <ChevronLeft size={20} />}
          </button>
        </div>

        <nav className="flex-1 space-y-1 px-3">
          {menuItems.map((item) => {
            const Icon = item.icon
            const isActive = pathname === item.href

            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
                  isActive
                    ? "bg-accent text-accent-foreground"
                    : "hover:bg-accent hover:text-accent-foreground",
                  isCollapsed && "justify-center"
                )}
                title={isCollapsed ? item.title : undefined}
              >
                <Icon className="h-5 w-5 flex-shrink-0" />
                {!isCollapsed && <span>{item.title}</span>}
              </Link>
            )
          })}
        </nav>

        <div className="border-t p-4">
          <div className={cn("space-y-1", isCollapsed && "hidden")}>
            <p className="text-xs font-medium text-muted-foreground">System Status</p>
            <div className="flex items-center gap-2">
              <div className="h-2 w-2 bg-green-500 rounded-full animate-pulse"></div>
              <span className="text-xs">All systems operational</span>
            </div>
          </div>
          {isCollapsed && (
            <div className="flex justify-center">
              <div className="h-2 w-2 bg-green-500 rounded-full animate-pulse"></div>
            </div>
          )}
        </div>
      </div>
    </aside>
  )
}