import { create } from "zustand"
import { persist } from "zustand/middleware"
import { User } from "@/types"
import { apiClient } from "@/lib/api-client"

interface AuthState {
  user: User | null
  token: string | null
  isAuthenticated: boolean
  isLoading: boolean
  error: string | null
  
  login: (email: string, password: string) => Promise<void>
  register: (email: string, password: string, name: string, username: string) => Promise<void>
  logout: () => void
  setUser: (user: User) => void
  setToken: (token: string) => void
  clearError: () => void
  checkAuth: () => Promise<void>
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      token: null,
      isAuthenticated: false,
      isLoading: false,
      error: null,

      login: async (email: string, password: string) => {
        set({ isLoading: true, error: null })
        try {
          const response = await apiClient.post<{ 
            access_token: string
            user_info: User 
          }>("/auth/login", {
            email,
            password,
          })

          if (response.error) {
            throw new Error(response.error)
          }

          const { user_info: user, access_token: token } = response.data
          apiClient.setAuthToken(token)
          
          set({
            user,
            token,
            isAuthenticated: true,
            isLoading: false,
          })
        } catch (error) {
          set({
            error: error instanceof Error ? error.message : "Login failed",
            isLoading: false,
          })
          throw error
        }
      },

      register: async (email: string, password: string, name: string, username: string) => {
        set({ isLoading: true, error: null })
        try {
          // First register the user
          const registerResponse = await apiClient.post<User>("/auth/register", {
            email,
            password,
            full_name: name,
            username,
          })

          if (registerResponse.error) {
            throw new Error(registerResponse.error)
          }

          // Then login to get the token
          const loginResponse = await apiClient.post<{
            access_token: string
            user_info: User
          }>("/auth/login", {
            email,
            password,
          })

          if (loginResponse.error) {
            throw new Error(loginResponse.error)
          }

          const { user_info: user, access_token: token } = loginResponse.data
          apiClient.setAuthToken(token)
          
          set({
            user,
            token,
            isAuthenticated: true,
            isLoading: false,
          })
        } catch (error) {
          set({
            error: error instanceof Error ? error.message : "Registration failed",
            isLoading: false,
          })
          throw error
        }
      },

      logout: () => {
        apiClient.clearAuthToken()
        set({
          user: null,
          token: null,
          isAuthenticated: false,
          error: null,
        })
      },

      setUser: (user: User) => {
        set({ user, isAuthenticated: true })
      },

      setToken: (token: string) => {
        apiClient.setAuthToken(token)
        set({ token })
      },

      clearError: () => {
        set({ error: null })
      },

      checkAuth: async () => {
        const token = get().token
        if (!token) {
          set({ isAuthenticated: false })
          return
        }

        set({ isLoading: true })
        try {
          const response = await apiClient.get<User>("/auth/me")
          
          if (response.error) {
            throw new Error(response.error)
          }

          set({
            user: response.data,
            isAuthenticated: true,
            isLoading: false,
          })
        } catch {
          set({
            user: null,
            token: null,
            isAuthenticated: false,
            isLoading: false,
          })
          apiClient.clearAuthToken()
        }
      },
    }),
    {
      name: "auth-storage",
      partialize: (state) => ({ 
        token: state.token,
        user: state.user,
        isAuthenticated: state.isAuthenticated 
      }),
      onRehydrateStorage: () => (state) => {
        // When the store is rehydrated from localStorage, set the token in the API client
        if (state?.token) {
          apiClient.setAuthToken(state.token)
        }
      },
    }
  )
)