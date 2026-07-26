import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { Toaster } from 'react-hot-toast'
import App from './App.jsx'
import './index.css'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
    <Toaster
      position="top-center"
      toastOptions={{
        duration: 3500,
        style: {
          fontFamily: '"Plus Jakarta Sans", sans-serif',
          fontSize: '14px',
          borderRadius: '12px',
          padding: '12px 16px',
          background: '#121822',
          color: '#e8eef8',
          border: '1px solid #243044',
          boxShadow: '0 8px 30px rgba(0, 0, 0, 0.35)',
        },
        success: {
          iconTheme: { primary: '#22c55e', secondary: '#07090d' },
        },
        error: {
          iconTheme: { primary: '#ef4444', secondary: '#fff' },
        },
      }}
    />
  </StrictMode>,
)
