import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { Toaster } from 'react-hot-toast'
import App from './App.jsx'
import './index.css'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
    <Toaster
      position="top-right"
      toastOptions={{
        duration: 3500,
        style: {
          fontFamily: '"Source Sans 3", sans-serif',
          fontSize: '14px',
          borderRadius: '10px',
          padding: '12px 16px',
          background: '#ffffff',
          color: '#0f172a',
          border: '1px solid #d8e0ea',
          boxShadow: '0 8px 24px rgba(15, 23, 42, 0.08)',
        },
        success: {
          iconTheme: { primary: '#059669', secondary: '#fff' },
        },
        error: {
          iconTheme: { primary: '#dc2626', secondary: '#fff' },
        },
      }}
    />
  </StrictMode>,
)
