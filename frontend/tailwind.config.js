/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        bgDark: "#030712",
        cardBg: "rgba(15, 23, 42, 0.65)",
        cardBorder: "rgba(255, 255, 255, 0.08)",
        neonPurple: "#8a2be2",
        neonCyan: "#00f2fe",
        neonGreen: "#10b981",
        neonAmber: "#f59e0b",
        neonRed: "#ef4444"
      },
      fontFamily: {
        sans: ['Outfit', 'sans-serif'],
        mono: ['Fira Code', 'monospace']
      }
    },
  },
  plugins: [],
}
