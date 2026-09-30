/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: { 900: "#0B0D12", 800: "#12161F", 700: "#1A1F2B" },
        gold: "#FFB020",
        ember: "#FF6B35",
        aqua: "#35D0FF",
        mist: { 100: "#F5F7FA", 300: "#A7B0C0", 500: "#6B7488" },
        ok: "#22C55E",
        warn: "#F59E0B",
        bad: "#EF4444",
      },
      fontFamily: {
        sans: ["Poppins", "PingFang SC", "system-ui", "sans-serif"],
      },
      opacity: {
        3: "0.03",
        4: "0.04",
        6: "0.06",
        8: "0.08",
        12: "0.12",
        15: "0.15",
        18: "0.18",
        22: "0.22",
        35: "0.35",
        45: "0.45",
        55: "0.55",
        65: "0.65",
        85: "0.85",
      },
      boxShadow: {
        glow: "0 0 24px rgba(255,176,32,0.22)",
        panel: "0 18px 48px rgba(0,0,0,0.45)",
      },
      keyframes: {
        fadeUp: {
          "0%": { opacity: "0", transform: "translateY(12px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        shimmer: {
          "0%": { backgroundPosition: "-500px 0" },
          "100%": { backgroundPosition: "500px 0" },
        },
      },
      animation: {
        fadeUp: "fadeUp .45s ease-out both",
        shimmer: "shimmer 1.6s linear infinite",
      },
    },
  },
  plugins: [],
};
