import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          950: "#08070b",
          900: "#0d0c12",
          800: "#16151c",
          700: "#211f29",
          600: "#2c2a36",
          400: "#74717f",
          300: "#a3a0ae",
          200: "#cfcdd5",
          100: "#ecebef",
        },
        accent: {
          green: "#21c98a",
          amber: "#f5b400",
          rose: "#ff5d6c",
          blue: "#5b8def",
          violet: "#a78bfa",
        },
      },
      fontFamily: {
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      keyframes: {
        pulse_dot: {
          "0%,100%": { opacity: "0.4" },
          "50%": { opacity: "1" },
        },
        token_in: {
          from: { opacity: "0", transform: "translateY(2px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
      },
      animation: {
        pulse_dot: "pulse_dot 1.6s ease-in-out infinite",
        token_in: "token_in 120ms ease-out",
      },
    },
  },
  plugins: [],
};
export default config;
