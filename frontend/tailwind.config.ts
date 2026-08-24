import type { Config } from "tailwindcss";

// Swiss International design system tokens. See frontend/DESIGN.md for the
// full rationale — this file is the single place these values are defined;
// components consume them by class name, never by re-declaring a hex value.
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: "class", // unused today (system is white-ground by design), reserved
  theme: {
    extend: {
      colors: {
        // The strict Swiss palette. "ink" instead of raw "black" so intent
        // reads at the call site (bg-ink, text-ink) — same colour, clearer name.
        paper: "#FFFFFF",
        ink: "#000000",
        muted: "#F2F2F2",
        accent: "#FF3000", // "Swiss red" — reserved exclusively for alerts/flags, never a class colour
        plate: "#000000", // the black image-well ground (see DESIGN.md "Ground")
      },
      fontFamily: {
        sans: [
          "Inter",
          "Helvetica Neue",
          "Helvetica",
          "Arial",
          "sans-serif",
        ],
      },
      borderRadius: {
        DEFAULT: "0px",
        none: "0px",
      },
      borderWidth: {
        DEFAULT: "1px",
        2: "2px",
        4: "4px",
      },
      letterSpacing: {
        tightest: "-0.04em",
        widest: "0.15em",
      },
      transitionDuration: {
        DEFAULT: "150ms",
      },
      transitionTimingFunction: {
        DEFAULT: "ease-out",
      },
    },
  },
  plugins: [],
} satisfies Config;
