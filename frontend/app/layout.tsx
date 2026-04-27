import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "NevUp AI Engine — Live dashboard",
  description:
    "Stateful trading-psychology coach. Cited evidence, persistent memory, streaming coaching, hallucination audit.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
