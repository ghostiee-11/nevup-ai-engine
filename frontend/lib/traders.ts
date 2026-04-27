// Source: nevup_seed_dataset.json — the 10 labelled trader profiles + 1 control.
export type Trader = {
  userId: string;
  name: string;
  pathology: string | null;
  totalTrades: number;
};

export const TRADERS: Trader[] = [
  {
    userId: "f412f236-4edc-47a2-8f54-8763a6ed2ce8",
    name: "Alex Mercer",
    pathology: "revenge_trading",
    totalTrades: 25,
  },
  {
    userId: "fcd434aa-2201-4060-aeb2-f44c77aa0683",
    name: "Jordan Lee",
    pathology: "overtrading",
    totalTrades: 80,
  },
  {
    userId: "84a6a3dd-f2d0-4167-960b-7319a6033d49",
    name: "Sam Rivera",
    pathology: "fomo_entries",
    totalTrades: 30,
  },
  {
    userId: "4f2f0816-f350-4684-b6c3-29bbddbb1869",
    name: "Casey Kim",
    pathology: "plan_non_adherence",
    totalTrades: 35,
  },
  {
    userId: "75076413-e8e8-44ac-861f-c7acb3902d6d",
    name: "Morgan Bell",
    pathology: "premature_exit",
    totalTrades: 35,
  },
  {
    userId: "8effb0f2-f16b-4b5f-87ab-7ffca376f309",
    name: "Taylor Grant",
    pathology: "loss_running",
    totalTrades: 30,
  },
  {
    userId: "50dd1053-73b0-43c5-8d0f-d2af88c01451",
    name: "Riley Stone",
    pathology: "session_tilt",
    totalTrades: 40,
  },
  {
    userId: "af2cfc5e-c132-4989-9c12-2913f89271fb",
    name: "Drew Patel",
    pathology: "time_of_day_bias",
    totalTrades: 48,
  },
  {
    userId: "9419073a-3d58-4ee6-a917-be2d40aecef2",
    name: "Quinn Torres",
    pathology: "position_sizing_inconsistency",
    totalTrades: 35,
  },
  {
    userId: "e84ea28c-e5a7-49ef-ac26-a873e32667bd",
    name: "Avery Chen",
    pathology: null,
    totalTrades: 30,
  },
];

export const PATHOLOGY_COLORS: Record<string, string> = {
  revenge_trading: "text-accent-rose border-accent-rose/40 bg-accent-rose/10",
  overtrading: "text-accent-amber border-accent-amber/40 bg-accent-amber/10",
  fomo_entries: "text-accent-violet border-accent-violet/40 bg-accent-violet/10",
  plan_non_adherence: "text-accent-blue border-accent-blue/40 bg-accent-blue/10",
  premature_exit: "text-accent-green border-accent-green/40 bg-accent-green/10",
  loss_running: "text-accent-rose border-accent-rose/40 bg-accent-rose/10",
  session_tilt: "text-accent-amber border-accent-amber/40 bg-accent-amber/10",
  time_of_day_bias: "text-accent-blue border-accent-blue/40 bg-accent-blue/10",
  position_sizing_inconsistency: "text-accent-violet border-accent-violet/40 bg-accent-violet/10",
  none: "text-ink-300 border-ink-600 bg-ink-700/40",
};
