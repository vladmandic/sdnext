import { create } from "zustand";
import type { Job } from "@/api/types/v2";

interface JobState {
  currentJobId: string | null;
  recentJobs: Job[];

  setCurrentJob: (id: string | null) => void;
  addRecentJob: (job: Job) => void;
  clearRecentJobs: () => void;
}

const MAX_RECENT_JOBS = 50;

export const useJobStore = create<JobState>()((set) => ({
  currentJobId: null,
  recentJobs: [],

  setCurrentJob: (id) => set({ currentJobId: id }),

  addRecentJob: (job) =>
    set((state) => ({
      recentJobs: [job, ...state.recentJobs.filter((j) => j.id !== job.id)].slice(0, MAX_RECENT_JOBS),
    })),

  clearRecentJobs: () => set({ recentJobs: [] }),
}));
