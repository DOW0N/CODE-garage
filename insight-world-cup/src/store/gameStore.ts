import { create } from 'zustand'
import { round32 } from '../data/matches/round32'
import { createNextRound, playRound } from '../engine/match/matchEngine'
import type { Match } from '../types/match'

interface GameState {
  matches: Match[] // 현재 라운드 경기 목록
  playCurrentRound: () => void // 경기 진행 버튼
  goToNextRound: () => void // 다음 라운드 버튼
  reset: () => void
}

export const useGameStore = create<GameState>((set) => ({
  matches: round32,
  playCurrentRound: () => set((s) => ({ matches: playRound(s.matches) })),
  goToNextRound: () => set((s) => ({ matches: createNextRound(s.matches) })),
  reset: () => set({ matches: round32 }),
}))
