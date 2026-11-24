import { create } from 'zustand'
import { round32 } from '../data/matches/round32'
import { createNextRound, playRound } from '../engine/match/matchEngine'
import type { Match } from '../types/match'
import { MAX_PLAYERS, MIN_PLAYERS, START_CHIPS, type Player } from '../types/player'

interface GameState {
  phase: 'setup' | 'play' // 플레이어 등록 중 / 게임 중
  players: Player[]
  matches: Match[] // 현재 라운드 경기 목록
  addPlayer: (name: string) => string | null // 실패하면 오류 메시지, 성공하면 null
  removePlayer: (name: string) => void
  startGame: () => void // 게임 시작 버튼
  playCurrentRound: () => void // 경기 진행 버튼
  goToNextRound: () => void // 다음 라운드 버튼
  reset: () => void // 처음부터 버튼
}

export const useGameStore = create<GameState>((set, get) => ({
  phase: 'setup',
  players: [],
  matches: round32,

  addPlayer: (rawName) => {
    const name = rawName.trim()
    const { players } = get()
    if (!name) return '닉네임을 입력해 주세요.'
    if (players.length >= MAX_PLAYERS) return `플레이어는 최대 ${MAX_PLAYERS}명입니다.`
    if (players.some((p) => p.name.toLowerCase() === name.toLowerCase())) return '이미 사용 중인 닉네임입니다.'
    set({ players: [...players, { name, chips: START_CHIPS }] })
    return null
  },

  removePlayer: (name) => set((s) => ({ players: s.players.filter((p) => p.name !== name) })),

  startGame: () => {
    if (get().players.length < MIN_PLAYERS) return
    set({ phase: 'play', matches: round32 })
  },

  playCurrentRound: () => set((s) => ({ matches: playRound(s.matches) })),
  goToNextRound: () => set((s) => ({ matches: createNextRound(s.matches) })),

  // 등록 화면으로 돌아간다. 닉네임은 남겨 두고 칩만 처음 값으로 되돌린다.
  reset: () =>
    set((s) => ({
      phase: 'setup',
      matches: round32,
      players: s.players.map((p) => ({ ...p, chips: START_CHIPS })),
    })),
}))
