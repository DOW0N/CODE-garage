export type Round = 'R32' | 'R16' | 'QF' | 'SF' | 'F'

export const ROUND_NAMES: Record<Round, string> = {
  R32: '32강',
  R16: '16강',
  QF: '8강',
  SF: '4강',
  F: '결승',
}

export const ROUND_ORDER: Round[] = ['R32', 'R16', 'QF', 'SF', 'F']

export interface Match {
  id: string // 예: "R32-1"
  round: Round
  home: string // 국가 코드
  away: string // 국가 코드
  winner: string | null // 경기 전에는 null
}
