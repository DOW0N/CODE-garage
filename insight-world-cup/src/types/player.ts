export const MIN_PLAYERS = 6
export const MAX_PLAYERS = 10
export const START_CHIPS = 30

export interface Player {
  name: string // 닉네임 (중복 불가)
  chips: number // 보유 칩
}
