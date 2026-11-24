// 서버(supabase/001_rooms.sql)에도 같은 규칙이 있다. 바꿀 때는 둘 다 바꾼다.
export const MIN_PLAYERS = 6
export const MAX_PLAYERS = 10

export interface Player {
  user_id: string
  nickname: string
  chips: number
}

export interface Room {
  id: string
  code: string // 참가용 6자리 숫자
  gm_id: string // 게임마스터
  phase: 'lobby' | 'playing'
}
