// 서버(supabase/003_chat.sql)에도 같은 제한이 있다. 바꿀 때는 둘 다 바꾼다.
export const MAX_MESSAGE_LENGTH = 200

export interface Message {
  id: number
  sender_id: string
  sender_name: string // 닉네임 또는 '게임마스터'
  to_id: string | null // null 이면 전체 채팅, 값이 있으면 귓속말
  body: string
}
