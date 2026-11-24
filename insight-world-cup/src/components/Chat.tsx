import { useEffect, useRef, useState } from 'react'
import { useRoomStore } from '../store/roomStore'
import { MAX_MESSAGE_LENGTH } from '../types/chat'

// 방 채팅: 전체 채팅 + 귓속말
export default function Chat() {
  const { userId, room, players, messages, sendMessage } = useRoomStore()
  const [body, setBody] = useState('')
  const [toId, setToId] = useState('') // '' = 전체
  const [error, setError] = useState<string | null>(null)
  const bottom = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottom.current?.scrollIntoView({ block: 'nearest' })
  }, [messages.length])

  if (!room) return null
  // 귓속말 상대: 게임마스터 + 다른 플레이어 (나는 빼고)
  const people = [{ id: room.gm_id, name: '게임마스터' }, ...players.map((p) => ({ id: p.user_id, name: p.nickname }))]
  const nameOf = (id: string) => people.find((p) => p.id === id)?.name ?? '(나간 사람)'

  const send = async () => {
    const err = await sendMessage(body, toId || null)
    setError(err)
    if (!err) setBody('')
  }

  return (
    <section className="border rounded p-2 space-y-2">
      <h2 className="font-bold">채팅</h2>

      <div className="h-60 overflow-y-auto space-y-1 text-sm">
        {messages.map((m) => {
          const whisper = m.to_id !== null
          const mine = m.sender_id === userId
          return (
            <p key={m.id} className={whisper ? 'text-purple-700' : ''}>
              <span className="font-bold">
                {whisper && (mine ? `[귓속말 → ${nameOf(m.to_id!)}] ` : '[귓속말] ')}
                {m.sender_name}
                {mine && ' (나)'}:
              </span>{' '}
              {m.body}
            </p>
          )
        })}
        <div ref={bottom} />
      </div>

      <select className="border rounded p-2 w-full" value={toId} onChange={(e) => setToId(e.target.value)}>
        <option value="">전체에게</option>
        {people
          .filter((p) => p.id !== userId)
          .map((p) => (
            <option key={p.id} value={p.id}>
              {p.name}에게 귓속말
            </option>
          ))}
      </select>

      <div className="flex gap-2">
        <input
          className="border rounded p-2 flex-1 min-w-0"
          placeholder="메시지 입력"
          maxLength={MAX_MESSAGE_LENGTH}
          value={body}
          onChange={(e) => setBody(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && !e.nativeEvent.isComposing && send()}
        />
        <button onClick={send} className="bg-blue-600 text-white px-4 py-2 rounded">
          보내기
        </button>
      </div>
      <p className="text-xs text-gray-500 text-right">
        {body.length}/{MAX_MESSAGE_LENGTH}
      </p>

      {error && <p className="text-red-600 text-sm">{error}</p>}
    </section>
  )
}
