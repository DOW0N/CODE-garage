import { useState } from 'react'
import { useRoomStore } from '../store/roomStore'
import { MAX_PLAYERS, MIN_PLAYERS } from '../types/player'

// 대기실: 방 코드 공유, 참가자 목록, 게임마스터의 게임 시작 버튼
export default function Lobby() {
  const { userId, room, players, leaveRoom, startGame } = useRoomStore()
  const [error, setError] = useState<string | null>(null)
  if (!room) return null
  const isGm = room.gm_id === userId

  return (
    <div className="p-4 max-w-md mx-auto space-y-4">
      <div>
        <p className="text-sm text-gray-500">방 코드</p>
        <p className="text-4xl font-bold tracking-widest">{room.code}</p>
        <p className="text-sm text-gray-500">{isGm ? '당신은 게임마스터입니다.' : '게임마스터가 시작할 때까지 기다려 주세요.'}</p>
      </div>

      <h2 className="font-bold">
        플레이어 {players.length}/{MAX_PLAYERS}
      </h2>
      <ul className="space-y-2">
        {players.map((p) => (
          <li key={p.user_id} className="border rounded p-2 flex justify-between">
            <span>
              {p.nickname}
              {p.user_id === userId && ' (나)'}
            </span>
            <span>칩 {p.chips}</span>
          </li>
        ))}
      </ul>

      {isGm ? (
        <button
          disabled={players.length < MIN_PLAYERS}
          onClick={async () => setError(await startGame())}
          className="bg-green-600 text-white px-4 py-2 rounded w-full disabled:opacity-50"
        >
          {players.length < MIN_PLAYERS ? `${MIN_PLAYERS}명 이상 모이면 시작할 수 있습니다` : '게임 시작'}
        </button>
      ) : (
        <button onClick={leaveRoom} className="bg-gray-500 text-white px-4 py-2 rounded w-full">
          나가기
        </button>
      )}

      {error && <p className="text-red-600">{error}</p>}
    </div>
  )
}
