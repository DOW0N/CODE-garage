import { useState } from 'react'
import { useRoomStore } from '../store/roomStore'

// 첫 화면: 게임마스터는 방 만들기, 플레이어는 방 코드 + 닉네임으로 참가
export default function Home() {
  const { createRoom, joinRoom } = useRoomStore()
  const [code, setCode] = useState('')
  const [nickname, setNickname] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const run = async (action: () => Promise<string | null>) => {
    setBusy(true)
    setError(await action())
    setBusy(false)
  }

  return (
    <div className="p-4 max-w-md mx-auto space-y-6">
      <h1 className="text-2xl font-bold">Insight World Cup</h1>

      <section className="space-y-2">
        <h2 className="font-bold">플레이어로 참가</h2>
        <input
          className="border rounded p-2 w-full"
          placeholder="방 코드 (6자리)"
          inputMode="numeric"
          maxLength={6}
          value={code}
          onChange={(e) => setCode(e.target.value)}
        />
        <input
          className="border rounded p-2 w-full"
          placeholder="닉네임 (12자 이하)"
          maxLength={12}
          value={nickname}
          onChange={(e) => setNickname(e.target.value)}
        />
        <button
          disabled={busy}
          onClick={() => run(() => joinRoom(code, nickname))}
          className="bg-blue-600 text-white px-4 py-2 rounded w-full disabled:opacity-50"
        >
          참가하기
        </button>
      </section>

      <section className="space-y-2">
        <h2 className="font-bold">게임마스터</h2>
        <button
          disabled={busy}
          onClick={() => run(createRoom)}
          className="bg-gray-700 text-white px-4 py-2 rounded w-full disabled:opacity-50"
        >
          방 만들기
        </button>
        <p className="text-sm text-gray-500">게임마스터가 다시 들어올 때는 방 코드만 입력하고 참가하기를 누르세요.</p>
      </section>

      {error && <p className="text-red-600">{error}</p>}
    </div>
  )
}
