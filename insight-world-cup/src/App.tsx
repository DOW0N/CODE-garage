import { useEffect, useState } from 'react'
import Home from './pages/Home'
import Lobby from './pages/Lobby'
import { useRoomStore } from './store/roomStore'

export default function App() {
  const { room, init } = useRoomStore()
  const [error, setError] = useState<string | null>(null)
  const [ready, setReady] = useState(false)

  useEffect(() => {
    init()
      .then(() => setReady(true))
      .catch((e) => setError(`서버 접속 실패: ${e.message}`))
  }, [init])

  if (error) return <p className="p-4 text-red-600">{error}</p>
  if (!ready) return <p className="p-4">접속 중...</p>
  if (!room) return <Home />
  if (room.phase === 'lobby') return <Lobby />
  // 경기 화면은 4단계(라운드·스코어 서버 진행)에서 연결한다
  return <p className="p-4 text-xl font-bold">게임이 시작되었습니다!</p>
}
