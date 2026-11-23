import { getCountry } from '../data/countryStats'
import { useGameStore } from '../store/gameStore'
import { ROUND_NAMES } from '../types/match'

export default function GamePlay() {
  const { matches, playCurrentRound, goToNextRound, reset } = useGameStore()
  const round = matches[0].round
  const played = matches.every((m) => m.winner)
  const isFinal = round === 'F'

  return (
    <div className="p-4 max-w-md mx-auto">
      <h1 className="text-2xl font-bold mb-4">{ROUND_NAMES[round]}</h1>

      <ul className="space-y-2 mb-4">
        {matches.map((m) => (
          <li key={m.id} className="border rounded p-2 flex justify-between">
            {[m.home, m.away].map((code) => (
              <span key={code} className={m.winner === code ? 'font-bold text-blue-600' : m.winner ? 'text-gray-400' : ''}>
                {getCountry(code).name}
                {m.winner === code && ' (승)'}
              </span>
            ))}
          </li>
        ))}
      </ul>

      {!played && (
        <button onClick={playCurrentRound} className="bg-blue-600 text-white px-4 py-2 rounded">
          경기 진행
        </button>
      )}
      {played && !isFinal && (
        <button onClick={goToNextRound} className="bg-green-600 text-white px-4 py-2 rounded">
          다음 라운드
        </button>
      )}
      {played && isFinal && (
        <div>
          <p className="text-xl font-bold mb-2">🏆 우승: {getCountry(matches[0].winner!).name}</p>
          <button onClick={reset} className="bg-gray-600 text-white px-4 py-2 rounded">
            처음부터
          </button>
        </div>
      )}
    </div>
  )
}
