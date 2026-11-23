import { getCountry } from '../data/countryStats'
import { useGameStore } from '../store/gameStore'
import { ROUND_NAMES } from '../types/match'

export default function GamePlay() {
  const { matches, playCurrentRound, goToNextRound, reset } = useGameStore()
  const played = matches.every((m) => m.winner)
  const final = matches.find((m) => m.round === 'F')
  const thirdPlace = matches.find((m) => m.round === '3P')

  return (
    <div className="p-4 max-w-md mx-auto">
      <h1 className="text-2xl font-bold mb-4">
        {[...new Set(matches.map((m) => ROUND_NAMES[m.round]))].join(' · ')}
      </h1>

      <ul className="space-y-2 mb-4">
        {matches.map((m) => (
          <li key={m.id} className="border rounded p-2">
            {thirdPlace && <div className="text-xs text-gray-500 mb-1">{ROUND_NAMES[m.round]}</div>}
            <div className="flex justify-between">
              {[m.home, m.away].map((code) => (
                <span key={code} className={m.winner === code ? 'font-bold text-blue-600' : m.winner ? 'text-gray-400' : ''}>
                  {getCountry(code).name}
                  {m.winner === code && ' (승)'}
                </span>
              ))}
            </div>
          </li>
        ))}
      </ul>

      {!played && (
        <button onClick={playCurrentRound} className="bg-blue-600 text-white px-4 py-2 rounded">
          경기 진행
        </button>
      )}
      {played && !final && (
        <button onClick={goToNextRound} className="bg-green-600 text-white px-4 py-2 rounded">
          다음 라운드
        </button>
      )}
      {played && final && (
        <div>
          <p className="text-xl font-bold">🏆 우승: {getCountry(final.winner!).name}</p>
          <p className="text-lg font-bold mb-2">🥉 3위: {getCountry(thirdPlace!.winner!).name}</p>
          <button onClick={reset} className="bg-gray-600 text-white px-4 py-2 rounded">
            처음부터
          </button>
        </div>
      )}
    </div>
  )
}
