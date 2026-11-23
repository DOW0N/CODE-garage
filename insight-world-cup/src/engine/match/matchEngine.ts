import { getCountry } from '../../data/countryStats'
import { ROUND_ORDER, type Match } from '../../types/match'

// 홈팀 승리 확률. 전력 차이 10이면 약 76%, 20이면 약 91%. (테스트용 공식)
export const homeWinChance = (homePower: number, awayPower: number): number =>
  1 / (1 + 10 ** ((awayPower - homePower) / 20))

// 경기 1개 진행 → 승자가 정해진 새 경기 반환
export const playMatch = (match: Match, random = Math.random): Match => {
  const chance = homeWinChance(getCountry(match.home).power, getCountry(match.away).power)
  return { ...match, winner: random() < chance ? match.home : match.away }
}

// 라운드의 모든 경기 진행
export const playRound = (matches: Match[], random = Math.random): Match[] =>
  matches.map((m) => playMatch(m, random))

// 승자끼리 2경기씩 묶어 다음 라운드 생성 (1·2경기 승자, 3·4경기 승자 ...)
export const createNextRound = (matches: Match[]): Match[] => {
  if (matches.some((m) => !m.winner)) throw new Error('아직 끝나지 않은 경기가 있습니다.')
  const next = ROUND_ORDER[ROUND_ORDER.indexOf(matches[0].round) + 1]
  if (!next) throw new Error('결승 이후 라운드는 없습니다.')

  const result: Match[] = []
  for (let i = 0; i < matches.length; i += 2) {
    result.push({
      id: `${next}-${i / 2 + 1}`,
      round: next,
      home: matches[i].winner!,
      away: matches[i + 1].winner!,
      winner: null,
    })
  }
  return result
}
