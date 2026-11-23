import type { Match } from '../../types/match'

// [홈, 원정] 16경기. 1경기 승자와 2경기 승자가 16강에서 만난다.
const pairs: [string, string][] = [
  ['BRA', 'CRC'],
  ['SUI', 'POL'],
  ['ARG', 'KSA'],
  ['MEX', 'AUS'],
  ['FRA', 'GHA'],
  ['DEN', 'SRB'],
  ['ENG', 'IRN'],
  ['USA', 'ECU'],
  ['ESP', 'CAN'],
  ['JPN', 'NGA'],
  ['GER', 'EGY'],
  ['COL', 'KOR'],
  ['POR', 'SEN'],
  ['BEL', 'CRO'],
  ['URU', 'ITA'],
  ['NED', 'MAR'],
]

export const round32: Match[] = pairs.map(([home, away], i) => ({
  id: `R32-${i + 1}`,
  round: 'R32',
  home,
  away,
  winner: null,
}))
