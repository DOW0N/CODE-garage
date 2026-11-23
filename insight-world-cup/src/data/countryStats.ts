import type { Country } from '../types/country'

// 테스트용 전력 값. 최종 밸런스는 미확정.
export const countries: Country[] = [
  { code: 'BRA', name: '브라질', power: 95 },
  { code: 'ARG', name: '아르헨티나', power: 94 },
  { code: 'FRA', name: '프랑스', power: 94 },
  { code: 'ENG', name: '잉글랜드', power: 92 },
  { code: 'ESP', name: '스페인', power: 92 },
  { code: 'GER', name: '독일', power: 91 },
  { code: 'POR', name: '포르투갈', power: 91 },
  { code: 'URU', name: '우루과이', power: 91 },
  { code: 'NED', name: '네덜란드', power: 90 },
  { code: 'MAR', name: '모로코', power: 90 },
  { code: 'BEL', name: '벨기에', power: 89 },
  { code: 'CRO', name: '크로아티아', power: 89 },
  { code: 'ITA', name: '이탈리아', power: 89 },
  { code: 'COL', name: '콜롬비아', power: 88 },
  { code: 'SEN', name: '세네갈', power: 87 },
  { code: 'JPN', name: '일본', power: 86 },
  { code: 'MEX', name: '멕시코', power: 86 },
  { code: 'USA', name: '미국', power: 85 },
  { code: 'SUI', name: '스위스', power: 85 },
  { code: 'DEN', name: '덴마크', power: 85 },
  { code: 'KOR', name: '대한민국', power: 84 },
  { code: 'NGA', name: '나이지리아', power: 84 },
  { code: 'POL', name: '폴란드', power: 83 },
  { code: 'SRB', name: '세르비아', power: 83 },
  { code: 'ECU', name: '에콰도르', power: 82 },
  { code: 'AUS', name: '호주', power: 81 },
  { code: 'IRN', name: '이란', power: 81 },
  { code: 'CAN', name: '캐나다', power: 80 },
  { code: 'EGY', name: '이집트', power: 80 },
  { code: 'KSA', name: '사우디아라비아', power: 78 },
  { code: 'GHA', name: '가나', power: 78 },
  { code: 'CRC', name: '코스타리카', power: 76 },
]

export const getCountry = (code: string): Country => {
  const country = countries.find((c) => c.code === code)
  if (!country) throw new Error(`알 수 없는 국가 코드: ${code}`)
  return country
}
