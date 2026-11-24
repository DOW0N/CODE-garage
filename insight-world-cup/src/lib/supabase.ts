import { createClient } from '@supabase/supabase-js'

// publishable 키는 공개용이라 앱 코드에 넣어도 안전하다. (secret 키는 절대 넣지 않는다)
export const supabase = createClient(
  'https://jxwrxmmsnzhpxixqjbkv.supabase.co',
  'sb_publishable_Zqa6NCaKqUCNrGFH0OTv7w_0osdM5F4',
  // 개발 중(npm run dev)에는 탭마다 다른 플레이어가 되도록 로그인을 탭별로 저장한다.
  // 실제 배포판은 브라우저에 저장해서, 앱을 껐다 켜도 같은 플레이어로 남는다.
  { auth: { storage: import.meta.env.DEV ? sessionStorage : localStorage } },
)
