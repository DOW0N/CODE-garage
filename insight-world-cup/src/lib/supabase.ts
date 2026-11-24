import { createClient } from '@supabase/supabase-js'

// publishable 키는 공개용이라 앱 코드에 넣어도 안전하다. (secret 키는 절대 넣지 않는다)
export const supabase = createClient(
  'https://jxwrxmmsnzhpxixqjbkv.supabase.co',
  'sb_publishable_Zqa6NCaKqUCNrGFH0OTv7w_0osdM5F4',
)
