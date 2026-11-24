-- 2단계: 채팅 (전체 채팅 + 귓속말)
-- Supabase 대시보드 → SQL Editor → 새 쿼리(+)에 전체를 붙여 넣고 Run 한다. (한 번만 실행)

create table messages (
  id bigint generated always as identity primary key,
  room_id uuid not null references rooms (id) on delete cascade,
  sender_id uuid not null,
  sender_name text not null, -- 닉네임 또는 '게임마스터'
  to_id uuid, -- 비어 있으면 전체 채팅, 값이 있으면 그 사람에게 귓속말
  body text not null,
  created_at timestamptz not null default now()
);
create index messages_room on messages (room_id, id);

-- 보안: 같은 방 사람만 읽기. 귓속말은 보낸 사람과 받는 사람만 읽기. 직접 쓰기는 막고 아래 함수로만 보낸다.
alter table messages enable row level security;
create policy "같은 방 전체 채팅 + 내 귓속말만 읽기" on messages for select using (
  is_room_member(room_id) and (to_id is null or to_id = auth.uid() or sender_id = auth.uid())
);
grant select on messages to authenticated;

-- 메시지 보내기. p_to 가 비어 있으면 전체 채팅, 값이 있으면 귓속말.
create function send_message(p_room uuid, p_body text, p_to uuid default null) returns void
language plpgsql security definer set search_path = public as $$
declare
  b text := btrim(coalesce(p_body, ''));
  r rooms;
  name text;
begin
  if auth.uid() is null then raise exception '로그인이 필요합니다.'; end if;
  select * into r from rooms where id = p_room;
  if not found then raise exception '방을 찾을 수 없습니다.'; end if;

  if r.gm_id = auth.uid() then
    name := '게임마스터';
  else
    select nickname into name from room_players where room_id = p_room and user_id = auth.uid();
    if name is null then raise exception '이 방의 참가자가 아닙니다.'; end if;
  end if;

  if b = '' then raise exception '메시지를 입력해 주세요.'; end if;
  if char_length(b) > 200 then raise exception '메시지는 200자 이하로 입력해 주세요.'; end if;

  if p_to is not null then
    if p_to = auth.uid() then raise exception '자기 자신에게는 귓속말을 보낼 수 없습니다.'; end if;
    if p_to <> r.gm_id and not exists (select 1 from room_players where room_id = p_room and user_id = p_to) then
      raise exception '귓속말 상대가 이 방에 없습니다.';
    end if;
  end if;

  insert into messages (room_id, sender_id, sender_name, to_id, body) values (p_room, auth.uid(), name, p_to, b);
end $$;

-- 실시간: 새 메시지를 바로 받는다 (귓속말은 보안 규칙 때문에 당사자에게만 전달된다)
alter publication supabase_realtime add table messages;
