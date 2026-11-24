-- 1단계: 방 만들기 / 참가 / 게임 시작
-- Supabase 대시보드 → SQL Editor 에 전체를 붙여 넣고 Run 한다. (한 번만 실행)

-- 방
create table rooms (
  id uuid primary key default gen_random_uuid(),
  code text not null unique, -- 참가용 6자리 숫자
  gm_id uuid not null, -- 게임마스터
  phase text not null default 'lobby' check (phase in ('lobby', 'playing')),
  created_at timestamptz not null default now()
);

-- 방에 참가한 플레이어 (게임마스터는 여기에 들어가지 않는다)
create table room_players (
  room_id uuid not null references rooms (id) on delete cascade,
  user_id uuid not null,
  nickname text not null,
  chips int not null default 30,
  joined_at timestamptz not null default now(),
  primary key (room_id, user_id)
);
create unique index room_players_nickname on room_players (room_id, lower(nickname));

-- 보안: 같은 방 사람만 읽을 수 있고, 직접 쓰기는 전부 막는다. 쓰기는 아래 함수로만 한다.
alter table rooms enable row level security;
alter table room_players enable row level security;

create function is_room_member(p_room uuid) returns boolean
language sql stable security definer set search_path = public as $$
  select exists (select 1 from rooms where id = p_room and gm_id = auth.uid())
      or exists (select 1 from room_players where room_id = p_room and user_id = auth.uid());
$$;

create policy "같은 방 사람만 방 정보 읽기" on rooms for select using (is_room_member(id));
create policy "같은 방 사람만 플레이어 목록 읽기" on room_players for select using (is_room_member(room_id));

-- 방 만들기: 만든 사람이 게임마스터
create function create_room() returns rooms
language plpgsql security definer set search_path = public as $$
declare
  c text;
  r rooms;
begin
  if auth.uid() is null then raise exception '로그인이 필요합니다.'; end if;
  loop
    c := lpad(floor(random() * 1000000)::int::text, 6, '0');
    exit when not exists (select 1 from rooms where code = c);
  end loop;
  insert into rooms (code, gm_id) values (c, auth.uid()) returning * into r;
  return r;
end $$;

-- 방 참가. 이미 참가한 사람(또는 게임마스터)이 같은 코드로 들어오면 다시 입장시킨다.
create function join_room(p_code text, p_nickname text) returns uuid
language plpgsql security definer set search_path = public as $$
declare
  r rooms;
  n text := btrim(coalesce(p_nickname, ''));
begin
  if auth.uid() is null then raise exception '로그인이 필요합니다.'; end if;
  -- for update: 동시에 여러 명이 들어와도 인원 제한이 지켜지도록 방을 잠근다
  select * into r from rooms where code = btrim(p_code) for update;
  if not found then raise exception '방 코드를 찾을 수 없습니다.'; end if;
  if r.gm_id = auth.uid() then return r.id; end if;
  if exists (select 1 from room_players where room_id = r.id and user_id = auth.uid()) then return r.id; end if;
  if r.phase <> 'lobby' then raise exception '이미 시작된 게임입니다.'; end if;
  if n = '' then raise exception '닉네임을 입력해 주세요.'; end if;
  if char_length(n) > 12 then raise exception '닉네임은 12자 이하로 입력해 주세요.'; end if;
  if (select count(*) from room_players where room_id = r.id) >= 10 then
    raise exception '방이 가득 찼습니다. (최대 10명)';
  end if;
  if exists (select 1 from room_players where room_id = r.id and lower(nickname) = lower(n)) then
    raise exception '이미 사용 중인 닉네임입니다.';
  end if;
  insert into room_players (room_id, user_id, nickname) values (r.id, auth.uid(), n);
  return r.id;
end $$;

-- 대기실에서 나가기 (플레이어만)
create function leave_room(p_room uuid) returns void
language plpgsql security definer set search_path = public as $$
begin
  delete from room_players
  where room_id = p_room and user_id = auth.uid()
    and (select phase from rooms where id = p_room) = 'lobby';
end $$;

-- 게임 시작: 게임마스터만, 플레이어 6명 이상
create function start_game(p_room uuid) returns void
language plpgsql security definer set search_path = public as $$
begin
  update rooms set phase = 'playing'
  where id = p_room and gm_id = auth.uid() and phase = 'lobby'
    and (select count(*) from room_players where room_id = p_room) >= 6;
  if not found then raise exception '게임을 시작할 수 없습니다. (게임마스터만, 플레이어 6명 이상)'; end if;
end $$;

-- 실시간: 방과 플레이어 목록이 바뀌면 모든 휴대폰에 알린다
alter publication supabase_realtime add table rooms, room_players;
