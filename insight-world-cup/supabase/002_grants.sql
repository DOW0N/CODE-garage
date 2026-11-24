-- 001 다음에 실행. 로그인한 사용자가 방 정보를 읽을 수 있게 허용한다.
-- (누가 무엇을 읽을 수 있는지는 001의 보안 규칙이 그대로 제한한다. 쓰기 권한은 주지 않는다.)
grant select on rooms, room_players to authenticated;
