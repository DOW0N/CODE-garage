import type { RealtimeChannel } from '@supabase/supabase-js'
import { create } from 'zustand'
import { supabase } from '../lib/supabase'
import type { Message } from '../types/chat'
import type { Player, Room } from '../types/player'

interface RoomState {
  userId: string | null // 이 휴대폰의 익명 사용자 ID
  room: Room | null
  players: Player[]
  messages: Message[] // 전체 채팅 + 내 귓속말 (서버가 걸러서 준다)
  init: () => Promise<void> // 앱 시작 시 익명 로그인
  createRoom: () => Promise<string | null> // 실패하면 오류 메시지
  joinRoom: (code: string, nickname: string) => Promise<string | null>
  leaveRoom: () => Promise<void>
  startGame: () => Promise<string | null>
  sendMessage: (body: string, toId: string | null) => Promise<string | null> // toId 가 있으면 귓속말
}

let channel: RealtimeChannel | null = null
let signIn: Promise<string> | null = null // 로그인을 한 번만 하도록 기억해 둔다

export const useRoomStore = create<RoomState>((set, get) => {
  // 서버에서 방과 플레이어 목록을 다시 읽는다
  const refresh = async (roomId: string) => {
    const [{ data: room }, { data: players }] = await Promise.all([
      supabase.from('rooms').select('id, code, gm_id, phase').eq('id', roomId).single(),
      supabase.from('room_players').select('user_id, nickname, chips').eq('room_id', roomId).order('joined_at'),
    ])
    set({ room: room as Room | null, players: (players ?? []) as Player[] })
  }

  // 방에 들어가면 실시간 구독: 누가 들어오거나 게임이 시작되면 모든 휴대폰이 다시 읽는다
  const enter = async (roomId: string) => {
    if (channel) await supabase.removeChannel(channel)
    set({ messages: [] })
    channel = supabase
      .channel(`room-${roomId}`)
      .on('postgres_changes', { event: '*', schema: 'public', table: 'rooms', filter: `id=eq.${roomId}` }, () => refresh(roomId))
      .on('postgres_changes', { event: '*', schema: 'public', table: 'room_players', filter: `room_id=eq.${roomId}` }, () => refresh(roomId))
      .on('postgres_changes', { event: 'INSERT', schema: 'public', table: 'messages', filter: `room_id=eq.${roomId}` }, ({ new: m }) => {
        const msg = m as Message
        if (!get().messages.some((x) => x.id === msg.id)) set({ messages: [...get().messages, msg] })
      })
      .subscribe()
    // shortcut: 최근 100개만 불러온다, 그 이전 대화가 필요해지면 "더 보기"를 추가한다
    const { data: recent } = await supabase
      .from('messages')
      .select('id, sender_id, sender_name, to_id, body')
      .eq('room_id', roomId)
      .order('id', { ascending: false })
      .limit(100)
    // 불러오는 사이에 실시간으로 먼저 도착한 메시지도 잃지 않게 합친다
    const loaded = ((recent ?? []) as Message[]).reverse()
    set({ messages: [...loaded, ...get().messages.filter((m) => !loaded.some((l) => l.id === m.id))] })
    await refresh(roomId)
    return get().room ? null : '방 정보를 불러오지 못했습니다. 잠시 후 다시 시도해 주세요.'
  }

  return {
    userId: null,
    room: null,
    players: [],
    messages: [],

    init: async () => {
      signIn ??= (async () => {
        const { data } = await supabase.auth.getSession()
        if (data.session) return data.session.user.id
        const { data: anon, error } = await supabase.auth.signInAnonymously()
        if (error) throw error
        return anon.user!.id
      })()
      set({ userId: await signIn })
    },

    createRoom: async () => {
      const { data, error } = await supabase.rpc('create_room')
      if (error) return error.message
      return enter((data as Room).id)
    },

    joinRoom: async (code, nickname) => {
      const { data, error } = await supabase.rpc('join_room', { p_code: code, p_nickname: nickname })
      if (error) return error.message
      return enter(data as string)
    },

    leaveRoom: async () => {
      const room = get().room
      if (!room) return
      await supabase.rpc('leave_room', { p_room: room.id })
      if (channel) await supabase.removeChannel(channel)
      channel = null
      set({ room: null, players: [], messages: [] })
    },

    startGame: async () => {
      const room = get().room
      if (!room) return null
      const { error } = await supabase.rpc('start_game', { p_room: room.id })
      return error ? error.message : null
    },

    sendMessage: async (body, toId) => {
      const room = get().room
      if (!room) return null
      const { error } = await supabase.rpc('send_message', { p_room: room.id, p_body: body, p_to: toId })
      return error ? error.message : null
    },
  }
})
