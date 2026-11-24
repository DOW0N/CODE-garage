import type { RealtimeChannel } from '@supabase/supabase-js'
import { create } from 'zustand'
import { supabase } from '../lib/supabase'
import type { Player, Room } from '../types/player'

interface RoomState {
  userId: string | null // 이 휴대폰의 익명 사용자 ID
  room: Room | null
  players: Player[]
  init: () => Promise<void> // 앱 시작 시 익명 로그인
  createRoom: () => Promise<string | null> // 실패하면 오류 메시지
  joinRoom: (code: string, nickname: string) => Promise<string | null>
  leaveRoom: () => Promise<void>
  startGame: () => Promise<string | null>
}

let channel: RealtimeChannel | null = null

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
    channel = supabase
      .channel(`room-${roomId}`)
      .on('postgres_changes', { event: '*', schema: 'public', table: 'rooms', filter: `id=eq.${roomId}` }, () => refresh(roomId))
      .on('postgres_changes', { event: '*', schema: 'public', table: 'room_players', filter: `room_id=eq.${roomId}` }, () => refresh(roomId))
      .subscribe()
    await refresh(roomId)
  }

  return {
    userId: null,
    room: null,
    players: [],

    init: async () => {
      const { data } = await supabase.auth.getSession()
      if (data.session) return set({ userId: data.session.user.id })
      const { data: signIn, error } = await supabase.auth.signInAnonymously()
      if (error) throw error
      set({ userId: signIn.user!.id })
    },

    createRoom: async () => {
      const { data, error } = await supabase.rpc('create_room')
      if (error) return error.message
      await enter((data as Room).id)
      return null
    },

    joinRoom: async (code, nickname) => {
      const { data, error } = await supabase.rpc('join_room', { p_code: code, p_nickname: nickname })
      if (error) return error.message
      await enter(data as string)
      return null
    },

    leaveRoom: async () => {
      const room = get().room
      if (!room) return
      await supabase.rpc('leave_room', { p_room: room.id })
      if (channel) await supabase.removeChannel(channel)
      channel = null
      set({ room: null, players: [] })
    },

    startGame: async () => {
      const room = get().room
      if (!room) return null
      const { error } = await supabase.rpc('start_game', { p_room: room.id })
      return error ? error.message : null
    },
  }
})
