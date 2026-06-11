const EventEmitter = require('events')
const { WorldView } = require('prismarine-viewer/viewer')

function isDroppedItemEntity (entity) {
  return entity && (entity.name === 'item' || entity.displayName === 'Item')
}

function normalizeEntityForViewer (entity) {
  const normalized = {
    id: entity.id,
    name: entity.name,
    pos: entity.position,
    width: entity.width,
    height: entity.height,
    username: entity.username
  }

  if (isDroppedItemEntity(entity)) {
    normalized.name = null
    normalized.width = Math.max(entity.width || 0, 0.35)
    normalized.height = Math.max(entity.height || 0, 0.35)
  }

  return normalized
}

function getDroppedItemPoints (bot) {
  return Object.values(bot.entities || {})
    .filter(isDroppedItemEntity)
    .map((entity) => {
      return {
        x: entity.position.x,
        y: entity.position.y + 0.25,
        z: entity.position.z
      }
    })
}

module.exports = (bot, {
  viewDistance = 5,
  firstPerson = false,
  port = 3000,
  host = '0.0.0.0',
  prefix = '',
  updateInterval = 100
}) => {
  const express = require('express')
  const app = express()
  const http = require('http').createServer(app)
  const io = require('socket.io')(http, { path: prefix + '/socket.io' })
  const { setupRoutes } = require('prismarine-viewer/lib/common')

  setupRoutes(app, prefix)

  const sockets = []
  const primitives = {}
  const intervals = new Map()

  bot.viewer = new EventEmitter()

  bot.viewer.erase = (id) => {
    delete primitives[id]
    for (const socket of sockets) {
      socket.emit('primitive', { id })
    }
  }

  bot.viewer.drawBoxGrid = (id, start, end, color = 'aqua') => {
    primitives[id] = { type: 'boxgrid', id, start, end, color }
    for (const socket of sockets) {
      socket.emit('primitive', primitives[id])
    }
  }

  bot.viewer.drawLine = (id, points, color = 0xff0000) => {
    primitives[id] = { type: 'line', id, points, color }
    for (const socket of sockets) {
      socket.emit('primitive', primitives[id])
    }
  }

  bot.viewer.drawPoints = (id, points, color = 0xffd400, size = 10) => {
    primitives[id] = { type: 'points', id, points, color, size }
    for (const socket of sockets) {
      socket.emit('primitive', primitives[id])
    }
  }

  io.on('connection', (socket) => {
    console.log(`ADAM viewer client connected: ${socket.id}`)
    socket.emit('version', bot.version)
    sockets.push(socket)

    const worldView = new WorldView(bot.world, viewDistance, bot.entity.position, socket)
    worldView.init(bot.entity.position)

    worldView.on('blockClicked', (block, face, button) => {
      bot.viewer.emit('blockClicked', block, face, button)
    })

    for (const id in primitives) {
      socket.emit('primitive', primitives[id])
    }

    function botPosition () {
      const packet = { pos: bot.entity.position, yaw: bot.entity.yaw, addMesh: true }
      if (firstPerson) {
        packet.pitch = bot.entity.pitch
      }
      socket.emit('position', packet)
      worldView.updatePosition(bot.entity.position)
    }

    function droppedItemsOverlay () {
      const points = getDroppedItemPoints(bot)
      if (points.length) {
        socket.emit('primitive', {
          type: 'points',
          id: 'adam_dropped_items',
          points,
          color: 0xffd400,
          size: 12
        })
      } else {
        socket.emit('primitive', { id: 'adam_dropped_items' })
      }
    }

    function refreshDynamicView () {
      botPosition()
      droppedItemsOverlay()
    }

    const originalEntitySpawn = (entity) => {
      if (entity === bot.entity) return
      socket.emit('entity', normalizeEntityForViewer(entity))
    }
    const originalEntityMoved = (entity) => {
      const payload = normalizeEntityForViewer(entity)
      payload.pitch = entity.pitch
      payload.yaw = entity.yaw
      socket.emit('entity', payload)
    }
    const originalEntityGone = (entity) => {
      socket.emit('entity', { id: entity.id, delete: true })
    }

    for (const id in bot.entities) {
      const entity = bot.entities[id]
      if (entity && entity !== bot.entity) {
        socket.emit('entity', normalizeEntityForViewer(entity))
      }
    }

    bot.on('move', botPosition)
    bot.on('entitySpawn', originalEntitySpawn)
    bot.on('entityMoved', originalEntityMoved)
    bot.on('entityGone', originalEntityGone)
    worldView.listenToBot(bot)

    const interval = setInterval(refreshDynamicView, updateInterval)
    intervals.set(socket, interval)
    refreshDynamicView()

    socket.on('disconnect', () => {
      console.log(`ADAM viewer client disconnected: ${socket.id}`)
      bot.removeListener('move', botPosition)
      bot.removeListener('entitySpawn', originalEntitySpawn)
      bot.removeListener('entityMoved', originalEntityMoved)
      bot.removeListener('entityGone', originalEntityGone)
      worldView.removeListenersFromBot(bot)
      clearInterval(intervals.get(socket))
      intervals.delete(socket)
      sockets.splice(sockets.indexOf(socket), 1)
    })
  })

  http.listen(port, host, () => {
    console.log(`ADAM viewer web server running on ${host}:${port}`)
  })

  bot.viewer.close = () => {
    for (const interval of intervals.values()) {
      clearInterval(interval)
    }
    intervals.clear()
    http.close()
    for (const socket of sockets) {
      socket.disconnect()
    }
  }
}
