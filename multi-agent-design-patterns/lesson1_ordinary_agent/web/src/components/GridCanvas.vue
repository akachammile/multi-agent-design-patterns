<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from "vue"

type GridNode = {
  id: string
  title: string
  x: number
  y: number
  width: number
  height: number
}

const WORLD_WIDTH = 2400
const WORLD_HEIGHT = 1400
const WORLD_PADDING = 20
const GRID_SIZE = 24
const MIN_SCALE = 0.6
const MAX_SCALE = 1.8
const ZOOM_STEP = 0.08

const viewportRef = ref<HTMLElement | null>(null)
const viewportWidth = ref(0)
const viewportHeight = ref(0)
const panX = ref(28)
const panY = ref(28)
const scale = ref(1)

const nodes = ref<GridNode[]>([
  { id: "agent-1", title: "Planner", x: 120, y: 120, width: 200, height: 96 },
  { id: "agent-2", title: "Executor", x: 384, y: 120, width: 200, height: 96 },
  { id: "agent-3", title: "Memory", x: 120, y: 288, width: 200, height: 96 },
])

const panState = ref({
  active: false,
  pointerId: -1,
  startClientX: 0,
  startClientY: 0,
  originPanX: 0,
  originPanY: 0,
})

const dragState = ref({
  active: false,
  pointerId: -1,
  nodeId: "",
  startClientX: 0,
  startClientY: 0,
  originNodeX: 0,
  originNodeY: 0,
})

const workspaceStyle = computed(() => ({
  transform: `translate(${panX.value}px, ${panY.value}px) scale(${scale.value})`,
}))

const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value))

const snapToGrid = (value: number) => Math.round(value / GRID_SIZE) * GRID_SIZE

const findNodeById = (id: string) => nodes.value.find((node) => node.id === id)

const refreshViewport = () => {
  const rect = viewportRef.value?.getBoundingClientRect()
  if (!rect) {
    return
  }
  viewportWidth.value = rect.width
  viewportHeight.value = rect.height
  clampPanToBounds()
}

const clampPanToBounds = () => {
  const scaledWorldWidth = WORLD_WIDTH * scale.value
  const scaledWorldHeight = WORLD_HEIGHT * scale.value

  const minPanX = viewportWidth.value - scaledWorldWidth - WORLD_PADDING
  const maxPanX = WORLD_PADDING
  const minPanY = viewportHeight.value - scaledWorldHeight - WORLD_PADDING
  const maxPanY = WORLD_PADDING

  panX.value = clamp(panX.value, Math.min(minPanX, maxPanX), Math.max(minPanX, maxPanX))
  panY.value = clamp(panY.value, Math.min(minPanY, maxPanY), Math.max(minPanY, maxPanY))
}

const stopPan = (pointerId: number) => {
  if (!panState.value.active) {
    return
  }
  panState.value.active = false
  viewportRef.value?.releasePointerCapture(pointerId)
}

const stopNodeDrag = (pointerId: number) => {
  if (!dragState.value.active) {
    return
  }
  dragState.value.active = false
  dragState.value.nodeId = ""
  viewportRef.value?.releasePointerCapture(pointerId)
}

const onViewportPointerDown = (event: PointerEvent) => {
  if (event.button !== 0) {
    return
  }
  if ((event.target as HTMLElement | null)?.closest(".grid-node")) {
    return
  }

  panState.value = {
    active: true,
    pointerId: event.pointerId,
    startClientX: event.clientX,
    startClientY: event.clientY,
    originPanX: panX.value,
    originPanY: panY.value,
  }
  viewportRef.value?.setPointerCapture(event.pointerId)
}

const onNodePointerDown = (node: GridNode, event: PointerEvent) => {
  if (event.button !== 0) {
    return
  }

  dragState.value = {
    active: true,
    pointerId: event.pointerId,
    nodeId: node.id,
    startClientX: event.clientX,
    startClientY: event.clientY,
    originNodeX: node.x,
    originNodeY: node.y,
  }
  viewportRef.value?.setPointerCapture(event.pointerId)
}

const onViewportPointerMove = (event: PointerEvent) => {
  if (dragState.value.active && event.pointerId === dragState.value.pointerId) {
    const node = findNodeById(dragState.value.nodeId)
    if (!node) {
      return
    }

    const deltaWorldX = (event.clientX - dragState.value.startClientX) / scale.value
    const deltaWorldY = (event.clientY - dragState.value.startClientY) / scale.value

    const rawX = dragState.value.originNodeX + deltaWorldX
    const rawY = dragState.value.originNodeY + deltaWorldY
    const maxX = WORLD_WIDTH - node.width
    const maxY = WORLD_HEIGHT - node.height

    node.x = clamp(snapToGrid(rawX), 0, maxX)
    node.y = clamp(snapToGrid(rawY), 0, maxY)
    return
  }

  if (panState.value.active && event.pointerId === panState.value.pointerId) {
    const deltaX = event.clientX - panState.value.startClientX
    const deltaY = event.clientY - panState.value.startClientY
    panX.value = panState.value.originPanX + deltaX
    panY.value = panState.value.originPanY + deltaY
    clampPanToBounds()
  }
}

const onViewportPointerUp = (event: PointerEvent) => {
  if (dragState.value.active && event.pointerId === dragState.value.pointerId) {
    stopNodeDrag(event.pointerId)
    return
  }
  if (panState.value.active && event.pointerId === panState.value.pointerId) {
    stopPan(event.pointerId)
  }
}

const onViewportWheel = (event: WheelEvent) => {
  event.preventDefault()
  if (!viewportRef.value) {
    return
  }

  const oldScale = scale.value
  const direction = event.deltaY < 0 ? 1 : -1
  const nextScale = clamp(oldScale + direction * ZOOM_STEP, MIN_SCALE, MAX_SCALE)

  if (nextScale === oldScale) {
    return
  }

  const rect = viewportRef.value.getBoundingClientRect()
  const cursorX = event.clientX - rect.left
  const cursorY = event.clientY - rect.top
  const worldX = (cursorX - panX.value) / oldScale
  const worldY = (cursorY - panY.value) / oldScale

  scale.value = nextScale
  panX.value = cursorX - worldX * nextScale
  panY.value = cursorY - worldY * nextScale
  clampPanToBounds()
}

onMounted(() => {
  refreshViewport()
  window.addEventListener("resize", refreshViewport)
})

onBeforeUnmount(() => {
  window.removeEventListener("resize", refreshViewport)
})
</script>

<template>
  <section
    ref="viewportRef"
    class="grid-viewport"
    :class="{ 'is-panning': panState.active, 'is-dragging-node': dragState.active }"
    @pointerdown="onViewportPointerDown"
    @pointermove="onViewportPointerMove"
    @pointerup="onViewportPointerUp"
    @pointercancel="onViewportPointerUp"
    @wheel="onViewportWheel"
  >
    <div class="workspace" :style="workspaceStyle">
      <div class="workspace-boundary" aria-hidden="true" />

      <article
        v-for="node in nodes"
        :key="node.id"
        class="grid-node"
        :style="{
          left: `${node.x}px`,
          top: `${node.y}px`,
          width: `${node.width}px`,
          height: `${node.height}px`,
        }"
        @pointerdown.stop="onNodePointerDown(node, $event)"
      >
        <h4>{{ node.title }}</h4>
        <p>{{ node.id }}</p>
      </article>
    </div>
  </section>
</template>

<style scoped>
.grid-viewport {
  position: relative;
  width: 100%;
  height: 100%;
  border: 1px solid color-mix(in srgb, var(--border-default) 80%, #fff);
  border-radius: var(--radius-lg);
  overflow: hidden;
  background: color-mix(in srgb, var(--bg-canvas) 65%, #fff);
}

.grid-viewport.is-panning {
  cursor: grabbing;
}

.workspace {
  position: absolute;
  left: 0;
  top: 0;
  width: 2400px;
  height: 1400px;
  transform-origin: 0 0;
  background-image:
    linear-gradient(to right, rgba(20, 38, 66, 0.12) 1px, transparent 1px),
    linear-gradient(to bottom, rgba(20, 38, 66, 0.12) 1px, transparent 1px);
  background-size: 24px 24px;
}

.workspace-boundary {
  position: absolute;
  inset: 0;
  border: 2px dashed color-mix(in srgb, var(--border-strong) 80%, #fff);
  border-radius: 14px;
  pointer-events: none;
}

.grid-node {
  position: absolute;
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: 0.2rem;
  padding: 0.8rem 0.95rem;
  border-radius: 14px;
  border: 1px solid color-mix(in srgb, var(--brand-500) 25%, #fff);
  background: linear-gradient(160deg, #ffffff 0%, #f2f7ff 95%);
  box-shadow: 0 10px 28px rgba(20, 38, 66, 0.08);
  cursor: grab;
  user-select: none;
}

.grid-node:active {
  cursor: grabbing;
}

.grid-node h4 {
  font-size: 0.95rem;
  font-weight: 700;
}

.grid-node p {
  font-size: 0.78rem;
  color: var(--text-muted);
}
</style>
