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
const MIN_SCALE = 0.8 // 限制最小缩放到 80% (原来是 0.6)
const MAX_SCALE = 1.2 // 限制最大放大到 120% (原来是 1.8)
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
  // 必须按住 Ctrl/Cmd 才能拖拽画布
  if (!event.ctrlKey && !event.metaKey) {
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

  // 不按 Ctrl/Cmd 时，执行画布上下/左右滚动
  if (!event.ctrlKey && !event.metaKey) {
    panX.value -= event.deltaX
    panY.value -= event.deltaY
    clampPanToBounds()
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
  /* 移除了边框和圆角，让它能够真正充当“整个页面的底层” */
  overflow: hidden;
  /* 温润的米黄色背景 */
  background: #fdfbf7; 
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
  /* 现代感的点阵网格，调整颜色以适应米黄色底色 */
  background-image: radial-gradient(circle, rgba(160, 150, 140, 0.25) 1.5px, transparent 1.5px);
  background-size: 24px 24px;
}

.workspace-boundary {
  position: absolute;
  inset: 0;
  border: 2px dashed rgba(160, 150, 140, 0.2);
  border-radius: 14px;
  pointer-events: none;
}

.grid-node {
  position: absolute;
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: 0.4rem;
  padding: 1rem 1.2rem;
  border-radius: 16px;
  /* 偏白且带边缘锐化 */
  border: 1px solid rgba(0, 0, 0, 0.06); 
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(12px);
  /* 强化阴影，与背景产生锐利对比 */
  box-shadow: 
    0 4px 12px -2px rgba(0, 0, 0, 0.05),
    0 2px 4px -1px rgba(0, 0, 0, 0.03),
    inset 0 1px 0 rgba(255, 255, 255, 1),
    inset 0 0 0 1px rgba(255, 255, 255, 0.5);
  cursor: grab;
  user-select: none;
  transition: box-shadow 0.2s ease, transform 0.2s ease;
  overflow: hidden;
}

/* 节点左侧的彩色强调条 */
.grid-node::before {
  content: "";
  position: absolute;
  left: 0;
  top: 0;
  bottom: 0;
  width: 6px;
  background: linear-gradient(180deg, #4facfe 0%, #00f2fe 100%);
  border-radius: 16px 0 0 16px;
}

/* 悬浮时的交互反馈：轻微上浮，阴影加深 */
.grid-node:hover {
  transform: translateY(-2px);
  box-shadow: 
    0 12px 20px -4px rgba(0, 0, 0, 0.08),
    0 4px 8px -2px rgba(0, 0, 0, 0.04),
    inset 0 1px 0 rgba(255, 255, 255, 1),
    inset 0 0 0 1px rgba(255, 255, 255, 0.5);
}

.grid-node:active {
  cursor: grabbing;
  transform: translateY(0);
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
}

.grid-node h4 {
  font-size: 1.05rem;
  font-weight: 700;
  color: #1e293b;
  margin: 0;
  padding-left: 8px;
}

.grid-node p {
  font-size: 0.8rem;
  color: #64748b;
  margin: 0;
  padding-left: 8px;
}
</style>
