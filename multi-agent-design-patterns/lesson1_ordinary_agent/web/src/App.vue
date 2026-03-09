<script setup lang="ts">
import SidebarComponent from "@/components/SidebarComponent.vue"
import FixedChatPanel from "@/components/FixedChatPanel.vue"
import GridCanvas from "@/components/GridCanvas.vue"
import DynamicIsland from "@/components/DynamicIsland.vue"
</script>

<template>
  <div class="app-root">
    <div class="background-canvas">
      <GridCanvas />
    </div>

    <div class="layout-container">
      <SidebarComponent />

      <div class="main-stage">
        <div class="ui-overlay">
          <main class="main-content">
            <DynamicIsland />
          </main>
        </div>
      </div>

      <FixedChatPanel />
    </div>
  </div>
</template>

<style scoped>
.app-root {
  position: relative;
  width: 100vw;
  height: 100vh;
  overflow: hidden;
}

.background-canvas {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  z-index: 0;
}

.layout-container {
  position: relative;
  z-index: 10;
  display: flex;
  width: 100%;
  height: 100%;
  background: transparent;
  transition: all 0.3s ease;
  pointer-events: none;
}

.layout-container > :deep(.capsule-sidebar),
.layout-container > :deep(.chat-shell) {
  pointer-events: auto;
}

.main-stage {
  position: relative;
  flex: 1;
  height: 100%;
  background: transparent;
  pointer-events: none;
}

.ui-overlay {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  display: flex;
}

.main-content {
  flex: 1;
  padding: 1.5rem;
  display: flex;
  flex-direction: column;
  pointer-events: none;
}

@media (max-width: 1024px) {
  .main-content {
    padding: 1rem;
  }
}
</style>
