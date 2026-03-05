<script setup lang="ts">
import SidebarComponent from "@/components/SidebarComponent.vue"
import FixedChatPanel from "@/components/FixedChatPanel.vue"
import GridCanvas from "@/components/GridCanvas.vue"
</script>

<template>
  <FixedChatPanel />
  <div class="layout-container">
    <SidebarComponent />
    <main class="main-content">
      <div class="main-card">
        <header class="card-head">
          <h1>Agent Canvas</h1>
          <p>Drag nodes on grid. Wheel to zoom canvas. Drag empty area to pan.</p>
        </header>
        <GridCanvas />
      </div>
    </main>
  </div>
</template>

<style scoped>
.layout-container {
  display: flex;
  height: 100vh;
  width: 100vw;
  overflow: hidden;
  background-color: var(--bg-canvas);
}

.main-content {
  flex: 1;
  padding: 2rem 360px 2rem 2rem;
  overflow: hidden;
}

.main-card {
  height: 100%;
  background: #fff;
  padding: 1.25rem;
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow-sm);
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.card-head h1 {
  margin-bottom: 0.3rem;
}

.card-head p {
  font-size: 0.92rem;
}

@media (max-width: 1024px) {
  .main-content {
    padding: 1rem;
  }
}
</style>
