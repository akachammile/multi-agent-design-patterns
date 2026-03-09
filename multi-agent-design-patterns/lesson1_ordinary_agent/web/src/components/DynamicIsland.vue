<template>
  <header class="floating-header">
    <h1>灵动岛</h1>
    <p>Drag nodes on grid. Wheel to zoom canvas. Drag empty area to pan.</p>
  </header>
</template>

<style scoped>
.floating-header {
  pointer-events: auto;
  align-self: center;
  background: rgba(255, 255, 255, 0.8) !important;
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  color: #1e2530;
  min-width: 180px;
  height: 44px;
  padding: 0 1.5rem;
  border-radius: 22px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  overflow: hidden;
  transition:
    width 0.4s cubic-bezier(0.4, 0, 0.2, 1),
    height 0.4s cubic-bezier(0.4, 0, 0.2, 1),
    border-radius 0.4s ease,
    background 0.3s ease,
    box-shadow 0.4s ease;
  border: 1px solid rgba(0, 0, 0, 0.1);
  box-shadow:
    0 20px 40px rgba(0, 0, 0, 0.3),
    inset 0 1px 1px rgba(255, 255, 255, 0.05);
  cursor: pointer;
  margin-bottom: auto;
}

.floating-header:hover {
  height: auto;
  min-height: 100px;
  min-width: 480px;
  padding: 1.2rem 2rem;
  border-radius: 28px;
  background: rgba(255, 255, 255, 0.95) !important;
  box-shadow: 0 30px 60px rgba(0, 0, 0, 0.5);
}

.floating-header h1 {
  margin: 0;
  font-size: 0.9rem;
  color: #1e2530 !important;
  font-weight: 600;
  transition: all 0.3s ease;
  white-space: nowrap;
}

.floating-header p {
  margin: 0;
  font-size: 0.85rem;
  color: #3a4a5c !important;
  line-height: 1.5;
  opacity: 0;
  max-height: 0;
  overflow: hidden;
  transform: translateY(10px);
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.floating-header:hover h1 {
  font-size: 1.15rem;
  margin-bottom: 0.6rem;
}

.floating-header:hover p {
  opacity: 1;
  max-height: 100px;
  transform: translateY(0);
}
</style>
