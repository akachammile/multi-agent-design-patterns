<script setup lang="ts">
import SidebarComponent from "@/components/SidebarComponent.vue"
import FixedChatPanel from "@/components/FixedChatPanel.vue"
import GridCanvas from "@/components/GridCanvas.vue"
</script>

<template>
  <!-- 整个应用的根容器 -->
  <div class="app-root">

    <!-- 画布层：全屏底色背景 -->
    <div class="background-canvas">
      <GridCanvas />
    </div>

    <!-- UI 交互层：使用 Flex 布局挤压空间，视觉完全透明 -->
    <div class="layout-container">
      <!-- 1. 左侧边栏 -->
      <SidebarComponent />

      <!-- 2. 主体区域：承载悬浮 UI -->
      <div class="main-stage">
        <div class="ui-overlay">
          <main class="main-content">
            <header class="floating-header">
              <h1>灵动岛</h1>
              <p>Drag nodes on grid. Wheel to zoom canvas. Drag empty area to pan.</p>
            </header>
          </main>
        </div>
      </div>

      <!-- 3. 右侧对话面板 -->
      <FixedChatPanel />
    </div>
  </div>
</template>

<style scoped>
/* 根容器充满全屏 */
.app-root {
  position: relative;
  width: 100vw;
  height: 100vh;
  overflow: hidden;
}

/* 背景画布：全屏拉伸 */
.background-canvas {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  z-index: 0;
}

/* 侧边栏与主舞台的水平容器 */
.layout-container {
  position: relative;
  z-index: 10;
  display: flex;
  width: 100%;
  height: 100%;
  background: transparent;
  transition: all 0.3s ease;
  pointer-events: none;
  /* 关键：允许鼠标事件穿透 */
}

/* 恢复左右侧边栏的交互 */
.layout-container> :deep(.capsule-sidebar),
.layout-container> :deep(.chat-shell) {
  pointer-events: auto;
}

/* 主舞台：承载悬浮 UI */
.main-stage {
  position: relative;
  flex: 1;
  height: 100%;
  background: transparent;
  pointer-events: none;
  /* 确保中间区域透明穿透 */
}

/* UI 悬浮层：覆盖层级，但不阻碍鼠标点击画布 */
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
  /* 关键：容器本身不响应点击，允许穿透到下层画布 */
  pointer-events: none;
}

/* 悬浮标题栏 - 灵动岛风格 (高级深色毛玻璃版) */
.floating-header {
  pointer-events: auto;
  align-self: center;

  /* 基础视觉：深色毛玻璃效果 */
  background: rgba(15, 23, 42, 0.8);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  color: #ffffff;

  /* 初始尺寸：紧凑胶囊型 */
  min-width: 180px;
  height: 44px;
  padding: 0 1.5rem;
  border-radius: 22px;

  /* 布局控制 */
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  overflow: hidden;

  /* 核心动效：贝塞尔曲线 */
  transition:
    width 0.4s cubic-bezier(0.4, 0, 0.2, 1),
    height 0.4s cubic-bezier(0.4, 0, 0.2, 1),
    border-radius 0.4s ease,
    background 0.3s ease,
    box-shadow 0.4s ease;

  /* 边框与阴影：增强立体感 */
  border: 1px solid rgba(255, 255, 255, 0.1);
  box-shadow:
    0 20px 40px rgba(0, 0, 0, 0.3),
    inset 0 1px 1px rgba(255, 255, 255, 0.05);
  cursor: pointer;
  margin-bottom: auto;
}

/* 灵动岛扩张状态 */
.floating-header:hover {
  height: auto;
  min-height: 100px;
  min-width: 480px;
  padding: 1.2rem 2rem;
  border-radius: 28px;
  background: rgba(15, 23, 42, 0.95);
  box-shadow: 0 30px 60px rgba(0, 0, 0, 0.5);
}

.floating-header h1 {
  margin: 0;
  font-size: 0.9rem;
  color: #ffffff;
  font-weight: 600;
  transition: all 0.3s ease;
  white-space: nowrap;
}

.floating-header p {
  margin: 0;
  font-size: 0.85rem;
  color: rgba(255, 255, 255, 0.6);
  line-height: 1.5;

  /* 内容显隐控制 */
  opacity: 0;
  max-height: 0;
  overflow: hidden;
  transform: translateY(10px);
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

/* 悬浮时内容平滑显现 */
.floating-header:hover h1 {
  font-size: 1.15rem;
  margin-bottom: 0.6rem;
}

.floating-header:hover p {
  opacity: 1;
  max-height: 100px;
  transform: translateY(0);
}

@media (max-width: 1024px) {
  .main-content {
    padding: 1rem;
  }
}
</style>