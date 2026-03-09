<script setup lang="ts">
import { ref } from "vue"
import { Home, LineChart, Settings } from 'lucide-vue-next'

const activeMenu = ref("overview")

const handleSelect = (key: string) => {
  activeMenu.value = key
}
</script>

<template>
  <aside class="capsule-sidebar">
    <!-- 导航菜单 -->
    <div class="sidebar-nav">
      <div class="nav-item" :class="{ active: activeMenu === 'overview' }" @click="handleSelect('overview')"
        data-title="首页概览">
        <Home :size="20" :stroke-width="2.5" />
      </div>
      <div class="nav-item" :class="{ active: activeMenu === 'analysis' }" @click="handleSelect('analysis')"
        data-title="数据分析">
        <LineChart :size="20" :stroke-width="2.5" />
      </div>
      <div class="nav-item" :class="{ active: activeMenu === 'settings' }" @click="handleSelect('settings')"
        data-title="系统设置">
        <Settings :size="20" :stroke-width="2.5" />
      </div>
    </div>
  </aside>
</template>

<style scoped>
/* 胶囊悬浮侧边栏 */
.capsule-sidebar {
  /* 调整尺寸，让图标有更充裕的空间居中 */
  width: 56px;
  height: auto;
  padding: 1.5rem 0;
  margin: auto 0 auto 1.5rem;
  align-self: center;

  display: flex;
  flex-direction: column;
  align-items: center;

  /* 使用主题变量统一背景，摒弃带有黑色杂质的毛玻璃 */
  background: var(--bg-elevated);
  border-radius: 32px;
  border: 1px solid var(--border-default);

  /* 彻底移除黑色阴影，使用系统全局的清爽卡片阴影 */
  box-shadow: var(--shadow-md);

  z-index: 20;
  transition: all 0.3s ease;
}

/* 导航图标区 */
.sidebar-nav {
  display: flex;
  flex-direction: column;
  gap: 16px;
  width: 100%;
  align-items: center;
}

.nav-item {
  width: 38px;
  height: 38px;
  display: flex;
  justify-content: center;
  align-items: center;
  border-radius: 10px;
  color: var(--text-secondary);
  cursor: pointer;
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
  background: transparent;
  position: relative;
}

/* Tooltip 弹出层 */
.nav-item::after {
  content: attr(data-title);
  position: absolute;
  left: 100%;
  margin-left: 14px;
  padding: 6px 10px;
  background: #000000;
  color: #ffffff;
  font-size: 12px;
  border-radius: 6px;
  white-space: nowrap;
  opacity: 0;
  visibility: hidden;
  transform: translateX(-10px);
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
  pointer-events: none;
  z-index: 100;
}

/* Tooltip 小三角箭头 */
.nav-item::before {
  content: '';
  position: absolute;
  left: 100%;
  margin-left: 6px;
  border: 4px solid transparent;
  border-right-color: #000000;
  opacity: 0;
  visibility: hidden;
  transform: translateX(-10px);
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
  pointer-events: none;
  z-index: 100;
}

.nav-item:hover::after,
.nav-item:hover::before {
  opacity: 1;
  visibility: visible;
  transform: translateX(0);
}

.nav-item:hover {
  background: #f3f4f6; /* 淡灰色 */
  color: var(--text-primary, #111827);
  border-radius: 10px;
  transform: scale(1.05);
}

.nav-item.active {
  background: #000000;
  color: #ffffff;
  border-radius: 10px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  transform: scale(1.05);
}
</style>
