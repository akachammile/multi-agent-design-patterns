<script setup lang="ts">
import { ref } from 'vue'

// 控制侧边栏展开/收起的状态
const isCollapsed = ref(false)

const toggleSidebar = () => {
  isCollapsed.value = !isCollapsed.value
}
</script>

<template>
  <aside class="sidebar" :class="{ 'collapsed': isCollapsed }">
    <div class="sidebar-header">
      <h3 v-show="!isCollapsed">菜单导航</h3>
      <button class="toggle-btn" @click="toggleSidebar">
        {{ isCollapsed ? '👉' : '👈' }}
      </button>
    </div>
    
    <nav class="sidebar-nav">
      <ul>
        <li>
          <a href="#">
            <span class="icon">🏠</span>
            <span class="text" v-show="!isCollapsed">首页概览</span>
          </a>
        </li>
        <li>
          <a href="#">
            <span class="icon">📊</span>
            <span class="text" v-show="!isCollapsed">数据分析</span>
          </a>
        </li>
        <li>
          <a href="#">
            <span class="icon">⚙️</span>
            <span class="text" v-show="!isCollapsed">系统设置</span>
          </a>
        </li>
      </ul>
    </nav>
  </aside>
</template>

<style scoped>
.sidebar {
  width: 240px;
  height: 100vh;
  background-color: #2c3e50;
  color: #ecf0f1;
  transition: width 0.3s ease;
  display: flex;
  flex-direction: column;
}

.sidebar.collapsed {
  width: 64px;
}

.sidebar-header {
  height: 60px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 16px;
  border-bottom: 1px solid #34495e;
}

.sidebar-header h3 {
  margin: 0;
  font-size: 16px;
  white-space: nowrap;
}

.toggle-btn {
  background: none;
  border: none;
  cursor: pointer;
  font-size: 16px;
  padding: 4px;
  color: #ecf0f1;
}

.toggle-btn:hover {
  opacity: 0.8;
}

.sidebar-nav {
  flex: 1;
  overflow-y: auto;
  padding-top: 10px;
}

.sidebar-nav ul {
  list-style: none;
  padding: 0;
  margin: 0;
}

.sidebar-nav li a {
  display: flex;
  align-items: center;
  padding: 12px 20px;
  color: #bdc3c7;
  text-decoration: none;
  transition: all 0.2s ease;
  white-space: nowrap;
}

.sidebar-nav li a:hover {
  background-color: #34495e;
  color: #fff;
}

.icon {
  font-size: 20px;
  margin-right: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 24px;
}

.sidebar.collapsed .sidebar-nav li a {
  padding: 12px 20px; /* 保持内边距一致 */
}

.sidebar.collapsed .icon {
  margin-right: 0; /* 折叠时去掉右边距，让图标居中 */
}
</style>
