<script setup lang="ts">
import { ref } from "vue"

const isCollapsed = ref(false)
const activeMenu = ref("overview")

const toggleSidebar = () => {
  isCollapsed.value = !isCollapsed.value
}

const handleSelect = (key: string) => {
  activeMenu.value = key
}
</script>

<template>
  <aside class="sidebar" :class="{ collapsed: isCollapsed }">
    <div class="sidebar-header">
      <div class="brand" v-show="!isCollapsed">
        <span class="brand-dot" />
        <span class="brand-text">Agent Console</span>
      </div>
      <el-button class="toggle-btn" circle text @click="toggleSidebar">
        <el-icon>
          <Expand v-if="isCollapsed" />
          <Fold v-else />
        </el-icon>
      </el-button>
    </div>

    <el-scrollbar class="sidebar-scroll">
      <el-menu class="sidebar-menu" :collapse="isCollapsed" :default-active="activeMenu" @select="handleSelect">
        <el-menu-item index="overview">
          <el-icon>
            <House />
          </el-icon>
          <template #title>首页概览</template>
        </el-menu-item>
        <el-menu-item index="analysis">
          <el-icon>
            <DataAnalysis />
          </el-icon>
          <template #title>数据分析</template>
        </el-menu-item>
        <el-menu-item index="settings">
          <el-icon>
            <Setting />
          </el-icon>
          <template #title>系统设置</template>
        </el-menu-item>
      </el-menu>
    </el-scrollbar>
  </aside>
</template>

<style scoped>
.sidebar {
  width: 240px;
  height: 100vh;
  background: #ffffff;
  color: #303133;
  transition: width 0.3s ease;
  display: flex;
  flex-direction: column;
  border-right: 1px solid #e4e7ed;
}

.sidebar.collapsed {
  width: 68px;
  /* 给折叠后的图标留出合适的空间 */
}

.sidebar-header {
  height: 60px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 16px;
  border-bottom: 1px solid #e4e7ed;
  overflow: hidden;
}

.sidebar.collapsed .sidebar-header {
  justify-content: center;
  padding: 0;
}

.brand {
  display: flex;
  align-items: center;
  gap: 8px;
  white-space: nowrap;
}

.brand-dot {
  width: 10px;
  height: 10px;
  border-radius: 999px;
  background: #67c23a;
  box-shadow: 0 0 8px rgba(103, 194, 58, 0.4);
}

.brand-text {
  font-size: 14px;
  font-weight: 700;
  letter-spacing: 0.02em;
}

.toggle-btn {
  color: #909399;
}

.toggle-btn:hover {
  color: #303133;
  background-color: #f4f4f5;
}

.sidebar-scroll {
  flex: 1;
}

:deep(.sidebar-menu) {
  border-right: none;
  background: transparent;
  --el-menu-bg-color: transparent;
  --el-menu-hover-bg-color: #f2f6fc;
  --el-menu-text-color: #606266;
  --el-menu-active-color: #146ef5;
}

:deep(.sidebar-menu .el-menu-item) {
  height: 46px;
  margin: 8px 12px;
  border-radius: 10px;
}

:deep(.sidebar-menu .el-menu-item .el-icon) {
  font-size: 20px;
}

:deep(.sidebar-menu .el-menu-item.is-active) {
  background: #e9f2ff;
  font-weight: 600;
}

:deep(.sidebar-menu.el-menu--collapse) {
  width: 68px;
}

:deep(.sidebar-menu.el-menu--collapse .el-menu-item) {
  margin: 8px 12px;
  padding: 0 !important;
  display: flex;
  justify-content: center;
  align-items: center;
  width: calc(100% - 24px);
  /* 让背景框居中对齐 */
}

/* Element Plus 折叠时，默认使用 el-tooltip 包装图标，需要强制其内联样式居中 */
:deep(.sidebar-menu.el-menu--collapse .el-menu-tooltip__trigger),
:deep(.sidebar-menu.el-menu--collapse .el-tooltip__trigger) {
  display: flex !important;
  justify-content: center !important;
  align-items: center !important;
  width: 100% !important;
  padding: 0 !important;
}

:deep(.sidebar-menu.el-menu--collapse .el-icon) {
  margin: 0 !important;
}
</style>
