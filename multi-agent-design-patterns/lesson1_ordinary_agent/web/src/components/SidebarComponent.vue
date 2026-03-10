<script setup lang="ts">
import { ref } from "vue"
import { Home, LineChart, Settings } from "lucide-vue-next"

const activeMenu = ref("overview")

const handleSelect = (key: string) => {
  activeMenu.value = key
}
</script>

<template>
  <aside
    class="pointer-events-auto ml-6 my-auto flex w-14 flex-col items-center self-center rounded-[32px] border border-[#d7dee8] bg-white py-6 shadow-[0_16px_48px_rgba(20,38,66,0.08)] transition-all duration-300"
  >
    <div class="flex w-full flex-col items-center gap-4">
      <div
        class="group relative flex h-[38px] w-[38px] items-center justify-center rounded-[10px] text-slate-500 transition-all duration-200 ease-[cubic-bezier(0.4,0,0.2,1)] hover:bg-slate-100 hover:text-slate-900 hover:scale-105"
        :class="activeMenu === 'overview' ? 'bg-black text-white shadow-[0_4px_12px_rgba(0,0,0,0.15)] scale-105' : ''"
        @click="handleSelect('overview')"
      >
        <Home :size="20" :stroke-width="2.5" />
        <span
          class="pointer-events-none absolute left-full ml-3.5 whitespace-nowrap rounded-md bg-black px-2.5 py-1.5 text-xs text-white opacity-0 translate-x-[-10px] transition-all duration-200 ease-[cubic-bezier(0.4,0,0.2,1)] group-hover:opacity-100 group-hover:translate-x-0"
        >
          棣栭〉姒傝
        </span>
        <span
          class="pointer-events-none absolute left-full ml-1.5 h-0 w-0 border-y-[4px] border-y-transparent border-r-[4px] border-r-black opacity-0 translate-x-[-10px] transition-all duration-200 ease-[cubic-bezier(0.4,0,0.2,1)] group-hover:opacity-100 group-hover:translate-x-0"
        />
      </div>
      <div
        class="group relative flex h-[38px] w-[38px] items-center justify-center rounded-[10px] text-slate-500 transition-all duration-200 ease-[cubic-bezier(0.4,0,0.2,1)] hover:bg-slate-100 hover:text-slate-900 hover:scale-105"
        :class="activeMenu === 'analysis' ? 'bg-black text-white shadow-[0_4px_12px_rgba(0,0,0,0.15)] scale-105' : ''"
        @click="handleSelect('analysis')"
      >
        <LineChart :size="20" :stroke-width="2.5" />
        <span
          class="pointer-events-none absolute left-full ml-3.5 whitespace-nowrap rounded-md bg-black px-2.5 py-1.5 text-xs text-white opacity-0 translate-x-[-10px] transition-all duration-200 ease-[cubic-bezier(0.4,0,0.2,1)] group-hover:opacity-100 group-hover:translate-x-0"
        >
          鏁版嵁鍒嗘瀽
        </span>
        <span
          class="pointer-events-none absolute left-full ml-1.5 h-0 w-0 border-y-[4px] border-y-transparent border-r-[4px] border-r-black opacity-0 translate-x-[-10px] transition-all duration-200 ease-[cubic-bezier(0.4,0,0.2,1)] group-hover:opacity-100 group-hover:translate-x-0"
        />
      </div>
      <div
        class="group relative flex h-[38px] w-[38px] items-center justify-center rounded-[10px] text-slate-500 transition-all duration-200 ease-[cubic-bezier(0.4,0,0.2,1)] hover:bg-slate-100 hover:text-slate-900 hover:scale-105"
        :class="activeMenu === 'settings' ? 'bg-black text-white shadow-[0_4px_12px_rgba(0,0,0,0.15)] scale-105' : ''"
        @click="handleSelect('settings')"
      >
        <Settings :size="20" :stroke-width="2.5" />
        <span
          class="pointer-events-none absolute left-full ml-3.5 whitespace-nowrap rounded-md bg-black px-2.5 py-1.5 text-xs text-white opacity-0 translate-x-[-10px] transition-all duration-200 ease-[cubic-bezier(0.4,0,0.2,1)] group-hover:opacity-100 group-hover:translate-x-0"
        >
          绯荤粺璁剧疆
        </span>
        <span
          class="pointer-events-none absolute left-full ml-1.5 h-0 w-0 border-y-[4px] border-y-transparent border-r-[4px] border-r-black opacity-0 translate-x-[-10px] transition-all duration-200 ease-[cubic-bezier(0.4,0,0.2,1)] group-hover:opacity-100 group-hover:translate-x-0"
        />
      </div>
    </div>
  </aside>
</template>
