<template>
  <div class="pointer-events-none relative mb-auto flex h-[100px] w-[480px] self-center justify-center">
    <header
      class="pointer-events-auto absolute top-0 flex items-center justify-center overflow-hidden border border-black/10 bg-white/95 backdrop-blur-[16px] transition-[border-radius,width,height,box-shadow,padding] duration-300 [transition-timing-function:cubic-bezier(0.34,1.56,0.64,1)]"
      :class="
        isCollapsed
          ? 'h-14 w-14 rounded-full p-0 shadow-[0_10px_25px_rgba(0,0,0,0.2)]'
          : 'h-[100px] w-[480px] rounded-[28px] px-8 py-5 shadow-[0_30px_60px_rgba(0,0,0,0.5)]'
      "
      @click="toggleIsland"
    >
      <div
        class="absolute flex flex-col items-center justify-center whitespace-nowrap text-center transition-opacity duration-150"
        :class="isCollapsed ? 'opacity-0 pointer-events-none' : 'opacity-100 delay-100'"
      >
        <h1 class="mb-2 text-[1.15rem] font-semibold text-slate-900">鍔ㄦ€佸矝</h1>
        <p class="text-[0.85rem] text-slate-600">Drag nodes on grid. Wheel to zoom canvas. Drag empty area to pan.</p>
      </div>

      <div
        class="absolute flex items-center justify-center transition-opacity duration-150"
        :class="isCollapsed ? 'opacity-100 delay-100' : 'opacity-0 pointer-events-none'"
      >
        <Orbit :size="24" :stroke-width="2" />
      </div>
    </header>
  </div>
</template>

<script setup>
import { ref } from "vue"
import { Orbit } from "lucide-vue-next"

const isCollapsed = ref(true)

const toggleIsland = () => {
  isCollapsed.value = !isCollapsed.value
}
</script>
