<script setup lang="ts">
import { ChevronDown, Paperclip, ArrowUp, Sparkles, MessageSquare, Zap } from "lucide-vue-next"
import { ref } from "vue"

const inputText = ref("")
const fileInput = ref<HTMLInputElement | null>(null)
const messages = ref<any[]>([])
const autoResize = (event: Event) => {
  const el = event.target as HTMLTextAreaElement
  el.style.height = "auto"
  el.style.height = el.scrollHeight + "px"
}

const triggerFileInput = () => {
  fileInput.value?.click()
}
</script>

<template>
  <section
    class="pointer-events-auto my-4 mr-4 ml-0 flex h-[calc(100%-2rem)] w-[340px] flex-col overflow-hidden rounded-[28px] border border-[#d7dee8] bg-white shadow-[0_16px_48px_rgba(20,38,66,0.08)] xl:w-[400px]"
  >
    <header class="px-5 py-4">
      <h3 class="text-base font-semibold text-slate-900">新对话</h3>
    </header>

    <div class="flex flex-1 flex-col gap-6 overflow-y-auto px-5 py-6">
      <div v-if="messages.length === 0" class="flex h-full flex-1 flex-col items-center justify-center text-center">
        <div
          class="mb-6 flex h-16 w-16 items-center justify-center rounded-full bg-gradient-to-br from-blue-500/10 via-purple-500/10 to-pink-500/10 shadow-[0_8px_32px_rgba(168,85,247,0.15)]"
        >
          <Sparkles
            :size="32"
            class="text-transparent bg-gradient-to-br from-blue-500 via-purple-500 to-pink-500 bg-clip-text"
          />
        </div>
        <h4 class="mb-2 text-lg font-semibold text-slate-900">今天想聊点什么？</h4>
        <p class="mb-8 text-sm text-slate-500">我可以帮您写代码、分析数据，或者提供创意灵感。</p>

        <div class="flex w-full max-w-[300px] flex-col gap-3">
          <div
            class="flex cursor-pointer items-center gap-3 rounded-[18px] border border-[#d7dee8] bg-[#f4f7fb] px-4 py-3 text-sm text-slate-500 transition-all duration-300 ease-[cubic-bezier(0.4,0,0.2,1)] hover:-translate-y-0.5 hover:border-blue-500 hover:bg-white hover:text-blue-500 hover:shadow-[0_8px_20px_rgba(0,0,0,0.06)]"
          >
            <div class="flex h-7 w-7 items-center justify-center rounded-lg bg-blue-500/10 text-blue-500">
              <MessageSquare :size="16" />
            </div>
            <span>解释量子计算</span>
          </div>
          <div
            class="flex cursor-pointer items-center gap-3 rounded-[18px] border border-[#d7dee8] bg-[#f4f7fb] px-4 py-3 text-sm text-slate-500 transition-all duration-300 ease-[cubic-bezier(0.4,0,0.2,1)] hover:-translate-y-0.5 hover:border-orange-500 hover:bg-white hover:text-orange-500 hover:shadow-[0_8px_20px_rgba(0,0,0,0.06)]"
          >
            <div class="flex h-7 w-7 items-center justify-center rounded-lg bg-orange-500/10 text-orange-500">
              <Zap :size="16" />
            </div>
            <span>帮我写一段快速排序</span>
          </div>
        </div>
      </div>

      <template v-else>
        <div
          v-for="(msg, index) in messages"
          :key="index"
          class="flex gap-3"
          :class="msg.role === 'user' ? 'flex-row-reverse' : ''"
        >
          <div
            v-if="msg.role === 'assistant'"
            class="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full bg-white p-1.5"
          >
            <img src="@/assets/vue.svg" alt="AI" class="h-full w-full" />
          </div>
          <div class="max-w-[75%]">
            <div
              class="rounded-[18px] px-4 py-3 text-sm leading-6"
              :class="
                msg.role === 'assistant'
                  ? 'rounded-tl-[4px] border border-[#d7dee8] bg-white text-slate-900'
                  : 'rounded-tr-[4px] bg-slate-700 text-white'
              "
            >
              <p>{{ msg.content }}</p>
            </div>
          </div>
        </div>
      </template>
    </div>

    <footer class="flex h-[22%] min-h-[140px] flex-col px-2 pb-2">
      <div
        class="flex flex-1 flex-col overflow-hidden rounded-[28px] border border-[#d7dee8] bg-[#f4f7fb] shadow-[0_4px_12px_rgba(0,0,0,0.03)] transition-[border-color,box-shadow] duration-200 focus-within:border-slate-500 focus-within:shadow-[0_8px_20px_rgba(0,0,0,0.06)]"
      >
        <textarea
          v-model="inputText"
          class="m-2 mb-0 flex-1 resize-none self-stretch border-0 bg-transparent px-2 pt-2 text-[14.5px] leading-[1.5] text-slate-900 outline-none"
          placeholder="Message AI Assistant..."
          rows="2"
          @input="autoResize"
        ></textarea>

        <div class="flex items-center justify-between px-3 pb-3 pt-2">
          <div class="flex items-center gap-2">
            <div
              class="flex h-8 w-8 cursor-pointer items-center justify-center rounded-full border border-[#b9c6d6] text-slate-500 transition-all duration-200 hover:bg-[#eef2f8] hover:text-slate-900"
              title="Add files"
              @click="triggerFileInput"
            >
              <input type="file" ref="fileInput" style="display: none;" multiple />
              <Paperclip :size="18" />
            </div>
            <div
              class="flex cursor-pointer items-center gap-1 rounded-lg bg-[#eef2f8] px-2.5 py-1 text-xs font-semibold text-slate-500"
            >
              <span>GPT-4o</span>
              <ChevronDown :size="14" />
            </div>
          </div>

          <div class="flex items-center">
            <div
              class="flex h-8 w-8 cursor-pointer items-center justify-center rounded-full bg-[#eef2f8] text-slate-400 transition-all duration-200"
              :class="inputText.trim() ? 'bg-zinc-900 text-white hover:-translate-y-0.5 hover:bg-black' : ''"
            >
              <ArrowUp :size="18" :stroke-width="2.5" />
            </div>
          </div>
        </div>
      </div>
    </footer>
  </section>
</template>
