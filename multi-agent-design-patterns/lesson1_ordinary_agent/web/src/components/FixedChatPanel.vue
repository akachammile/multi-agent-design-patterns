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
  <section class="chat-panel glass-panel rounded-panel">
    <header class="chat-header">
      <h3 class="chat-title">新对话</h3>
    </header>

    <div class="chat-body">
      <div v-if="messages.length === 0" class="empty-state">
        <div class="sparkle-icon-container">
          <Sparkles :size="32" class="sparkle-icon" />
        </div>
        <h4 class="welcome-title">今天想聊点什么？</h4>
        <p class="welcome-desc">我可以帮您写代码、分析数据，或者提供创意灵感。</p>

        <div class="suggestion-list">
          <div class="suggestion-item suggestion-blue">
            <div class="suggestion-icon bg-blue">
              <MessageSquare :size="16" />
            </div>
            <span>解释量子计算</span>
          </div>
          <div class="suggestion-item suggestion-orange">
            <div class="suggestion-icon bg-orange">
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
          class="message-row"
          :class="msg.role === 'user' ? 'flex-row-reverse' : ''"
        >
          <div v-if="msg.role === 'assistant'" class="avatar-container">
            <img src="@/assets/vue.svg" alt="AI" class="avatar-img" />
          </div>
          <div class="message-bubble-wrapper">
            <div
              class="message-bubble"
              :class="
                msg.role === 'assistant'
                  ? 'assistant-bubble'
                  : 'user-bubble'
              "
            >
              <p>{{ msg.content }}</p>
            </div>
          </div>
        </div>
      </template>
    </div>

    <footer class="chat-footer">
      <div class="input-container">
        <textarea
          v-model="inputText"
          class="chat-textarea"
          placeholder="Message AI Assistant..."
          rows="2"
          @input="autoResize"
        ></textarea>

        <div class="input-actions-bar">
          <div class="left-actions">
            <div class="action-btn" title="Add files" @click="triggerFileInput">
              <input type="file" ref="fileInput" class="hidden" multiple />
              <Paperclip :size="18" />
            </div>
            <div class="model-selector">
              <span>GPT-4o</span>
              <ChevronDown :size="14" />
            </div>
          </div>

          <div class="right-actions">
            <div
              class="send-btn"
              :class="inputText.trim() ? 'send-btn-active' : ''"
            >
              <ArrowUp :size="18" :stroke-width="2.5" />
            </div>
          </div>
        </div>
      </div>
    </footer>
  </section>
</template>

<style scoped>
@reference "tailwindcss";

/* Main Container */
.chat-panel {
  @apply pointer-events-auto my-4 mr-4 ml-0 flex h-[calc(100%-2rem)] w-[340px] flex-col overflow-hidden;
}

@media (min-width: 1280px) {
  .chat-panel {
    width: 400px;
  }
}

/* Header */
.chat-header {
  @apply px-5 py-4;
}
.chat-title {
  @apply text-base font-semibold text-slate-900;
}

/* Body & Empty State */
.chat-body {
  @apply flex flex-1 flex-col gap-6 overflow-y-auto px-5 py-6;
}
.empty-state {
  @apply flex h-full flex-1 flex-col items-center justify-center text-center;
}
.sparkle-icon-container {
  @apply mb-6 flex h-16 w-16 items-center justify-center rounded-full bg-gradient-to-br from-blue-500/10 via-purple-500/10 to-pink-500/10 shadow-[0_8px_32px_rgba(168,85,247,0.15)];
}
.sparkle-icon {
  @apply text-transparent bg-gradient-to-br from-blue-500 via-purple-500 to-pink-500 bg-clip-text;
}
.welcome-title {
  @apply mb-2 text-lg font-semibold text-slate-900;
}
.welcome-desc {
  @apply mb-8 text-sm text-slate-500;
}

/* Suggestions */
.suggestion-list {
  @apply flex w-full max-w-[300px] flex-col gap-3;
}
.suggestion-item {
  @apply flex cursor-pointer items-center gap-3 rounded-[18px] border border-[#d7dee8] bg-[#f4f7fb] px-4 py-3 text-sm text-slate-500 transition-all duration-300 ease-[cubic-bezier(0.4,0,0.2,1)] hover:-translate-y-0.5 hover:bg-white hover:shadow-[0_8px_20px_rgba(0,0,0,0.06)];
}
.suggestion-blue:hover {
  @apply border-blue-500 text-blue-500;
}
.suggestion-orange:hover {
  @apply border-orange-500 text-orange-500;
}
.suggestion-icon {
  @apply flex h-7 w-7 items-center justify-center rounded-lg;
}
.suggestion-icon.bg-blue {
  @apply bg-blue-500/10 text-blue-500;
}
.suggestion-icon.bg-orange {
  @apply bg-orange-500/10 text-orange-500;
}

/* Messages */
.message-row {
  @apply flex gap-3;
}
.avatar-container {
  @apply flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full bg-white p-1.5;
}
.avatar-img {
  @apply h-full w-full;
}
.message-bubble-wrapper {
  @apply max-w-[75%];
}
.message-bubble {
  @apply rounded-[18px] px-4 py-3 text-sm leading-6;
}
.assistant-bubble {
  @apply rounded-tl-[4px] border border-[#d7dee8] bg-white text-slate-900;
}
.user-bubble {
  @apply rounded-tr-[4px] bg-slate-700 text-white;
}

/* Footer & Input */
.chat-footer {
  @apply flex h-[22%] min-h-[140px] flex-col px-2 pb-2;
}
.input-container {
  @apply flex flex-1 flex-col overflow-hidden rounded-[28px] border border-[#d7dee8] bg-[#f4f7fb] shadow-[0_4px_12px_rgba(0,0,0,0.03)] transition-[border-color,box-shadow] duration-200 focus-within:border-slate-500 focus-within:shadow-[0_8px_20px_rgba(0,0,0,0.06)];
}
.chat-textarea {
  @apply m-2 mb-0 flex-1 resize-none self-stretch border-0 bg-transparent px-2 pt-2 text-[14.5px] leading-[1.5] text-slate-900 outline-none;
}
.input-actions-bar {
  @apply flex items-center justify-between px-3 pb-3 pt-2;
}
.left-actions {
  @apply flex items-center gap-2;
}
.action-btn {
  @apply flex h-8 w-8 cursor-pointer items-center justify-center rounded-full border border-[#b9c6d6] text-slate-500 transition-all duration-200 hover:bg-[#eef2f8] hover:text-slate-900;
}
.model-selector {
  @apply flex cursor-pointer items-center gap-1 rounded-lg bg-[#eef2f8] px-2.5 py-1 text-xs font-semibold text-slate-500;
}
.right-actions {
  @apply flex items-center;
}
.send-btn {
  @apply flex h-8 w-8 cursor-pointer items-center justify-center rounded-full bg-[#eef2f8] text-slate-400 transition-all duration-200;
}
.send-btn-active {
  @apply bg-zinc-900 text-white hover:-translate-y-0.5 hover:bg-black;
}
</style>
