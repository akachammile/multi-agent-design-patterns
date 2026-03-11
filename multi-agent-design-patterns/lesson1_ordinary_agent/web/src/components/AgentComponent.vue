<script setup lang="ts">
import { Paperclip, ArrowUp, Sparkles, MessageSquare, Zap, SlidersHorizontal, Check } from "lucide-vue-next"
import { ref } from "vue"
import { agentApi } from "@/api/agent"

const inputText = ref("")
const fileInput = ref<HTMLInputElement | null>(null)
const messages = ref<any[]>([])
const modelMenuOpen = ref(false)
const selectedModel = ref("Gemini")
const modelOptions = ["Gemini", "GPT-4o", "Claude 3.7"]

const autoResize = (event: Event) => {
  const el = event.target as HTMLTextAreaElement
  el.style.height = "auto"
  el.style.height = el.scrollHeight + "px"
}

const triggerFileInput = () => {
  fileInput.value?.click()
}

const toggleModelMenu = () => {
  modelMenuOpen.value = !modelMenuOpen.value
}

const pickModel = (model: string) => {
  selectedModel.value = model
  modelMenuOpen.value = false
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
        <h4 class="welcome-title">尝试使用以下</h4>
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
        <div v-for="(msg, index) in messages" :key="index" class="message-row"
          :class="msg.role === 'user' ? 'flex-row-reverse' : ''">
          <div v-if="msg.role === 'assistant'" class="avatar-container">
            <img src="@/assets/vue.svg" alt="AI" class="avatar-img" />
          </div>
          <div class="message-bubble-wrapper">
            <div class="message-bubble" :class="msg.role === 'assistant'
              ? 'assistant-bubble'
              : 'user-bubble'
              ">
              <p>{{ msg.content }}</p>
            </div>
          </div>
        </div>
      </template>
    </div>

    <footer class="chat-footer">
      <div class="input-container">
        <textarea v-model="inputText" class="chat-textarea" placeholder="Message AI Assistant..." rows="2"
          @input="autoResize"></textarea>

        <div class="input-actions-bar">
          <div class="left-actions">
            <div class="action-btn" title="Add files" @click="triggerFileInput">
              <input type="file" ref="fileInput" class="hidden" multiple />
              <Paperclip :size="18" />
            </div>
          </div>

          <div class="right-actions">
            <div class="model-picker">
              <button class="model-trigger" type="button" @click="toggleModelMenu">
                <SlidersHorizontal :size="16" />
              </button>

              <Transition name="model-pop">
                <div v-if="modelMenuOpen" class="model-popover">
                  <button v-for="model in modelOptions" :key="model" type="button" class="model-pill"
                    :class="selectedModel === model ? 'model-pill-active' : ''" @click="pickModel(model)">
                    <span class="model-pill-icon">{{ model.charAt(0) }}</span>
                    <span class="model-pill-name">{{ model }}</span>
                    <span class="model-pill-check" :class="selectedModel === model ? 'is-checked' : ''">
                      <Check :size="12" />
                    </span>
                  </button>
                </div>
              </Transition>
            </div>
            <div class="send-btn" :class="inputText.trim() ? 'send-btn-active' : ''" @click="toggleModelMenu">
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
  font-size: 16px;
  line-height: 20px;
  font-weight: 600;
  color: #0f172a;
}

/* Body & Empty State */
.chat-body {
  @apply flex flex-1 flex-col gap-6 overflow-y-auto px-5 py-6;
}

.empty-state {
  @apply flex h-full flex-1 flex-col items-center justify-center text-center;
}

.sparkle-icon-container {
  @apply mb-6 flex h-16 w-16 items-center justify-center rounded-full bg-gradient-to-br from-blue-500/10 via-purple-500/10 to-pink-500/10;
  box-shadow: 0 8px 32px rgba(168, 85, 247, 0.15);
}

.sparkle-icon {
  @apply text-transparent bg-gradient-to-br from-blue-500 via-purple-500 to-pink-500 bg-clip-text;
}

.welcome-title {
  @apply mb-2;
  font-size: 16px;
  line-height: 20px;
  font-weight: 600;
  color: #0f172a;
}

.welcome-desc {
  @apply mb-8;
  font-size: 13px;
  line-height: 18px;
  font-weight: 400;
  color: #888888;
}

/* Suggestions */
.suggestion-list {
  @apply flex w-full max-w-[300px] flex-col gap-3;
}

.suggestion-item {
  @apply flex cursor-pointer items-center gap-3 rounded-[18px] border border-[#d7dee8] bg-[#f4f7fb] px-4 py-3 text-slate-500 transition-all duration-300 hover:-translate-y-0.5 hover:bg-white;
  font-size: 14px;
  line-height: 20px;
  font-weight: 400;
  transition-timing-function: cubic-bezier(0.4, 0, 0.2, 1);
}

.suggestion-item:hover {
  box-shadow: 0 8px 20px rgba(0, 0, 0, 0.06);
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
  @apply rounded-[18px] px-4 py-3;
  font-size: 14px;
  line-height: 20px;
  font-weight: 400;
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
  @apply flex flex-1 flex-col overflow-hidden rounded-[28px] border border-[#d7dee8] bg-[#f4f7fb] duration-200 focus-within:border-slate-500;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
  transition-property: border-color, box-shadow;
  transition-duration: 200ms;
}

.input-container:focus-within {
  box-shadow: 0 8px 20px rgba(0, 0, 0, 0.06);
}

.chat-textarea {
  @apply m-2 mb-0 flex-1 resize-none self-stretch border-0 bg-transparent px-2 pt-2 text-slate-900 outline-none;
  font-size: 16px;
  line-height: 20px;
  font-weight: 400;
  scrollbar-width: none;
  -ms-overflow-style: none;
}

.chat-textarea::-webkit-scrollbar {
  display: none;
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

.right-actions {
  @apply relative flex items-center gap-2;
}

.send-btn {
  @apply flex h-8 w-8 cursor-pointer items-center justify-center rounded-full bg-[#eef2f8] text-slate-400 transition-all duration-200;
}

.send-btn-active {
  @apply bg-zinc-900 text-white hover:-translate-y-0.5 hover:bg-black;
}

.model-picker {
  @apply relative;
}

.model-trigger {
  @apply flex h-8 w-8 cursor-pointer items-center justify-center rounded-full border border-[#b9c6d6] text-slate-500 transition-all duration-200 hover:bg-[#eef2f8] hover:text-slate-900;
}

.model-popover {
  @apply absolute bottom-10 right-0 z-20 flex min-w-[210px] flex-col gap-2 rounded-2xl border border-[#d7dee8] bg-white p-2 shadow-xl;
}

.model-pill {
  @apply flex w-full items-center gap-2 rounded-full border border-[#d5deea] bg-[#f4f7fb] px-2.5 py-1.5 text-[12px] font-medium text-slate-600 transition-all duration-200 hover:border-slate-500 hover:bg-white;
}

.model-pill-active {
  @apply border-slate-900 bg-slate-900 text-white;
}

.model-pill-icon {
  @apply flex h-4 w-4 items-center justify-center rounded-full bg-white/80 text-[10px] font-semibold text-slate-700;
}

.model-pill-name {
  @apply text-[12px] font-medium;
}

.model-pill-check {
  @apply flex h-4 w-4 items-center justify-center rounded-full border border-[#cfd8e3] text-transparent;
}

.model-pill-check.is-checked {
  @apply border-white/70 bg-white/20 text-white;
}

.model-pop-enter-active,
.model-pop-leave-active {
  transition: opacity 180ms ease, transform 180ms ease;
}

.model-pop-enter-from,
.model-pop-leave-to {
  opacity: 0;
  transform: translateY(8px) scale(0.96);
}
</style>
