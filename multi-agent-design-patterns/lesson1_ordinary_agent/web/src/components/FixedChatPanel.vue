<script setup lang="ts">
import { ChevronDown, Paperclip, ArrowUp, Plus, Sparkles, MessageSquare, Zap } from 'lucide-vue-next'
import { ref } from 'vue'

const inputText = ref('')
const fileInput = ref<HTMLInputElement | null>(null)
const messages = ref<any[]>([]) // 初始化为空数组，模拟未开启对话的状态

// 自动调整 textarea 高度（简单示例）
const autoResize = (event: Event) => {
  const el = event.target as HTMLTextAreaElement
  el.style.height = 'auto'
  el.style.height = el.scrollHeight + 'px'
}

const triggerFileInput = () => {
  fileInput.value?.click()
}
</script>

<template>
  <section class="chat-shell">
    <header class="chat-header">
      <div class="header-title">
        <h3>新对话</h3>
      </div>
    </header>

    <div class="chat-body">
      <!-- 空状态 (无消息时显示) -->
      <div v-if="messages.length === 0" class="empty-state">
        <div class="empty-icon-wrapper colorful-glow">
          <Sparkles :size="32" class="empty-icon colorful-text" />
        </div>
        <h4>今天想聊点什么？</h4>
        <p>我可以帮您写代码、分析数据，或者提供创意灵感。</p>
        
        <div class="suggestion-cards">
          <div class="suggestion-card card-blue">
            <div class="card-icon-wrapper">
              <MessageSquare :size="16" />
            </div>
            <span>解释量子计算</span>
          </div>
          <div class="suggestion-card card-orange">
            <div class="card-icon-wrapper">
              <Zap :size="16" />
            </div>
            <span>帮我写一段快排代码</span>
          </div>
        </div>
      </div>

      <!-- 对话列表 (有消息时显示) -->
      <template v-else>
        <div v-for="(msg, index) in messages" :key="index" class="message" :class="msg.role">
          <div v-if="msg.role === 'assistant'" class="avatar">
            <img src="@/assets/vue.svg" alt="AI" />
          </div>
          <div class="content">
            <div class="message-bubble">
              <p>{{ msg.content }}</p>
            </div>
          </div>
        </div>
      </template>
    </div>

    <footer class="chat-footer">
      <!-- 独立的大矩形对话框 -->
      <div class="large-input-card">
        <textarea class="main-textarea" v-model="inputText" placeholder="Message AI Assistant..." rows="2"
          @input="autoResize"></textarea>

        <div class="input-actions">
          <div class="left-group">
            <div class="action-item" title="Add files" @click="triggerFileInput">
              <input type="file" ref="fileInput" style="display: none;" multiple />
              <Paperclip :size="18" />
            </div>
            <div class="model-badge">
              <span>GPT-4o</span>
              <ChevronDown :size="14" />
            </div>
          </div>

          <div class="right-group">
            <div class="send-action" :class="{ 'has-text': inputText.trim() }">
              <ArrowUp :size="18" :stroke-width="2.5" />
            </div>
          </div>
        </div>
      </div>
    </footer>
  </section>
</template>

<style scoped>
.chat-shell {
  width: 400px;
  height: calc(100% - 2rem);
  margin: 1rem 1rem 1rem 0;
  display: flex;
  flex-direction: column;
  background-color: var(--bg-elevated);
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow-md);
  border: 1px solid var(--border-default);
  overflow: hidden;
  pointer-events: auto;
}

/* Header */
.chat-header {
  padding: 16px 20px;
}

.header-title h3 {
  margin: 0;
  font-size: 16px;
  font-weight: 600;
  color: var(--text-primary);
}

/* Body */
.chat-body {
  flex: 1;
  overflow-y: auto;
  padding: 24px 20px;
  display: flex;
  flex-direction: column;
  gap: 24px;
}

/* Empty State */
.empty-state {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  height: 100%;
}

.empty-icon-wrapper {
  width: 64px;
  height: 64px;
  border-radius: 50%;
  background: var(--bg-canvas);
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 24px;
  box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.02);
}

.colorful-glow {
  background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(168, 85, 247, 0.1), rgba(236, 72, 153, 0.1));
  box-shadow: 0 8px 32px rgba(168, 85, 247, 0.15);
}

.colorful-text {
  color: transparent;
  background: linear-gradient(135deg, #3b82f6, #a855f7, #ec4899);
  background-clip: text;
  -webkit-background-clip: text;
}

.empty-state h4 {
  font-size: 18px;
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 8px;
}

.empty-state p {
  font-size: 14px;
  color: var(--text-secondary);
  margin-bottom: 32px;
}

.suggestion-cards {
  display: flex;
  flex-direction: column;
  gap: 12px;
  width: 100%;
  max-width: 300px;
}

.suggestion-card {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 16px;
  background-color: var(--bg-canvas);
  border: 1px solid var(--border-default);
  border-radius: var(--radius-md);
  color: var(--text-secondary);
  font-size: 14px;
  cursor: pointer;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.card-icon-wrapper {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  border-radius: 8px;
  transition: all 0.3s ease;
}

.card-blue .card-icon-wrapper {
  background-color: rgba(59, 130, 246, 0.1);
  color: #3b82f6;
}

.card-orange .card-icon-wrapper {
  background-color: rgba(249, 115, 22, 0.1);
  color: #f97316;
}

.suggestion-card:hover {
  background-color: #ffffff;
  transform: translateY(-2px);
  box-shadow: 0 8px 20px rgba(0, 0, 0, 0.06);
}

.card-blue:hover {
  border-color: #3b82f6;
  color: #3b82f6;
}

.card-orange:hover {
  border-color: #f97316;
  color: #f97316;
}

.message {
  display: flex;
  gap: 12px;
}

.message.user {
  flex-direction: row-reverse;
}

.avatar {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  background-color: var(--bg-elevated);
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  padding: 6px;
}

.message-bubble {
  padding: 12px 16px;
  border-radius: var(--radius-md);
  font-size: 14px;
  line-height: 1.6;
}

.message.assistant .message-bubble {
  background-color: var(--bg-elevated);
  color: var(--text-primary);
  border: 1px solid var(--border-default);
  border-top-left-radius: 4px;
}

.message.user .message-bubble {
  background-color: var(--brand-500);
  color: #ffffff;
  border-top-right-radius: 4px;
}

/* Footer - 占据约1/5高度且下左右隔离 */
.chat-footer {
  height: 22%;
  /* 约五分之一高度 */
  min-height: 140px;
  padding: 0 8px 8px 8px;
  /* 减小隔离距离 */
  display: flex;
  flex-direction: column;
}

.large-input-card {
  flex: 1;
  background-color: var(--bg-canvas);
  border: 1px solid var(--border-default);
  border-radius: var(--radius-lg);
  display: flex;
  flex-direction: column;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
  transition: border-color 0.2s, box-shadow 0.2s;
  overflow: hidden;
}

.large-input-card:focus-within {
  border-color: var(--brand-500);
  box-shadow: 0 8px 20px rgba(0, 0, 0, 0.06);
}

.main-textarea {
  flex: 1;
  margin: 10px 10px 0 10px;
  /* 缩小与外部 large-input-card 的距离 */
  align-self: stretch;
  /* 确保去除 width: 100% 后依然自动撑满宽度 */
  border: none;
  background: transparent;
  border-radius: 0;
  /* 修改为直角矩形 */
  resize: none;
  font-size: 14.5px;
  line-height: 1.5;
  color: var(--text-primary);
  outline: none;
  font-family: inherit;
  padding: 10px 10px 0 10px;
  /* 也略微缩小内部文本的边距 */
  box-sizing: border-box;
  overflow-y: auto;
  scrollbar-width: none;
  /* Firefox 隐藏滚动条 */
}

.main-textarea::-webkit-scrollbar {
  display: none;
  /* Chrome, Safari, Edge 隐藏滚动条 */
}

.input-actions {
  padding: 8px 12px 12px 12px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.left-group,
.right-group {
  display: flex;
  align-items: center;
  gap: 8px;
}

.action-item {
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  /* 圆形 */
  border: 1px solid var(--border-strong);
  /* 灰色线条包裹 */
  background-color: transparent;
  /* 无色背景 */
  color: var(--text-secondary);
  cursor: pointer;
  transition: all 0.2s;
}

.action-item:hover {
  background-color: var(--bg-subtle);
  color: var(--text-primary);
}

.model-badge {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 4px 10px;
  border-radius: 8px;
  background-color: var(--bg-subtle);
  color: var(--text-secondary);
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
}

.send-action {
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  /* 修改为完美的圆形 */
  background-color: var(--bg-subtle);
  color: var(--text-muted);
  cursor: pointer;
  transition: all 0.2s;
}

.send-action.has-text {
  background-color: #18181b;
  /* 极深灰/近乎纯黑 */
  color: #ffffff;
}

.send-action.has-text:hover {
  background-color: #000000;
  /* hover 时纯黑 */
  transform: translateY(-1px);
}


/* Scrollbar */
.chat-body::-webkit-scrollbar {
  width: 5px;
}

.chat-body::-webkit-scrollbar-thumb {
  background-color: var(--border-strong);
  border-radius: 10px;
}

@media (max-width: 1280px) {
  .chat-shell {
    width: 340px;
  }
}
</style>
