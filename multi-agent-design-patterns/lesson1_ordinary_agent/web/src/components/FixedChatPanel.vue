<template>
  <section class="chat-shell">
    <header class="chat-header">
      <h3>AI 对话</h3>
      <span class="status-dot" />
    </header>

    <div class="chat-body">
      <article class="bubble bubble-assistant">你好，这里是左侧固定对话框雏形。</article>
      <article class="bubble bubble-user">后续可接入真实消息流与历史会话。</article>
    </div>

    <footer class="chat-footer">
      <input type="text" placeholder="输入消息..." disabled />
      <button type="button" disabled>发送</button>
    </footer>
  </section>
</template>

<style scoped>
.chat-shell {
  position: fixed;
  right: 16px;
  top: 84px;
  width: 320px;
  height: calc(100vh - 108px);
  display: flex;
  flex-direction: column;
  border: 1px solid var(--border-default);
  border-radius: var(--radius-lg);
  background: #ffffff;
  box-shadow: var(--shadow-md);
  z-index: 20;
  overflow: hidden;
}

.chat-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 16px;
  border-bottom: 1px solid var(--border-default);
}

.chat-header h3 {
  margin: 0;
  font-size: 15px;
}

.status-dot {
  width: 10px;
  height: 10px;
  border-radius: 999px;
  background: #22c55e;
}

.chat-body {
  flex: 1;
  overflow-y: auto;
  padding: 14px;
  display: flex;
  flex-direction: column;
  gap: 10px;
  background: linear-gradient(180deg, #f9fbff 0%, #f3f7fd 100%);
}

.bubble {
  max-width: 90%;
  padding: 12px 16px;
  border-radius: 20px;
  font-size: 13.5px;
  line-height: 1.5;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.02);
}

.bubble-assistant {
  align-self: flex-start;
  background: #ffffff;
  border: 1px solid #eef2f8;
  border-bottom-left-radius: 4px;
}

.bubble-user {
  align-self: flex-end;
  background: var(--brand-500);
  color: #ffffff;
  border-bottom-right-radius: 4px;
}

.chat-footer {
  border-top: 1px solid var(--border-default);
  padding: 12px;
  display: flex;
  gap: 8px;
}

.chat-footer input {
  flex: 1;
}

.chat-footer button {
  padding: 0.55rem 0.9rem;
}

@media (max-width: 1280px) {
  .chat-shell {
    width: 280px;
  }
}

@media (max-width: 1024px) {
  .chat-shell {
    display: none;
  }
}
</style>
