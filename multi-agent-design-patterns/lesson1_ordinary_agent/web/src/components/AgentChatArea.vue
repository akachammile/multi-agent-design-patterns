<script lang="ts">
import { ref } from "vue"
const modelMenuOpen = ref(false)
const inputText = ref("")
</script>

<template>
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
                                    :class="selectedModel === model ? 'model-pill-active' : ''"
                                    @click="pickModel(model)">
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




</template>