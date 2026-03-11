<script setup lang="ts">
import { Paperclip } from "lucide-vue-next"
import { ref } from "vue"
import { agentApi } from "@/api/agent"
import MessageInputComponent from "@/components/MessageInputComponent.vue"

const inputText = ref("")
const fileInput = ref<HTMLInputElement | null>(null)
const sending = ref(false)

const triggerFileInput = () => {
  fileInput.value?.click()
}

type SendPayload = {
  message: string
  model: string
}

const seedMessage2Agent = async ({ message, model }: SendPayload) => {
  sending.value = true
  try {
    await agentApi.send2Agent(
      "ordinary-agent",
      {
        messages: [{ role: "user", content: message }],
      },
      {
        model,
        stream: false,
      },
    )

    inputText.value = ""
  } finally {
    sending.value = false
  }
}
</script>

<template>
  <MessageInputComponent
    v-model="inputText"
    :sending="sending"
    @send="seedMessage2Agent"
    @attachment-click="triggerFileInput"
  >
    <template #attachment>
      <input ref="fileInput" type="file" class="hidden" multiple />
      <Paperclip :size="18" />
    </template>
  </MessageInputComponent>
</template>
