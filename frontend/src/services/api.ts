export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

export interface ChatResponse {
  id: string;
  choices: {
    message: ChatMessage;
    finish_reason: string;
  }[];
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
  memforge_telemetry?: {
    tokens_per_second: number;
    acceptance_rate?: number;
    time_to_first_token_ms?: number;
  };
}

export async function sendChatCompletion(
  messages: ChatMessage[],
  useSpeculative: boolean = true,
  maxTokens: number = 128
): Promise<ChatResponse> {
  const response = await fetch('/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      messages,
      use_speculative: useSpeculative,
      max_new_tokens: maxTokens,
    }),
  });

  if (!response.ok) {
    throw new Error(`API Error: ${response.statusText}`);
  }
  return response.json();
}
