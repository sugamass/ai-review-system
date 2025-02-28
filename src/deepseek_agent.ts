import OpenAI from "openai";
import { AgentFunction, AgentFunctionInfo } from "graphai";
import {
  GraphAILLMInputBase,
  getMergeValue,
  getMessages,
} from "@graphai/llm_utils";

type DeepSeekInputs = {
  model?: string;
  tools?: OpenAI.ChatCompletionTool[];
  tool_choice?: OpenAI.ChatCompletionToolChoiceOption;
  max_tokens?: number;
  verbose?: boolean;
  temperature?: number;
  messages?: Array<OpenAI.ChatCompletionMessageParam>;
  response_format?:
    | OpenAI.ResponseFormatText
    | OpenAI.ResponseFormatJSONObject
    | OpenAI.ResponseFormatJSONSchema;
} & GraphAILLMInputBase;

type DeepSeekConfig = {
  baseURL?: string;
  apiKey?: string;
  stream?: boolean;
  forWeb?: boolean;
  model?: string;
};

type DeepSeekParams = DeepSeekInputs & DeepSeekConfig;

type DeepSeekResult = Record<string, any> | string;

const convToolCall = (
  tool_call: OpenAI.Chat.Completions.ChatCompletionMessageToolCall
) => {
  return {
    id: tool_call.id,
    name: tool_call.function.name,
    arguments: (() => {
      try {
        return JSON.parse(tool_call.function.arguments);
      } catch (__e) {
        console.log(__e);
        return undefined;
      }
    })(),
  };
};

const convertDeepSeekChatCompletion = (
  response: OpenAI.ChatCompletion,
  messages: OpenAI.ChatCompletionMessageParam[]
) => {
  const newMessage =
    response?.choices[0] && response?.choices[0].message
      ? response?.choices[0].message
      : null;
  const text = newMessage && newMessage.content ? newMessage.content : null;

  const functionResponses =
    newMessage?.tool_calls && Array.isArray(newMessage?.tool_calls)
      ? newMessage?.tool_calls
      : [];
  // const functionId = message?.tool_calls && message?.tool_calls[0] ? message?.tool_calls[0]?.id : null;

  const tool_calls = functionResponses.map(convToolCall);
  const tool = tool_calls && tool_calls.length > 0 ? tool_calls[0] : undefined;

  const message = (() => {
    if (newMessage) {
      const { content, role, tool_calls } = newMessage;
      if (tool_calls && tool_calls.length > 0) {
        return {
          content,
          role,
          tool_calls,
        };
      }
      return {
        content,
        role,
      };
    }
    return null;
  })();

  if (message) {
    messages.push(message);
  }
  return {
    ...response,
    text,
    tool,
    tool_calls,
    message,
    messages,
  };
};

export const deepSeekAgent: AgentFunction<
  DeepSeekParams,
  DeepSeekResult,
  DeepSeekInputs,
  DeepSeekConfig
> = async ({ filterParams, params, namedInputs, config }) => {
  const { verbose, system, temperature, max_tokens, prompt, messages } = {
    ...params,
    ...namedInputs,
  };

  const { apiKey, stream, forWeb, model } = {
    ...(config || {}),
    ...params,
  };

  const userPrompt = getMergeValue(
    namedInputs,
    params,
    "mergeablePrompts",
    prompt
  );
  const systemPrompt = getMergeValue(
    namedInputs,
    params,
    "mergeableSystem",
    system
  );

  const messagesCopy = getMessages<OpenAI.ChatCompletionMessageParam>(
    systemPrompt,
    messages
  );

  if (userPrompt) {
    messagesCopy.push({
      role: "user",
      content: userPrompt,
    });
  }

  if (verbose) {
    console.log(messagesCopy);
  }

  const deepseek = new OpenAI({
    apiKey,
    baseURL: "https://api.deepseek.com/v1",
    dangerouslyAllowBrowser: !!forWeb,
  });

  const modelName = model || "deepseek-chat";
  const chatParams: OpenAI.ChatCompletionCreateParams = {
    model: modelName,
    messages: messagesCopy as unknown as OpenAI.ChatCompletionMessageParam[],
    temperature: temperature ?? 0.7,
    max_tokens: max_tokens ?? 1024,
  };

  if (!stream) {
    const result = await deepseek.chat.completions.create(chatParams);
    return convertDeepSeekChatCompletion(result, messagesCopy);
  }

  const chatStream = deepseek.beta.chat.completions.stream({
    ...chatParams,
    stream: true,
  });

  // streaming
  for await (const message of chatStream) {
    const token = message.choices[0].delta.content;
    if (filterParams && filterParams.streamTokenCallback && token) {
      filterParams.streamTokenCallback(token);
    }
  }

  const chatCompletion = await chatStream.finalChatCompletion();
  return convertDeepSeekChatCompletion(chatCompletion, messagesCopy);
};

const deepseekAgentInfo: AgentFunctionInfo = {
  name: "deepseekAgent",
  agent: deepSeekAgent,
  mock: deepSeekAgent,
  inputs: {
    type: "object",
    properties: {
      model: { type: "string" },
      system: { type: "string" },
      tools: { type: "object" },
      tool_choice: {
        anyOf: [{ type: "array" }, { type: "object" }],
      },
      max_tokens: { type: "number" },
      verbose: { type: "boolean" },
      temperature: { type: "number" },
      baseURL: { type: "string" },
      apiKey: {
        anyOf: [{ type: "string" }, { type: "object" }],
      },
      stream: { type: "boolean" },
      prompt: {
        type: "string",
        description: "query string",
      },
      messages: {
        anyOf: [{ type: "string" }, { type: "object" }, { type: "array" }],
        description: "chat messages",
      },
    },
  },
  output: {
    type: "object",
    properties: {
      id: {
        type: "string",
      },
      object: {
        type: "string",
      },
      created: {
        type: "integer",
      },
      model: {
        type: "string",
      },
      choices: {
        type: "array",
        items: [
          {
            type: "object",
            properties: {
              index: {
                type: "integer",
              },
              message: {
                type: "array",
                items: [
                  {
                    type: "object",
                    properties: {
                      content: {
                        type: "string",
                      },
                      role: {
                        type: "string",
                      },
                    },
                    required: ["content", "role"],
                  },
                ],
              },
            },
            required: ["index", "message", "logprobs", "finish_reason"],
          },
        ],
      },
      usage: {
        type: "object",
        properties: {
          prompt_tokens: {
            type: "integer",
          },
          completion_tokens: {
            type: "integer",
          },
          total_tokens: {
            type: "integer",
          },
        },
        required: ["prompt_tokens", "completion_tokens", "total_tokens"],
      },
      text: {
        type: "string",
      },
      tool: {
        arguments: {
          type: "object",
        },
        name: {
          type: "string",
        },
      },
      message: {
        type: "object",
        properties: {
          content: {
            type: "string",
          },
          role: {
            type: "string",
          },
        },
        required: ["content", "role"],
      },
    },
    required: ["id", "object", "created", "model", "choices", "usage"],
  },
  params: {
    type: "object",
    properties: {
      model: { type: "string" },
      system: { type: "string" },
      tools: { type: "object" },
      tool_choice: { anyOf: [{ type: "array" }, { type: "object" }] },
      max_tokens: { type: "number" },
      verbose: { type: "boolean" },
      temperature: { type: "number" },
      baseURL: { type: "string" },
      apiKey: { anyOf: [{ type: "string" }, { type: "object" }] },
      stream: { type: "boolean" },
      prompt: { type: "string", description: "query string" },
      messages: {
        anyOf: [{ type: "string" }, { type: "object" }, { type: "array" }],
        description: "chat messages",
      },
    },
  },
  outputFormat: {
    llmResponse: {
      key: "choices.$0.message.content",
      type: "string",
    },
  },
  samples: [],
  description: "DeepSeek Agent",
  category: ["llm"],
  author: "",
  repository: "",
  license: "",
  stream: true,
  npms: [],
  environmentVariables: ["DEEPSEEK_API_KEY"],
};

export default deepseekAgentInfo;
