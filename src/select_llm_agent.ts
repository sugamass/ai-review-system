import { AgentFunction, AgentFunctionInfo } from "graphai";

export type LLMOption = {
  name: string;
  agentName: string;
  model: string;
  apiKey?: string;
};

const selectLLMAgent: AgentFunction = async ({ params, namedInputs }) => {
  const { llmOptions } = params;
  const { selectedLLMName } = namedInputs;

  const selectedLLM: LLMOption =
    llmOptions.find((llm: LLMOption) => llm.name === selectedLLMName) || null;

  const notSelectedLLM: LLMOption[] = llmOptions.filter(
    (llm: LLMOption) => llm.name !== selectedLLMName
  );

  return {
    selectedLLM,
    proofreadLLM_A: notSelectedLLM[0],
    proofreadLLM_B: notSelectedLLM[1],
  };
};

export const selectLLMAgentInfo: AgentFunctionInfo = {
  name: "selectLLMAgent",
  agent: selectLLMAgent,
  mock: selectLLMAgent,
  samples: [],
  description: "",
  category: [],
  author: "",
  repository: "",
  license: "",
};
