import { AgentFunction, AgentFunctionInfo } from "graphai";

// 入力値が数値でない場合の処理も必要
const averageScoreAgent: AgentFunction = async ({ namedInputs }) => {
  const { scores } = namedInputs;

  const scoreNumbers: number[] = [];
  scores.forEach((score: string) => {
    const scoreNum = Number(score);
    // 数値でない場合、無効とする
    // LLMが、例として「主観的な表現を含むため、評価が難しいですが、一般的な認識として 50 とします。」という出力をすることがあるため。
    if (!isNaN(scoreNum)) {
      scoreNumbers.push(scoreNum);
    }
  });

  const averageScore =
    scoreNumbers.reduce((acc, score) => acc + score, 0) / scoreNumbers.length;

  return {
    score: averageScore,
  };
};

export const averageScoreAgentInfo: AgentFunctionInfo = {
  name: "averageScoreAgent",
  agent: averageScoreAgent,
  mock: averageScoreAgent,
  samples: [],
  description: "",
  category: [],
  author: "",
  repository: "",
  license: "",
};
