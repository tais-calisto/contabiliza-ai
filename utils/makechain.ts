import { OpenAI } from 'langchain/llms/openai';
import { PineconeStore } from 'langchain/vectorstores/pinecone';
import { ConversationalRetrievalQAChain } from 'langchain/chains';

const CONDENSE_PROMPT = `
Pergunta: {question}`;

const QA_PROMPT = `Atue, responda, fale e interaja como um professor de contabilidade. Fale usando linguagem simples e assertiva. Se não for primeira pergunta, use as seguintes partes de contexto para responder à pergunta no final. Se você não sabe a resposta, apenas diga que não sabe. NÃO tente inventar uma resposta. Se a pergunta não estiver relacionada ao contexto, responda educadamente que você está sintonizado para responder apenas perguntas relacionadas ao contexto.

{context}

Question: {question}
Helpful answer in markdown:`;

export const makeChain = (vectorstore: PineconeStore) => {
  const model = new OpenAI({
    temperature: 0, // increase temepreature to get more creative answers
    modelName: 'gpt-3.5-turbo', //change this to gpt-4 if you have access
  });

  const chain = ConversationalRetrievalQAChain.fromLLM(
    model,
    vectorstore.asRetriever(),
    {
      qaTemplate: QA_PROMPT,
      questionGeneratorTemplate: CONDENSE_PROMPT,
      returnSourceDocuments: false, //The number of source documents returned is 4 by default
    },
  );
  return chain;
};
