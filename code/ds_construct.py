import argparse
import os
from textwrap import dedent
import time
import json
from typing import Optional
from tqdm import tqdm
from string import Template

from llm_service import get_llm_chat, get_available_models
from utils import logger, load_json, restore_progress


def concate_entity_information(info):
    from textwrap import dedent
    # 把 infobox 转成箭头连接的字符串
    infobox_str = ', '.join(f"{k} → {v}" for k, v in info['infobox'].items())
    # 把 triples 转成箭头连接的字符串列表
    info['triples'] = [f'{item[1]} → {item[2]}' for item in info['triples']]
    triples_str = ", ".join(info['triples'])    # 拼接 context 字符串
    return dedent(f"""\
        - Text: {info['description']}
        - Table: {infobox_str}
        - Triples: {triples_str}""")


class QuestionGenerator:
    """
    问题生成
    """

    def __init__(
            self, 
            data_path: str, 
            output_dir: str = "./output"
        ) -> None:
        """
        问题评判
        Args:
            data_path: 原始语料读取路径
            output_dir: 输出根目录
        """

        self.data_path = data_path
        self.output_dir = output_dir

        # 读入原始数据
        self.raw_data = load_json(self.data_path)
        os.makedirs(self.output_dir, exist_ok=True)

        self.multi = Template(dedent("""\
            You are an expert at creating question-answering tasks for ambiguous entities based on the provided data.  
            You are given an ambiguous name and a set of information about the ambiguous entity with different interpretations (including the text, table, and triple data for each entity).  

            Your task:  
            - Generate **exactly one standalone factual question** about an ambiguous entity, such that the question can be answered by **as many entities as possible (at least two)**, and their answers should be different if the data allows.  
            - The question must:  
            - Be based solely on verifiable information in the provided data.  
            - Include the ambiguous name explicitly.  
            - Not rely on opinions or hypotheticals.  
            - For each entity:  
            - Provide a concise answer if the data allows.  
            - If an entity cannot provide an answer to the generated question, output `"N/A"`.  
            - List all modalities (e.g., text, tables, triples) that were used as sources.  

            **Important:** You must rely solely on the provided data. Do not use any external knowledge.  

            Return the results strictly in the following JSON format:  
            {
            "question": "<generated natural-language question that includes '$ambiguous_name'>",
            "answers": [
                {
                "entity name": "<full name of the first entity>",
                "answer": "<answer or N/A>",
                "source": "<source list, e.g. ['text', 'triples'], or N/A>"
                },
                {
                "entity name": "<full name of the second entity>",
                "answer": "<answer or N/A>",
                "source": "<source list, e.g. ['text', 'tables'], or N/A>"
                }
            ]
            }

            Here is the data you need to process:  
            Ambiguous Name: $ambiguous_name
            $information

            Output:"""))

        self.single = Template(dedent("""\
            You are an expert in generating natural, fact-based questions for ambiguous entities.  
            You are given information about an ambiguous name, including the ambiguous name itself and detailed information about multiple possible entities.  
            Your task is to generate a **natural, fact-based question** that **can only** be answered by one of the entities.  

            ### Important rules:
            1. To simulate a user who is unaware that the name is ambiguous, you must **use the ambiguous name** when asking the question.  
            - Do **NOT** directly reveal the type, profession or year of the selected entity in parentheses (e.g., profession, film, year, ).  
            2. The question must be **natural, fact-driven, and specific**. It should not be a statement or a yes/no question.  
            3. The answer should have some **depth and direction**, and must **not** simply be one of the entity names provided in the context.  
            4. Do **NOT** generate questions that start with **"Which + ambiguous name"** (e.g., "Which $ambiguous_name...").  
            5. You must also provide an **explanation** describing:  
            - why the entity was chosen,  
            - why the question ensures uniqueness ,has depth and doesn't directly reveal the type, profession or year of the selected entity,  
            - and why the answer is correct while not being the entity name itself from the provided context or the ambiguous name.  

            The output must strictly follow this JSON format:

            {
            "explanation": "<why the output is according to the rules>",
            "selected_entity": "<entity name>",
            "question": "<a standalone, natural question must include '$ambiguous_name'>",
            "answer": "<short but fully answer the question answer(no long sentences)>",
            "modalities": ["<list of source modalities used (text, tables, triples)>"]
            }

            Here is the data you need to process:  
            Ambiguous name: $ambiguous_name 
            Information: $information
            Output:"""))


    def generate(self):
        pass

        


class QuestionJudger:

    def __init__(
        self,
        data_path: str,
        output_dir: str = 'output',
    ) -> None:
        """
        问题评判
        Args:
            data_path: 要评判的问题读取路径
            output_dir: 输出根目录
        """

        self.question_path = data_path
        self.output_dir = output_dir

        self.raw_data = load_json(self.data_path)

        # 检查问题使用的提示词
        self.judge_template = Template(dedent("""\
            You are an impartial annotator. You will receive a context in three formats: "text", "table" and "triple", a question and an answer. Your task is to judge whether the given answer is supported by provided context. You must only use provided context to make your judgment. Do not use outside knowledge.
            
            ## Output JSON Format
            {
                "label": true | false,
                "answer": "only filled when 'label' is false", 
                "source": ["list formats used"]
            }
            
            ## Rules
            1. If the answer is fully supported by provided context, output "label": true and "answer": "".
            2. If the answer is incorrect or not supported, output "label": false and provide the correct, short answer in field "answer".
            3. Additionally, if provided context is insufficient to answer the question, output "answer": "N/A".
            4. Field "source" should be a list indicating which formats are used to make the judgment.
            
            ## Input
            Context:
            $context
            Question: $question
            Answer: $answer

            ## Output\
        """))

    def judge_multi(
        self, 
        question_path: str, 
        last_output_path: Optional[str]=None
    ) -> str:
        """检查: 多实体问题, 共有属性"""
        qa_pairs = load_json(question_path)
        output_path = os.path.join(self.output_dir, "judged_" + time.strftime('%m%d_%H%M%S') + "_" + os.path.basename(question_path))
        logger.info("多实体问题检查结果保存路径: {output_path}")
        last_saved_idx = restore_progress(last_output_path, output_path)
        f = open(output_path, "a", encoding="utf-8")

        for idx, ambig_dict in enumerate(tqdm(qa_pairs, desc="正在检查多实体问题...")):
            if last_saved_idx >= idx:
                continue

            # 获取歧义名, 歧义数据, 问题
            ambig_name = ambig_dict["ambiguous_name"]
            ambig_data = self.raw_data.get(ambig_name)
            question = ambig_dict["question"]
            answers = []

            # 标注每个实体的答案
            for ent_dict in ambig_dict["answers"]:
                # {"entity name": "", "answer": "", "source": []}
                ent_data = ambig_data.get(ent_dict["entity_name"])
                if ent_data is None:
                    continue
                _context = concate_entity_information(ent_data)
                _answer =ent_dict.get("answer", "N/A")
                if _answer == "N/A":
                    _answer = "This question cannot be answered based on the provided context."
                prompt = self.judge_template.safe_substitute(
                    context=_context,
                    question=question,
                    answer=_answer
                )
                resp_dict = json.loads(get_llm_chat(prompt, json_format=True))

                # 标注为 true, 表示原有答案正确，保留原有答案
                if resp_dict.get('label').lower() == "yes":
                    resp_dict.pop("answer", None)
                if resp_dict.get('answer') == ent_dict.get("answer"):
                    resp_dict['label'] = "YES"
                ent_dict.update(resp_dict)
                # 如果答案为 N/A, 则清空来源
                if ent_dict.get("answer") == "N/A":
                    ent_dict['source'] = []

                answers.append(ent_dict)

            ambig_dict["answers"] = answers
            f.write(json.dumps(ambig_dict, ensure_ascii=False) + "\n")
        f.close()
        return output_path


    def judge_single(
        self, 
        question_path: str, 
        last_output_path: Optional[str]=None
    ):
        """检查: 单实体问题 - 实体独有属性"""
        qa_pairs = load_json(question_path)
        output_path = os.path.join(self.output_dir, "judged_" + time.strftime('%m%d_%H%M%S') + "_" + os.path.basename(question_path))
        logger.info("单一实体问题检查结果保存路径: {output_path}")
        last_saved_idx = self._load_last_output(last_output_path, output_path)
        f = open(output_path, "a", encoding="utf-8")

        for idx, dist_dict in enumerate(tqdm(qa_pairs, desc="正在检查单一实体问题...")):
            if last_saved_idx >= idx:
                continue

            # 获取歧义名, 歧义数据, 问题
            ambig_name = dist_dict["ambiguous_name"]
            ambig_data = self.raw_data.get(ambig_name)
            question = dist_dict["question"]
            selected_entity = dist_dict.get("selected_entity")

            # 所有实体的答案
            entity_answers = []
            # 全局标签
            global_label = True

            for ent_name, ent_data in ambig_data.items():
                _context = "Entity name:" + ent_name + "\n" + concate_entity_information(ent_data)
                _answer = "This question cannot be answered based on the provided context."
                
                # 如果是目标实体，则使用原始答案
                if ent_name == selected_entity:
                    _answer = dist_dict.get("answer")

                prompt = self.judge_template.safe_substitute(
                    context=_context,
                    question=question,
                    answer=_answer
                )

                resp_dict = json.loads(get_llm_chat(prompt, json_format=True))

                # 是目标实体, 给出了不同答案, 不管是否为空，都保存, 留待进一步检查
                if ent_name == selected_entity:
                    if resp_dict.get('label').lower() == "no":
                        global_label = False
                        entity_answers.append({
                            "entity_name": ent_name,
                            "answer": resp_dict.get('answer'),
                            "source": resp_dict.get('source', [])
                        })

                # 不是目标实体, 仍然有答案, 并且不为空
                elif resp_dict.get('label').lower() == "no" and resp_dict.get('answer') != "N/A":
                    entity_answers.append({
                        "entity_name": ent_name,
                        "answer": resp_dict.get('answer'),
                        "source": resp_dict.get('source', [])
                    })

            dist_dict["is_distinct"] = global_label
            dist_dict["other_answers"] = entity_answers
            dist_dict.pop("modalities", None)
            f.write(json.dumps(dist_dict, ensure_ascii=False) + "\n")
        f.close()
        return output_path


class QuestionAnswer:
    def __init__(
        self, 
        data_path: str = None, 
        output_dir: str = "./output"
    ) -> None:

        self.data_path = data_path
        self.output_dir = output_dir

        # 读入原始数据
        if self.data_path:
            self.raw_data = load_json(self.data_path)
        os.makedirs(self.output_dir, exist_ok=True)


        # 生成答案使用的提示词
        self.ansewr_template = Template(dedent("""\
            You are given a question and a context. Answer for the given question using only the provided context.  
            If different entities in the context provide different valid answers, include them all.  
            If you cannot reason out the answer, output "N/A".  

            Return the output strictly in JSON format like this:
            {
            "generated_answers": {
                "<entity name>": "<answer>",
                ...
            }
            }

            Question: $question
            Context: $context
            Answer:"""))
    
    
    def judge_multienity(
        self, 
        question_path: str, 
        last_output_path: Optional[str]=None
    ) -> str:
        """检查: 多实体问题, 共有属性"""
        ambiguous_questions = load_json(question_path)
        output_path = os.path.join(self.output_dir, "judged_" + time.strftime('%m%d_%H%M%S') + "_" + os.path.basename(question_path))
        logger.info("歧义性问题检查结果保存路径: {output_path}")
        last_saved_idx = self._load_last_output(last_output_path, output_path)
        f = open(output_path, "a", encoding="utf-8")

        for idx, ambig_dict in enumerate(tqdm(ambiguous_questions, desc="正在检查歧义问题...")):
            if last_saved_idx >= idx:
                continue

            # 获取歧义名, 歧义数据, 问题
            ambiguous_name = ambig_dict["ambiguous_name"]
            ambiguous_data = self.raw_data.get(ambiguous_name)
            question = ambig_dict["question"]
            answers = []

            # 标注每个实体的答案
            for ent_dict in ambig_dict["answers"]:
                # {"entity name": "", "answer": "", "source": []}
                ent_data = ambiguous_data.get(ent_dict["entity_name"])
                if ent_data is None:
                    continue
                _context = concate_entity_information(ent_data)
                _answer =ent_dict.get("answer", "N/A")
                if _answer == "N/A":
                    _answer = "This question cannot be answered based on the provided context."
                prompt = self.judge_template.safe_substitute(
                    context=_context,
                    question=question,
                    answer=_answer
                )
                resp_dict = json.loads(get_llm_chat(prompt, json_format=True))

                # 标注为 true, 表示原有答案正确，保留原有答案
                if resp_dict.get('label').lower() == "yes":
                    resp_dict.pop("answer", None)
                if resp_dict.get('answer') == ent_dict.get("answer"):
                    resp_dict['label'] = "YES"
                ent_dict.update(resp_dict)
                # 如果答案为 N/A, 则清空来源
                if ent_dict.get("answer") == "N/A":
                    ent_dict['source'] = []

                answers.append(ent_dict)

            ambig_dict["answers"] = answers
            f.write(json.dumps(ambig_dict, ensure_ascii=False) + "\n")
        f.close()
        return output_path

    def judge_singlentity(
        self, 
        question_path: str, 
        last_output_path: Optional[str]=None
    ):
        """检查: 单实体问题 - 实体独有属性"""
        distinct_questions = load_json(question_path)
        output_path = os.path.join(self.output_dir, "judged_" + time.strftime('%m%d_%H%M%S') + "_" + os.path.basename(question_path))
        logger.info("区分性问题检查结果保存路径: {output_path}")
        last_saved_idx = self._load_last_output(last_output_path, output_path)
        f = open(output_path, "a", encoding="utf-8")

        for idx, dist_dict in enumerate(tqdm(distinct_questions, desc="正在检查区分性问题...")):
            if last_saved_idx >= idx:
                continue

            # 获取歧义名, 歧义数据, 问题
            ambig_name = dist_dict["ambiguous_name"]
            ambig_data = self.raw_data.get(ambig_name)
            question = dist_dict["question"]
            selected_entity = dist_dict.get("selected_entity")

            # 所有实体的答案
            entity_answers = []
            # 全局标签
            global_label = True

            for ent_name, ent_data in ambig_data.items():
                _context = "Entity name:" + ent_name + "\n" + concate_entity_information(ent_data)
                _answer = "This question cannot be answered based on the provided context."
                
                # 如果是目标实体，则使用原始答案
                if ent_name == selected_entity:
                    _answer = dist_dict.get("answer")

                prompt = self.judge_template.safe_substitute(
                    context=_context,
                    question=question,
                    answer=_answer
                )

                resp_dict = json.loads(get_llm_chat(prompt, json_format=True))

                # 是目标实体, 给出了不同答案, 不管是否为空，都保存, 留待进一步检查
                if ent_name == selected_entity:
                    if resp_dict.get('label').lower() == "no":
                        global_label = False
                        entity_answers.append({
                            "entity_name": ent_name,
                            "answer": resp_dict.get('answer'),
                            "source": resp_dict.get('source', [])
                        })

                # 不是目标实体, 仍然有答案, 并且不为空
                elif resp_dict.get('label').lower() == "no" and resp_dict.get('answer') != "N/A":
                    entity_answers.append({
                        "entity_name": ent_name,
                        "answer": resp_dict.get('answer'),
                        "source": resp_dict.get('source', [])
                    })

            dist_dict["is_distinct"] = global_label
            dist_dict["other_answers"] = entity_answers
            dist_dict.pop("modalities", None)
            f.write(json.dumps(dist_dict, ensure_ascii=False) + "\n")
        f.close()
        return output_path
    

    def generate_answers(
        self, 
        question_path: str, 
        last_output_path: Optional[str]=None, 
        output_path: Optional[str]=None,
        model: Optional[str]=None,
        enable_thinking: bool=False,
        is_grouped: bool=True,
    ):
        """生成答案: 单一实体问题 - 实体独有属性"""
        dist_qas = load_json(question_path)
        output_path = output_path or os.path.join(self.output_dir, "distinct_answers.jsonl")
        logger.info(f"区分性问题答案生成结果保存路径: {output_path}")
        last_saved_idx = self._load_last_output(last_output_path, output_path)
        f = open(output_path, "a", encoding="utf-8")

        for idx, qa_dict in enumerate(tqdm(dist_qas, desc="正在生成区分性问题答案...")):
            if last_saved_idx >= idx:
                continue

            # 获取歧义名, 歧义数据, 问题
            ambig_name = qa_dict.get("ambiguous_name")
            question = qa_dict.get("question")
            context = qa_dict.get("grouped_context") if is_grouped else qa_dict.get("ungrouped_context")
            selected_entity = qa_dict.get("selected_entity")

            prompt = self.ansewr_template.safe_substitute(
                context=context,
                question=question,
            )
            try:
                resp_dict = json.loads(get_llm_chat(prompt, model, json_format=True, enable_thinking=enable_thinking))
            except Exception as e:
                logger.warning(f"LLM 调用错误, 使用空答案. 错误信息: {e}")
                resp_dict = {"generated_answers": {}}
            resp_dict = {
                "ambiguous_name": ambig_name,
                "question": question,
                "selected_entity": selected_entity,
                "generated_answers": resp_dict.get("generated_answers", {})
            }
            f.write(json.dumps(resp_dict, ensure_ascii=False) + "\n")
            time.sleep(0.1)

        f.close()
        return output_path

def main():
    parser = argparse.ArgumentParser(description="多来源问题生成、模态检查、答案检查管道")

    parser.add_argument("--mode", choices=["jmulti", "jsingle", "answer"], default=None, help="运行模式")
    parser.add_argument("--data_path", default="data/wikipedia.json", type=str, help="原始 Wikipedia 数据路径")
    parser.add_argument("--question_path", default="data/distinguish_questions_v4.jsonl", type=str, help="待检查的问题的路径")
    # parser.add_argument("--qa_path", default="output_qa/rag_qa_distinct.json", type=str, help="待生成答案的问题路径")
    parser.add_argument("--output_dir", default="output_answer_distinct", type=str, help="输出目录")
    parser.add_argument("--last_output_path", default=None, type=str, help="上次保存的输出路径, 用于恢复进度, 可用于生成问题、检查模态")
    parser.add_argument("--model", default=None, type=str, help="使用的模型")
    parser.add_argument("--grouped", action="store_true", help="是否使用分组后的上下文")
    parser.add_argument("--enable_thinking", action="store_true", help="是否启用大模型的思考模式")

    args = parser.parse_args()

    logger.info(f"运行参数: {args.__dict__}")

    # 初始化 Pipeline
    pipeline = QuestionAnswer(
        data_path=args.data_path,
        output_dir=args.output_dir,
    )

    # 检查模型是否提供
    if not args.model or not args.model in get_available_models():
        raise ValueError(f"请通过 `--model` 指定模型, 可选: {', '.join(get_available_models())}")

    os.makedirs(args.output_dir, exist_ok=True)

    match args.mode:
        case "jmulti":
            logger.info("检查多实体问题的质量...")
            if not args.question_path:
                logger.error("`--question_path` is required for judge mode")
                return
            judge_file = pipeline.judge_multienity(args.question_path, args.last_output_path)

            logger.info(f"Question judge completed and saved to: {judge_file}")
        case "jsingle":
            logger.info("检查单实体问题的质量...")
            if not args.question_path:
                logger.error("`--question_path` is required for judge mode")
                return
            judge_file = pipeline.judge_singlentity(args.question_path)
            logger.info(f"Question judge completed and saved to: {judge_file}")

        case "answer":
            logger.info("生成问题的答案...")
            if not args.question_path:
                logger.error("`--question_path` is required for answer generation mode")
                return
            
            _output_path = os.path.join(
                args.output_dir, 
                args.model + "_" + time.strftime("%Y%m%d-%H%M%S") + ("" if args.grouped else "ungrouped") + ".jsonl"
            )

            answer_file = pipeline.generate_answers(
                args.qa_path,
                args.last_output_path, 
                output_path=_output_path, 
                model=args.model,
                is_grouped=args.grouped,
                enable_thinking=args.enable_thinking,
            )
            logger.info(f"答案生成完成, 结果已保存至: {answer_file}")

        case _:
            logger.error(f"不支持的运行模式: {args.mode}")
    
    logger.info("任务执行完成.")


if __name__ == "__main__":
    """
    使用分组后的上下文生成区分性问题答案:
    python ambigmm.py --mode answer --model XXX --grouped

    不使用分组后的上下文生成区分性问题答案:
    python ambigmm.py --mode answer --model XXX

    从上次保存的结果继续生成:
    python ambigmm.py --mode answer --model XXX --last_output_path /path/to/last_output.jsonl

    """
    main()
