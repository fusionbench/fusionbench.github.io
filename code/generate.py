import argparse
import json
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal, Optional

from tqdm import tqdm

from fusionrag import FusionRAG
from llms import get_available_models, get_llm_chat
from utils import load_json, logger, restore_progress, sanitize_filename

def get_no_rag_prompt(question: str) -> str:
    prompt = f"""You are an expert in question answering.
Please respond with the exact answer only. If you are not sure, please reply "N/A".
Do not be verbose or provide extra information.
If there are multiple correct answers, please list them all.

Return the output strictly in JSON format like this:
{{    "generated_answers": {{
        "<entity name>": "<answer>",
        ...
    }}
}}

Question: {question}
Answer:"""
    return prompt

def get_rag_prompt(model: str, question: str, context: str) -> str:
    prompt = f"""You are given a question and a context.  
Answer for the given question using only the provided context.  
If different entities in the context provide different valid answers, include them all.  
If you cannot reason out the answer, output 'N/A'.  

Return the output strictly in JSON format like this:
{
    "generated_answers": {
        "<entity name>": "<answer>",
        ...
    }
}

Question: {question}
Context: {context}
Answer:"""
    return prompt

def process_single_question(qa_dict, ctx_name, pipeline, lock, fp, idx):
    """处理单个问题的线程函数"""
    try:
        # 获取歧义名, 歧义数据, 问题
        question = qa_dict.get("question")
        context = qa_dict.get(ctx_name)
        # 获取答案
        resp = pipeline.run(question, context)
        resp = json.loads(resp).get('final_answer', {}) if resp else {}
        resp = {} if resp == 'N/A' else resp

    except Exception as e:
        logger.error(f"问题 {idx} 发生错误, 将使用空答案. 错误信息: {e}")
        resp = {}
    
    qa_dict['generated_answers'] = resp

    # 删除多余字段, 保留问题 id、问题、真实答案、生成答案
    qa_dict.pop("grouped_context", None)
    qa_dict.pop("ungrouped_context", None)
    qa_dict.pop("retrieval_context", None)
    
    # 线程安全地写入文件
    with lock:
        fp.write(json.dumps(qa_dict, ensure_ascii=False) + "\n")
        fp.flush()
    
    return idx, qa_dict


def generate_answers(
    data_path: str,
    model: str,
    ctx_name: Literal['grouped_context', 'ungrouped_context', 'retrieval_context'],
    output_path: str,
    restore_path: Optional[str]=None,    
    use_think: bool=False,
    use_pipeline: bool=False,
    max_workers: int=4,
    **kwargs,
):
    """
    生成答案

    Args:
        model: 要测试的模型ID
        output_path: 结果保存路径
        last_output_path: 恢复进度的保存路径
        use_align: 是否使用对齐的上下文
        use_think: 是否启动推理模式
        use_pipeline: 是否使用 FusionPipeline
        max_workers: 最大线程数
    """
    qa_pairs = load_json(data_path)
    pipeline = FusionRAG(model)

    # 恢复进度
    if restore_path:
        logger.info(f"尝试从 {restore_path} 恢复进度")
        last_saved_idx = restore_progress(restore_path, output_path)
    else:
        last_saved_idx= -1

    # 创建线程锁用于文件写入
    lock = threading.Lock()
    
    # 边生成边写入
    fp = open(output_path, "a", encoding="utf-8")

    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交任务
        future_to_idx = {}
        for idx, qa_dict in enumerate(qa_pairs):
            if last_saved_idx >= idx:
                continue
            
            future = executor.submit(process_single_question, qa_dict, ctx_name, pipeline, lock, fp, idx)
            future_to_idx[future] = idx

        # 使用进度条跟踪完成情况
        completed_count = 0
        total_tasks = len(future_to_idx)
        
        with tqdm(total=total_tasks, desc="正在生成答案...") as pbar:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result_idx, result_dict = future.result()
                    completed_count += 1
                    pbar.update(1)
                except Exception as e:
                    logger.error(f"处理问题 {idx} 时发生异常: {e}")

    fp.close()
    logger.info(f"答案生成完成, 结果已保存至: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--qtype", choices=["multi", "single"], type=str, help="qtype")
    parser.add_argument("--model", choices=get_available_models(), type=str, required=True, help="使用的模型")
    parser.add_argument("--restore_path", type=str, default=None, help="上次的保存路径, 用于恢复进度")
    parser.add_argument("--under_rag", action="store_true", help="是否在 RAG 模式下开展任务")
    parser.add_argument("--use_align", action="store_true", help="OpenBook 模式下是否使用分组后的上下文, RAG 模式仅支持不分组的上下文")
    parser.add_argument("--use_think", action="store_true", help="是否启用推理模型的思考模式")
    parser.add_argument("--use_pipeline", action="store_true", help="是否启用 Piepeline")
    parser.add_argument("--max_workers", type=int, default=5, help="并行处理的最大线程数")

    args = parser.parse_args()
    logger.info(f"运行参数: {args.__dict__}")

    # 判断是否启用 FusionPipeline
    logger.info(f'是否使用 FusionPipeline 开展任务: {args.use_pipeline}')

    # 判断使用的问答数据集
    data_path = f'data/{args.qtype}-answer_pairs.json'
    logger.info(f"使用的问答数据集为: {data_path}, 类型为: {args.qtype} 实体问题")

    # 是否对齐
    logger.info(f"是否使用对齐的知识: {args.use_align}")

    # 判断使用的推理模式
    logger.info(f"是否使用思考模式: {args.use_think}")

    # 判断使用的问答上下文的字段
    if args.under_rag:
        ctx_name = 'retrieval_context'
    elif args.use_align: # Evidence-aware 有两类 Context
        ctx_name = 'grouped_context'
    else:
        ctx_name = 'ungrouped_context'
    logger.info(f'使用的问答上下文字段为: {ctx_name}')

    # 构造输出结果的保存路径
    if args.under_rag:
        # RAG 模式下保存到 rag 子目录
        output_dir = os.path.join("output/fusionrag", sanitize_filename(args.model), "rag")
    else:
        # 非 RAG 模式下保存到 evidence 子目录
        if args.qtype:
            # 如果指定了 qtype，保存到 qtype 子目录下
            output_dir = os.path.join("output/fusionrag", sanitize_filename(args.model), "evidence", args.qtype)
        else:
            # 如果没有指定 qtype，直接保存到 evidence 子目录
            output_dir = os.path.join("output/fusionrag", sanitize_filename(args.model), "evidence")
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"grpuped_{time.strftime('%Y%m%d_%H%M%S')}.jsonl" if args.use_align else f"ungrouped_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
    logger.info(f'问答结果将保存到: {output_path}')

    generate_answers(
        data_path=data_path,
        model=args.model,
        ctx_name=ctx_name,
        output_path=output_path,
        restore_path=args.restore_path,
        use_think=args.use_think,
        use_pipeline=args.use_pipeline,
        max_workers=args.max_workers
    )

    logger.info("结束")
    

if __name__ == "__main__":

    main()
