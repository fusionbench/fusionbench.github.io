import os
import json
import random

def load_json(filepath):
    """
    读取 .json 或 .jsonl 文件
    :param filepath: 文件路径
    :return: Python 对象(list 或 dict)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件未找到: {filepath}")

    _, ext = os.path.splitext(filepath)
    ext = ext.lower()

    with open(filepath, "r", encoding="utf-8") as f:
        if ext == ".json":
            return json.load(f)
        elif ext == ".jsonl":
            return [json.loads(line) for line in f.readlines() if line.strip()]
        else:
            raise ValueError(f"不支持的文件类型: {ext}")
        

def save_json(data, filepath):
    """
    保存数据到 .json 文件
    :param data: Python 对象(list 或 dict)
    :param filepath: 文件路径
    """
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    return filepath


def convert_json_to_jsonl(input_dir, output_dir='temp'):
    """将 JSON 文件转换为 JSONL 文件"""

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):

        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename.replace(".json", ".jsonl"))

        data = load_json(input_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"转换完成: {input_path} -> {output_path}")


def process_ambiguous_questions(data_path, ambiguous_filepath, judged_filepath):
    """ 
    处理歧义性问题的结果，统计 YES 和 NO 的数量，并保存 NO 的相关信息到 CSV 文件
    
    Args:
        data_path (str): 原始数据文件路径，包含实体的详细信息
        ambiguous_filepath (str): 包含歧义性问题及其答案的文件路径
        judged_filepath (str): 包含判断结果的文件路径
    
    """
    raw_data = load_json(data_path)
    ambiguous_questions = load_json(ambiguous_filepath)
    judged_questions = load_json(judged_filepath)
    
    # 统计数量
    num_questions = 0
    num_yes = 0
    num_no = 0
    num_no_origin_unknown = 0  # 原始答案是 N/A 的 NO 数量
    num_no_judged_unknown = 0         # 判断后答案是 N/A 的 NO 数量

    # 存储 label_no 的相关信息
    import pandas as pd
    # 列：歧义名, 问题, 实体名, 答案
    label_no_df = pd.DataFrame(columns=['ambiguous_name', 'question', 'entity_name', 'origin_answer', 'judged_answer'])

    for ambig_group, judged_group in zip(ambiguous_questions, judged_questions):
        ambig_name = ambig_group['ambiguous_name']
        assert ambig_name == judged_group['ambiguous_name'], f"歧义名不匹配: {ambig_name} != {judged_group['ambiguous_name']}"
        
        for origin_dict, judged_dict in zip(ambig_group['answers'], judged_group["answers"]):
            num_questions += 1
            
            if judged_dict['label'] == 'YES':
                num_yes += 1
            
            elif judged_dict['label'] == 'NO':
                num_no += 1

                # 获取实体的详细信息
                entity_name = judged_dict['entity_name']
                entity_data = raw_data.get(ambig_name).get(entity_name)

                origin_answer = origin_dict['answer']
                judged_answer = judged_dict['answer']

                if origin_answer == 'N/A':
                    num_no_origin_unknown += 1
                    origin_answer = 'UNANSWERABLE'
                if judged_answer == 'N/A':
                    num_no_judged_unknown += 1
                    judged_answer = 'UNANSWERABLE'
                
                # 追加到 DataFrame 中
                label_no_df = label_no_df._append({
                    'ambiguous_name': ambig_name,
                    'question': judged_group['question'],
                    'entity_name': entity_name,
                    'origin_answer': origin_answer,
                    'judged_answer': judged_answer,
                    'context': concate_entity_information(entity_data)
                }, ignore_index=True)

    # 保存 label_no 的相关信息到 CSV 文件
    os.makedirs("output_stats", exist_ok=True)
    label_no_df.to_csv(f"output_stats/{ambiguous_filepath.split('/')[-1].replace('.jsonl', '_label_no.csv')}", index=False, encoding='utf-8-sig')
    
    print(f"总共有 {len(judged_questions)} 个歧义性问题。")
    print(f"总共有 {num_questions} 个答案。")
    print(f"- 其中 YES 有 {num_yes} 个，占比 {num_yes / num_questions:.2%}")
    print(f"- 其中 NO 有 {num_no} 个，占比 {num_no / num_questions:.2%}")
    print(f"  - 其中原始答案是 N/A 的有 {num_no_origin_unknown} 个，占比 {num_no_origin_unknown / num_no:.2%}")
    print(f"  - 其中判断后答案是 N/A 的有 {num_no_judged_unknown} 个，占比 {num_no_judged_unknown / num_no:.2%}")


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


def process_distinct_questions(data_path, judged_filepath):
    """
    处理区分性问题的判断结果，统计 is_distinct 为 True 和 False 的数量，并保存 False 的相关信息到 CSV 文件
    """
    raw_data = load_json(data_path)
    judged_questions = load_json(judged_filepath)

    # 统计数量
    num_questions = 0
    num_true = 0
    num_false = 0

    # 存储 is_distinct 为 False 的相关信息
    import pandas as pd
    false_df = pd.DataFrame(columns=['ambiguous_name', 'question', 'selected_entity', 'answer', 'other_answers'])

    for ques_dict in judged_questions:
        ambig_name = ques_dict.get('ambiguous_name')
        question = ques_dict.get('question')

        num_questions += 1

        if ques_dict.get('is_distinct') == True:
            num_true += 1
        else:
            num_false += 1

            answer = ques_dict.get('answer', 'UNKNOWN')
            other_answers = "\n\n".join(
                [
                    f"{idx + 1}. {item['entity_name']}\n- Answer: {item['answer']}\n{concate_entity_information(raw_data.get(ambig_name).get(item['entity_name']))}" 
                    for idx, item in enumerate(ques_dict.get('other_answers', []))
                ]
            )

            # 追加到 DataFrame 中
            false_df = false_df._append({
                'ambiguous_name': ambig_name,
                'question': question,
                'selected_entity': ques_dict.get('selected_entity'),
                'answer': answer,
                'other_answers': other_answers
            }, ignore_index=True)

    # 保存 is_distinct 为 False 的相关信息到 CSV 文件
    os.makedirs("output_stats", exist_ok=True)
    false_df.to_csv(f"output_stats/{judged_filepath.split('/')[-1].replace('.jsonl', '_is_distinct_false.csv')}", index=False, encoding='utf-8-sig')
    print(f"总共有 {len(judged_questions)} 个区分性问题。")
    print(f"总共有 {num_questions} 个答案。")
    print(f"- 其中 is_distinct 为 True 有 {num_true} 个，占比 {num_true / num_questions:.2%}")
    print(f"- 其中 is_distinct 为 False 有 {num_false} 个，占比 {num_false / num_questions:.2%}")



def clear_ambiguous_questions(data_path, judged_filepath):
    """删除歧义性问题中被标记为 NO 的实体, 并保存清理后的结果"""
    raw_data = load_json(data_path)
    judged_questions = load_json(judged_filepath)
    clean_questions = []

    for ques_dict in judged_questions:
        ambig_name = ques_dict.get('ambiguous_name')
        question = ques_dict.get('question')
        ambig_data = raw_data.get(ambig_name)

        new_answers = []
        for answer_dict in ques_dict.get('answers', []):
            if answer_dict.get('label') == 'YES':
                ent_name = answer_dict.get('entity_name')
                ent_data = ambig_data.get(ent_name)
                new_answers.append({
                    'entity': ent_name,
                    'answer': answer_dict.get('answer'),
                    'text': ent_data.get('description'),
                    'table': ent_data.get('infobox'),
                    'triple': ent_data.get('triples'),
                })

        clean_questions.append({
            'ambiguous_name': ambig_name,
            'question': question,
            'entities': new_answers
        })

    os.makedirs("output_cleaned", exist_ok=True)
    with open(f"output_cleaned/{judged_filepath.split('/')[-1].replace('.jsonl', '_cleaned.json')}", 'w', encoding='utf-8') as f:
        json.dump(clean_questions, f, ensure_ascii=False, indent=4)


def clear_distinct_questions(data_path, judged_filepath):
    """删除区分性问题中被标记为 is_distinct 为 False 的问题, 并保存清理后的结果"""
    raw_data = load_json(data_path)
    judged_questions = load_json(judged_filepath)
    clean_questions = []

    for ques_dict in judged_questions:
        ambig_name = ques_dict.get('ambiguous_name')
        question = ques_dict.get('question')
        ambig_data = raw_data.get(ambig_name)
        selected_entity = ques_dict.get('selected_entity')
        
        if not ambig_data:
            continue

        if ques_dict.get('is_distinct') and selected_entity in list(ambig_data.keys()):
            entities = []
            for ent_name, ent_data in ambig_data.items():
                entities.append({
                    'entity': ent_name,
                    'text': ent_data.get('description'),
                    'table': ent_data.get('infobox'),
                    'triple': ent_data.get('triples'),
                })

            clean_questions.append({
                'ambiguous_name': ambig_name,
                'question': question,
                'answer': ques_dict.get('answer'),
                'selected_entity': selected_entity,
                'entities': entities,
            })

    output_path = f"output_cleaned/{judged_filepath.split('/')[-1].replace('.jsonl', '_cleaned.json')}"
    os.makedirs("output_cleaned", exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(clean_questions, f, ensure_ascii=False, indent=4)

    print(f"清理完成，共有{len(judged_questions)} 个区分性问题, 保留了 {len(clean_questions)} 个区分性问题。")
    print(f"保存路径: {output_path}")



def convert_dict_to_markdown_table(data: dict) -> str:
    """
        将字典转换为 Markdown 格式的表格
        :param data: dict, key-value 对
        :return: str, markdown 格式表格
    """
        # 表头
    headers = ["Field", "Value"]
    markdown = f"| {headers[0]} | {headers[1]} |\n"
    markdown += f"|{'-' * (len(headers[0]) + 2)}|{'-' * (len(headers[1]) + 2)}|\n"

        # 表格内容
    for key, value in data.items():
        markdown += f"| {key} | {value} |\n"

    return markdown
    

def convert_triples_to_text(triples):
    """
        将三元组转换为文本
        A -- B --> C
    """
    return "\n".join([f"{s} -- {p} --> {o}" for s, p, o in triples])


def construct_ambiguous_qa(ambiguous_clean_filepath):
    """构造上下文问题"""

    ambiguous_questions = load_json(ambiguous_clean_filepath)

    new_qas = []

    for ques_dict in ambiguous_questions:
        ambig_name = ques_dict.get('ambiguous_name')
        question = ques_dict.get('question')
        
        # List[Dict[str, str]]
        grouped_contexts = []
        answers = []
        ungrouped_contexts = []
        
        for entity_dict in ques_dict.get('entities'):
            entity_name = entity_dict.get('entity')
            text = entity_dict.get('text', '')
            table = convert_dict_to_markdown_table(entity_dict['table'])
            triple = convert_triples_to_text(entity_dict['triple'])
            
            grouped_contexts.append(f"{entity_name}\n:{text}\n{table}\n{triple}")

            ungrouped_contexts += [
                text,
                entity_name + "\n" + table,
                triple,
            ]

            answers.append({
                'entity': entity_name,
                'answer': entity_dict['answer'],
            })
        
        # shuffle ungrouped_contexts
        random.shuffle(ungrouped_contexts)

        new_qas.append({
            'ambiguous_name': ambig_name,
            'question': question,
            'answers': answers,
            'grouped_context': "\n\n".join(grouped_contexts),
            'ungrouped_context': "\n\n".join(ungrouped_contexts),
        })


    os.makedirs("output_qa", exist_ok=True)
    with open(f"output_qa/{ambiguous_clean_filepath.split('/')[-1].replace('.json', '_context_qa.json')}", 'w', encoding='utf-8') as f:
        json.dump(new_qas, f, ensure_ascii=False, indent=4)



def construct_distinct_qa(distinct_clean_filepath):
    """构造上下文问题"""

    distinct_questions = load_json(distinct_clean_filepath)

    new_qas = []

    for ques_dict in distinct_questions:
        ambig_name = ques_dict.get('ambiguous_name')
        question = ques_dict.get('question')
        
        grouped_contexts = []
        ungrouped_contexts = []

                
        for entity_dict in ques_dict.get('entities'):
            entity_name = entity_dict.get('entity')
            text = entity_dict.get('text')
            table = convert_dict_to_markdown_table(entity_dict['table'])
            triple = convert_triples_to_text(entity_dict['triple'])
            
            grouped_contexts.append(f"{entity_name}\n:{text}\n{table}\n{triple}")

            ungrouped_contexts += [
                text,
                entity_name + "\n" + table,
                triple,
            ]
        
        # shuffle ungrouped_contexts
        random.shuffle(ungrouped_contexts)

        new_qas.append({
            'ambiguous_name': ambig_name,
            'question': question,
            'answer': ques_dict.get('answer'),
            'selected_entity': ques_dict.get('selected_entity'),
            'grouped_context': "\n\n".join(grouped_contexts),
            'ungrouped_context': "\n\n".join(ungrouped_contexts),
        })

    output_path = f"output_qa/{distinct_clean_filepath.split('/')[-1].replace('.json', '_context_qa.json')}"
    os.makedirs("output_qa", exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_qas, f, ensure_ascii=False, indent=4)

    print(f"构造完成，共有{len(new_qas)} 个区分性问题。")
    print(f"保存路径: {output_path}")


def clear_unvliad_ambiguous_answers():
    """使用更新的数据清洗掉不合法的歧义性问题答案"""
    valid_ambiguous_names = load_json("data/distinct_valid_ambiguous_names.json")
    
    # 列出目录下的所有文件 "output_answer_distinct"
    input_dir = "temp"
    output_dir = "temp2"
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename.replace(".jsonl", ".json"))

        data = load_json(input_path)
        cleaned_data = []

        for item in data:
            ambig_name = item.get('ambiguous_name')
            if ambig_name in valid_ambiguous_names:
                cleaned_data.append(item)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, ensure_ascii=False, indent=4)
        
        print(f"清理完成: {filename}, 原始数量: {len(data)}, 清理后数量: {len(cleaned_data)}")
    

def count_multianswers(input_dir):
    """统计每个文件中的答案数量, 将大于1的问题记录下来"""
    total_results = {}

    for filename in os.listdir(input_dir):
        _, ext = os.path.splitext(filename)
        if ext.lower() not in ['.json', '.jsonl']:
            continue
        
        input_path = os.path.join(input_dir, filename)
        data = load_json(input_path)
        print(f"处理文件: {filename}, 共 {len(data)} 条数据")

        for item in data:
            ambig_name = item['ambiguous_name']
            generated_answers = item.get('generated_answers', {})
            # print(f"{ambig_name}: {generated_answers}")
            # 排除掉 N/A
            answers = []
            if not isinstance(generated_answers, dict):
                continue
            for ans in list(generated_answers.values()):
                if isinstance(ans, str):
                    ans = ans.strip()
                    if ans and ans.upper() != 'N/A':
                        answers.append(ans)
            if(len(answers) > 1):
                # 统计该 ambig_name 的数量
                if ambig_name not in total_results.keys():
                    total_results[ambig_name] = {
                        'ambiguous_name': ambig_name,
                        'question': item['question'],
                        'selected_entity': item.get('selected_entity', ''),
                        'count': 1,
                    }
                else:
                    total_results[ambig_name]['count'] += 1

    # 统计一个各个count数量的分布
    # count_distribution = {}
    # for ambig_name, info in total_results.items():
    #     count = info['count']
    #     if count not in count_distribution:
    #         count_distribution[count] = 1
    #     else:
    #         count_distribution[count] += 1
    # print("答案数量分布:")
    # for count, num in sorted(count_distribution.items()):
    #     print(f"{count}: {num} 个")
    # 按照 count 降序排列
    # total_results = dict(sorted(total_results.items(), key=lambda x: x[1]['count'], reverse=True))
    # total_results = list(total_results.values())
    # 将分布写入文件
    # dist_path = 'temp/distincts_multiple_answers_distribution.json'
    # with open(dist_path, 'w', encoding='utf-8') as f:
    #     json.dump(count_distribution, f, ensure_ascii=False, indent=4)
    

    # 保存结果
    output_path = 'temp/distinct_multiple_answers.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(total_results, f, ensure_ascii=False, indent=4)
    print(f"统计完成, 共有{len(total_results)}个问题得到了大于1个答案, 保存路径: {output_path}")


def convert_rag():
    print("开始清理数据...")

    # count_multianswers("output_answer_distinct_cleaned")

    # data = load_json("data/filtered_distinct_questions.json")
    # ambig_names = [item['ambiguous_name'] for item in data]
    # with open("data/distinct_valid_ambiguous_names.json", 'w', encoding='utf-8') as f:
    #     json.dump(ambig_names, f, ensure_ascii=False, indent=4)
    # clear_unvliad_ambiguous_answers()

    # 获取chunk的ID2Content映射
    data = load_json("data/rag/10times_merged_corpus.json")
    chunkid2content = {item['id']: item['text'] for item in data}
    print(f"获取chunk的ID2Content映射完成, 共 {len(chunkid2content)} 条数据")

    # 获取实体名到实体id的映射
    entityname2id = load_json("data/rag/entity_id_mapping.json")
    entityid2name = {v: k for k, v in entityname2id.items()}

    # 获取问题召回结果映射(entityid -> List[chunkid])
    data = load_json("data/rag/result_bge_2.json")
    entityid2chunkids = {}
    for entityid, retrievals in data.items():
        # for chunkid, score in retrievals.items():
        #     if entityid not in entityid2chunkids:
        #         entityid2chunkids[entityid] = [chunkid]
        #     else:
        #         entityid2chunkids[entityid].append(chunkid)
        entityid2chunkids[entityid] = list(retrievals.keys())
        # print(entityid, entityid2chunkids[entityid])
    print(f"获取问题召回结果映射完成, 共 {len(entityid2chunkids)} 条数据")
    
    # 构建 entity2context 映射
    entityname2context = {}
    for entityid, chunkids in entityid2chunkids.items():
        entity_name = entityid2name.get(entityid, None)
        # print(entityid, entity_name)
        if not entity_name:
            print(f"警告: 实体ID {entityid} 未找到对应的实体名称!")
            continue
        entityname2context[entity_name] = "\n".join([chunkid2content[chunkid] for chunkid in chunkids])
        # print(f"实体: {entity_name}, 上下文: {entityname2context[entity_name]}")
        # print(chunkids)
    print(f"构建 entity2context 映射完成, 共 {len(entityname2context)} 条数据")

    
    # 读取 qa 数据
    data = load_json("data/rag/question_entity_mapping.json")
    rag_qa = []
    for question, ques_dict in data.items():
        ambig_name = ques_dict['ambiguous_name']
        entity_name = ques_dict['selected_entity']
        context = entityname2context.get(entity_name, "")
        if not context:
            print(f"警告: 实体 {entity_name} 没有找到对应的上下文!")
            continue
        rag_qa.append({
                'ambiguous_name': ambig_name,
                'question': question,
                'answer': ques_dict['answer'],
                'selected_entity': entity_name,
                'grouped_context': context,
            })
    # 前后数量比较
    print(f"共有 {len(data)} 个问题, 构建后共有 {len(rag_qa)} 个问题。")

    save_json(rag_qa, "output_qa/rag_qa_distinct2.json")



def count_sources():
    print("统计各类modalities类型的使用情况")

    valid_names = load_json("data/name/qa_names_single.json")

    qas = load_json("data/temp/single.jsonl")
    print(len(qas))

    qas = [item for item in qas if item['ambiguous_name'] in valid_names]
    print(len(qas))

    # 统计每个 modalities 类型在 qas 中的出现频率
    modality_counts = {"text": 0, "table": 0, "triples": 0}
    total_items = 0

    for item in qas:
        modalities = item.get("modalities", [])
        total_items += 1
        if "text" in modalities:
            modality_counts["text"] += 1
        if "table" in modalities:
            modality_counts["table"] += 1
        if "triples" in modalities:
            modality_counts["triples"] += 1

    if total_items > 0:
        print("\nmodalities字段中各类型出现的百分比：")
        print(f"text型占比：{modality_counts['text'] / total_items * 100:.2f}%")
        print(f"table型占比：{modality_counts['table'] / total_items * 100:.2f}%")
        print(f"triples型占比：{modality_counts['triples'] / total_items * 100:.2f}%")
        print(f"问题总数量: {total_items}")
    else:
        print("没有有效的数据，无法统计。")


def list_json_and_jsonl_files(directory):
    """
    列出指定目录下所有 .json 和 .jsonl 文件（不遍历子目录）
    返回这些文件的绝对路径列表
    """
    files = []
    for fname in os.listdir(directory):
        # 只看本目录文件
        path = os.path.join(directory, fname)
        if os.path.isfile(path):
            if fname.lower().endswith('.json') or fname.lower().endswith('.jsonl'):
                files.append(path)
    return files


# from pipeline import QUESTION_ANSWERING, KNOWLEDGE_GROUPING


# if __name__ == '__main__':
# # def construct_batch_tasks():

#     # 示例：
#     # {"custom_id": "request-3", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "qwen-max", "messages": [{"role": "system", "content": "你是计算小助手."}, {"role": "user", "content": "1+3等于几?"}]}}

#     mode = "multi"
#     # data = load_json(f"data/{mode}_entity_pairs.json")
#     data = load_json(f"batch/rag_{mode}new.json")
#     out = []
#     model = "qwen-plus-latest"

#     for idx, qa_dict in enumerate(data):
#         question = qa_dict["question"]
#         context = qa_dict["ctx"]
#         prompt = QUESTION_ANSWERING.safe_substitute(
#             context=context,
#             question=question,
#         )
#         # prompt = KNOWLEDGE_GROUPING.safe_substitute(context=context)

#         # 构造消息内容，可以自定义你的 system prompt
#         messages = [
#             {"role": "system", "content": "你是问题回复小助手."},
#             {"role": "user", "content": prompt}
#         ]
#         req = {
#             "custom_id": qa_dict.get('ambiguous_name'),
#             "method": "POST",
#             "url": "/v1/chat/completions",
#             "body": {
#                 "model": model,
#                 "messages": messages,
#                 # "response_format": {"type": "json_object"},
#                 # "extra_body": {"enable_thinking": True},
#             }
#         }
#         out.append(req)

#     # 保存批量推理数据集
#     with open(f"batch/{model}_rag_{mode}.jsonl", "w", encoding="utf-8") as fout:
#         for item in out:
#             fout.write(json.dumps(item, ensure_ascii=False) + "\n")





# if __name__ == '__main__':

#     mode = "single"
#     mode = "multi"
#     files = list_json_and_jsonl_files(f'batch/{mode}')

#     for filepath in files:
#         raw_qa_pairs = load_json(f'data/{mode}_entity_pairs.json')

#         print(filepath)
#         resps = load_json(filepath)
#         resps_dict = {}
#         cnt = 0
#         for r in resps:
#             try:
#                 content = json.loads(r['response']['body']['choices'][0]['message']['content']).get('generated_answers', {})
#                 resps_dict[r['custom_id']] = content
#                 # print(content[:1000])
#             except Exception as e:
#                 # print(r['custom_id'])
#                 cnt += 1
#                 print(e)
#         print(f"错误数量: {cnt}")
            

#         out = []
#         for p in raw_qa_pairs:
#             # question = p["question"]
#             # context = resps_dict.get(p['ambiguous_name'], p['ungrouped_context'])
#             # prompt = QUESTION_ANSWERING.safe_substitute(
#             #     question=question,
#             #     context=context,
#             # )
#             # prompt = KNOWLEDGE_GROUPING.safe_substitute(context=context)

#             # 构造消息内容，可以自定义你的 system prompt
#             # messages = [
#             #     {"role": "system", "content": "你是问题回复小助手."},
#             #     {"role": "user", "content": prompt}
#             # ]
#             # req = {
#             #     "custom_id": p.get('ambiguous_name'),
#             #     "method": "POST",
#             #     "url": "/v1/chat/completions",
#             #     "body": {
#             #         "model": "qwen-plus-latest",
#             #         "messages": messages,
#             #         "response_format": {"type": "json_object"},
#             #         # "extra_body": {"enable_thinking": True},
#             #     }
#             # }
#             p['generated_answers'] = resps_dict.get(p['ambiguous_name'], {})
#             p.pop("grouped_context", None)
#             p.pop("ungrouped_context", None)
#             p.pop("retrieval_context", None)

#             out.append(p)

#         save_json(out, filepath.replace(".jsonl", "_batch.json"))

#         # with open(filepath.replace(".jsonl", "_batch.jsonl"), "w", encoding="utf-8") as fout:
#         #     for item in out:
#         #         fout.write(json.dumps(item, ensure_ascii=False) + "\n")


def convert_jsonl_to_json(input_dir):
    """将 JSONL 文件转换为 JSON 文件"""
    for filename in os.listdir(input_dir):
        if filename.endswith(".jsonl"):
            data = load_json(os.path.join(input_dir, filename))
            save_json(data, os.path.join(input_dir, filename.replace(".jsonl", ".json")))
            print(f"转换完成: {filename}")

            

convert_jsonl_to_json("output/glm-4-air-250414")